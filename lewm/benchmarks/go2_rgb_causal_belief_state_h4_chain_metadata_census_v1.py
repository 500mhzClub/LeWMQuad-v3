"""Stdlib contract for the recurrent-H4 development-metadata census.

This module is source-only. Importing it opens no files.  The scientific unit
is six exact pair edges: two history transitions followed by four predicted
future transitions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


PREREGISTRATION = {
    "path": (
        "docs/lewm_go2_rgb_causal_belief_state_h4_chain_metadata_census_"
        "preregistration_2026-07-27.md"
    ),
    "commit": "3795cb60e6a14cbb4d8236cb8da386e0c6ef1126",
    "file_sha256": "1fde9198e1221a69167bd37be4c9dc91a26f8bb8e19c949b119f02201f6d98f2",
    "byte_count": 16_646,
}
AUTHORIZATION_PATH = (
    "docs/lewm_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1_"
    "execution_authorization_2026-07-27.json"
)
REVIEW_PATH = (
    "docs/lewm_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1_"
    "source_review_2026-07-27.json"
)

RAW_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
MANIFEST_PATH = f"{RAW_ROOT}/manifest.json"
PAIRS_PATH = f"{RAW_ROOT}/pairs.jsonl"
OUTPUT_PATH = (
    ".generated/go2_rgb_causal_belief_state_h4_chain_metadata_census_v1/"
    "receipt.json"
)
MANIFEST_FILE_SHA256 = (
    "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360"
)
MANIFEST_CONTENT_SHA256 = (
    "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a"
)
MANIFEST_BYTE_COUNT = 311_598
PAIRS_FILE_SHA256 = (
    "5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d"
)
PAIRS_BYTE_COUNT = 6_207_286
PAIR_COUNT = 5_172
ORDERED_PAIR_CONTENT_SHA256 = (
    "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
)

ROLES = ("train", "checkpoint_selection", "probability_calibration")
ELIGIBLE_ROLES = ("train", "checkpoint_selection")
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
PRIMITIVES = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
ROLE_COUNTS = {
    "train": {"pair_count": 4_262, "unique_endpoint_count": 7_777, "scene_count": 72},
    "checkpoint_selection": {
        "pair_count": 495,
        "unique_endpoint_count": 924,
        "scene_count": 8,
    },
    "probability_calibration": {
        "pair_count": 415,
        "unique_endpoint_count": 759,
        "scene_count": 8,
    },
}
PAIR_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_pair_v1"
PAIR_FIELDS = frozenset(
    {
        "schema",
        "dataset_role",
        "global_row",
        "scene_id",
        "family",
        "episode_id",
        "env_index",
        "reset_count",
        "source_split",
        "frames_jsonl_sha256",
        "scene_manifest_sha256",
        "primitive",
        "relative_se2_current_frame",
        "current_endpoint_sha256",
        "next_endpoint_sha256",
        "label_shard_path_metadata_only",
        "label_shard_sha256",
        "label_shard_row",
        "sidecar_row_identity_sha256",
        "content_sha256",
    }
)
PROJECTED_FIELDS = (
    "content_sha256",
    "dataset_role",
    "global_row",
    "scene_id",
    "family",
    "episode_id",
    "env_index",
    "reset_count",
    "source_split",
    "frames_jsonl_sha256",
    "scene_manifest_sha256",
    "primitive",
    "current_endpoint_sha256",
    "next_endpoint_sha256",
)
CONTEXT_FIELDS = (
    "dataset_role",
    "family",
    "scene_id",
    "env_index",
    "episode_id",
    "reset_count",
    "source_split",
    "frames_jsonl_sha256",
    "scene_manifest_sha256",
)
CONTEXT_COUNTERS = {
    "dataset_role": "cross_role_endpoint_count",
    "family": "cross_family_endpoint_count",
    "scene_id": "cross_scene_endpoint_count",
    "env_index": "cross_environment_endpoint_count",
    "episode_id": "cross_episode_endpoint_count",
    "reset_count": "cross_reset_endpoint_count",
    "source_split": "cross_split_endpoint_count",
    "frames_jsonl_sha256": "cross_frames_source_endpoint_count",
    "scene_manifest_sha256": "cross_scene_manifest_endpoint_count",
}
INTEGRITY_COUNTERS = (
    "input_validation_failure_count",
    "malformed_row_count",
    "duplicate_global_row_extra_count",
    "duplicate_pair_content_extra_count",
    "self_edge_row_count",
    "duplicate_current_owner_endpoint_count",
    "duplicate_next_owner_endpoint_count",
    *tuple(CONTEXT_COUNTERS.values()),
    "cycle_component_count",
    "cycle_edge_count",
    "uncovered_edge_row_count",
)
RECEIPT_SCHEMA = (
    "lewm_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1_receipt_v1"
)
AUTHORIZATION_SCHEMA = (
    "lewm_go2_rgb_causal_belief_state_h4_chain_metadata_census_v1_"
    "execution_authorization_v1"
)
RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "decision",
        "preregistration",
        "input_bindings",
        "populations",
        "integrity",
        "graph",
        "adequacy",
        "access",
        "work",
        "content_sha256",
    }
)


class CensusValidationError(RuntimeError):
    """The exact metadata contract could not be validated."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    if type(core) is not dict or "content_sha256" in core:
        raise TypeError("self-hashed core must be a plain dict without its hash")
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CensusValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_canonical_json_line(raw: bytes, *, name: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise CensusValidationError(f"{name} must be one canonical JSON line")
    try:
        value = json.loads(
            raw[:-1].decode("ascii"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CensusValidationError(f"{name} is not ASCII JSON") from error
    if type(value) is not dict or canonical_json_bytes(value) + b"\n" != raw:
        raise CensusValidationError(f"{name} is not canonical JSON")
    return value


def _plain_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _nonempty_string(value: object) -> bool:
    return type(value) is str and bool(value)


def validate_manifest(raw: bytes) -> dict[str, Any]:
    if len(raw) != MANIFEST_BYTE_COUNT or hashlib.sha256(raw).hexdigest() != MANIFEST_FILE_SHA256:
        raise CensusValidationError("manifest file binding changed")
    value = parse_canonical_json_line(raw, name="raw manifest")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if declared != MANIFEST_CONTENT_SHA256 or canonical_json_sha256(core) != declared:
        raise CensusValidationError("manifest content binding changed")
    expected_fields = {
        "schema", "status", "evidence_schema", "raster_schema", "roles",
        "pair_counts", "endpoint_instance_count", "unique_endpoint_counts",
        "scene_shard_count", "ordered_pair_sha256", "ordered_endpoint_sha256",
        "pair_index", "endpoint_index", "array_layout", "shards", "files",
        "input_provenance", "access_ledger", "independent_audit_precommit",
        "parallel_contract", "publication", "licenses", "content_sha256",
    }
    if set(value) != expected_fields:
        raise CensusValidationError("manifest field set changed")
    if (
        value.get("roles") != list(ROLES)
        or value.get("pair_counts")
        != {role: ROLE_COUNTS[role]["pair_count"] for role in ROLES}
        or value.get("unique_endpoint_counts")
        != {role: ROLE_COUNTS[role]["unique_endpoint_count"] for role in ROLES}
        or value.get("ordered_pair_sha256") != ORDERED_PAIR_CONTENT_SHA256
        or value.get("pair_index")
        != {"path": "pairs.jsonl", "row_count": PAIR_COUNT, "file_sha256": PAIRS_FILE_SHA256}
    ):
        raise CensusValidationError("manifest pair population changed")
    pair_inventory = [
        item for item in value.get("files", [])
        if type(item) is dict and item.get("path") == "pairs.jsonl"
    ]
    if pair_inventory != [
        {"path": "pairs.jsonl", "byte_count": PAIRS_BYTE_COUNT, "file_sha256": PAIRS_FILE_SHA256}
    ]:
        raise CensusValidationError("manifest pair inventory changed")
    return {
        "path": MANIFEST_PATH,
        "expected_file_sha256": MANIFEST_FILE_SHA256,
        "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_content_sha256": MANIFEST_CONTENT_SHA256,
        "observed_content_sha256": declared,
        "expected_byte_count": MANIFEST_BYTE_COUNT,
        "observed_byte_count": len(raw),
    }


def _validate_pair(value: object, raw_line: bytes, *, line_number: int) -> dict[str, Any]:
    if type(value) is not dict or set(value) != PAIR_FIELDS:
        raise CensusValidationError(f"pair line {line_number} field set changed")
    if canonical_json_bytes(value) + b"\n" != raw_line:
        raise CensusValidationError(f"pair line {line_number} is not canonical")
    sha_fields = (
        "frames_jsonl_sha256", "scene_manifest_sha256",
        "current_endpoint_sha256", "next_endpoint_sha256",
        "label_shard_sha256", "sidecar_row_identity_sha256", "content_sha256",
    )
    relative = value.get("relative_se2_current_frame")
    if (
        value.get("schema") != PAIR_SCHEMA
        or value.get("dataset_role") not in ROLES
        or value.get("family") not in FAMILIES
        or value.get("primitive") not in PRIMITIVES
        or any(not _plain_nonnegative_int(value.get(field)) for field in (
            "global_row", "env_index", "reset_count", "label_shard_row"
        ))
        or any(not _nonempty_string(value.get(field)) for field in (
            "scene_id", "family", "episode_id", "source_split",
            "label_shard_path_metadata_only",
        ))
        or any(not is_sha256(value.get(field)) for field in sha_fields)
        or value.get("current_endpoint_sha256") == value.get("next_endpoint_sha256")
        or type(relative) is not list
        or len(relative) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in relative
        )
    ):
        raise CensusValidationError(f"pair line {line_number} value contract changed")
    core = dict(value)
    declared = core.pop("content_sha256")
    if canonical_json_sha256(core) != declared:
        raise CensusValidationError(f"pair line {line_number} self hash changed")
    return {field: value[field] for field in PROJECTED_FIELDS}


def validate_pairs(raw: bytes) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if len(raw) != PAIRS_BYTE_COUNT or hashlib.sha256(raw).hexdigest() != PAIRS_FILE_SHA256:
        raise CensusValidationError("pair-index file binding changed")
    lines = raw.splitlines(keepends=True)
    if len(lines) != PAIR_COUNT or any(not line.endswith(b"\n") for line in lines):
        raise CensusValidationError("pair-index row population changed")
    rows: list[dict[str, Any]] = []
    global_rows: set[int] = set()
    content_ids: set[str] = set()
    ordered_content_ids: list[str] = []
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(
                line[:-1].decode("ascii"), object_pairs_hook=_reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CensusValidationError(f"pair line {number} is not ASCII JSON") from error
        row = _validate_pair(value, line, line_number=number)
        if row["global_row"] in global_rows:
            raise CensusValidationError("pair global-row identity repeated")
        if row["content_sha256"] in content_ids:
            raise CensusValidationError("pair content identity repeated")
        global_rows.add(row["global_row"])
        content_ids.add(row["content_sha256"])
        ordered_content_ids.append(row["content_sha256"])
        rows.append(row)
    ordered_sha = canonical_json_sha256(ordered_content_ids)
    if ordered_sha != ORDERED_PAIR_CONTENT_SHA256:
        raise CensusValidationError("ordered pair-content identity changed")
    populations: dict[str, Any] = {}
    for role in ROLES:
        selected = [row for row in rows if row["dataset_role"] == role]
        endpoints = {
            endpoint
            for row in selected
            for endpoint in (row["current_endpoint_sha256"], row["next_endpoint_sha256"])
        }
        observed = {
            "pair_count": len(selected),
            "scene_count": len({row["scene_id"] for row in selected}),
            "unique_endpoint_count": len(endpoints),
        }
        if observed != ROLE_COUNTS[role]:
            raise CensusValidationError(f"{role} population changed")
        populations[role] = observed
    binding = {
        "path": PAIRS_PATH,
        "expected_file_sha256": PAIRS_FILE_SHA256,
        "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_content_sha256": None,
        "observed_content_sha256": None,
        "expected_byte_count": PAIRS_BYTE_COUNT,
        "observed_byte_count": len(raw),
        "expected_row_count": PAIR_COUNT,
        "observed_row_count": len(rows),
        "expected_ordered_content_sha256": ORDERED_PAIR_CONTENT_SHA256,
        "observed_ordered_content_sha256": ordered_sha,
    }
    return rows, populations, binding


def _context(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in CONTEXT_FIELDS)


def _empty_integrity() -> dict[str, int]:
    return {name: 0 for name in INTEGRITY_COUNTERS}


def _head_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*_context(row), row["current_endpoint_sha256"])


def _window_counts(length: int) -> dict[str, int]:
    return {f"H{h}": max(length - h + 1, 0) for h in range(1, 7)}


def census_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return integrity, graph and adequacy objects for validated projections."""

    integrity = _empty_integrity()
    current_owners: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    next_owners: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        current_owners[str(row["current_endpoint_sha256"])].append(row)
        next_owners[str(row["next_endpoint_sha256"])].append(row)
        if row["current_endpoint_sha256"] == row["next_endpoint_sha256"]:
            integrity["self_edge_row_count"] += 1
    integrity["duplicate_current_owner_endpoint_count"] = sum(
        len(owners) > 1 for owners in current_owners.values()
    )
    integrity["duplicate_next_owner_endpoint_count"] = sum(
        len(owners) > 1 for owners in next_owners.values()
    )
    all_endpoints = set(current_owners) | set(next_owners)
    for endpoint in all_endpoints:
        owners = current_owners.get(endpoint, []) + next_owners.get(endpoint, [])
        for field, counter in CONTEXT_COUNTERS.items():
            if len({owner[field] for owner in owners}) > 1:
                integrity[counter] += 1

    paths_by_role: dict[str, list[list[Mapping[str, Any]]]] = {
        role: [] for role in ELIGIBLE_ROLES
    }
    for role in ELIGIBLE_ROLES:
        role_rows = [row for row in rows if row["dataset_role"] == role]
        outgoing = {str(row["current_endpoint_sha256"]): row for row in role_rows}
        incoming = {str(row["next_endpoint_sha256"]): row for row in role_rows}
        heads = sorted(
            (row for row in role_rows if row["current_endpoint_sha256"] not in incoming),
            key=_head_key,
        )
        visited: set[str] = set()
        for head in heads:
            path: list[Mapping[str, Any]] = []
            row: Mapping[str, Any] | None = head
            while row is not None and row["content_sha256"] not in visited:
                visited.add(str(row["content_sha256"]))
                path.append(row)
                row = outgoing.get(str(row["next_endpoint_sha256"]))
            if path:
                paths_by_role[role].append(path)
        unvisited = [row for row in role_rows if row["content_sha256"] not in visited]
        remaining = {str(row["content_sha256"]): row for row in unvisited}
        while remaining:
            integrity["cycle_component_count"] += 1
            start_id = min(remaining)
            row = remaining[start_id]
            component: set[str] = set()
            while row["content_sha256"] not in component:
                identity = str(row["content_sha256"])
                component.add(identity)
                next_row = outgoing.get(str(row["next_endpoint_sha256"]))
                if next_row is None:
                    break
                row = next_row
            integrity["cycle_edge_count"] += len(component)
            for identity in component:
                remaining.pop(identity, None)
        integrity["uncovered_edge_row_count"] += len(unvisited)

    sliding: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    packed: dict[str, Any] = {}
    path_histograms: dict[str, Any] = {}
    train_tuples_by_family: dict[str, list[list[str]]] = {family: [] for family in FAMILIES}
    train_histograms = {
        f"p{position}": {primitive: 0 for primitive in PRIMITIVES}
        for position in range(2, 6)
    }
    for role in ELIGIBLE_ROLES:
        role_paths = paths_by_role[role]
        sliding[role] = {f"H{h}": 0 for h in range(1, 7)}
        by_family[role] = {
            family: {f"H{h}": 0 for h in range(1, 7)} for family in FAMILIES
        }
        packed_by_family = {family: 0 for family in FAMILIES}
        length_histogram: Counter[str] = Counter()
        leftovers = 0
        packed_count = 0
        for path in role_paths:
            length = len(path)
            family = str(path[0]["family"])
            counts = _window_counts(length)
            for horizon, count in counts.items():
                sliding[role][horizon] += count
                by_family[role][family][horizon] += count
            number = length // 6
            packed_count += number
            packed_by_family[family] += number
            leftovers += length % 6
            length_histogram[str(length)] += 1
            for start in range(max(length - 5, 0)):
                window = path[start : start + 6]
                if len(window) != 6:
                    continue
                actions = [str(item["primitive"]) for item in window[2:6]]
                if role == "train":
                    train_tuples_by_family[family].append(actions)
                    for offset, action in enumerate(actions, start=2):
                        train_histograms[f"p{offset}"][action] += 1
        packed[role] = {
            "count": packed_count,
            "by_family": packed_by_family,
            "leftover_edge_count": leftovers,
        }
        path_histograms[role] = dict(sorted(length_histogram.items(), key=lambda item: int(item[0])))

    tuple_hashes = {
        family: canonical_json_sha256({
            "domain": "train_future_action_tuple_multiset_v1",
            "role": "train",
            "family": family,
            "tuples": sorted(train_tuples_by_family[family]),
        })
        for family in FAMILIES
    }
    aggregate_tuples = sorted(
        tuple(action_tuple)
        for values in train_tuples_by_family.values()
        for action_tuple in values
    )
    aggregate_tuple_hash = canonical_json_sha256({
        "domain": "train_future_action_tuple_multiset_v1",
        "role": "train",
        "family": "aggregate",
        "tuples": aggregate_tuples,
    })
    integrity_all_zero = all(value == 0 for value in integrity.values())
    primitive_coverage = all(
        train_histograms[position][primitive] > 0
        for position in train_histograms
        for primitive in PRIMITIVES
    )
    predicates = {
        "input_contract_valid": True,
        "integrity_all_zero": integrity_all_zero,
        "train_row_disjoint_h6_at_least_64": packed["train"]["count"] >= 64,
        "train_each_family_row_disjoint_h6_at_least_1": all(
            packed["train"]["by_family"][family] >= 1 for family in FAMILIES
        ),
        "train_all_primitives_each_future_position": primitive_coverage,
        "selection_row_disjoint_h6_at_least_8": (
            packed["checkpoint_selection"]["count"] >= 8
        ),
        "selection_each_family_row_disjoint_h6_at_least_1": all(
            packed["checkpoint_selection"]["by_family"][family] >= 1
            for family in FAMILIES
        ),
        "all_chains_context_exact": integrity_all_zero,
    }
    failed = sorted(name for name, passed in predicates.items() if not passed)
    graph = {
        "scientific_unit": {
            "edge_count": 6,
            "endpoint_count": 7,
            "history_positions": [0, 1],
            "future_positions": [2, 3, 4, 5],
        },
        "eligible_roles": list(ELIGIBLE_ROLES),
        "sliding_candidate_counts": sliding,
        "sliding_candidate_counts_by_family": by_family,
        "row_disjoint_h6": packed,
        "maximal_path_length_histograms": path_histograms,
        "train_future_primitive_histograms": train_histograms,
        "train_future_primitive_tuple_multiset_sha256_by_family": tuple_hashes,
        "train_future_primitive_tuple_multiset_sha256": aggregate_tuple_hash,
    }
    adequacy = {"predicates": predicates, "failed_predicates": failed}
    return {"counts": integrity, "all_zero": integrity_all_zero}, graph, adequacy


def census_from_raw(manifest_raw: bytes, pairs_raw: bytes) -> dict[str, Any]:
    manifest_binding = validate_manifest(manifest_raw)
    rows, populations, pair_binding = validate_pairs(pairs_raw)
    integrity, graph, adequacy = census_rows(rows)
    return {
        "input_bindings": {"manifest": manifest_binding, "pairs": pair_binding},
        "populations": populations,
        "integrity": integrity,
        "graph": graph,
        "adequacy": adequacy,
    }


def build_receipt(
    census: Mapping[str, Any],
    *,
    access: Mapping[str, Any],
    work: Mapping[str, Any],
) -> dict[str, Any]:
    access_value = {name: dict(value) if isinstance(value, Mapping) else value for name, value in access.items()}
    work_value = dict(work)
    predicates = dict(census["adequacy"]["predicates"])
    forbidden = access_value.get("forbidden")
    predicates["all_forbidden_access_zero"] = (
        type(forbidden) is dict
        and set(forbidden) == set(default_access_receipt()["forbidden"])
        and all(type(value) is int and value == 0 for value in forbidden.values())
        and access_value.get("all_forbidden_zero") is True
    )
    predicates["allowed_input_opens_exact"] = access_value.get("allowed") == {
        "manifest_open_attempt_count": 1,
        "manifest_read_success_count": 1,
        "pairs_open_attempt_count": 1,
        "pairs_read_success_count": 1,
    }
    predicates["work_scope_exact"] = work_value == default_work_receipt()
    failed = sorted(name for name, passed in predicates.items() if passed is not True)
    adequacy = {"predicates": predicates, "failed_predicates": failed}
    decision = "H4_METADATA_FEASIBLE" if not failed else "STOP_H4_METADATA_INADEQUATE"
    core = {
        "schema": RECEIPT_SCHEMA,
        "status": "COMPLETE",
        "decision": decision,
        "preregistration": dict(PREREGISTRATION),
        "input_bindings": dict(census["input_bindings"]),
        "populations": dict(census["populations"]),
        "integrity": dict(census["integrity"]),
        "graph": dict(census["graph"]),
        "adequacy": adequacy,
        "access": access_value,
        "work": work_value,
    }
    return with_content_sha256(core)


def _unobserved_input_bindings() -> dict[str, Any]:
    return {
        "manifest": {
            "path": MANIFEST_PATH,
            "expected_file_sha256": MANIFEST_FILE_SHA256,
            "observed_file_sha256": None,
            "expected_content_sha256": MANIFEST_CONTENT_SHA256,
            "observed_content_sha256": None,
            "expected_byte_count": MANIFEST_BYTE_COUNT,
            "observed_byte_count": None,
        },
        "pairs": {
            "path": PAIRS_PATH,
            "expected_file_sha256": PAIRS_FILE_SHA256,
            "observed_file_sha256": None,
            "expected_content_sha256": None,
            "observed_content_sha256": None,
            "expected_byte_count": PAIRS_BYTE_COUNT,
            "observed_byte_count": None,
            "expected_row_count": PAIR_COUNT,
            "observed_row_count": None,
            "expected_ordered_content_sha256": ORDERED_PAIR_CONTENT_SHA256,
            "observed_ordered_content_sha256": None,
        },
    }


def build_stop_receipt(
    *,
    input_bindings: Mapping[str, Any] | None,
    access: Mapping[str, Any],
    work: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete aggregate STOP receipt for a post-reservation failure."""

    counters = _empty_integrity()
    counters["input_validation_failure_count"] = 1
    zero_horizons = {f"H{h}": 0 for h in range(1, 7)}
    zero_by_family = {
        family: dict(zero_horizons) for family in FAMILIES
    }
    graph = {
        "scientific_unit": {
            "edge_count": 6,
            "endpoint_count": 7,
            "history_positions": [0, 1],
            "future_positions": [2, 3, 4, 5],
        },
        "eligible_roles": list(ELIGIBLE_ROLES),
        "sliding_candidate_counts": {
            role: dict(zero_horizons) for role in ELIGIBLE_ROLES
        },
        "sliding_candidate_counts_by_family": {
            role: {family: dict(values) for family, values in zero_by_family.items()}
            for role in ELIGIBLE_ROLES
        },
        "row_disjoint_h6": {
            role: {
                "count": 0,
                "by_family": {family: 0 for family in FAMILIES},
                "leftover_edge_count": 0,
            }
            for role in ELIGIBLE_ROLES
        },
        "maximal_path_length_histograms": {role: {} for role in ELIGIBLE_ROLES},
        "train_future_primitive_histograms": {
            f"p{position}": {primitive: 0 for primitive in PRIMITIVES}
            for position in range(2, 6)
        },
        "train_future_primitive_tuple_multiset_sha256_by_family": {
            family: canonical_json_sha256({
                "domain": "train_future_action_tuple_multiset_v1",
                "role": "train",
                "family": family,
                "tuples": [],
            })
            for family in FAMILIES
        },
        "train_future_primitive_tuple_multiset_sha256": canonical_json_sha256({
            "domain": "train_future_action_tuple_multiset_v1",
            "role": "train",
            "family": "aggregate",
            "tuples": [],
        }),
    }
    predicate_names = (
        "input_contract_valid",
        "integrity_all_zero",
        "train_row_disjoint_h6_at_least_64",
        "train_each_family_row_disjoint_h6_at_least_1",
        "train_all_primitives_each_future_position",
        "selection_row_disjoint_h6_at_least_8",
        "selection_each_family_row_disjoint_h6_at_least_1",
        "all_chains_context_exact",
        "all_forbidden_access_zero",
        "allowed_input_opens_exact",
        "work_scope_exact",
    )
    census = {
        "input_bindings": (
            _unobserved_input_bindings()
            if input_bindings is None
            else {name: dict(value) for name, value in input_bindings.items()}
        ),
        "populations": {
            role: {"pair_count": 0, "scene_count": 0, "unique_endpoint_count": 0}
            for role in ROLES
        },
        "integrity": {"counts": counters, "all_zero": False},
        "graph": graph,
        "adequacy": {
            "predicates": {name: False for name in predicate_names},
            "failed_predicates": list(predicate_names),
        },
    }
    return build_receipt(census, access=access, work=work)


def validate_receipt(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != RECEIPT_FIELDS:
        raise CensusValidationError("receipt top-level fields changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    failed = value.get("adequacy", {}).get("failed_predicates") if type(value.get("adequacy")) is dict else None
    predicates = value.get("adequacy", {}).get("predicates") if type(value.get("adequacy")) is dict else None
    expected_decision = (
        "H4_METADATA_FEASIBLE"
        if type(failed) is list and not failed
        else "STOP_H4_METADATA_INADEQUATE"
    )
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("status") != "COMPLETE"
        or value.get("decision") not in {
            "H4_METADATA_FEASIBLE", "STOP_H4_METADATA_INADEQUATE"
        }
        or value.get("preregistration") != PREREGISTRATION
        or type(predicates) is not dict
        or type(failed) is not list
        or failed != sorted(name for name, passed in predicates.items() if passed is not True)
        or value.get("decision") != expected_decision
        or set(value.get("input_bindings", {})) != {"manifest", "pairs"}
        or set(value.get("populations", {})) != set(ROLES)
        or type(value.get("integrity")) is not dict
        or type(value.get("graph")) is not dict
        or type(value.get("access")) is not dict
        or type(value.get("work")) is not dict
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise CensusValidationError("receipt identity changed")
    return dict(value)


def validate_authorization(
    raw: bytes,
    *,
    expected_file_sha256: str,
    source_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise CensusValidationError("authorization file hash changed")
    value = parse_canonical_json_line(raw, name="execution authorization")
    required = {
        "schema", "status", "authority", "preregistration", "source_bindings",
        "review", "runtime_inputs", "output", "attempt", "content_sha256",
    }
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        set(value) != required
        or value.get("schema") != AUTHORIZATION_SCHEMA
        or value.get("status") != "PASS_EXACTLY_ONE_METADATA_CENSUS"
        or value.get("authority") != "execute_exactly_one_recurrent_h4_metadata_census"
        or value.get("preregistration") != PREREGISTRATION
        or value.get("source_bindings") != list(source_bindings)
        or value.get("runtime_inputs")
        != {
            "manifest": {
                "path": MANIFEST_PATH,
                "file_sha256": MANIFEST_FILE_SHA256,
                "content_sha256": MANIFEST_CONTENT_SHA256,
                "byte_count": MANIFEST_BYTE_COUNT,
            },
            "pairs": {
                "path": PAIRS_PATH,
                "file_sha256": PAIRS_FILE_SHA256,
                "byte_count": PAIRS_BYTE_COUNT,
                "row_count": PAIR_COUNT,
                "ordered_content_sha256": ORDERED_PAIR_CONTENT_SHA256,
            },
        }
        or value.get("output") != {"path": OUTPUT_PATH, "must_be_fresh": True}
        or value.get("attempt")
        != {"maximum_execution_count": 1, "retry_authorized": False}
        or type(value.get("review")) is not dict
        or value["review"] != {
            "path": REVIEW_PATH,
            "source_commit": value["review"].get("source_commit"),
            "file_sha256": value["review"].get("file_sha256"),
            "content_sha256": value["review"].get("content_sha256"),
            "byte_count": value["review"].get("byte_count"),
            "status": "PASS_SOURCE_ONLY_ZERO_FINDINGS",
        }
        or not is_sha256(value["review"].get("source_commit"))
        or not is_sha256(value["review"].get("file_sha256"))
        or not is_sha256(value["review"].get("content_sha256"))
        or not _plain_nonnegative_int(value["review"].get("byte_count"))
        or value["review"].get("byte_count") == 0
        or not is_sha256(declared)
        or canonical_json_sha256(core) != declared
    ):
        raise CensusValidationError("execution authorization contract changed")
    return dict(value)


def default_access_receipt() -> dict[str, Any]:
    forbidden = {
        "rgb_open_count": 0,
        "label_or_shard_open_count": 0,
        "endpoint_index_open_count": 0,
        "sidecar_or_pose_source_open_count": 0,
        "schedule_open_count": 0,
        "checkpoint_or_trace_open_count": 0,
        "prior_runtime_output_open_count": 0,
        "gpu_use_count": 0,
        "training_or_navigation_count": 0,
        "heldout_sealed_g2_g8_open_count": 0,
    }
    return {
        "allowed": {
            "manifest_open_attempt_count": 1,
            "manifest_read_success_count": 1,
            "pairs_open_attempt_count": 1,
            "pairs_read_success_count": 1,
        },
        "forbidden": forbidden,
        "all_forbidden_zero": True,
    }


def default_work_receipt() -> dict[str, Any]:
    return {
        "stdlib_only": True,
        "real_input_run": True,
        "exclusive_receipt_write_count": 1,
        "retained_pair_field_count": len(PROJECTED_FIELDS),
        "nonallowlisted_retained_field_count": 0,
        "calibration_chain_eligible": False,
    }
