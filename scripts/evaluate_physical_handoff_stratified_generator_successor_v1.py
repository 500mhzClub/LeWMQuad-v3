#!/usr/bin/env python3
"""Independent reducer for the stratified handoff generator successor V1.

The evaluator has no simulator, model, or training path.  It reopens ordinary
JSON/JSONL/NPZ evidence, independently rebuilds the generator and conditional
handoff metrics, and publishes only the frozen 9- or 19-leaf official tree.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.safety import physical_handoff_stratified_generator_successor_v1_contract as CONTRACT
from lewm.safety import physical_handoff_stratified_generator_successor_v1_metrics as METRICS
from scripts import run_physical_graph_edge_handoff_qualification_v1 as V1
from scripts import run_physical_graph_edge_handoff_qualification_v4 as V4


class RegenerationError(RuntimeError):
    """Persisted successor evidence failed independent regeneration."""


EXPERIMENT_ID = CONTRACT.EXPERIMENT_ID
DEFAULT_OUTPUT_ROOT = Path(CONTRACT.OUTPUT_ROOT)
DEFAULT_MATERIAL_ROOT = Path(CONTRACT.MATERIAL_ROOT)
FILE_HASHES_SCHEMA = (
    "physical_handoff_stratified_generator_successor_v1.file_hashes.v1"
)


def canonical_bytes(value: Any) -> bytes:
    return CONTRACT.canonical_json_bytes(value).rstrip(b"\n") + b"\n"


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RegenerationError(f"{label} is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RegenerationError(f"{label} is not an ordinary regular file")
    cursor = path.parent
    while True:
        if cursor.is_symlink():
            raise RegenerationError(f"{label} has a symlink ancestor")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    return info


def _load_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    _require_regular(path, label)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegenerationError(f"invalid {label}") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise RegenerationError(f"noncanonical {label}")
    return raw, copy.deepcopy(value)


def _load_jsonl(path: Path, label: str) -> tuple[bytes, list[dict[str, Any]]]:
    _require_regular(path, label)
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            raise RegenerationError(f"{label} row {index} lacks newline")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RegenerationError(f"invalid {label} row {index}") from exc
        if not isinstance(value, dict) or line != canonical_bytes(value):
            raise RegenerationError(f"noncanonical {label} row {index}")
        rows.append(copy.deepcopy(value))
    if not rows:
        raise RegenerationError(f"{label} is empty")
    return raw, rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    V1.atomic_bytes(path, canonical_bytes(dict(value)))


def _binding(path: Path, *, root: Path) -> dict[str, Any]:
    info = _require_regular(path, str(path))
    return {
        "path": str(path.relative_to(root)),
        "bytes": int(info.st_size),
        "sha256": sha256_file(path),
        "nlink": int(info.st_nlink),
        "ordinary_regular_file": True,
        "resolved_path_ancestor_symlink_count": 0,
    }


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    function = getattr(METRICS, name, None)
    if not callable(function):
        raise RegenerationError(f"required reducer API is absent: {name}")
    try:
        return function(*args, **kwargs)
    except RegenerationError:
        raise
    except Exception as exc:
        raise RegenerationError(f"reducer API failed: {name}") from exc


def _require_no_successor_external_receipt(root: Path) -> None:
    """Reject conventional sibling receipts; this experiment publishes none."""

    try:
        authority = CONTRACT.validate_content_digest(
            CONTRACT.PROHIBITED_EXTERNAL_PUBLICATION_AUTHORITY
        )
    except Exception as exc:
        raise RegenerationError(
            "prohibited external-publication authority drift"
        ) from exc
    prohibited = [
        Path(value) for value in CONTRACT.PROHIBITED_EXTERNAL_PUBLICATION_PATHS
    ]
    if (
        authority.get("paths") != [str(path) for path in prohibited]
        or len(prohibited) != 3
        or any(
            path.parent != root.parent
            or not path.name.startswith(f"{root.name}_")
            for path in prohibited
        )
    ):
        raise RegenerationError("prohibited external-publication path drift")
    opened = [path for path in prohibited if path.exists() or path.is_symlink()]
    if opened:
        raise RegenerationError(
            "successor external regeneration/custody publication is prohibited: "
            + ", ".join(str(path) for path in opened)
        )


def _git(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=REPO_ROOT,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RegenerationError(
            f"git {' '.join(arguments)} failed: {exc.output}"
        ) from exc


def _observe_source_freeze(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Independently bind the live tree to the freeze or its empty result child."""

    freeze = str(runtime["source_freeze_commit"])
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RegenerationError("source worktree is not clean")
    if (
        _git("show", "-s", "--format=%s", freeze)
        != CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT
        or _git("rev-list", "--parents", "-n", "1", freeze).split()
        != [freeze, CONTRACT.SOURCE_PARENT_COMMIT]
    ):
        raise RegenerationError("source freeze commit ancestry/subject drift")
    freeze_tree = _git("rev-parse", f"{freeze}^{{tree}}")
    if head != freeze:
        if (
            _git("show", "-s", "--format=%s", head)
            != CONTRACT.RESULT_COMMIT_SUBJECT
            or _git("rev-list", "--parents", "-n", "1", head).split()
            != [head, freeze]
            or _git("rev-parse", f"{head}^{{tree}}") != freeze_tree
            or _git("diff", "--name-only", freeze, head)
        ):
            raise RegenerationError(
                "HEAD is neither the source freeze nor its empty result child"
            )
    closure_path = REPO_ROOT / (
        f"{CONTRACT.DOC_PREFIX}_source_closure_2026-09-04.json"
    )
    closure_raw, closure = _load_json(closure_path, "source closure")
    try:
        CONTRACT.validate_content_digest(closure)
    except Exception as exc:
        raise RegenerationError("source closure content digest drift") from exc
    rows = closure.get("rows")
    if (
        not isinstance(rows, list)
        or [row.get("path") for row in rows]
        != list(CONTRACT.SOURCE_CLOSURE_PATHS)
    ):
        raise RegenerationError("source closure row inventory/order drift")
    for row in rows:
        relative = str(row["path"])
        path = REPO_ROOT / relative
        info = _require_regular(path, f"source closure file {relative}")
        raw = path.read_bytes()
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{freeze}:{relative}"],
                cwd=REPO_ROOT,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            raise RegenerationError(
                f"source closure path is absent from freeze: {relative}"
            ) from exc
        if (
            raw != committed
            or int(row.get("bytes", -1)) != int(info.st_size)
            or row.get("sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise RegenerationError(f"source closure file drift: {relative}")
    observation = CONTRACT.build_source_freeze_observation(
        source_freeze_commit=freeze,
        source_freeze_tree_oid=freeze_tree,
        source_closure_content_digest=str(closure["content_digest"]),
        source_closure_file_sha256=hashlib.sha256(closure_raw).hexdigest(),
        # Git publication is operational custody, not scientific identity.
        # The actual result child was checked above; the persisted reduction
        # always receives this canonical freeze-only observation so rebuilding
        # after the empty result commit is byte-identical.
        observed_head_commit_at_scientific_reduction=freeze,
    )
    return CONTRACT.validate_source_freeze_observation(
        observation, source_freeze_commit=freeze
    )


def _sorted_projection_sha256(values: Sequence[str | int]) -> str:
    return hashlib.sha256(
        CONTRACT.canonical_json_bytes(sorted(set(values)))[:-1]
    ).hexdigest()


def _independent_disposition_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    observed = Counter(str(row["disposition"]) for row in rows)
    if set(observed) - set(CONTRACT.STATE_DISPOSITIONS):
        raise RegenerationError("unknown disposition in terminal population")
    return {
        disposition: int(observed.get(disposition, 0))
        for disposition in CONTRACT.STATE_DISPOSITIONS
    }


def _independent_candidate_spec(row: Mapping[str, Any]) -> dict[str, Any]:
    return CONTRACT.build_candidate_spec(
        str(row["family"]), int(row["stratum_index"]), int(row["attempt_index"])
    )


def _independent_common_physical_parameters(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    specs = [_independent_candidate_spec(row) for row in rows]
    first = specs[0]
    failed_specs = [
        spec for spec, row in zip(specs, rows) if not bool(row["qualified"])
    ]
    projections = []
    for spec in failed_specs:
        projections.append(
            {
                "family": spec["family"],
                "stratum_index": spec["stratum_index"],
                "edge_direction": spec["route_direction"],
                "opening_width_id": spec["passage_width_id"],
                "opening_width_m": spec["passage_width_m"],
                "port_offset_id": spec["port_distance_id"],
                "port_offset_m": spec["port_distance_m"],
                "base_spawn_lateral_offset_m": spec["spawn_lateral_offset_m"],
                "starting_bearing_rad": spec["spawn_yaw_offset_rad"],
                "geometry_jitter_x_m": spec["geometry_jitter_xy_m"][0],
                "geometry_jitter_y_m": spec["geometry_jitter_xy_m"][1],
                **{
                    name: spec["variant_adjustments"][name]
                    for name in CONTRACT.HASH_PERTURBATION_PARAMETER_IDS[2:]
                },
            }
        )
    counts: dict[str, list[dict[str, Any]]] = {}
    constants: dict[str, Any] = {}
    for name in CONTRACT.COMMON_FAILURE_PARAMETER_IDS:
        grouped: dict[bytes, dict[str, Any]] = {}
        for projection in projections:
            value = copy.deepcopy(projection[name])
            key = CONTRACT.canonical_json_bytes(value)
            if key not in grouped:
                grouped[key] = {"value": value, "count": 0}
            grouped[key]["count"] += 1
        counts[name] = [grouped[key] for key in sorted(grouped)]
        constants[name] = (
            copy.deepcopy(counts[name][0]["value"])
            if len(counts[name]) == 1
            else None
        )
    result = {
        "family": first["family"],
        "stratum_index": first["stratum_index"],
        "failed_attempt_count": len(failed_specs),
        "parameter_order": list(CONTRACT.COMMON_FAILURE_PARAMETER_IDS),
        "canonical_value_counts": counts,
        "constant_across_all_failed_attempts": constants,
        "predeclared_parameters_only": True,
        "post_hoc_factor_added": False,
    }
    if set(result) != set(CONTRACT.COMMON_PHYSICAL_PARAMETER_FIELDS):
        raise RegenerationError("independent common physical parameter drift")
    for spec in specs:
        if (
            spec["family"] != first["family"]
            or spec["stratum_index"] != first["stratum_index"]
            or spec["geometry"]["selected_directed_edge"]["opening_width_m"]
            != spec["passage_width_m"]
        ):
            raise RegenerationError("stream physical parameters changed")
    return result


def _independent_hash_perturbation_ranges(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {
        name: [] for name in CONTRACT.HASH_PERTURBATION_PARAMETER_IDS
    }
    for row in rows:
        spec = _independent_candidate_spec(row)
        values["geometry_jitter_x_m"].append(
            float(spec["geometry_jitter_xy_m"][0])
        )
        values["geometry_jitter_y_m"].append(
            float(spec["geometry_jitter_xy_m"][1])
        )
        for name in CONTRACT.HASH_PERTURBATION_PARAMETER_IDS[2:]:
            values[name].append(float(spec["variant_adjustments"][name]))
    return {
        name: {"minimum": min(values[name]), "maximum": max(values[name])}
        for name in CONTRACT.HASH_PERTURBATION_PARAMETER_IDS
    }


def _independent_stream_metric(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    supplied = [copy.deepcopy(dict(row)) for row in rows]
    first = supplied[0]
    qualified = [row for row in supplied if row["qualified"]]
    target = len(qualified) == CONTRACT.TARGET_QUALIFIED_PER_STREAM
    selected = (
        min(
            qualified,
            key=lambda row: (
                row["canonical_spec_sha256"], row["candidate_spec_id"]
            ),
        )
        if target
        else None
    )
    dispositions = _independent_disposition_counts(supplied)
    nonqualified = len(supplied) - len(qualified)
    initial_valid = sum(
        not any(row["state_disposition"]["initial_termination_flags"].values())
        for row in supplied
    )
    teacher_rows = [
        row for row in supplied if row["state_disposition"]["teacher_executed"]
    ]
    teacher_left_source = sum(
        bool(
            row["state_disposition"]["teacher_criteria"][
                "teacher_left_source_region"
            ]
        )
        for row in teacher_rows
    )
    correct_port_ever = sum(
        bool(
            row["state_disposition"]["teacher_criteria"][
                "teacher_crossed_directed_port"
            ]
        )
        for row in teacher_rows
    )
    contact_count = dispositions["TEACHER_PHYSICS_CONTACT"]
    other_rejections = [
        count
        for disposition, count in dispositions.items()
        if disposition not in {"QUALIFIED", "TEACHER_PHYSICS_CONTACT"}
    ]
    qualification_ordinals = [
        offset + 1 for offset, row in enumerate(supplied) if row["qualified"]
    ]
    return {
        "stream_index": first["stream_index"],
        "stream_id": first["stream_id"],
        "family": first["family"],
        "stratum_index": first["stratum_index"],
        "attempt_count": len(supplied),
        "attempt_indices": [row["attempt_index"] for row in supplied],
        "candidate_indices": [row["candidate_index"] for row in supplied],
        "qualified_count": len(qualified),
        "nonqualified_count": nonqualified,
        "qualified_candidate_indices": [
            row["candidate_index"] for row in qualified
        ],
        "disposition_counts": dispositions,
        "disposition_rates": {
            name: count / len(supplied) for name, count in dispositions.items()
        },
        "attempts_to_first_qualified": (
            qualification_ordinals[0] if qualification_ordinals else None
        ),
        "attempts_to_fourth_qualified": (
            qualification_ordinals[3]
            if len(qualification_ordinals)
            >= CONTRACT.TARGET_QUALIFIED_PER_STREAM
            else None
        ),
        "valid_initial_state_count": initial_valid,
        "valid_initial_state_fraction": initial_valid / len(supplied),
        "teacher_executed_count": len(teacher_rows),
        "teacher_left_source_count": teacher_left_source,
        "teacher_left_source_fraction_of_teacher_executed": (
            teacher_left_source / len(teacher_rows) if teacher_rows else 0.0
        ),
        "correct_port_ever_count": correct_port_ever,
        "correct_port_ever_fraction_of_teacher_executed": (
            correct_port_ever / len(teacher_rows) if teacher_rows else 0.0
        ),
        "contact_rejection_count": contact_count,
        "contact_rejection_fraction": (
            contact_count / nonqualified if nonqualified else 0.0
        ),
        "contact_dominant": bool(
            contact_count > 0
            and all(contact_count > count for count in other_rejections)
        ),
        "common_physical_parameters": _independent_common_physical_parameters(
            supplied
        ),
        "observed_hash_perturbation_ranges": (
            _independent_hash_perturbation_ranges(supplied)
        ),
        "termination_reason": (
            "TARGET_QUALIFIED_REACHED" if target else "ATTEMPT_LIMIT_REACHED"
        ),
        "target_reached": target,
        "yield_fraction": len(qualified) / len(supplied),
        "qualification_rate": len(qualified) / len(supplied),
        "rejection_rate": nonqualified / len(supplied),
        "selected_candidate_index": (
            None if selected is None else selected["candidate_index"]
        ),
        "selected_candidate_spec_id": (
            None if selected is None else selected["candidate_spec_id"]
        ),
    }


def _independent_aggregate_counts(
    streams: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts = len(records)
    qualified = sum(bool(row["qualified"]) for row in records)
    return {
        "stream_count": len(streams),
        "attempt_count": attempts,
        "qualified_count": qualified,
        "nonqualified_count": attempts - qualified,
        "qualification_rate": qualified / attempts,
        "rejection_rate": (attempts - qualified) / attempts,
        "disposition_counts": _independent_disposition_counts(records),
        "disposition_rates": {
            name: count / attempts
            for name, count in _independent_disposition_counts(records).items()
        },
        "zero_yield_stream_count": sum(
            row["qualified_count"] == 0 for row in streams
        ),
        "low_yield_stream_count": sum(
            0 < row["qualified_count"] < CONTRACT.TARGET_QUALIFIED_PER_STREAM
            for row in streams
        ),
        "target_reached_stream_count": sum(
            bool(row["target_reached"]) for row in streams
        ),
        "valid_initial_state_count": sum(
            int(row["valid_initial_state_count"]) for row in streams
        ),
        "teacher_executed_count": sum(
            int(row["teacher_executed_count"]) for row in streams
        ),
        "teacher_left_source_count": sum(
            int(row["teacher_left_source_count"]) for row in streams
        ),
        "correct_port_ever_count": sum(
            int(row["correct_port_ever_count"]) for row in streams
        ),
        "contact_rejection_count": sum(
            int(row["contact_rejection_count"]) for row in streams
        ),
        "contact_dominant_stream_count": sum(
            bool(row["contact_dominant"]) for row in streams
        ),
    }


def _independent_factor_level(stream: Mapping[str, Any], factor: str) -> Any:
    spec = CONTRACT.build_candidate_spec(
        str(stream["family"]), int(stream["stratum_index"]), 0
    )
    field = {
        "family": "family",
        "edge_direction": "route_direction",
        "opening_width_id": "passage_width_id",
        "opening_width_m": "passage_width_m",
        "port_offset_id": "port_distance_id",
        "port_offset_m": "port_distance_m",
        "starting_bearing_rad": "spawn_yaw_offset_rad",
        "base_spawn_lateral_offset_m": "spawn_lateral_offset_m",
    }[factor]
    return copy.deepcopy(spec[field])


def _independent_failure_next_decision(families: Sequence[str]) -> str:
    ordered = [family for family in CONTRACT.FAMILY_IDS if family in set(families)]
    if list(families) != ordered or not ordered:
        raise RegenerationError("failure family projection drift")
    if ordered == ["TURNING_JUNCTION"]:
        return CONTRACT.TURNING_JUNCTION_GENERATOR_NEXT_DECISION
    if ordered == ["OFFSET_OPENING"]:
        return CONTRACT.OFFSET_OPENING_GENERATOR_NEXT_DECISION
    return CONTRACT.PHYSICAL_EDGE_STRATUM_CONTRACT_NEXT_DECISION


def _independent_v4_shortfall_resolution(
    streams: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild the frozen twelve-stream V4 shortfall projection locally."""

    by_identity = {
        (str(row["family"]), int(row["stratum_index"])): row
        for row in streams
    }
    rows: list[dict[str, Any]] = []
    for family, stratum_index in CONTRACT.V4_CANONICAL_SHORTFALL_STREAMS:
        stream = by_identity.get((family, stratum_index))
        if stream is None:
            raise RegenerationError("canonical V4 shortfall stream is absent")
        qualified = int(stream["qualified_count"])
        resolution_status = (
            "RESOLVED_AT_FOUR_QUALIFIED"
            if qualified >= CONTRACT.TARGET_QUALIFIED_PER_STREAM
            else "PARTIAL_YIELD_BELOW_FOUR"
            if qualified > 0
            else "ZERO_YIELD_AT_ATTEMPT_LIMIT"
        )
        row = {
            "family": family,
            "stratum_index": stratum_index,
            "v4_fixed_four_shortfall": True,
            "successor_attempt_count": int(stream["attempt_count"]),
            "successor_qualified_count": qualified,
            "attempts_to_first_qualified": stream[
                "attempts_to_first_qualified"
            ],
            "attempts_to_fourth_qualified": stream[
                "attempts_to_fourth_qualified"
            ],
            "successor_target_reached": bool(stream["target_reached"]),
            "resolution_status": resolution_status,
        }
        if set(row) != set(CONTRACT.V4_SHORTFALL_RESOLUTION_ROW_FIELDS):
            raise RegenerationError("V4 shortfall resolution row field drift")
        rows.append(row)
    resolved = sum(
        row["resolution_status"] == "RESOLVED_AT_FOUR_QUALIFIED"
        for row in rows
    )
    partial = sum(
        row["resolution_status"] == "PARTIAL_YIELD_BELOW_FOUR"
        for row in rows
    )
    zero = sum(
        row["resolution_status"] == "ZERO_YIELD_AT_ATTEMPT_LIMIT"
        for row in rows
    )
    conclusion = (
        CONTRACT.GENERATOR_FEASIBILITY_NO_GO
        if zero
        else CONTRACT.V4_SHORTFALL_INSUFFICIENT
        if partial
        else CONTRACT.V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING
    )
    result = {
        "authority_content_digest": CONTRACT.V4_SHORTFALL_RESOLUTION_AUTHORITY[
            "content_digest"
        ],
        "canonical_v4_shortfall_stream_count": len(rows),
        "stream_rows": rows,
        "resolved_stream_count": resolved,
        "partial_stream_count": partial,
        "zero_yield_stream_count": zero,
        "all_canonical_v4_shortfalls_resolved": resolved == len(rows),
        "conclusion": conclusion,
        "supports_shallow_sampling_as_primary_cause": (
            conclusion == CONTRACT.V4_SHORTFALL_SUPPORTS_SHALLOW_SAMPLING
        ),
        "intrinsic_infeasibility_established": False,
    }
    if set(result) != set(CONTRACT.V4_SHORTFALL_RESOLUTION_FIELDS):
        raise RegenerationError("V4 shortfall resolution field drift")
    return result


def _independent_generator_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(row)) for row in records]
    indices = [int(row["candidate_index"]) for row in rows]
    if not rows or indices != sorted(set(indices)):
        raise RegenerationError("terminal record candidate ordering drift")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stream_index"])].append(row)
        if row["hard_stop"] or not row["continuation_authorized"]:
            raise RegenerationError("terminal population contains a hard stop")
    if set(grouped) != set(range(CONTRACT.STREAM_COUNT)):
        raise RegenerationError("terminal population omits a stream")
    for index in range(CONTRACT.STREAM_COUNT):
        stream_rows = grouped[index]
        attempts = [int(row["attempt_index"]) for row in stream_rows]
        qualified = [row for row in stream_rows if row["qualified"]]
        if attempts != list(range(len(stream_rows))):
            raise RegenerationError("stream attempt prefix drift")
        if len(qualified) == CONTRACT.TARGET_QUALIFIED_PER_STREAM:
            if stream_rows[-1]["qualified"] is not True:
                raise RegenerationError("stream continued beyond fourth qualified")
        elif len(qualified) < CONTRACT.TARGET_QUALIFIED_PER_STREAM:
            if len(stream_rows) != CONTRACT.MAX_ATTEMPTS_PER_STREAM:
                raise RegenerationError("stream stopped before attempt cap")
        else:
            raise RegenerationError("stream exceeds qualification target")
    streams = [
        _independent_stream_metric(grouped[index])
        for index in range(CONTRACT.STREAM_COUNT)
    ]
    zero = [
        {"family": row["family"], "stratum_index": row["stratum_index"]}
        for row in streams
        if row["qualified_count"] == 0
    ]
    low = [
        {"family": row["family"], "stratum_index": row["stratum_index"]}
        for row in streams
        if 0 < row["qualified_count"] < CONTRACT.TARGET_QUALIFIED_PER_STREAM
    ]
    reached = [
        {"family": row["family"], "stratum_index": row["stratum_index"]}
        for row in streams
        if row["target_reached"]
    ]
    status = (
        CONTRACT.GENERATOR_FEASIBILITY_NO_GO
        if zero
        else CONTRACT.GENERATOR_LOW_YIELD
        if low
        else CONTRACT.GENERATOR_PANEL_AVAILABLE
    )
    selected_indices = (
        [int(row["selected_candidate_index"]) for row in streams]
        if status == CONTRACT.GENERATOR_PANEL_AVAILABLE
        else []
    )
    selected_ids = (
        [str(row["selected_candidate_spec_id"]) for row in streams]
        if status == CONTRACT.GENERATOR_PANEL_AVAILABLE
        else []
    )
    family_rows: list[dict[str, Any]] = []
    for family in CONTRACT.FAMILY_IDS:
        family_streams = [row for row in streams if row["family"] == family]
        family_records = [row for row in rows if row["family"] == family]
        family_rows.append(
            {
                "family": family,
                **_independent_aggregate_counts(family_streams, family_records),
                "minimum_stream_qualified_count": min(
                    int(row["qualified_count"]) for row in family_streams
                ),
            }
        )
    factor_rows: list[dict[str, Any]] = []
    for factor in CONTRACT.STRATUM_FACTOR_IDS:
        levels: list[Any] = []
        for stream in streams:
            level = _independent_factor_level(stream, factor)
            if level not in levels:
                levels.append(level)
        for level in levels:
            factor_streams = [
                stream
                for stream in streams
                if _independent_factor_level(stream, factor) == level
            ]
            stream_indices = {
                int(stream["stream_index"]) for stream in factor_streams
            }
            factor_records = [
                row for row in rows if int(row["stream_index"]) in stream_indices
            ]
            factor_rows.append(
                {
                    "factor": factor,
                    "level": copy.deepcopy(level),
                    **_independent_aggregate_counts(
                        factor_streams, factor_records
                    ),
                }
            )
    below_target = zero + low
    failure_families = [
        family
        for family in CONTRACT.FAMILY_IDS
        if any(row["family"] == family for row in below_target)
    ]
    next_decision = (
        None
        if status == CONTRACT.GENERATOR_PANEL_AVAILABLE
        else _independent_failure_next_decision(failure_families)
    )
    failure_streams = [
        stream
        for stream in streams
        if stream["qualified_count"] < CONTRACT.TARGET_QUALIFIED_PER_STREAM
    ]
    qualified_count = sum(bool(row["qualified"]) for row in rows)
    nonqualified_count = len(rows) - qualified_count
    return CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "generator_metrics.v1"
            ),
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "v4_context_authority_content_digest": (
                CONTRACT.V4_CONTEXT_AUTHORITY["content_digest"]
            ),
            "stream_manifest_content_digest": (
                CONTRACT.STREAM_MANIFEST_CONTENT_DIGEST
            ),
            "allocation_authority_content_digest": (
                CONTRACT.GENERATOR_ALLOCATION_AUTHORITY["content_digest"]
            ),
            "terminal_record_count": len(rows),
            "terminal_unique_identity_count": len(
                {row["candidate_spec_id"] for row in rows}
            ),
            "maximum_candidate_count": CONTRACT.MAX_CANDIDATE_COUNT,
            "expected_stream_count": CONTRACT.STREAM_COUNT,
            "completed_stream_count": len(grouped),
            "missing_stream_count": CONTRACT.STREAM_COUNT - len(grouped),
            "duplicated_candidate_identity_count": (
                len(rows) - len({row["candidate_spec_id"] for row in rows})
            ),
            "stream_count": CONTRACT.STREAM_COUNT,
            "maximum_attempts_per_stream": CONTRACT.MAX_ATTEMPTS_PER_STREAM,
            "target_qualified_per_stream": CONTRACT.TARGET_QUALIFIED_PER_STREAM,
            "qualified_count": qualified_count,
            "nonqualified_count": nonqualified_count,
            "hard_stop_count": 0,
            "teacher_execution_count": sum(
                bool(row["state_disposition"]["teacher_executed"])
                for row in rows
            ),
            "disposition_counts": _independent_disposition_counts(rows),
            "disposition_rates": {
                name: count / len(rows)
                for name, count in _independent_disposition_counts(rows).items()
            },
            "stream_rows": streams,
            "family_rows": family_rows,
            "stratum_factor_rows": factor_rows,
            "v4_shortfall_resolution": (
                _independent_v4_shortfall_resolution(streams)
            ),
            "qualification_rate": qualified_count / len(rows),
            "rejection_rate": nonqualified_count / len(rows),
            "zero_yield_streams": zero,
            "low_yield_streams": low,
            "target_reached_streams": reached,
            "selected_candidate_indices": selected_indices,
            "selected_candidate_spec_ids": selected_ids,
            "status": status,
            "primary_classification": status,
            "next_decision": next_decision,
            "next_decision_evidence": {
                "below_target_stream_count": len(failure_streams),
                "below_target_families": failure_families,
                "failure_streams": [
                    {
                        "family": stream["family"],
                        "stratum_index": stream["stratum_index"],
                        "qualified_count": stream["qualified_count"],
                        "nonqualified_count": stream["nonqualified_count"],
                        "disposition_counts": stream["disposition_counts"],
                        "disposition_rates": stream["disposition_rates"],
                        "valid_initial_state_count": stream[
                            "valid_initial_state_count"
                        ],
                        "valid_initial_state_ever": bool(
                            stream["valid_initial_state_count"]
                        ),
                        "teacher_left_source_count": stream[
                            "teacher_left_source_count"
                        ],
                        "teacher_left_source_ever": bool(
                            stream["teacher_left_source_count"]
                        ),
                        "correct_port_ever_count": stream[
                            "correct_port_ever_count"
                        ],
                        "correct_port_ever": bool(
                            stream["correct_port_ever_count"]
                        ),
                        "contact_rejection_count": stream[
                            "contact_rejection_count"
                        ],
                        "contact_rejection_fraction": stream[
                            "contact_rejection_fraction"
                        ],
                        "contact_dominant": stream["contact_dominant"],
                        "common_physical_parameters": stream[
                            "common_physical_parameters"
                        ],
                    }
                    for stream in failure_streams
                ],
                "teacher_or_policy_change_inferred": False,
            },
            "downstream_scientific_execution_authorized": (
                status == CONTRACT.GENERATOR_PANEL_AVAILABLE
            ),
            "downstream_outcomes_opened": 0,
            "models_trained": 0,
            "development_only": True,
            "final_evaluation_eligible": False,
        }
    )


def _independent_mean(values: Sequence[Any], label: str) -> float:
    rows = [float(value) for value in values]
    if not rows or any(not math.isfinite(value) for value in rows):
        raise RegenerationError(f"{label} population is empty or nonfinite")
    return math.fsum(rows) / len(rows)


def _independent_correct_candidate(row: Mapping[str, Any]) -> bool:
    """Reapply the frozen correct-edge predicate without a metrics reducer call."""

    return bool(
        row["oracle_admissible"]
        and row["entered_correct_edge"]
        and row["successor_viable"]
        and row["positive_port_progress"]
        and not row["physics_contact"]
        and not row["stuck"]
    )


def _independent_materially_outperforms(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> bool:
    authority = CONTRACT.TARGET_MATERIALITY
    return bool(
        float(candidate["selected_correct_edge_execution_rate"])
        - float(baseline["selected_correct_edge_execution_rate"])
        >= float(authority["selected_correct_edge_execution_rate_improvement"])
        or float(candidate["correct_edge_top3_rate"])
        - float(baseline["correct_edge_top3_rate"])
        >= float(authority["correct_edge_top3_rate_improvement"])
        or float(baseline["normalized_port_regret"])
        - float(candidate["normalized_port_regret"])
        >= float(authority["normalized_port_regret_reduction"])
        or float(candidate["mean_selected_port_progress_m"])
        - float(baseline["mean_selected_port_progress_m"])
        >= float(authority["selected_port_progress_m_improvement"])
    )


def _independent_command_tracking(
    heldout_rows: Sequence[Mapping[str, Any]],
    fanout_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    ticks: list[Mapping[str, Any]] = []
    selected_conditions = {
        "FROZEN_CURRENT_VISUAL_RANKER",
        "ORACLE_BEST_ADMISSIBLE_CANDIDATE",
    }
    for row in heldout_rows:
        if row["condition_id"] not in selected_conditions:
            continue
        selected = row["selected_candidate_index"]
        if not isinstance(selected, int) or isinstance(selected, bool):
            raise RegenerationError("heldout selected candidate index drift")
        branches = fanout_by_state.get(str(row["state_id"]))
        if branches is None or not 0 <= selected < len(branches):
            raise RegenerationError("heldout selected branch is absent")
        ticks.extend(branches[selected]["command_tracking_rows"])
    vx_errors: list[float] = []
    yaw_errors: list[float] = []
    vy_values: list[float] = []
    sign_matches: list[float] = []
    for tick in ticks:
        command = tick["post_slew_command"]
        achieved = tick["mean_achieved_body_velocity"]
        vy_values.append(abs(float(achieved[1])))
        if tick["active_vx"]:
            vx_errors.append(abs(float(achieved[0]) - float(command[0])))
            sign_matches.append(
                float(float(achieved[0]) * float(command[0]) > 0.0)
            )
        if tick["active_yaw"]:
            yaw_errors.append(abs(float(achieved[2]) - float(command[2])))
            sign_matches.append(
                float(float(achieved[2]) * float(command[2]) > 0.0)
            )
    vx_mae = _independent_mean(vx_errors, "active vx") if vx_errors else 0.0
    vy_mean = _independent_mean(vy_values, "absolute vy")
    yaw_mae = _independent_mean(yaw_errors, "active yaw") if yaw_errors else 0.0
    sign_rate = (
        _independent_mean(sign_matches, "commanded sign") if sign_matches else 1.0
    )
    authority = CONTRACT.COMMAND_TRACKING_AUTHORITY
    passed = bool(
        vx_mae <= float(authority["vx_mae_maximum_mps"])
        and vy_mean <= float(authority["vy_absolute_mean_maximum_mps"])
        and yaw_mae <= float(authority["yaw_rate_mae_maximum_rad_s"])
        and sign_rate >= float(authority["commanded_sign_agreement_minimum"])
    )
    return {
        "selected_branch_count": len(ticks)
        // int(CONTRACT.HORIZON_TICKS[CONTRACT.PRIMARY_HORIZON]),
        "command_tick_count": len(ticks),
        "active_vx_component_count": len(vx_errors),
        "active_yaw_component_count": len(yaw_errors),
        "vx_mae_mps": vx_mae,
        "vy_absolute_mean_mps": vy_mean,
        "yaw_rate_mae_rad_s": yaw_mae,
        "commanded_sign_agreement_rate": sign_rate,
        "passed": passed,
    }


def _independent_classification(
    classification_input: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen V1/V4 component tree locally in the evaluator."""

    gate = CONTRACT.HANDOFF_GATE
    teacher_count = int(classification_input["teacher_correct_execution_count"])
    coverage = float(classification_input["coverage_rate"])
    top1 = float(classification_input["ranker_correct_edge_top1_rate"])
    top3 = float(classification_input["ranker_correct_edge_top3_rate"])
    execution = float(
        classification_input["ranker_selected_correct_edge_execution_rate"]
    )
    regret = float(classification_input["ranker_normalized_port_regret"])
    oracle_covered = float(
        classification_input["oracle_covered_state_correct_execution_rate"]
    )
    repeatability = float(classification_input["repeatability_rate"])
    family_minimum = int(
        classification_input["minimum_family_correct_execution_count"]
    )
    tracking = bool(classification_input["command_tracking_pass"])
    coverage_failure = coverage < float(gate["coverage_rate_minimum"])
    repeat_pass = repeatability >= float(gate["repeatability_rate_minimum"])
    oracle_succeeds = oracle_covered >= 1.0
    low_level_failure = bool(
        coverage > 0.0
        and teacher_count == int(gate["teacher_correct_execution_count"])
        and (not oracle_succeeds or not repeat_pass or not tracking)
    )
    ranker_gate_pass = bool(
        top1 >= float(gate["ranker_correct_edge_top1_rate_minimum"])
        and top3 >= float(gate["ranker_correct_edge_top3_rate_minimum"])
        and execution
        >= float(gate["ranker_selected_correct_edge_execution_rate_minimum"])
        and regret <= float(gate["ranker_normalized_port_regret_maximum"])
        and family_minimum >= int(gate["minimum_correct_execution_per_family"])
    )
    full_gate = bool(
        teacher_count == int(gate["teacher_correct_execution_count"])
        and not coverage_failure
        and oracle_succeeds
        and ranker_gate_pass
        and repeat_pass
        and tracking
    )
    if classification_input["selected_target_passes_handoff_gate"] is not full_gate:
        raise RegenerationError("independent full-handoff gate projection drift")
    target_failure = bool(
        classification_input["selected_target_id"] != "TARGET_NODE_CENTRE"
        and full_gate
        and classification_input[
            "selected_target_materially_outperforms_node_centre"
        ]
        is True
    )
    components = {
        "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT": target_failure,
        "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO": coverage_failure,
        "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO": bool(
            not coverage_failure
            and oracle_succeeds
            and repeat_pass
            and tracking
            and not ranker_gate_pass
        ),
        "LOW_LEVEL_PREFIX_EXECUTION_NO_GO": low_level_failure,
    }
    frozen_v1 = CONTRACT.V4.V3.V2.V1
    active = [
        name for name in frozen_v1.COMPONENT_PRECEDENCE if components[name]
    ]
    if len(active) > 1:
        primary = "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO"
    elif target_failure:
        primary = "GRAPH_EDGE_TARGET_REPRESENTATION_DEFECT"
    elif full_gate:
        primary = "PHYSICAL_GRAPH_EDGE_HANDOFF_SIGNAL"
    elif coverage_failure:
        primary = "LOCAL_ACTION_BANK_EDGE_COVERAGE_NO_GO"
    elif components["CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO"]:
        primary = "CURRENT_VISUAL_RANKER_EDGE_SELECTION_NO_GO"
    elif low_level_failure:
        primary = "LOW_LEVEL_PREFIX_EXECUTION_NO_GO"
    else:
        raise RegenerationError("independent classification tree has no disposition")
    next_decision = (
        frozen_v1.COMPOSITE_NEXT_BY_EARLIEST_COMPONENT[active[0]]
        if primary == "PHYSICAL_GRAPH_EDGE_HANDOFF_COMPOSITE_NO_GO"
        else CONTRACT.NEXT_DECISION_BY_CLASSIFICATION[primary]
    )
    return {
        "handoff_gate_passed": full_gate,
        "component_failures": components,
        "active_components_in_precedence_order": active,
        "earliest_failing_component": active[0] if active else None,
        "primary_classification": primary,
        "next_decision": next_decision,
    }


def _independent_validate_comparator_alias_authority(value: Any) -> None:
    """Require the user-facing teacher alias without renaming stored rows."""

    try:
        alias = CONTRACT.validate_content_digest(value)
    except (
        TypeError,
        ValueError,
        CONTRACT.PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError,
    ) as exc:
        raise RegenerationError(
            "independent heldout comparator alias authority is invalid"
        ) from exc
    expected_alias_semantics = {
        "ordinal_base": 1,
        "internal_condition_order": list(CONTRACT.HELDOUT_CONDITION_IDS),
        "user_facing_comparator_order": [
            "DETERMINISTIC_KINEMATICS",
            "FROZEN_CURRENT_VISUAL_RANKER",
            "ORACLE_BEST_ADMISSIBLE_CANDIDATE",
            "PHYSICAL_TEACHER",
        ],
        "user_facing_to_internal_condition_id": {
            "DETERMINISTIC_KINEMATICS": "DETERMINISTIC_KINEMATICS",
            "FROZEN_CURRENT_VISUAL_RANKER": "FROZEN_CURRENT_VISUAL_RANKER",
            "ORACLE_BEST_ADMISSIBLE_CANDIDATE": (
                "ORACLE_BEST_ADMISSIBLE_CANDIDATE"
            ),
            "PHYSICAL_TEACHER": "TEACHER_TRACE",
        },
        "physical_teacher_comparator_ordinal": 4,
        "internal_condition_ids_renamed": False,
        "new_teacher_execution": False,
        "model_policy_formula_threshold_or_gate_change": False,
    }
    if (
        canonical_bytes(alias)
        != canonical_bytes(CONTRACT.HELDOUT_COMPARATOR_ALIAS_AUTHORITY)
        or any(
            alias.get(key) != value
            for key, value in expected_alias_semantics.items()
        )
    ):
        raise RegenerationError(
            "independent heldout comparator alias authority drift"
        )


def _independent_validate_downstream_decision(
    metrics: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    """Cross-check decisive success gates directly from official row evidence."""

    if metrics["generator_status"] != CONTRACT.GENERATOR_PANEL_AVAILABLE:
        return
    panel = evidence["panel_manifest.json"]
    selection = evidence["development_target_selection.json"]
    fanout = evidence["candidate_fanout.jsonl"]
    heldout = evidence["heldout_scores.jsonl"]
    repeats = evidence["repeatability.jsonl"]
    states = list(panel["states"])
    heldout_states = [
        row for row in states if row["role"] == "DEVELOPMENT_HELDOUT"
    ]
    if len(states) != 64 or len(heldout_states) != 16:
        raise RegenerationError("independent downstream panel cardinality drift")
    fanout_by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fanout:
        fanout_by_state[str(row["state_id"])].append(row)
    for state in states:
        state_id = str(state["state_id"])
        rows = sorted(
            fanout_by_state.get(state_id, []),
            key=lambda row: int(row["branch_candidate_index"]),
        )
        if [row["branch_candidate_index"] for row in rows] != list(
            range(len(CONTRACT.CANDIDATE_IDS))
        ):
            raise RegenerationError("independent fanout branch population drift")
        fanout_by_state[state_id] = rows
    correct_counts = {
        str(state["state_id"]): sum(
            _independent_correct_candidate(row)
            for row in fanout_by_state[str(state["state_id"])]
        )
        for state in heldout_states
    }
    coverage = {
        state_id: count >= 1 for state_id, count in correct_counts.items()
    }
    coverage_rate = _independent_mean(coverage.values(), "heldout coverage")
    by_condition = {
        condition: [row for row in heldout if row["condition_id"] == condition]
        for condition in CONTRACT.HELDOUT_CONDITION_IDS
    }
    if any(len(rows) != 16 for rows in by_condition.values()):
        raise RegenerationError("independent heldout condition population drift")
    teachers = by_condition["TEACHER_TRACE"]
    rankers = by_condition["FROZEN_CURRENT_VISUAL_RANKER"]
    oracles = by_condition["ORACLE_BEST_ADMISSIBLE_CANDIDATE"]
    covered_oracles = [row for row in oracles if coverage[str(row["state_id"])]]
    if not covered_oracles:
        raise RegenerationError("independent oracle covered population is empty")
    teacher_count = sum(bool(row["teacher_correct_execution"]) for row in teachers)
    top1 = _independent_mean(
        [row["correct_edge_top1"] for row in rankers], "ranker top1"
    )
    top3 = _independent_mean(
        [row["correct_edge_top3"] for row in rankers], "ranker top3"
    )
    execution = _independent_mean(
        [row["selected_correct_edge_execution"] for row in rankers],
        "ranker correct execution",
    )
    regret = _independent_mean(
        [row["normalized_port_regret"] for row in rankers], "ranker regret"
    )
    oracle_execution = _independent_mean(
        [row["selected_correct_edge_execution"] for row in oracles],
        "oracle execution",
    )
    oracle_covered = _independent_mean(
        [row["selected_correct_edge_execution"] for row in covered_oracles],
        "covered oracle execution",
    )
    repeatability = _independent_mean(
        [row["repeat_success"] for row in repeats], "repeatability"
    )
    family_minimum = min(
        sum(
            bool(row["selected_correct_edge_execution"])
            for row in rankers
            if row["family"] == family
        )
        for family in CONTRACT.FAMILY_IDS
    )
    tracking = _independent_command_tracking(heldout, fanout_by_state)
    summaries = {
        row["target_id"]: row for row in selection["target_summaries"]
    }
    if set(summaries) != set(CONTRACT.TARGET_IDS):
        raise RegenerationError("independent target summary population drift")
    selected_target = selection["selected_target_id"]
    materiality = _independent_materially_outperforms(
        summaries[selected_target], summaries["TARGET_NODE_CENTRE"]
    )
    gate = CONTRACT.HANDOFF_GATE
    full_gate = bool(
        teacher_count == int(gate["teacher_correct_execution_count"])
        and coverage_rate >= float(gate["coverage_rate_minimum"])
        and oracle_covered >= 1.0
        and top1 >= float(gate["ranker_correct_edge_top1_rate_minimum"])
        and top3 >= float(gate["ranker_correct_edge_top3_rate_minimum"])
        and execution
        >= float(gate["ranker_selected_correct_edge_execution_rate_minimum"])
        and regret <= float(gate["ranker_normalized_port_regret_maximum"])
        and repeatability >= float(gate["repeatability_rate_minimum"])
        and tracking["passed"]
        and family_minimum >= int(gate["minimum_correct_execution_per_family"])
    )
    classification_input = {
        "teacher_correct_execution_count": teacher_count,
        "coverage_rate": coverage_rate,
        "ranker_correct_edge_top1_rate": top1,
        "ranker_correct_edge_top3_rate": top3,
        "ranker_selected_correct_edge_execution_rate": execution,
        "ranker_normalized_port_regret": regret,
        "oracle_selected_correct_edge_execution_rate": oracle_execution,
        "oracle_covered_state_correct_execution_rate": oracle_covered,
        "repeatability_rate": repeatability,
        "command_tracking_pass": tracking["passed"],
        "minimum_family_correct_execution_count": family_minimum,
        "selected_target_id": selected_target,
        "selected_target_passes_handoff_gate": full_gate,
        "selected_target_materially_outperforms_node_centre": materiality,
    }
    decision = _independent_classification(classification_input)
    downstream = metrics["downstream"]
    _independent_validate_comparator_alias_authority(
        downstream.get("heldout", {}).get("comparator_alias_authority")
    )
    expected_projection = {
        "classification_input": classification_input,
        "command_tracking": tracking,
        "gate": {
            "authority": copy.deepcopy(CONTRACT.HANDOFF_GATE),
            "passed": decision["handoff_gate_passed"],
        },
        "component_failures": decision["component_failures"],
        "active_components_in_precedence_order": decision[
            "active_components_in_precedence_order"
        ],
        "earliest_failing_component": decision["earliest_failing_component"],
    }
    supplied_projection = {key: downstream[key] for key in expected_projection}
    if canonical_bytes(supplied_projection) != canonical_bytes(expected_projection):
        raise RegenerationError(
            "downstream decisive metrics differ from independent evaluator reduction"
        )
    if (
        metrics["primary_classification"] != decision["primary_classification"]
        or metrics["next_decision"] != decision["next_decision"]
        or metrics["runtime_environments"]["encoder"]["checkpoint_sha256"]
        != "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
        or metrics["runtime_environments"]["ranker"]["checkpoint_sha256"]
        != "e1a2a58ff527b4d2bc210f6b1c8fd1d00d87873691db64e8c3ccb2257fa6c127"
        or CONTRACT.PRIMARY_HORIZON != "H3"
        or float(CONTRACT.ROUTE_LOOKAHEAD_DISTANCE_M) != 0.50
    ):
        raise RegenerationError("independent downstream decision/checkpoint drift")


def _rebuild_predecessor_identity_and_seed_nonoverlap() -> dict[str, Any]:
    """Independently reconstruct all bound predecessor identity registries."""

    from scripts import run_occluded_goal_topological_belief_v1 as OGTB

    documents = OGTB._load_bound_prior_authorities()  # noqa: SLF001
    prior = OGTB._project_prior_identities(documents)  # noqa: SLF001
    inherited = sorted(int(value) for value in prior["numeric_seed"])
    binding = CONTRACT.V4.PRIOR_SCENE_EXCLUSION_AUTHORITY["v2_panel_binding"]
    path = Path(str(binding["path"]))
    info = _require_regular(path, "bound predecessor V2 panel")
    if (
        int(info.st_size) != int(binding["bytes"])
        or sha256_file(path) != str(binding["sha256"])
    ):
        raise RegenerationError("bound predecessor V2 panel drift")
    try:
        panel = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise RegenerationError("bound predecessor V2 panel is invalid") from exc
    stack: list[Any] = [panel]
    while stack:
        value = stack.pop()
        if isinstance(value, Mapping):
            scene = value.get("scene_id")
            if isinstance(scene, str) and scene:
                prior["scene_identity"].add(scene)
                prior["scene_identity_sha256"].add(
                    hashlib.sha256(scene.encode("utf-8")).hexdigest()
                )
            for field in ("state_id", "episode_id", "graph_id"):
                identity = value.get(field)
                if isinstance(identity, str) and identity:
                    prior["episode_or_state_identity"].add(identity)
            seed = value.get("procedural_seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                prior["numeric_seed"].add(seed)
            for field in (
                "episode_path_id",
                "geometry_digest",
                "geometry_sha256",
                "scene_dir",
                "source_path",
            ):
                identity = value.get(field)
                if isinstance(identity, str) and identity:
                    prior["textual_path_geometry_or_source_identity"].add(identity)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    broad = CONTRACT.V4.PRIOR_SCENE_EXCLUSION_AUTHORITY[
        "broad_prior_plus_v2_projection"
    ]
    checks = (
        (
            "scene_identity",
            "scene_identity_count",
            "scene_identity_canonical_json_sha256",
            False,
        ),
        (
            "scene_identity_sha256",
            "scene_identity_sha256_count",
            "scene_identity_sha256_canonical_json_sha256",
            False,
        ),
        (
            "episode_or_state_identity",
            "episode_state_or_graph_identity_count",
            "episode_state_or_graph_identity_canonical_json_sha256",
            False,
        ),
        (
            "numeric_seed",
            "numeric_seed_count",
            "numeric_seed_canonical_json_sha256",
            True,
        ),
        (
            "textual_path_geometry_or_source_identity",
            "textual_path_geometry_or_source_identity_count",
            "textual_path_geometry_or_source_identity_canonical_json_sha256",
            False,
        ),
    )
    for key, count_key, digest_key, numeric in checks:
        values = list(prior[key])
        if (
            len(values) != int(broad[count_key])
            or OGTB._identity_projection_digest(  # noqa: SLF001
                values, numeric=numeric
            )
            != str(broad[digest_key])
        ):
            raise RegenerationError(f"broad predecessor identity drift: {key}")
    structured = list(prior["structured_waypoint_or_sequence_path"])
    if (
        len(structured)
        != int(broad["structured_waypoint_or_sequence_path_count"])
        or OGTB._structured_identity_projection_digest(structured)  # noqa: SLF001
        != str(
            broad[
                "structured_waypoint_or_sequence_path_canonical_json_sha256"
            ]
        )
    ):
        raise RegenerationError("broad predecessor structured identity drift")
    fixtures = V4._nonregistered_family_fixture_specs()  # noqa: SLF001
    seed_registries = {
        "v4_inherited_broad_numeric_seed_registry": inherited,
        "v4_registered_candidate_seeds": sorted(
            int(value) for value in CONTRACT.V4.PROCEDURAL_SEED_VALUES
        ),
        "v4_nonregistered_family_fixture_seeds": sorted(
            int(row["procedural_seed"]) for row in fixtures
        ),
        "v4_broad_predecessor_numeric_seed_registry": sorted(
            int(value) for value in prior["numeric_seed"]
        ),
    }
    seed_result = CONTRACT.validate_seed_nonoverlap(seed_registries)
    candidates = CONTRACT.build_candidate_identity_manifest()
    fields = ("candidate_spec_id", "scene_id", "state_id", "episode_id", "graph_id")
    current_by_field = {
        field: {str(row[field]) for row in candidates} for field in fields
    }
    current = set().union(*current_by_field.values())
    if any(
        not value.startswith(f"{CONTRACT.IDENTITY_NAMESPACE}-")
        or value.startswith(("pgehq-", "ogtb-"))
        for value in current
    ):
        raise RegenerationError("successor identity namespace drift")
    v4_specs = CONTRACT.V4.build_candidate_specs()
    v4_prior = {
        str(row[field])
        for row in (*v4_specs, *fixtures)
        for field in fields
        if field in row
    }
    broad_strings = set().union(
        set(prior["scene_identity"]),
        set(prior["episode_or_state_identity"]),
        set(prior["textual_path_geometry_or_source_identity"]),
    )
    scene_hashes = {
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in current_by_field["scene_id"]
    }
    fixture_seeds = {int(row["procedural_seed"]) for row in fixtures}
    overlaps = {
        "broad_predecessor_identity_overlap_count": len(current & broad_strings),
        "broad_predecessor_scene_hash_overlap_count": len(
            scene_hashes & set(prior["scene_identity_sha256"])
        ),
        "v4_registered_and_fixture_identity_overlap_count": len(current & v4_prior),
        "v4_fixture_seed_overlap_count": len(
            fixture_seeds & set(CONTRACT.PROCEDURAL_SEED_VALUES)
        ),
    }
    if any(overlaps.values()):
        raise RegenerationError(f"successor predecessor overlap: {overlaps}")
    broad_projection = {
        key: {
            "count": len(prior[key]),
            "canonical_sorted_unique_no_lf_sha256": _sorted_projection_sha256(
                list(prior[key])
            ),
        }
        for key in (
            "scene_identity",
            "scene_identity_sha256",
            "episode_or_state_identity",
            "numeric_seed",
            "textual_path_geometry_or_source_identity",
        )
    }
    broad_projection["structured_waypoint_or_sequence_path"] = {
        "count": len(structured),
        "canonical_sorted_unique_no_lf_sha256": hashlib.sha256(
            CONTRACT.canonical_json_bytes(
                sorted(
                    {
                        CONTRACT.canonical_json_bytes(list(value))[:-1].decode(
                            "utf-8"
                        )
                        for value in structured
                    }
                )
            )[:-1]
        ).hexdigest(),
        "intersection_not_applied": (
            "structured family and stratum semantics are scientifically invariant"
        ),
    }
    projection = {
        "seed_nonoverlap": seed_result,
        "broad_predecessor_identity_projection": broad_projection,
        "v4_registered_and_fixture_identity_count": len(v4_prior),
        "v4_registered_and_fixture_identity_projection_sha256": (
            _sorted_projection_sha256(list(v4_prior))
        ),
        "successor_identity_count": len(current),
        "successor_identity_projection_sha256": _sorted_projection_sha256(
            list(current)
        ),
        "successor_namespace": CONTRACT.IDENTITY_NAMESPACE,
        "successor_namespace_required_prefix": f"{CONTRACT.IDENTITY_NAMESPACE}-",
        **overlaps,
        "structured_semantic_overlap_is_not_an_identity_gate": True,
        "all_identity_and_seed_overlap_counts_zero": True,
    }
    try:
        return CONTRACT.validate_identity_and_seed_nonoverlap(projection)
    except Exception as exc:
        raise RegenerationError(
            "predecessor identity/seed projection differs from frozen authority"
        ) from exc


def _leaf_names(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise RegenerationError("official root is not an ordinary directory")
    names: set[str] = set()
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise RegenerationError("official root contains a non-file leaf")
            names.add(entry.name)
    return names


def _load_common_evidence(root: Path) -> dict[str, Any]:
    _contract_raw, runtime = _load_json(root / "contract.json", "runtime contract")
    _context_raw, context = _load_json(root / "V4_context.json", "V4 context")
    _manifest_raw, manifest = _load_json(
        root / "generator_stream_manifest.json", "generator stream manifest"
    )
    terminal_raw, terminal_rows = _load_jsonl(
        root / "generator_terminal_records.jsonl", "generator terminal records"
    )
    _generator_raw, generator = _load_json(
        root / "generator_metrics.json", "generator metrics"
    )
    try:
        runtime = CONTRACT.validate_runtime_contract(runtime)
        context = CONTRACT.validate_v4_context(context)
        manifest = CONTRACT.validate_generator_stream_manifest(manifest)
    except Exception as exc:
        raise RegenerationError("contract or generator authority drift") from exc
    terminal_rows = _call(
        "validate_generator_terminal_records_jsonl", terminal_raw
    )
    generator = _call("validate_generator_metrics", generator, terminal_rows)
    independently_rebuilt_generator = _independent_generator_metrics(
        terminal_rows
    )
    if canonical_bytes(generator) != canonical_bytes(
        independently_rebuilt_generator
    ):
        raise RegenerationError(
            "generator metrics differ from independent evaluator reduction"
        )
    if generator["stream_manifest_content_digest"] != manifest["content_digest"]:
        raise RegenerationError("generator metric/manifest cross-link drift")
    return {
        "runtime_contract": runtime,
        "v4_context": context,
        "generator_stream_manifest": manifest,
        "generator_terminal_records_jsonl": terminal_raw,
        "generator_terminal_records": terminal_rows,
        "generator_metrics": generator,
        "common_bindings": {
            "contract.json": _binding(root / "contract.json", root=root),
            "V4_context.json": _binding(root / "V4_context.json", root=root),
            "generator_stream_manifest.json": _binding(
                root / "generator_stream_manifest.json", root=root
            ),
            "generator_terminal_records.jsonl": _binding(
                root / "generator_terminal_records.jsonl", root=root
            ),
            "generator_metrics.json": _binding(
                root / "generator_metrics.json", root=root
            ),
        },
    }


def _load_downstream_evidence(root: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for leaf in CONTRACT.DOWNSTREAM_OUTPUT_LEAVES:
        path = root / leaf
        if leaf.endswith(".jsonl"):
            raw, value = _load_jsonl(path, leaf)
        else:
            raw, value = _load_json(path, leaf)
        evidence[leaf] = value
        evidence[f"{leaf}:raw"] = raw
    return evidence


def _require_real_generator_runtime_marker(material: Path) -> None:
    """Fail production publication before accepting truthfully fake evidence."""

    _raw, runtime = _load_json(
        material / "generator_runtime_environment.json",
        "generator runtime environment",
    )
    if runtime.get("fake_runtime") is not False:
        raise RegenerationError(
            "production generator runtime evidence is absent or marked fake"
        )


def _load_material_shard(
    material: Path, relative: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    directory = material / relative
    if directory.is_symlink() or not directory.is_dir():
        raise RegenerationError("terminal material directory drift")
    _, metadata = _load_json(directory / "metadata.json", "terminal metadata")
    payload_path = directory / "payload.npz"
    binding = _binding(payload_path, root=material)
    payload = metadata.get("payload")
    if (
        not isinstance(payload, Mapping)
        or payload.get("role") != "material_shard_payload"
        or payload.get("kind") != "npz"
        or {
            key: payload.get(key)
            for key in CONTRACT.MATERIAL_FILE_BINDING_FIELDS
        }
        != binding
    ):
        raise RegenerationError("terminal payload binding drift")
    with zipfile.ZipFile(payload_path, mode="r") as archive:
        _call("validate_npz_archive_comment", archive.comment)
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }
    _call(
        "validate_persisted_array_evidence",
        metadata.get("persisted_array_evidence"),
        reopened_arrays=arrays,
    )
    return metadata, arrays


def _material_tree_inventory(material: Path) -> tuple[set[str], set[str]]:
    """Return exact ordinary relative files/directories, rejecting links."""

    files: set[str] = set()
    directories: set[str] = set()
    pending = [material]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = str(path.relative_to(material))
                if entry.is_symlink():
                    raise RegenerationError("material tree contains a symlink")
                if entry.is_dir(follow_symlinks=False):
                    directories.add(relative)
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    files.add(relative)
                else:
                    raise RegenerationError("material tree contains a special file")
    return files, directories


def _require_no_v4_hardlink_reuse(
    output_root: Path, material_root: Path, material_files: Sequence[str]
) -> dict[str, Any]:
    v4_root = Path(CONTRACT.V4.OUTPUT_ROOT)
    v4_material = Path(CONTRACT.V4.MATERIAL_ROOT)
    hashes_raw, manifest = _load_json(
        v4_root / "file_hashes.json", "V4 file hashes"
    )
    hashes_binding = CONTRACT.V4_CONTEXT_AUTHORITY["bindings"]["file_hashes"]
    if (
        len(hashes_raw) != int(hashes_binding["bytes"])
        or hashlib.sha256(hashes_raw).hexdigest() != str(hashes_binding["sha256"])
    ):
        raise RegenerationError("V4 file-hash authority binding drift")
    official_rows = manifest.get("files")
    expected_official = {
        "contract.json",
        "metrics.json",
        "panel_adequacy.json",
        "qualification_state_dispositions.jsonl",
        "result.json",
        "result.md",
        "scientific_invariance_receipt.json",
        "v1_v2_v3_custody_and_nonreuse.json",
    }
    if (
        not isinstance(official_rows, list)
        or len(official_rows) != 8
        or {row.get("path") for row in official_rows if isinstance(row, Mapping)}
        != expected_official
    ):
        raise RegenerationError("V4 official inventory authority drift")
    predecessor_paths = [v4_root / "file_hashes.json"]
    for row in official_rows:
        path = v4_root / str(row["path"])
        info = _require_regular(path, "V4 official evidence")
        if int(info.st_size) != int(row["bytes"]) or sha256_file(path) != str(
            row["sha256"]
        ):
            raise RegenerationError("V4 official binding drift")
        predecessor_paths.append(path)
    _, ledger = _load_jsonl(
        v4_root / "qualification_state_dispositions.jsonl", "V4 qualification ledger"
    )
    if len(ledger) != 256 or [row.get("pool_index") for row in ledger] != list(range(256)):
        raise RegenerationError("V4 qualification ledger population drift")
    material_paths = [
        v4_material / "material_contract.json",
        v4_material / "prospective_pool.json",
    ]
    v4_physical_shard_sha256s: list[str] = []
    _, v4_material_contract = _load_json(
        material_paths[0], "V4 material contract"
    )
    _, v4_pool = _load_json(material_paths[1], "V4 prospective pool")
    try:
        CONTRACT.V4.validate_content_digest(v4_material_contract)
        CONTRACT.V4.validate_content_digest(v4_pool)
    except Exception as exc:
        raise RegenerationError("V4 material initialization digest drift") from exc
    if (
        v4_pool.get("experiment_id") != CONTRACT.V4.EXPERIMENT_ID
        or v4_pool.get("source_freeze_commit") != CONTRACT.V4_SOURCE_FREEZE_COMMIT
        or v4_pool.get("specs") != CONTRACT.V4.build_candidate_specs()
        or v4_material_contract.get("experiment_id") != CONTRACT.V4.EXPERIMENT_ID
        or v4_material_contract.get("source_freeze_commit")
        != CONTRACT.V4_SOURCE_FREEZE_COMMIT
        or v4_material_contract.get("prospective_pool_sha256")
        != sha256_file(material_paths[1])
        or v4_material_contract.get("official_contract_sha256")
        != sha256_file(v4_root / "contract.json")
    ):
        raise RegenerationError("V4 material initialization binding drift")
    for row in ledger:
        for field in ("material_metadata_binding", "material_payload_binding"):
            binding = row.get(field)
            if not isinstance(binding, Mapping):
                raise RegenerationError("V4 material binding absent")
            path = v4_material / str(binding.get("path"))
            info = _require_regular(path, "V4 material evidence")
            if int(info.st_size) != int(binding.get("bytes", -1)) or sha256_file(
                path
            ) != str(binding.get("sha256")):
                raise RegenerationError("V4 material binding drift")
            material_paths.append(path)
            v4_physical_shard_sha256s.append(str(binding["sha256"]))
    if len(v4_physical_shard_sha256s) != 512:
        raise RegenerationError("V4 physical-shard inventory drift")
    if len({str(path) for path in material_paths}) != 514:
        raise RegenerationError("V4 material inventory cardinality drift")
    receipt = Path(CONTRACT.V4.EXTERNAL_REGENERATION_RECEIPT)
    receipt_binding = CONTRACT.V4_CONTEXT_AUTHORITY["bindings"][
        "independent_reducer_receipt"
    ]
    receipt_info = _require_regular(receipt, "V4 independent receipt")
    if (
        int(receipt_info.st_size) != int(receipt_binding["bytes"])
        or sha256_file(receipt) != str(receipt_binding["sha256"])
    ):
        raise RegenerationError("V4 independent receipt binding drift")
    predecessor_paths.extend(material_paths)
    predecessor_paths.append(receipt)
    predecessor_inodes: set[tuple[int, int]] = set()
    for path in predecessor_paths:
        info = _require_regular(path, "V4 evidence")
        inode = (int(info.st_dev), int(info.st_ino))
        if inode in predecessor_inodes:
            raise RegenerationError("V4 evidence contains an inode alias")
        predecessor_inodes.add(inode)
    successor_paths = [
        *(output_root / name for name in _leaf_names(output_root)),
        *(material_root / relative for relative in material_files),
    ]
    observed: set[tuple[int, int]] = set()
    for path in successor_paths:
        info = _require_regular(path, "successor evidence")
        inode = (int(info.st_dev), int(info.st_ino))
        if inode in predecessor_inodes:
            raise RegenerationError("successor evidence hard-links V4 context")
        if inode in observed:
            raise RegenerationError("successor evidence contains a hard-link alias")
        observed.add(inode)
    v4_physical_digests = set(v4_physical_shard_sha256s)
    successor_physical_paths = [
        material_root / relative
        for relative in material_files
        if Path(relative).name in {"metadata.json", "payload.npz"}
    ]
    overlap_count = sum(
        sha256_file(path) in v4_physical_digests
        for path in successor_physical_paths
    )
    if overlap_count:
        raise RegenerationError("successor material copies V4 physical shards")
    return {
        "v4_physical_shard_file_count": len(v4_physical_shard_sha256s),
        "v4_physical_shard_sha256_overlap_count": overlap_count,
        "v4_physical_shard_copy_reuse_detected": False,
    }


def _validate_persisted_material(
    material: Path,
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    *,
    downstream_evidence: Mapping[str, Any] | None = None,
    allow_fake_runtime: bool = False,
) -> dict[str, Any]:
    if material.is_symlink() or not material.is_dir():
        raise RegenerationError("material root is absent or invalid")
    _, material_contract = _load_json(
        material / "material_contract.json", "material contract"
    )
    try:
        material_contract = CONTRACT.validate_material_contract(material_contract)
    except Exception as exc:
        raise RegenerationError("material contract digest drift") from exc
    if (
        material_contract.get("experiment_id") != EXPERIMENT_ID
        or material_contract.get("source_freeze_commit")
        != runtime["source_freeze_commit"]
        or material_contract.get("runtime_contract_content_digest")
        != runtime["content_digest"]
        or material_contract.get("models_trained") != 0
        or material_contract.get("stream_manifest_content_digest")
        != runtime["scientific_contract"]["generator_allocation_authority"][
            "stream_manifest_content_digest"
        ]
    ):
        raise RegenerationError("material/runtime contract drift")
    rebuilt_nonoverlap = _rebuild_predecessor_identity_and_seed_nonoverlap()
    if (
        material_contract.get("predecessor_identity_and_seed_nonoverlap")
        != rebuilt_nonoverlap
    ):
        raise RegenerationError(
            "material predecessor identity/seed nonoverlap drift"
        )
    metadata_rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    grouped: dict[int, list[Mapping[str, Any]]] = {
        index: [] for index in range(CONTRACT.STREAM_COUNT)
    }
    for record in records:
        index = int(record["candidate_index"])
        family = str(record["family"])
        stratum = int(record["stratum_index"])
        attempt = int(record["attempt_index"])
        spec = CONTRACT.build_candidate_spec(family, stratum, attempt)
        if index != spec["candidate_index"]:
            raise RegenerationError("terminal candidate coordinate drift")
        grouped[int(spec["stream_index"])].append(record)
        expected_directory = (
            f"qualification/{spec['stream_id']}/attempt-{attempt:02d}"
        )
        expected_metadata = f"{expected_directory}/metadata.json"
        expected_payload = f"{expected_directory}/payload.npz"
        if (
            record["material_metadata_binding"]["path"] != expected_metadata
            or record["material_payload_binding"]["path"] != expected_payload
        ):
            raise RegenerationError("terminal material path drift")
        metadata, arrays = _load_material_shard(material, expected_directory)
        if _binding(material / expected_metadata, root=material) != record[
            "material_metadata_binding"
        ] or _binding(material / expected_payload, root=material) != record[
            "material_payload_binding"
        ]:
            raise RegenerationError("terminal material file binding drift")
        validation = _call(
            "validate_qualification_material_shard",
            metadata,
            reopened_arrays=arrays,
            expected_pool_index=index,
        )
        evidence = metadata["persisted_array_evidence"]
        rebuilt = _call(
            "build_generator_terminal_record",
            metadata["state_disposition"],
            source_freeze_commit=metadata["source_freeze_commit"],
            runtime_contract_content_digest=metadata[
                "runtime_contract_content_digest"
            ],
            material_metadata_binding=record["material_metadata_binding"],
            material_payload_binding=record["material_payload_binding"],
            persisted_array_evidence_sha256=hashlib.sha256(
                CONTRACT.canonical_json_bytes(evidence)[:-1]
            ).hexdigest(),
        )
        if canonical_bytes(rebuilt) != canonical_bytes(record):
            raise RegenerationError("terminal record/material cross-link drift")
        metadata_rows.append(metadata)
        validations.append(validation)
    expected_files = {"material_contract.json", "generator_runtime_environment.json"}
    expected_directories = {
        "qualification",
        "selected",
        "fanout",
        "repeatability",
        "downstream_workspace",
    }
    for family in CONTRACT.FAMILY_IDS:
        for stratum in range(CONTRACT.STRATA_PER_FAMILY):
            stream = CONTRACT.stream_index(family, stratum)
            rows = grouped[stream]
            attempts = [int(row["attempt_index"]) for row in rows]
            if attempts != list(range(len(rows))):
                raise RegenerationError("stream attempted prefix drift")
            stream_id = str(CONTRACT.build_candidate_spec(family, stratum, 0)["stream_id"])
            stream_relative = f"qualification/{stream_id}"
            expected_directories.add(stream_relative)
            completion_path = material / stream_relative / "stream_completion.json"
            _, completion = _load_json(completion_path, "stream completion")
            try:
                CONTRACT.validate_content_digest(completion)
            except Exception as exc:
                raise RegenerationError("stream completion digest drift") from exc
            qualified = sum(bool(row["qualified"]) for row in rows)
            expected_completion = CONTRACT.attach_content_digest(
                {
                    "schema": (
                        "physical_handoff_stratified_generator_successor_v1."
                        "stream_completion.v1"
                    ),
                    "experiment_id": EXPERIMENT_ID,
                    "stream_index": stream,
                    "stream_id": stream_id,
                    "family": family,
                    "stratum_index": stratum,
                    "attempt_count": len(rows),
                    "qualified_count": qualified,
                    "terminal_candidate_indices": [
                        int(row["candidate_index"]) for row in rows
                    ],
                    "termination_reason": (
                        "TARGET_QUALIFIED_REACHED"
                        if qualified == CONTRACT.TARGET_QUALIFIED_PER_STREAM
                        else "ATTEMPT_LIMIT_REACHED"
                    ),
                    "source_freeze_commit": runtime["source_freeze_commit"],
                    "runtime_contract_content_digest": runtime["content_digest"],
                }
            )
            if canonical_bytes(completion) != canonical_bytes(expected_completion):
                raise RegenerationError("stream completion differs from terminal prefix")
            expected_files.add(f"{stream_relative}/stream_completion.json")
            for attempt in attempts:
                attempt_relative = f"{stream_relative}/attempt-{attempt:02d}"
                expected_directories.add(attempt_relative)
                expected_files.update(
                    {
                        f"{attempt_relative}/metadata.json",
                        f"{attempt_relative}/payload.npz",
                    }
                )
    runtime_environment = _call(
        "build_generator_runtime_environment",
        metadata_rows,
        runtime,
        allow_fake_runtime=allow_fake_runtime,
    )
    _call(
        "validate_generator_runtime_environment",
        runtime_environment,
        runtime,
        metadata_rows=metadata_rows,
        allow_fake_runtime=allow_fake_runtime,
    )
    _, persisted_runtime = _load_json(
        material / "generator_runtime_environment.json",
        "generator runtime environment",
    )
    if canonical_bytes(persisted_runtime) != canonical_bytes(runtime_environment):
        raise RegenerationError("persisted generator runtime differs from rebuild")
    files, directories = _material_tree_inventory(material)
    available = all(
        sum(bool(row["qualified"]) for row in grouped[index])
        == CONTRACT.TARGET_QUALIFIED_PER_STREAM
        for index in range(CONTRACT.STREAM_COUNT)
    )
    if not available:
        if files != expected_files or directories != expected_directories:
            raise RegenerationError("generator-terminal material inventory drift")
        downstream_validation: dict[str, Any] | None = None
    else:
        if downstream_evidence is None:
            raise RegenerationError("success material lacks downstream evidence")
        documents = {
            leaf: downstream_evidence[leaf]
            for leaf in (
                "panel_manifest.json",
                "split_manifest.json",
                "state_snapshot_index.json",
                "teacher_trace_index.json",
                "edge_port_index.json",
                "target_contracts.json",
            )
        }
        handoff = _call(
            "build_panel_handoff", records, downstream_evidence["generator_metrics"]
        )
        specs = sorted(
            (
                CONTRACT.validate_candidate_spec(spec)
                for spec in handoff["selected_candidate_specs"]
            ),
            key=lambda spec: spec["state_id"],
        )
        selected_validations: list[dict[str, Any]] = []
        for panel_index, spec in enumerate(specs):
            relative = f"selected/{spec['state_id']}"
            expected_directories.add(relative)
            expected_files.update(
                {f"{relative}/metadata.json", f"{relative}/payload.npz"}
            )
            metadata, arrays = _load_material_shard(material, relative)
            selected_validations.append(
                _call(
                    "validate_selected_reset_material_shard",
                    metadata,
                    reopened_arrays=arrays,
                    selected_candidate_spec=spec,
                    expected_panel_index=panel_index,
                    material_metadata_binding=_binding(
                        material / relative / "metadata.json", root=material
                    ),
                )
            )
        _call(
            "validate_frozen_panel_documents",
            documents,
            records,
            downstream_evidence["generator_metrics"],
            handoff,
            validations,
            selected_validations,
            runtime,
            allow_fake_runtime=allow_fake_runtime,
        )
        encoding_relative = "downstream_workspace/encoding"
        expected_directories.add(encoding_relative)
        expected_files.update(
            {
                f"{encoding_relative}/metadata.json",
                f"{encoding_relative}/payload.npz",
            }
        )
        encoding_metadata, encoding_arrays = _load_material_shard(
            material, encoding_relative
        )
        encoding_validation = _call(
            "validate_encoding_material_shard",
            encoding_metadata,
            reopened_arrays=encoding_arrays,
            panel_documents=documents,
            runtime_contract=runtime,
            allow_fake_runtime=allow_fake_runtime,
            material_metadata_binding=_binding(
                material / encoding_relative / "metadata.json", root=material
            ),
        )
        panel_states = [dict(row) for row in documents["panel_manifest.json"]["states"]]
        spec_by_state = {str(spec["state_id"]): spec for spec in specs}
        fanout_validations: list[dict[str, Any]] = []
        for state in panel_states:
            state_id = str(state["state_id"])
            relative = f"fanout/{state_id}"
            expected_directories.add(relative)
            expected_files.update(
                {f"{relative}/metadata.json", f"{relative}/payload.npz"}
            )
            metadata, arrays = _load_material_shard(material, relative)
            selected_validation = next(
                value
                for value in selected_validations
                if value["metadata"]["state_id"] == state_id
            )
            port = next(
                row
                for row in documents["edge_port_index.json"]["records"]
                if row["state_id"] == state_id
            )
            fanout_validations.append(
                _call(
                    "validate_fanout_material_shard",
                    metadata,
                    reopened_arrays=arrays,
                    panel_state=state,
                    candidate_spec=spec_by_state[state_id],
                    selected_material=selected_validation,
                    directed_port_record=port,
                    runtime_contract=runtime,
                    allow_fake_runtime=allow_fake_runtime,
                    material_metadata_binding=_binding(
                        material / relative / "metadata.json", root=material
                    ),
                )
            )
        selection = _call(
            "validate_development_target_selection",
            downstream_evidence["development_target_selection.json"],
            panel_documents=documents,
            encoding_material=encoding_validation,
            fanout_material=[
                value
                for value in fanout_validations
                if value["metadata"]["role"] == "DEVELOPMENT"
            ],
            runtime_contract=runtime,
            allow_fake_runtime=allow_fake_runtime,
        )
        heldout_rows = _call(
            "validate_heldout_scores",
            downstream_evidence["heldout_scores.jsonl"],
            panel_documents=documents,
            development_target_selection=selection,
            encoding_material=encoding_validation,
            fanout_material=fanout_validations,
            runtime_contract=runtime,
            allow_fake_runtime=allow_fake_runtime,
        )
        fanout_rows = _call(
            "validate_candidate_fanout_rows",
            downstream_evidence["candidate_fanout.jsonl"],
            panel_documents=documents,
            fanout_material=fanout_validations,
            development_target_selection=selection,
            runtime_contract=runtime,
            allow_fake_runtime=allow_fake_runtime,
        )
        repeat_validations: list[dict[str, Any]] = []
        for state in panel_states:
            if state["role"] != "DEVELOPMENT_HELDOUT":
                continue
            state_id = str(state["state_id"])
            relative = f"repeatability/{state_id}"
            expected_directories.add(relative)
            expected_files.update(
                {f"{relative}/metadata.json", f"{relative}/payload.npz"}
            )
            metadata, arrays = _load_material_shard(material, relative)
            selected_validation = next(
                value
                for value in selected_validations
                if value["metadata"]["state_id"] == state_id
            )
            fanout_validation = next(
                value
                for value in fanout_validations
                if value["metadata"]["state_id"] == state_id
            )
            port = next(
                row
                for row in documents["edge_port_index.json"]["records"]
                if row["state_id"] == state_id
            )
            repeat_validations.append(
                _call(
                    "validate_repeatability_material_shard",
                    metadata,
                    reopened_arrays=arrays,
                    panel_state=state,
                    candidate_spec=spec_by_state[state_id],
                    selected_material=selected_validation,
                    source_fanout_material=fanout_validation,
                    heldout_score_rows=heldout_rows,
                    directed_port_record=port,
                    runtime_contract=runtime,
                    allow_fake_runtime=allow_fake_runtime,
                    material_metadata_binding=_binding(
                        material / relative / "metadata.json", root=material
                    ),
                )
            )
        repeatability_rows = _call(
            "validate_repeatability_rows",
            downstream_evidence["repeatability.jsonl"],
            panel_documents=documents,
            candidate_fanout_rows=fanout_rows,
            heldout_score_rows=heldout_rows,
            repeatability_material=repeat_validations,
            runtime_contract=runtime,
            allow_fake_runtime=allow_fake_runtime,
        )
        if files != expected_files or directories != expected_directories:
            raise RegenerationError("success material inventory drift")
        expected_counts = CONTRACT.expected_material_inventory_counts(
            len(records), panel_available=True
        )
        if (
            len(files) != expected_counts["file_count"]
            or len(directories) != expected_counts["directory_count"]
        ):
            raise RegenerationError("success material count authority drift")
        downstream_validation = {
            "panel_documents": documents,
            "selected_material": selected_validations,
            "encoding_material": encoding_validation,
            "fanout_material": fanout_validations,
            "candidate_fanout_rows": fanout_rows,
            "development_target_selection": selection,
            "heldout_scores": heldout_rows,
            "repeatability_material": repeat_validations,
            "repeatability_rows": repeatability_rows,
        }
    nonreuse = _require_no_v4_hardlink_reuse(
        output_root, material, sorted(files)
    )
    inventory = CONTRACT.attach_content_digest(
        {
            "schema": (
                "physical_handoff_stratified_generator_successor_v1."
                "material_inventory_projection.v1"
            ),
            "experiment_id": EXPERIMENT_ID,
            "root": str(material),
            "branch": (
                "SUCCESS" if available else "GENERATOR_TERMINAL"
            ),
            "attempted_candidate_count": len(records),
            "file_count": len(files),
            "directory_count": len(directories),
            "files": [
                _binding(material / relative, root=material)
                for relative in sorted(files)
            ],
            "directories": sorted(directories),
            "unexpected_file_count": 0,
            "unexpected_directory_count": 0,
            **nonreuse,
        }
    )
    return {
        "terminal_record_count": len(records),
        "metadata_rows": metadata_rows,
        "material_validations": validations,
        "generator_runtime_environment": runtime_environment,
        "common_expected_material_files": sorted(expected_files),
        "common_expected_material_directories": sorted(expected_directories),
        "observed_material_files": sorted(files),
        "observed_material_directories": sorted(directories),
        "material_inventory_projection": inventory,
        "downstream_validation": downstream_validation,
        "models_trained": 0,
    }


def build_reduction(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    material_root: Path | str | None = None,
    publication_leaves_present: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    _require_no_successor_external_receipt(root)
    material = (
        Path(material_root)
        if material_root is not None
        else root.parent / f"{root.name}_material"
    )
    evidence = _load_common_evidence(root)
    evidence["source_freeze_observation"] = _observe_source_freeze(
        evidence["runtime_contract"]
    )
    evidence["material_root"] = str(material)
    available = (
        evidence["generator_metrics"].get("status")
        == CONTRACT.GENERATOR_PANEL_AVAILABLE
    )
    scientific_leaves = set(
        CONTRACT.SUCCESS_OUTPUT_LEAVES[:15]
        if available
        else CONTRACT.GENERATOR_TERMINAL_OUTPUT_LEAVES[:5]
    )
    expected = (
        set(
            CONTRACT.SUCCESS_OUTPUT_LEAVES
            if available
            else CONTRACT.GENERATOR_TERMINAL_OUTPUT_LEAVES
        )
        if publication_leaves_present
        else scientific_leaves
    )
    if _leaf_names(root) != expected:
        raise RegenerationError("official inventory drift before reduction")
    if available:
        evidence.update(_load_downstream_evidence(root))
    _require_real_generator_runtime_marker(material)
    evidence["material_validation"] = _validate_persisted_material(
        material,
        root,
        evidence["generator_terminal_records"],
        evidence["runtime_contract"],
        downstream_evidence=evidence if available else None,
    )
    reduction = copy.deepcopy(_call("recompute_metrics", evidence))
    _call("validate_recomputed_metrics", reduction)
    _independent_validate_downstream_decision(reduction, evidence)
    _require_no_successor_external_receipt(root)
    return reduction


def _publication_bindings(root: Path, *, available: bool) -> dict[str, Any]:
    leaves = (
        CONTRACT.SUCCESS_OUTPUT_LEAVES
        if available
        else CONTRACT.GENERATOR_TERMINAL_OUTPUT_LEAVES
    )
    return {
        leaf: _binding(root / leaf, root=root)
        for leaf in leaves
        if leaf not in {"result.json", "result.md", "file_hashes.json"}
    }


def reduce_publish_and_validate(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    material_root: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    _require_no_successor_external_receipt(root)
    metrics = build_reduction(root, material_root=material_root)
    _require_no_successor_external_receipt(root)
    available = (
        metrics["generator_status"] == CONTRACT.GENERATOR_PANEL_AVAILABLE
    )
    _atomic_json(root / "metrics.json", metrics)
    bindings = _publication_bindings(root, available=available)
    projection = _call(
        "build_result_publication_projection",
        metrics,
        bindings,
    )
    result = copy.deepcopy(projection["result_document"])
    _atomic_json(root / "result.json", result)
    V1.atomic_bytes(root / "result.md", _call("build_result_report_bytes", projection))
    leaves = (
        CONTRACT.SUCCESS_OUTPUT_LEAVES
        if available
        else CONTRACT.GENERATOR_TERMINAL_OUTPUT_LEAVES
    )
    files = [
        _binding(root / leaf, root=root)
        for leaf in leaves
        if leaf != "file_hashes.json"
    ]
    manifest = CONTRACT.attach_content_digest(
        {
            "schema": FILE_HASHES_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "root": str(root),
            "files": files,
            "file_count_excluding_self": len(files),
            "bytes_excluding_self": sum(int(row["bytes"]) for row in files),
            "file_hashes_self_sha256_excluded": True,
        }
    )
    _atomic_json(root / "file_hashes.json", manifest)
    validate_existing_publication(root, material_root=material_root)
    _require_no_successor_external_receipt(root)
    return result


def validate_existing_publication(
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    material_root: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    _require_no_successor_external_receipt(root)
    _, supplied_metrics = _load_json(root / "metrics.json", "metrics")
    _, supplied_result = _load_json(root / "result.json", "result")
    report_raw = _require_regular(root / "result.md", "result report") and (
        root / "result.md"
    ).read_bytes()
    _, supplied_hashes = _load_json(root / "file_hashes.json", "file hashes")
    rebuilt = build_reduction(
        root,
        material_root=material_root,
        publication_leaves_present=True,
    )
    if canonical_bytes(rebuilt) != canonical_bytes(supplied_metrics):
        raise RegenerationError("metrics differ from independent rebuild")
    available = rebuilt["generator_status"] == CONTRACT.GENERATOR_PANEL_AVAILABLE
    bindings = _publication_bindings(root, available=available)
    projection = _call(
        "validate_result_publication_projection",
        supplied_result,
        recomputed_metrics=rebuilt,
        scientific_bindings=bindings,
    )
    if available:
        _independent_validate_comparator_alias_authority(
            supplied_result.get("downstream_summary", {})
            .get("heldout", {})
            .get("comparator_alias_authority")
        )
    if report_raw != _call("build_result_report_bytes", projection):
        raise RegenerationError("result report differs from exact rebuild")
    leaves = (
        CONTRACT.SUCCESS_OUTPUT_LEAVES
        if available
        else CONTRACT.GENERATOR_TERMINAL_OUTPUT_LEAVES
    )
    expected_files = [
        _binding(root / leaf, root=root)
        for leaf in leaves
        if leaf != "file_hashes.json"
    ]
    expected_hashes = CONTRACT.attach_content_digest(
        {
            "schema": FILE_HASHES_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "root": str(root),
            "files": expected_files,
            "file_count_excluding_self": len(expected_files),
            "bytes_excluding_self": sum(int(row["bytes"]) for row in expected_files),
            "file_hashes_self_sha256_excluded": True,
        }
    )
    if canonical_bytes(expected_hashes) != canonical_bytes(supplied_hashes):
        raise RegenerationError("file hash manifest differs from exact rebuild")
    if _leaf_names(root) != set(leaves):
        raise RegenerationError("final official inventory drift")
    _require_no_successor_external_receipt(root)
    return supplied_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--material-root", type=Path, default=DEFAULT_MATERIAL_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if (arguments.root / "result.json").exists():
        result = validate_existing_publication(
            arguments.root, material_root=arguments.material_root
        )
    else:
        result = reduce_publish_and_validate(
            arguments.root, material_root=arguments.material_root
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
