#!/usr/bin/env python3
"""Independently replay non-greedy local-subgoal score metrics.

This reducer consumes only persisted JSON/JSONL products.  It deliberately
constructs or executes no model, tensor program, inference, training, archive,
or simulator operation.  The pure metrics module may import frozen model class
definitions, but this reducer never instantiates or calls them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS_MODULE = "lewm.safety.non_greedy_local_subgoal_jepa_planning_metrics_v1"
REDUCER_SOURCE_PATH = "scripts/evaluate_non_greedy_local_subgoal_jepa_planning_v1.py"
METRICS_SOURCE_PATH = (
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_metrics_v1.py"
)
STAGE_A_FILE = "stage_a_scores.jsonl"
STAGE_B_FILE = "conditional_stage_b_scores.jsonl"
METRICS_FILE = "metrics.json"
REGENERATION_RECEIPT_FILE = "independent_regeneration_receipt.json"
RECEIPT_SCHEMA = (
    "non_greedy_local_subgoal_jepa_planning_v1.regeneration_receipt.v1"
)
AUTHORITY_SCHEMA = (
    "non_greedy_local_subgoal_jepa_planning_v1.score_row_authority.v1"
)
IDENTITY_FIELDS = (
    "stage_id",
    "state_id",
    "model_id",
    "source_id",
    "candidate_index",
)
SCORE_ROW_FIELDS = {
    "stage_id",
    "state_id",
    "family",
    "split_role",
    "candidate_index",
    "model_id",
    "source_id",
    "score",
    "geodesic_progress_m",
    "euclidean_progress_m",
    "remaining_geodesic_m",
    "heading_error_to_next_shortest_segment_rad",
    "oracle_admissible",
    "immediate_contact",
    "committed_prefix_contact",
    "successor_viable",
    "stuck",
    "dead_end",
    "completed",
}
FINITE_NUMERIC_FIELDS = (
    "score",
    "geodesic_progress_m",
    "euclidean_progress_m",
    "remaining_geodesic_m",
    "heading_error_to_next_shortest_segment_rad",
)
EXPECTED_STATE_COUNT = 16
EXPECTED_MODEL_SOURCE_PAIRS = {
    "stage_a": (
        ("DETERMINISTIC_KINEMATICS", "KINEMATIC"),
        ("NO_LATENT_NON_GREEDY_RANKER", "NO_LATENT"),
        ("CURRENT_VISUAL_REACTIVE_RANKER", "CURRENT_VISUAL"),
        ("TRUE_FUTURE_JEPA_TRAJECTORY_RANKER", "TRUE_FUTURE"),
        (
            "FUTURE_TRAJECTORY_DERANGEMENT",
            "TRUE_FUTURE_DERANGED_CANDIDATE",
        ),
        (
            "FUTURE_TIME_ORDER_DERANGEMENT",
            "TRUE_FUTURE_DERANGED_TIME",
        ),
    ),
    "conditional_stage_b": (
        ("TRUE_FUTURE_JEPA_TRAJECTORY_RANKER", "R1"),
        ("TRUE_FUTURE_JEPA_TRAJECTORY_RANKER", "RR"),
    ),
}
STAGE_AUTHORITY_FIELDS = {
    "stage_id",
    "file_name",
    "required_fields",
    "finite_numeric_fields",
    "state_ids",
    "model_source_pairs",
    "candidate_count",
}


class RegenerationError(ValueError):
    """Raised when persisted reducer input is not exact and replayable."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RegenerationError("value contains non-finite or unsupported canonical JSON") from exc


def canonical_document_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise RegenerationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RegenerationError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def parse_canonical_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegenerationError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RegenerationError) as exc:
        if isinstance(exc, RegenerationError):
            raise
        raise RegenerationError(f"{label} is not valid JSON") from exc
    if canonical_document_bytes(value) != raw:
        raise RegenerationError(f"{label} is not canonical JSON with one LF")
    _reject_nonfinite(value, label=label)
    return value


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegenerationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, label=f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, label=f"{label}.{key}")
        return
    raise RegenerationError(f"{label} contains an unsupported JSON value")


def _read_regular_at(root_fd: int, leaf: str, *, optional: bool = False) -> bytes | None:
    if not leaf or leaf in {".", ".."} or "/" in leaf:
        raise RegenerationError(f"unsafe reducer input leaf: {leaf!r}")
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except FileNotFoundError:
        if optional:
            return None
        raise RegenerationError(f"required reducer input is absent: {leaf}")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RegenerationError(f"reducer input is not a single-link regular file: {leaf}")
        blocks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            blocks.append(block)
        raw = b"".join(blocks)
        after = os.fstat(fd)
        projection = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_uid,
            row.st_gid,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if projection(before) != projection(after) or len(raw) != before.st_size:
            raise RegenerationError(f"reducer input changed while read: {leaf}")
        return raw
    finally:
        os.close(fd)


def _parse_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        raise RegenerationError(f"{label} must be nonempty and LF terminated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw[:-1].split(b"\n"), start=1):
        if not line:
            raise RegenerationError(f"{label} contains an empty line")
        value = parse_canonical_json(line + b"\n", label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise RegenerationError(f"{label}:{line_number} must be an object")
        rows.append(value)
    return rows


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise RegenerationError(f"{label} must be a nonempty sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise RegenerationError(f"{label} must contain nonempty strings")
    if len(result) != len(set(result)):
        raise RegenerationError(f"{label} must not contain duplicates")
    return result


def _model_source_pairs(value: Any, *, label: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise RegenerationError(f"{label} must be a nonempty sequence")
    result: list[dict[str, str]] = []
    observed: set[tuple[str, str]] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"model_id", "source_id"}:
            raise RegenerationError(f"{label}[{index}] field set drift")
        model_id = item["model_id"]
        source_id = item["source_id"]
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(source_id, str)
            or not source_id
        ):
            raise RegenerationError(f"{label}[{index}] values must be nonempty strings")
        pair = (model_id, source_id)
        if pair in observed:
            raise RegenerationError(f"{label} contains a duplicate pair: {pair}")
        observed.add(pair)
        result.append({"model_id": model_id, "source_id": source_id})
    return tuple(result)


def _validate_stage_authority(value: Any, *, logical_stage: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != STAGE_AUTHORITY_FIELDS:
        raise RegenerationError(f"{logical_stage} authority field set drift")
    stage = dict(value)
    if not isinstance(stage["stage_id"], str) or not stage["stage_id"]:
        raise RegenerationError(f"{logical_stage}.stage_id must be a nonempty string")
    expected_file = STAGE_A_FILE if logical_stage == "stage_a" else STAGE_B_FILE
    if stage["file_name"] != expected_file:
        raise RegenerationError(f"{logical_stage}.file_name drift")
    required = _string_sequence(
        stage["required_fields"], label=f"{logical_stage}.required_fields"
    )
    finite = _string_sequence(
        stage["finite_numeric_fields"],
        label=f"{logical_stage}.finite_numeric_fields",
    )
    if set(required) != SCORE_ROW_FIELDS:
        raise RegenerationError(f"{logical_stage} exact score-row field authority drift")
    if tuple(finite) != FINITE_NUMERIC_FIELDS:
        raise RegenerationError(f"{logical_stage} finite numeric field authority drift")
    if stage["candidate_count"] != 12 or isinstance(stage["candidate_count"], bool):
        raise RegenerationError(f"{logical_stage}.candidate_count must equal 12")
    state_ids = _string_sequence(
        stage["state_ids"], label=f"{logical_stage}.state_ids"
    )
    if len(state_ids) != EXPECTED_STATE_COUNT:
        raise RegenerationError(
            f"{logical_stage}.state_ids must contain exactly {EXPECTED_STATE_COUNT} states"
        )
    pairs = _model_source_pairs(
        stage["model_source_pairs"],
        label=f"{logical_stage}.model_source_pairs",
    )
    observed_pairs = tuple((pair["model_id"], pair["source_id"]) for pair in pairs)
    if observed_pairs != EXPECTED_MODEL_SOURCE_PAIRS[logical_stage]:
        raise RegenerationError(f"{logical_stage} ordered model/source authority drift")
    return {
        **stage,
        "required_fields": required,
        "finite_numeric_fields": finite,
        "state_ids": state_ids,
        "model_source_pairs": pairs,
    }


def validate_score_row_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"schema", "stages"}:
        raise RegenerationError("score-row authority field set drift")
    if value["schema"] != AUTHORITY_SCHEMA:
        raise RegenerationError("score-row authority schema drift")
    stages = value["stages"]
    if not isinstance(stages, Mapping) or set(stages) != {"stage_a", "conditional_stage_b"}:
        raise RegenerationError("score-row stage authority set drift")
    stage_a = _validate_stage_authority(stages["stage_a"], logical_stage="stage_a")
    stage_b = _validate_stage_authority(
        stages["conditional_stage_b"], logical_stage="conditional_stage_b"
    )
    if stage_a["state_ids"] != stage_b["state_ids"]:
        raise RegenerationError("Stage-A and Stage-B heldout state authority drift")
    return {
        "schema": AUTHORITY_SCHEMA,
        "stages": {
            "stage_a": stage_a,
            "conditional_stage_b": stage_b,
        },
    }


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["state_id"]),
        str(row["model_id"]),
        str(row["source_id"]),
        int(row["candidate_index"]),
    )


def validate_stage_rows(
    rows: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = set(authority["required_fields"])
    state_ids = tuple(authority["state_ids"])
    model_source_pairs = tuple(
        (pair["model_id"], pair["source_id"])
        for pair in authority["model_source_pairs"]
    )
    candidate_count = int(authority["candidate_count"])
    expected = {
        (state_id, model_id, source_id, candidate_index)
        for state_id in state_ids
        for model_id, source_id in model_source_pairs
        for candidate_index in range(candidate_count)
    }
    if len(rows) != len(expected):
        raise RegenerationError(
            f"{label} cardinality drift: observed {len(rows)}, expected {len(expected)}"
        )
    observed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    state_metadata: dict[str, tuple[str, str]] = {}
    for index, original in enumerate(rows):
        if not isinstance(original, Mapping) or set(original) != required:
            raise RegenerationError(f"{label}[{index}] row schema drift")
        row = dict(original)
        if row["stage_id"] != authority["stage_id"]:
            raise RegenerationError(f"{label}[{index}] stage identity drift")
        for field in ("state_id", "model_id", "source_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise RegenerationError(f"{label}[{index}].{field} must be a string")
        if not isinstance(row["family"], str) or not row["family"]:
            raise RegenerationError(f"{label}[{index}].family must be a nonempty string")
        if row["split_role"] != "DEVELOPMENT_HELDOUT":
            raise RegenerationError(f"{label}[{index}].split_role drift")
        for field in (
            "oracle_admissible",
            "immediate_contact",
            "committed_prefix_contact",
            "successor_viable",
            "stuck",
            "dead_end",
            "completed",
        ):
            if type(row[field]) is not bool:
                raise RegenerationError(f"{label}[{index}].{field} must be Boolean")
        metadata = (row["family"], row["split_role"])
        prior_metadata = state_metadata.setdefault(row["state_id"], metadata)
        if prior_metadata != metadata:
            raise RegenerationError(f"{label}[{index}] state family/role drift")
        candidate_index = row["candidate_index"]
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < candidate_count
        ):
            raise RegenerationError(f"{label}[{index}] candidate_index drift")
        for field in authority["finite_numeric_fields"]:
            number = row[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise RegenerationError(f"{label}[{index}].{field} must be numeric")
            if not math.isfinite(float(number)):
                raise RegenerationError(f"{label}[{index}].{field} must be finite")
        _reject_nonfinite(row, label=f"{label}[{index}]")
        identity = _row_identity(row)
        if identity in observed:
            raise RegenerationError(f"{label} duplicate row identity: {identity}")
        observed[identity] = row
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise RegenerationError(
            f"{label} identity coverage drift: missing={missing[:3]}, extra={extra[:3]}"
        )
    ordered_identities = [
        (state_id, model_id, source_id, candidate_index)
        for state_id in state_ids
        for model_id, source_id in model_source_pairs
        for candidate_index in range(candidate_count)
    ]
    ordered = [observed[identity] for identity in ordered_identities]
    summary = {
        "rows": len(ordered),
        "expected_rows": len(expected),
        "states": len(state_ids),
        "model_source_pairs": len(model_source_pairs),
        "models": len({model_id for model_id, _ in model_source_pairs}),
        "sources": len({source_id for _, source_id in model_source_pairs}),
        "candidates_per_state_model_source": candidate_count,
        "row_identity_digest": canonical_digest(
            [
                {
                    "state_id": identity[0],
                    "model_id": identity[1],
                    "source_id": identity[2],
                    "candidate_index": identity[3],
                }
                for identity in ordered_identities
            ]
        ),
    }
    return ordered, summary


def validate_candidate_payload_consistency(
    stage_a_rows: Sequence[Mapping[str, Any]],
    conditional_stage_b_rows: Sequence[Mapping[str, Any]] | None,
) -> None:
    """Require model/source substitutions to leave candidate facts unchanged."""

    excluded = {"stage_id", "model_id", "source_id", "score"}

    def projections(
        rows: Sequence[Mapping[str, Any]], *, label: str
    ) -> dict[tuple[str, int], bytes]:
        result: dict[tuple[str, int], bytes] = {}
        for index, row in enumerate(rows):
            key = (str(row["state_id"]), int(row["candidate_index"]))
            projection = canonical_json_bytes(
                {field: value for field, value in row.items() if field not in excluded}
            )
            prior = result.setdefault(key, projection)
            if prior != projection:
                raise RegenerationError(
                    f"{label}[{index}] candidate payload differs across model/source pairs"
                )
        return result

    stage_a = projections(stage_a_rows, label="stage_a")
    if conditional_stage_b_rows is not None:
        stage_b = projections(
            conditional_stage_b_rows,
            label="conditional_stage_b",
        )
        if stage_b != stage_a:
            raise RegenerationError(
                "conditional Stage-B candidate payload differs from Stage A"
            )


def _binding(leaf: str, raw: bytes, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": leaf,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _load_metrics_module() -> Any:
    return importlib.import_module(METRICS_MODULE)


def _metrics_module_name(module: Any) -> str:
    name = getattr(module, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return module.__class__.__name__


def _git_bytes(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegenerationError(
            f"source-freeze Git observation failed: {' '.join(arguments)}"
        ) from exc
    return completed.stdout


def _git_text(*arguments: str) -> str:
    raw = _git_bytes(*arguments)
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RegenerationError("source-freeze Git output is not ASCII") from exc


def _read_repository_source(relative: str) -> bytes:
    parts = Path(relative).parts
    if (
        not parts
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise RegenerationError(f"unsafe repository source path: {relative!r}")
    fds: list[int] = []
    try:
        current_fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(current_fd)
        for component in parts[:-1]:
            current_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            info = os.fstat(current_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 2:
                raise RegenerationError(
                    f"repository source parent is not a live directory: {relative}"
                )
            fds.append(current_fd)
        raw = _read_regular_at(current_fd, parts[-1])
        assert raw is not None
        return raw
    except OSError as exc:
        raise RegenerationError(
            f"repository source cannot be opened without following links: {relative}"
        ) from exc
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _source_binding(relative: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _observe_source_freeze(module: Any) -> dict[str, Any]:
    if _metrics_module_name(module) != METRICS_MODULE:
        raise RegenerationError("production reduction requires the exact metrics module")
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or Path(module_path).absolute() != (
        ROOT / METRICS_SOURCE_PATH
    ):
        raise RegenerationError("metrics module source path drift")
    head = _git_text("rev-parse", "HEAD")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RegenerationError("source-freeze HEAD is not canonical 40-hex")
    if _git_bytes("status", "--porcelain=v1", "--untracked-files=all"):
        raise RegenerationError("independent reduction requires a clean source-freeze worktree")
    bindings: dict[str, dict[str, Any]] = {}
    for key, relative in (
        ("independent_reducer", REDUCER_SOURCE_PATH),
        ("metrics_module", METRICS_SOURCE_PATH),
    ):
        raw = _read_repository_source(relative)
        if _git_bytes("show", f"{head}:{relative}") != raw:
            raise RegenerationError(f"source bytes differ from source-freeze HEAD: {relative}")
        bindings[key] = _source_binding(relative, raw)
    return {
        "head_commit": head,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": bindings,
    }


def _validate_source_freeze_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "head_commit",
        "worktree_clean",
        "sources_exactly_equal_head",
        "sources",
    }:
        raise RegenerationError("source-freeze observation field set drift")
    head = value["head_commit"]
    if (
        not isinstance(head, str)
        or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise RegenerationError("source-freeze observation HEAD drift")
    if value["worktree_clean"] is not True:
        raise RegenerationError("source-freeze observation is not clean")
    if value["sources_exactly_equal_head"] is not True:
        raise RegenerationError("source bytes are not exact at source-freeze HEAD")
    sources = value["sources"]
    expected_paths = {
        "independent_reducer": REDUCER_SOURCE_PATH,
        "metrics_module": METRICS_SOURCE_PATH,
    }
    if not isinstance(sources, Mapping) or set(sources) != set(expected_paths):
        raise RegenerationError("source binding identity set drift")
    normalized: dict[str, dict[str, Any]] = {}
    for key, relative in expected_paths.items():
        binding = sources[key]
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise RegenerationError(f"source binding field set drift: {key}")
        byte_count = binding["bytes"]
        digest = binding["sha256"]
        if (
            binding["path"] != relative
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RegenerationError(f"source binding value drift: {key}")
        normalized[key] = dict(binding)
    return {
        "head_commit": head,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": normalized,
    }


def build_regeneration_receipt(
    output_root: Path | str,
    *,
    metrics_module: Any | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    if not root.is_absolute() or ".." in root.parts:
        raise RegenerationError("--output-root must be an absolute lexical path")
    module = metrics_module if metrics_module is not None else _load_metrics_module()
    authority_builder = getattr(module, "score_row_authority", None)
    reducer = getattr(module, "recompute_metrics_and_gates", None)
    if not callable(authority_builder) or not callable(reducer):
        raise RegenerationError(
            "metrics module must expose score_row_authority and recompute_metrics_and_gates"
        )
    authority = validate_score_row_authority(authority_builder())
    source_freeze = _validate_source_freeze_observation(
        _observe_source_freeze(module)
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        stage_a_raw = _read_regular_at(root_fd, STAGE_A_FILE)
        stage_b_raw = _read_regular_at(root_fd, STAGE_B_FILE, optional=True)
        metrics_raw = _read_regular_at(root_fd, METRICS_FILE)
    finally:
        os.close(root_fd)
    assert stage_a_raw is not None and metrics_raw is not None
    stage_a_input = _parse_jsonl(stage_a_raw, label=STAGE_A_FILE)
    stage_a_rows, stage_a_summary = validate_stage_rows(
        stage_a_input,
        authority["stages"]["stage_a"],
        label="stage_a",
    )
    stage_b_rows: list[dict[str, Any]] | None = None
    stage_b_summary: dict[str, Any] | None = None
    if stage_b_raw is not None:
        stage_b_input = _parse_jsonl(stage_b_raw, label=STAGE_B_FILE)
        stage_b_rows, stage_b_summary = validate_stage_rows(
            stage_b_input,
            authority["stages"]["conditional_stage_b"],
            label="conditional_stage_b",
        )
    validate_candidate_payload_consistency(stage_a_rows, stage_b_rows)
    supplied_metrics = parse_canonical_json(metrics_raw, label=METRICS_FILE)
    if not isinstance(supplied_metrics, dict):
        raise RegenerationError("metrics.json must contain an object")
    recomputed_metrics = reducer(
        tuple(stage_a_rows),
        None if stage_b_rows is None else tuple(stage_b_rows),
    )
    if not isinstance(recomputed_metrics, Mapping):
        raise RegenerationError("metrics reducer must return a mapping")
    recomputed_metrics = dict(recomputed_metrics)
    _reject_nonfinite(recomputed_metrics, label="recomputed_metrics")
    recomputed_raw = canonical_document_bytes(recomputed_metrics)
    if metrics_raw != recomputed_raw:
        raise RegenerationError("supplied metrics.json differs from exact recomputation")
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "pass": True,
        "mode": "INDEPENDENT_PERSISTED_ROW_REDUCTION_ONLY",
        "metrics_module": _metrics_module_name(module),
        "source_freeze": source_freeze,
        "score_row_authority_digest": canonical_digest(authority),
        "inputs": {
            "stage_a_scores": _binding(
                STAGE_A_FILE, stage_a_raw, rows=len(stage_a_rows)
            ),
            "conditional_stage_b_scores": (
                None
                if stage_b_raw is None
                else _binding(STAGE_B_FILE, stage_b_raw, rows=len(stage_b_rows or ()))
            ),
            "metrics": _binding(METRICS_FILE, metrics_raw),
        },
        "stage_a_validation": stage_a_summary,
        "conditional_stage_b_validation": stage_b_summary,
        "conditional_stage_b_present": stage_b_rows is not None,
        "metrics_and_gates_recomputed": True,
        "candidate_payload_consistency_validated": True,
        "supplied_metrics_exact_byte_equal": True,
        "recomputed_metrics_sha256": hashlib.sha256(recomputed_raw).hexdigest(),
        "scientific_execution_counters": {
            "archive_opens": 0,
            "v2_root_opens": 0,
            "model_initializations": 0,
            "training_steps": 0,
            "inference_calls": 0,
            "simulator_steps": 0,
        },
        "content_digest": "",
    }
    digest_input = dict(receipt)
    digest_input.pop("content_digest")
    receipt["content_digest"] = canonical_digest(digest_input)
    return receipt


def emit_regeneration_receipt(
    output_root: Path | str,
    receipt: Mapping[str, Any],
) -> bytes:
    root = Path(output_root)
    raw = canonical_document_bytes(dict(receipt))
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open(
                REGENERATION_RECEIPT_FILE,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=root_fd,
            )
        except FileExistsError:
            existing = _read_regular_at(root_fd, REGENERATION_RECEIPT_FILE)
            if existing != raw:
                raise RegenerationError("existing regeneration receipt bytes drift")
        else:
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(fd, raw[offset:])
                    if written <= 0:
                        raise OSError("short regeneration receipt write")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(root_fd)
    finally:
        os.close(root_fd)
    return raw


def verify_and_emit(
    output_root: Path | str,
    *,
    metrics_module: Any | None = None,
) -> dict[str, Any]:
    receipt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
    )
    emit_regeneration_receipt(output_root, receipt)
    return receipt


def validate_existing_regeneration_receipt(
    output_root: Path | str,
    *,
    metrics_module: Any | None = None,
) -> dict[str, Any]:
    """Rebuild the receipt and require exact canonical equality to the durable leaf."""

    root = Path(output_root)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        existing_raw = _read_regular_at(root_fd, REGENERATION_RECEIPT_FILE)
    finally:
        os.close(root_fd)
    assert existing_raw is not None
    existing = parse_canonical_json(
        existing_raw,
        label=REGENERATION_RECEIPT_FILE,
    )
    if not isinstance(existing, dict):
        raise RegenerationError("existing regeneration receipt must be an object")
    rebuilt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
    )
    if existing_raw != canonical_document_bytes(rebuilt):
        raise RegenerationError("existing regeneration receipt differs from exact rebuild")
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify persisted non-greedy planning scores"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="recompute all persisted metrics and gates and emit the receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, metrics_module: Any | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.verify:
        parser.error("--verify is required")
    receipt = verify_and_emit(args.output_root, metrics_module=metrics_module)
    print(canonical_document_bytes(receipt).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
