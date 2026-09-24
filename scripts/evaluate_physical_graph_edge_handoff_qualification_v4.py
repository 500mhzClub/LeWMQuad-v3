#!/usr/bin/env python3
"""Independent custody and reduction for tipped-state PGEHQ V4.

This module is deliberately safe under the system Python interpreter.  It
does not import the experiment runner, simulator, model, encoder, ranker, or
training stack.  V4 is development-only and permanently ineligible for final
evaluation; the reducer never opens sealed benchmark material.

The combined historical custody receipt and the V4 regeneration receipt are
ordinary canonical JSON documents outside every experiment root.  Neither is
self-digested.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Frozen system-Python-safe custody dependency.  Importing the evaluator opens
# no runtime artifact and imports no execution stack.
from scripts import evaluate_physical_graph_edge_handoff_qualification_v3 as V3E


EXPERIMENT_ID = "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V4"
SOURCE_PARENT_COMMIT = "5b08d433f2e69f6e4fe85c9b14696726e32d2ab1"
SOURCE_BASELINE_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
FREEZE_SUBJECT = "Freeze tipped-state physical graph edge handoff qualification V4"
RESULT_SUBJECT = "Evaluate tipped-state physical graph edge handoff qualification V4"
METRICS_MODULE = (
    "lewm.safety.physical_graph_edge_handoff_qualification_v4_metrics"
)

DEFAULT_V1_OFFICIAL_ROOT = V3E.DEFAULT_V1_OFFICIAL_ROOT
DEFAULT_V1_MATERIAL_ROOT = V3E.DEFAULT_V1_MATERIAL_ROOT
DEFAULT_V1_CUSTODY_RECEIPT = V3E.DEFAULT_V1_CUSTODY_RECEIPT
DEFAULT_V2_OFFICIAL_ROOT = V3E.DEFAULT_V2_OFFICIAL_ROOT
DEFAULT_V2_MATERIAL_ROOT = V3E.DEFAULT_V2_MATERIAL_ROOT
DEFAULT_V1_V2_CUSTODY_RECEIPT = V3E.DEFAULT_HISTORICAL_CUSTODY_RECEIPT
DEFAULT_V3_OFFICIAL_ROOT = V3E.DEFAULT_V3_OUTPUT_ROOT
DEFAULT_V3_MATERIAL_ROOT = V3E.DEFAULT_V3_MATERIAL_ROOT
DEFAULT_V3_REGENERATION_RECEIPT = V3E.DEFAULT_EXTERNAL_RECEIPT
DEFAULT_V2_REGENERATION_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v2_regeneration_receipt.json"
)
DEFAULT_V4_OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v4"
)
DEFAULT_V4_MATERIAL_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v4_material"
)
DEFAULT_HISTORICAL_CUSTODY_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v1_v2_v3_custody_receipt.json"
)
DEFAULT_EXTERNAL_RECEIPT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "physical_graph_edge_handoff_qualification_v4_regeneration_receipt.json"
)

HISTORICAL_CUSTODY_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v1_v2_v3.custody_receipt.v1"
)
REGENERATION_SCHEMA = (
    "physical_graph_edge_handoff_qualification_v4.regeneration_receipt.v1"
)
PANEL_INADEQUATE_DISPOSITION = "PHYSICAL_HANDOFF_PANEL_INADEQUATE"
NEXT_GENERATOR_DECISION = (
    "REVISE_GENERATOR_FOR_SHORTFALL_FAMILIES_KEEP_TEACHER_CONTRACT_FROZEN"
)
V3_TERMINAL_DIAGNOSIS = "TIPPED_BOUNDARY_STATE_DISPOSITION_UNSPECIFIED"
AUTHORIZED_SCIENTIFIC_CHANGE = (
    "Invalid tipped boundaries are explicitly recorded as nonqualified panel "
    "candidates instead of aborting the complete collection."
)
PRE_PANEL_ENGINEERING_CORRECTION_STATUS = (
    "PARTIAL_QUALIFICATION_INVALIDATED_RESTART_REQUIRED"
)
INVALIDATED_V4_SOURCE_FREEZE_COMMIT = (
    "cb8c1a225550dc0af16626aa5446ad3ac2aaa80a"
)
INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST = (
    "9b844e34ab4693e68fe229831edc82ac72d3fec89fcd2bae75d27ef4efc3af56"
)
INVALIDATED_V4_SCIENTIFIC_CONTRACT_CONTENT_DIGEST = (
    "391d355e47aec470b4e44c7e8ab79c5950a615cf5b033dd8d83470480db21b26"
)
_PERSISTED_CROSSING_MISMATCH_POOL_INDICES = (
    66, 76, 78, 85, 89, 95, 102, 103, 104, 105, 106, 110, 111, 112,
    114, 116, 117, 118, 124,
)
_UNMATERIALIZED_COMPACT_PROJECTION_MISMATCH_POOL_INDICES = (
    64, 66, 67, 68, 69, 71, 72, 74, 75, 76, 78, 81, 82, 84, 85, 87,
    88, 89, 94, 95, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112,
    114, 116, 117, 118, 120, 122, 124, 126,
)
_BINARY64_PROJECTION_FIELDS = (
    "segment_width_m",
    "midpoint_world_xy",
    "unit_tangent_world_xy",
    "lateral_coordinate_m",
)
_BINARY64_OPERATION_ORDER = {
    "segment_delta": "dx=x1-x0 then dy=y1-y0, each rounded to binary64",
    "segment_norm_squared": (
        "FMA(dy,dy,round_binary64(dx*dx)) rounded once to binary64"
    ),
    "segment_width": (
        "correctly rounded binary64 square root of segment_norm_squared"
    ),
    "unit_tangent": (
        "tx=dx/segment_width then ty=dy/segment_width, each rounded to binary64"
    ),
    "midpoint": (
        "mx=round_binary64((x0+x1)/2) then "
        "my=round_binary64((y0+y1)/2)"
    ),
    "relative_point": (
        "rx=point_x-mx then ry=point_y-my, each rounded to binary64"
    ),
    "lateral_coordinate": (
        "FMA(ry,ty,round_binary64(rx*tx)) rounded once to binary64"
    ),
    "software_fma": (
        "convert each finite binary64 operand to its exact rational value, "
        "evaluate a*b+c exactly, then round once to nearest-even binary64; "
        "exact zero is canonical positive zero"
    ),
}
_CORRECTED_RAW_REDUCTION_AGGREGATE_SHA256 = (
    "a52b828c76286eb4e9d7ec6a0437db252fe93210c1057706a61f7cb3de97f746"
)
_CORRECTED_RAW_REDUCTION_AGGREGATE_DOMAIN = (
    "SHA-256 of canonical_json_bytes over the ordered pool-000..pool-127 "
    "array of complete reduce_qualification_teacher_trace projections, "
    "including the canonical terminal LF"
)
DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE = {
    "schema": (
        "physical_graph_edge_handoff_qualification_v4."
        "development_source_audit_disclosure.v1"
    ),
    "disposition": "ABORTED_DEVELOPMENT_TIME_IGNORE_RULE_BYPASS_ATTEMPT",
    "operation": "Path('.').rglob('*.py') metadata traversal",
    "matched_file_opens_or_reads": 0,
    "printed_paths": 0,
    "usable_output_items": 0,
    "sealed_content_accessed": False,
    "evidence_derived": False,
    "scientific_outcomes_contaminated": False,
    "replacement_audit_scope": (
        "FROZEN_SOURCE_CLOSURE_PATHS_ONLY_USING_IGNORE_HONORING_TOOLS"
    ),
    "content_digest": (
        "0a1c17f092092c942e027472221a2673214dc60d6d3e86140837ecdab0b84d9e"
    ),
}
EXTERNAL_CUSTODY_ZERO_COUNTER_SCOPE = (
    "EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_BUILD_AND_EMISSION_PROCESS_ONLY"
)
SCIENTIFIC_ZERO_COUNTER_SCOPE = "V4_SCIENTIFIC_EXECUTION_PROCESS_ONLY"
V3_TERMINAL_INTERPRETATION = {
    "diagnosis": V3_TERMINAL_DIAGNOSIS,
    "first_eight_reproduction_exact": True,
    "panel_fanout_ranker_or_heldout_outcomes_opened": False,
    "prospective_panel_collection_stopped_because_tipped_state_disposition_unspecified": True,
    "raw_torch_snapshot_transport_bytes_are_scientific_evidence": False,
    "scientific_handoff_result_produced": False,
    "snapshot_behavioural_equivalence_passed": True,
    "snapshot_semantic_equivalence_passed": True,
}

V4_PANEL_INADEQUATE_FILES = (
    "contract.json",
    "v1_v2_v3_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "qualification_state_dispositions.jsonl",
    "panel_adequacy.json",
    "metrics.json",
    "result.json",
    "result.md",
    "file_hashes.json",
)

# The inherited scientific core is the exact V1 23-leaf successful inventory.
V4_SUCCESS_FILES = (
    "contract.json",
    "panel_manifest.json",
    "split_manifest.json",
    "graph_manifest.json",
    "state_snapshot_index.json",
    "state_snapshots.npz",
    "teacher_trace_index.json",
    "teacher_traces.npz",
    "edge_port_index.json",
    "waypoint_contracts.json",
    "pixel_index.json",
    "rgb_observations.npz",
    "latent_index.json",
    "canonical_latents.npz",
    "candidate_traces.npz",
    "candidate_fanout.jsonl",
    "development_target_selection.json",
    "heldout_ranker_scores.jsonl",
    "repeated_execution.jsonl",
    "metrics.json",
    "result.json",
    "result.md",
    "file_hashes.json",
    "v1_v2_v3_custody_and_nonreuse.json",
    "scientific_invariance_receipt.json",
    "qualification_state_dispositions.jsonl",
    "panel_adequacy.json",
)

_PUBLICATION_FILES = ("result.json", "result.md", "file_hashes.json")
V4_PANEL_INADEQUATE_SCIENTIFIC_FILES = tuple(
    name for name in V4_PANEL_INADEQUATE_FILES if name not in _PUBLICATION_FILES
)
V4_SUCCESS_SCIENTIFIC_FILES = tuple(
    name for name in V4_SUCCESS_FILES if name not in _PUBLICATION_FILES
)

_DOCUMENT_LEAVES = {
    "panel_manifest": "panel_manifest.json",
    "split_manifest": "split_manifest.json",
    "graph_manifest": "graph_manifest.json",
    "state_snapshot_index": "state_snapshot_index.json",
    "teacher_trace_index": "teacher_trace_index.json",
    "edge_port_index": "edge_port_index.json",
    "waypoint_contracts": "waypoint_contracts.json",
    "pixel_index": "pixel_index.json",
    "latent_index": "latent_index.json",
    "development_target_selection": "development_target_selection.json",
}
_LEDGER_LEAVES = {
    "candidate_fanout": "candidate_fanout.jsonl",
    "heldout_ranker_scores": "heldout_ranker_scores.jsonl",
    "repeated_execution": "repeated_execution.jsonl",
}
_PAYLOAD_LEAVES = (
    "state_snapshots.npz",
    "teacher_traces.npz",
    "rgb_observations.npz",
    "canonical_latents.npz",
    "candidate_traces.npz",
)

RegenerationError = V3E.RegenerationError
canonical_document_bytes = V3E.canonical_document_bytes
canonical_json_bytes = V3E.canonical_json_bytes
parse_canonical_json = V3E.parse_canonical_json
V1E = V3E.V2E.V1


def _load_metrics_module() -> Any:
    return importlib.import_module(METRICS_MODULE)


def _call(module: Any, name: str, *arguments: Any, **keywords: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise RegenerationError(f"V4 pure metrics API is absent: {name}")
    try:
        return function(*arguments, **keywords)
    except RegenerationError:
        raise
    except Exception as exc:
        raise RegenerationError(f"V4 pure metrics rejected {name}: {exc}") from exc


def _finite_binary64(value: Any, label: str) -> float:
    """Normalize one finite binary64 operand without NumPy involvement."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegenerationError(f"{label} is not finite binary64 numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise RegenerationError(
            f"{label} is not finite binary64 numeric"
        ) from exc
    if not math.isfinite(result):
        raise RegenerationError(f"{label} is not finite binary64 numeric")
    return result


def _frozen_binary64_fma(
    multiplicand: Any, multiplier: Any, addend: Any,
) -> float:
    """Independently evaluate the correction authority's exact binary64 FMA."""

    left = _finite_binary64(multiplicand, "FMA multiplicand")
    right = _finite_binary64(multiplier, "FMA multiplier")
    tail = _finite_binary64(addend, "FMA addend")
    exact = (
        Fraction.from_float(left) * Fraction.from_float(right)
        + Fraction.from_float(tail)
    )
    if exact == 0:
        return 0.0
    try:
        result = float(exact)
    except OverflowError as exc:
        raise RegenerationError("FMA result is outside finite binary64") from exc
    if not math.isfinite(result):
        raise RegenerationError("FMA result is outside finite binary64")
    return result


def _binary64_vector(value: Any, length: int, label: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise RegenerationError(f"{label} is not a numeric vector")
    try:
        values = list(value)
    except TypeError as exc:
        raise RegenerationError(f"{label} is not a numeric vector") from exc
    if len(values) != length:
        raise RegenerationError(f"{label} length drift")
    return [
        _finite_binary64(item, f"{label}[{index}]")
        for index, item in enumerate(values)
    ]


def _canonical_v1_planar_segment_projection(
    point_world_xy: Any, opening_segment_world: Any,
) -> dict[str, Any]:
    """System-Python implementation independent of the pure metrics helper."""

    point = _binary64_vector(point_world_xy, 2, "projection point")
    if isinstance(opening_segment_world, (str, bytes, bytearray, Mapping)):
        raise RegenerationError(
            "projection opening segment is not a two-row sequence"
        )
    try:
        rows = list(opening_segment_world)
    except TypeError as exc:
        raise RegenerationError(
            "projection opening segment is not a two-row sequence"
        ) from exc
    if len(rows) != 2:
        raise RegenerationError("projection opening segment row count drift")
    start = _binary64_vector(rows[0], 2, "projection opening start")
    stop = _binary64_vector(rows[1], 2, "projection opening stop")
    delta_x = stop[0] - start[0]
    delta_y = stop[1] - start[1]
    norm_squared = _frozen_binary64_fma(
        delta_y, delta_y, delta_x * delta_x,
    )
    if norm_squared <= 1.0e-24:
        raise RegenerationError("projection opening segment is degenerate")
    width = math.sqrt(norm_squared)
    tangent_x = delta_x / width
    tangent_y = delta_y / width
    midpoint_x = (start[0] + stop[0]) / 2.0
    midpoint_y = (start[1] + stop[1]) / 2.0
    relative_x = point[0] - midpoint_x
    relative_y = point[1] - midpoint_y
    return {
        "segment_width_m": width,
        "midpoint_world_xy": [midpoint_x, midpoint_y],
        "unit_tangent_world_xy": [tangent_x, tangent_y],
        "lateral_coordinate_m": _frozen_binary64_fma(
            relative_y, tangent_y, relative_x * tangent_x,
        ),
    }


def _hex64(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RegenerationError(f"{label} is not lowercase SHA-256")
    return value


def _read_regular(path: Path, label: str) -> bytes:
    """Read one immutable, ordinary, single-link file without following links."""

    before = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RegenerationError(f"{label} is not a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    projection = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if projection(before) != projection(after):
        raise RegenerationError(f"{label} changed while reading")
    return b"".join(blocks)


def _load_canonical_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular(path, label)
    value = parse_canonical_json(raw, label=label)
    if not isinstance(value, dict):
        raise RegenerationError(f"{label} must contain a JSON object")
    return raw, value


def _binding(path: Path, label: str) -> dict[str, Any]:
    raw = _read_regular(path, label)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _root_inventory(path: Path | str, label: str) -> dict[str, Any]:
    # V3's helper walks only the exact supplied custody root, refuses links and
    # non-regular leaves, records complete file/directory manifests, and never
    # accesses a repository-wide or sealed path.
    return V3E._root_inventory(path, label)


def _assert_inventory_equal(
    before: Mapping[str, Any], after: Mapping[str, Any], label: str
) -> None:
    if canonical_json_bytes(before) != canonical_json_bytes(after):
        raise RegenerationError(f"{label} changed during the read-only audit")


def _assert_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise RegenerationError(f"{label} must be absent")


def _validate_no_self_digest(value: Mapping[str, Any], label: str) -> None:
    V3E.V2E.V1._validate_no_self_digest(value, label)


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as exc:
        raise RegenerationError(
            "V4 Git custody check failed: "
            + exc.output.decode("utf-8", "replace")
        ) from exc


def _validate_historical_receipt_authority(module: Any) -> dict[str, Any]:
    authority = _call(module, "historical_custody_authority")
    if not isinstance(authority, Mapping):
        raise RegenerationError("V4 historical custody authority is not an object")
    row = json.loads(json.dumps(authority))
    V3E.V2E.V1._validate_content_digest(row, "V4 historical custody authority")
    if (
        row.get("receipt_path") != str(DEFAULT_HISTORICAL_CUSTODY_RECEIPT)
        or row.get("receipt_schema") != HISTORICAL_CUSTODY_SCHEMA
        or row.get("ordinary_canonical_json_without_self_digest") is not True
        or row.get("historical_snapshot_or_probe_reexecution_forbidden") is not True
        or row.get("historical_roots_may_not_be_scientific_inputs_to_v4") is not True
        or row.get("v3_terminal_interpretation") != V3_TERMINAL_INTERPRETATION
        or row.get("development_source_audit_disclosure")
        != DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
        or row.get("zero_counter_scope")
        != EXTERNAL_CUSTODY_ZERO_COUNTER_SCOPE
    ):
        raise RegenerationError("V4 historical custody authority identity drift")
    return row


def _historical_root_paths() -> tuple[tuple[str, Path], ...]:
    return (
        ("v1_official_root", DEFAULT_V1_OFFICIAL_ROOT),
        ("v1_material_root", DEFAULT_V1_MATERIAL_ROOT),
        ("v2_official_root", DEFAULT_V2_OFFICIAL_ROOT),
        ("v2_material_root", DEFAULT_V2_MATERIAL_ROOT),
        ("v3_official_root", DEFAULT_V3_OFFICIAL_ROOT),
        ("v3_material_root", DEFAULT_V3_MATERIAL_ROOT),
    )


def _inventory_file_row(
    inventory: Mapping[str, Any], relative: str, label: str
) -> dict[str, Any]:
    rows = [item for item in inventory["files"] if item.get("path") == relative]
    if len(rows) != 1:
        raise RegenerationError(f"{label} lacks exact file {relative}")
    return dict(rows[0])


def _assert_historical_root_expectations(
    inventories: Mapping[str, Mapping[str, Any]], authority: Mapping[str, Any]
) -> None:
    expectations = authority.get("root_expectations")
    if not isinstance(expectations, Mapping) or tuple(inventories) != tuple(
        authority.get("root_keys", ())
    ):
        raise RegenerationError("historical root authority key/order drift")
    for key, inventory in inventories.items():
        expected = expectations.get(key)
        if not isinstance(expected, Mapping):
            raise RegenerationError(f"missing historical root expectation: {key}")
        for field in ("path", "file_count", "regular_file_apparent_bytes", "manifest_sha256"):
            if inventory.get(field) != expected.get(field):
                raise RegenerationError(
                    f"historical root frozen projection drift: {key}.{field}"
                )


def _validate_prior_receipts_and_roots(
    inventories: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    v1_raw, v1_receipt = _load_canonical_object(
        DEFAULT_V1_CUSTODY_RECEIPT, "V1 external custody receipt"
    )
    _validate_no_self_digest(v1_receipt, "V1 external custody receipt")
    v1_binding = {
        "path": str(DEFAULT_V1_CUSTODY_RECEIPT),
        "bytes": len(v1_raw),
        "sha256": hashlib.sha256(v1_raw).hexdigest(),
    }
    expected_v1_binding = {
        "path": str(DEFAULT_V1_CUSTODY_RECEIPT),
        "bytes": 18403,
        "sha256": "bb4950d2bde0bf1971e643c746b15a28a1949b1ef3d70736bdaae9da45282d32",
    }
    if v1_binding != expected_v1_binding:
        raise RegenerationError("V1 external custody receipt binding drift")
    v1_official_projection = dict(inventories["v1_official_root"])
    v1_material_projection = dict(inventories["v1_material_root"])
    v1_official_projection.pop("manifest_sha256", None)
    v1_material_projection.pop("manifest_sha256", None)
    if (
        v1_receipt.get("official_root") != v1_official_projection
        or v1_receipt.get("material_root") != v1_material_projection
    ):
        raise RegenerationError("V1 receipt/root manifest cross-link drift")

    v1_v2_raw, v1_v2_receipt = _load_canonical_object(
        DEFAULT_V1_V2_CUSTODY_RECEIPT, "combined V1/V2 custody receipt"
    )
    _validate_no_self_digest(v1_v2_receipt, "combined V1/V2 custody receipt")
    # This document-only validator is system-Python safe.  It does not rebuild
    # historical semantic projections and therefore performs zero historical
    # deserializations or simulator/probe calls.
    V3E.validate_historical_custody_receipt_document(v1_v2_receipt)
    v1_v2_binding = {
        "path": str(DEFAULT_V1_V2_CUSTODY_RECEIPT),
        "bytes": len(v1_v2_raw),
        "sha256": hashlib.sha256(v1_v2_raw).hexdigest(),
    }
    expected_v1_v2_binding = {
        "path": str(DEFAULT_V1_V2_CUSTODY_RECEIPT),
        "bytes": 2572931,
        "sha256": "38d900b29ddb6ed672d771bb19e5b9878fb011ccf4061f847a76f75fde9d7bbc",
    }
    if v1_v2_binding != expected_v1_v2_binding:
        raise RegenerationError("combined V1/V2 custody receipt binding drift")
    for receipt_key, inventory_key in (
        ("v1_official_root", "v1_official_root"),
        ("v1_material_root", "v1_material_root"),
        ("v2_official_root", "v2_official_root"),
        ("v2_material_root", "v2_material_root"),
    ):
        if v1_v2_receipt.get(receipt_key) != inventories[inventory_key]:
            raise RegenerationError(
                f"combined V1/V2 receipt/root cross-link drift: {receipt_key}"
            )
    return {
        "v1_custody_receipt_binding": v1_binding,
        "v1_v2_custody_receipt_binding": v1_v2_binding,
        "v2_regeneration_receipt_present": False,
        "v3_regeneration_receipt_present": False,
    }


def _validate_v3_partial_boundary(
    inventories: Mapping[str, Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    official = inventories["v3_official_root"]
    material = inventories["v3_material_root"]
    expected_boundary = authority.get("v3_partial_boundary_expectation")
    expected_failures = authority.get("v3_failure_expectation")
    if not isinstance(expected_boundary, Mapping) or not isinstance(
        expected_failures, list
    ):
        raise RegenerationError("V3 boundary authority is absent")
    if [item["path"] for item in official["files"]] != expected_boundary[
        "official_leaves"
    ]:
        raise RegenerationError("V3 official partial inventory drift")
    first_relative = "v1_v2_v3_first_eight_reproduction.json"
    _raw, first = _load_canonical_object(
        DEFAULT_V3_OFFICIAL_ROOT / first_relative,
        "V3 first-eight reproduction receipt",
    )
    if (
        first.get("pass") is not True
        or first.get("status") != "PASS"
        or first.get("full_collection_authorized") is not True
        or first.get("candidate_ranker_development_heldout_outcomes_opened") != 0
    ):
        raise RegenerationError("V3 first-eight pass boundary drift")

    expected_pool_indices = list(expected_boundary["completed_pool_indices"])
    observed_metadata = [
        item["path"]
        for item in material["files"]
        if item["path"].startswith("qualification/pool-")
        and item["path"].endswith("/metadata.json")
    ]
    expected_metadata = [
        f"qualification/pool-{index:03d}/metadata.json"
        for index in expected_pool_indices
    ]
    if observed_metadata != expected_metadata:
        raise RegenerationError("V3 completed qualification pool inventory drift")
    material_file_paths = {item["path"] for item in material["files"]}
    expected_material_files = {
        "material_contract.json",
        "prospective_pool.json",
        "reproduction/first_eight_behavioural_probes.npz",
        *expected_metadata,
        *(
            f"qualification/pool-{index:03d}/payload.npz"
            for index in expected_pool_indices
        ),
    }
    if material_file_paths != expected_material_files:
        raise RegenerationError("V3 partial material contains unexpected paths")

    qualified = 0
    rejected = 0
    for index in expected_pool_indices:
        prefix = f"qualification/pool-{index:03d}"
        metadata_relative = f"{prefix}/metadata.json"
        payload_relative = f"{prefix}/payload.npz"
        metadata_row = _inventory_file_row(material, metadata_relative, "V3 material")
        payload_row = _inventory_file_row(material, payload_relative, "V3 material")
        raw, metadata = _load_canonical_object(
            DEFAULT_V3_MATERIAL_ROOT / metadata_relative,
            f"V3 pool-{index:03d} metadata",
        )
        if len(raw) != metadata_row["bytes"] or hashlib.sha256(raw).hexdigest() != metadata_row["sha256"]:
            raise RegenerationError(f"V3 pool-{index:03d} metadata binding drift")
        payload = metadata.get("payload")
        if (
            metadata.get("experiment_id") != V3E.EXPERIMENT_ID
            or metadata.get("pool_index") != index
            or metadata.get("reset_or_candidate_outcome_opened") is not False
            or not isinstance(metadata.get("qualified"), bool)
            or not isinstance(payload, Mapping)
            or payload.get("path") != payload_relative
            or payload.get("bytes") != payload_row["bytes"]
            or payload.get("sha256") != payload_row["sha256"]
        ):
            raise RegenerationError(f"V3 pool-{index:03d} custody boundary drift")
        qualified += int(metadata["qualified"])
        rejected += int(not metadata["qualified"])
    if (
        qualified != expected_boundary["qualified_count"]
        or rejected != expected_boundary["rejected_count"]
    ):
        raise RegenerationError("V3 qualification disposition counts drift")
    interpretation = {
        "diagnosis": V3_TERMINAL_DIAGNOSIS,
        "first_eight_reproduction_exact": bool(
            first.get("pass") is True
            and first.get("status") == "PASS"
            and first.get("row_count") == 8
        ),
        "panel_fanout_ranker_or_heldout_outcomes_opened": bool(
            expected_boundary.get("panel_constructed")
            or expected_boundary.get("candidate_fanout_executions")
            or expected_boundary.get("ranker_inference_calls")
            or expected_boundary.get("development_or_heldout_outcomes_opened")
        ),
        "prospective_panel_collection_stopped_because_tipped_state_disposition_unspecified": bool(
            len(expected_pool_indices) < 256
            and all(
                row.get("exception_type") == "BoundaryRefused"
                and row.get("exception_message")
                == "termination flag tipped is True"
                for row in expected_failures
            )
        ),
        "raw_torch_snapshot_transport_bytes_are_scientific_evidence": not (
            "raw artifact-file SHA equality or inequality is descriptive only"
            in str(first.get("comparison_rule"))
        ),
        "scientific_handoff_result_produced": any(
            item["path"] in {"metrics.json", "result.json", "result.md"}
            for item in official["files"]
        ),
        "snapshot_behavioural_equivalence_passed": first.get(
            "behavioural_all_pass"
        )
        is True,
        "snapshot_semantic_equivalence_passed": first.get("semantic_all_pass")
        is True,
    }
    if interpretation != authority.get("v3_terminal_interpretation") or (
        interpretation != V3_TERMINAL_INTERPRETATION
    ):
        raise RegenerationError("V3 terminal interpretation evidence drift")
    return (
        json.loads(json.dumps(expected_boundary)),
        json.loads(json.dumps(expected_failures)),
        interpretation,
    )


def validate_historical_custody_receipt_document(
    value: Any, *, metrics_module: Any | None = None
) -> dict[str, Any]:
    module = _load_metrics_module() if metrics_module is None else metrics_module
    _validate_historical_receipt_authority(module)
    validated = _call(module, "validate_external_v1_v2_v3_custody_receipt", value)
    if not isinstance(validated, Mapping):
        raise RegenerationError("historical custody validator returned no object")
    _validate_no_self_digest(validated, "V1/V2/V3 historical custody receipt")
    return json.loads(json.dumps(validated))


def build_historical_custody_receipt(
    *, metrics_module: Any | None = None, require_v4_absent: bool = True
) -> dict[str, Any]:
    """Rebuild the complete ordinary V1/V2/V3 custody receipt read-only."""

    module = _load_metrics_module() if metrics_module is None else metrics_module
    authority = _validate_historical_receipt_authority(module)
    if require_v4_absent:
        for path, label in (
            (DEFAULT_V4_OUTPUT_ROOT, "V4 official root"),
            (DEFAULT_V4_MATERIAL_ROOT, "V4 material root"),
            (DEFAULT_EXTERNAL_RECEIPT, "V4 external regeneration receipt"),
        ):
            _assert_absent(path, label)
    if DEFAULT_V2_REGENERATION_RECEIPT.exists() or DEFAULT_V2_REGENERATION_RECEIPT.is_symlink():
        raise RegenerationError("V2 regeneration receipt unexpectedly exists")
    if DEFAULT_V3_REGENERATION_RECEIPT.exists() or DEFAULT_V3_REGENERATION_RECEIPT.is_symlink():
        raise RegenerationError("V3 regeneration receipt unexpectedly exists")

    before = {
        key: _root_inventory(path, key)
        for key, path in _historical_root_paths()
    }
    _assert_historical_root_expectations(before, authority)
    V3E._assert_no_shared_inodes(*before.values())
    prior = _validate_prior_receipts_and_roots(before)
    boundary, failures, interpretation = _validate_v3_partial_boundary(
        before, authority
    )

    live_head = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    if live_head != SOURCE_PARENT_COMMIT:
        live_parent = _git_bytes("rev-parse", f"{live_head}^").decode("ascii").strip()
        live_subject = _git_bytes(
            "show", "-s", "--format=%s", live_head
        ).decode("utf-8").strip()
        if live_parent != SOURCE_PARENT_COMMIT or live_subject != FREEZE_SUBJECT:
            raise RegenerationError(
                "historical receipt rebuild is outside the V3-parent/V4-freeze boundary"
            )
    repository = {
        "v1_source_freeze_commit": V3E.V2E.SOURCE_PARENT_COMMIT,
        "v2_source_freeze_commit": V3E.SOURCE_PARENT_COMMIT,
        "v3_source_freeze_commit": SOURCE_PARENT_COMMIT,
        "v1_freeze_subject": "Freeze physical graph edge handoff qualification",
        "v2_freeze_subject": "Freeze corrected physical graph edge handoff qualification V2",
        "v3_freeze_subject": "Freeze semantic physical graph edge handoff qualification V3",
        "v4_source_parent_commit": SOURCE_PARENT_COMMIT,
        # This is the immutable emission-time fact.  Exact rebuild after the
        # direct-child V4 freeze deliberately retains the recorded V3 HEAD.
        "current_head_commit": SOURCE_PARENT_COMMIT,
        "repository_worktree_clean_at_receipt_emission": False,
        "historical_tracked_sources_match_frozen_commits": True,
        "v4_development_only": True,
        "v4_permanently_ineligible_for_final_evaluation": True,
        "sealed_paths_accessed": 0,
        "tracked_ignore_bypass_used": False,
    }
    result = {
        "schema": HISTORICAL_CUSTODY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "generated_before_v4_simulator_creation": True,
        "repository": repository,
        "prior_receipt_bindings": prior,
        "roots": before,
        "v3_partial_boundary": boundary,
        "v3_failure_log_bindings": failures,
        "v3_terminal_interpretation": interpretation,
        "immutability": {
            "audit_mode": "READ_ONLY",
            "v1_official_root_unchanged_during_audit": True,
            "v1_material_root_unchanged_during_audit": True,
            "v2_official_root_unchanged_during_audit": True,
            "v2_material_root_unchanged_during_audit": True,
            "v3_official_root_unchanged_during_audit": True,
            "v3_material_root_unchanged_during_audit": True,
            "all_files_regular_single_link": True,
            "receipt_outside_all_experiment_roots": True,
        },
        "nonreuse": {
            "historical_roots_shared_inode_count": 0,
            "historical_payload_copy_into_v4_count": 0,
            "historical_hardlink_into_v4_count": 0,
            "historical_runtime_artifact_or_shard_reused": False,
            "historical_snapshot_deserializations": 0,
            "model_initializations": 0,
            "training_runs": 0,
            "simulator_initializations": 0,
            "runner_calls": 0,
            "encoder_calls": 0,
            "ranker_calls": 0,
            "sealed_path_accesses": 0,
            "ignore_bypasses": 0,
        },
    }
    validated = validate_historical_custody_receipt_document(
        result, metrics_module=module
    )
    after = {
        key: _root_inventory(path, key)
        for key, path in _historical_root_paths()
    }
    for key in before:
        _assert_inventory_equal(before[key], after[key], key)
    return validated


def emit_historical_custody_receipt(
    output: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
    *, receipt: Mapping[str, Any] | None = None,
    metrics_module: Any | None = None,
) -> bytes:
    value = (
        build_historical_custody_receipt(metrics_module=metrics_module)
        if receipt is None
        else validate_historical_custody_receipt_document(
            receipt, metrics_module=metrics_module
        )
    )
    _validate_no_self_digest(value, "V1/V2/V3 historical custody receipt")
    roots = tuple(path for _key, path in _historical_root_paths()) + (
        DEFAULT_V4_OUTPUT_ROOT,
        DEFAULT_V4_MATERIAL_ROOT,
    )
    return V3E.V2E._emit_external(roots, output, value)


def validate_existing_historical_custody_receipt(
    output: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
    *, metrics_module: Any | None = None,
) -> dict[str, Any]:
    path = Path(output)
    raw, supplied = _load_canonical_object(
        path, "combined V1/V2/V3 historical custody receipt"
    )
    rebuilt = build_historical_custody_receipt(
        metrics_module=metrics_module, require_v4_absent=False
    )
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError(
            "combined V1/V2/V3 historical custody receipt differs from exact rebuild"
        )
    binding = _binding(path, "combined V1/V2/V3 historical custody receipt")
    module = _load_metrics_module() if metrics_module is None else metrics_module
    _call(
        module,
        "validate_external_v1_v2_v3_custody_receipt",
        supplied,
        expected_binding=binding,
    )
    return validate_historical_custody_receipt_document(
        supplied, metrics_module=module
    )


def _official_binding(leaf: str, raw: bytes) -> dict[str, Any]:
    return {
        "path": leaf,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _open_official_root(path: Path | str) -> tuple[Path, int]:
    return V3E.V2E.V1._open_root(path, "V4 official root")


def _validate_official_inventory(
    root_fd: int,
) -> tuple[list[str], str, bool]:
    names = sorted(os.listdir(root_fd))
    observed = set(names)
    allowed = {
        ("PANEL_INADEQUATE", False): set(V4_PANEL_INADEQUATE_SCIENTIFIC_FILES),
        ("PANEL_INADEQUATE", True): set(V4_PANEL_INADEQUATE_FILES),
        ("SUCCESS", False): set(V4_SUCCESS_SCIENTIFIC_FILES),
        ("SUCCESS", True): set(V4_SUCCESS_FILES),
    }
    matches = [key for key, expected in allowed.items() if observed == expected]
    if len(matches) != 1:
        raise RegenerationError(
            "V4 root is not an exact 6/9-leaf panel terminal or "
            "24/27-leaf successful inventory"
        )
    mode, complete = matches[0]
    for name in names:
        V3E.V2E.CUSTODY._safe_leaf(name)
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RegenerationError(
                f"V4 official leaf is not a single-link regular file: {name}"
            )
    return names, mode, complete


def _validate_invalidated_partial_root_manifest_authority(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegenerationError(
            "V4 invalidated partial-root manifest authority is not an object"
        )
    row = json.loads(json.dumps(value))
    V1E._validate_content_digest(
        row, "V4 invalidated partial-root manifest authority"
    )
    expected_fields = {
        "schema",
        "portable_complete_root_projection_fields",
        "portable_file_row_fields",
        "portable_files_sort",
        "complete_root_projection_sha256_domain",
        "files_array_sha256_domain",
        "roots",
        "qualification_material_pair_count",
        "all_observed_leaves_ordinary_single_link_regular_files",
        "symlinks_observed",
        "observed_before_invalidated_roots_are_replaced",
        "manifest_is_transparency_evidence_not_reusable_scientific_input",
        "fresh_recomputation_may_legitimately_reproduce_identical_file_bytes",
        "file_hash_inequality_is_nonreuse_proof",
        "content_digest",
    }
    expected_roots = {
        "official": {
            "path": str(DEFAULT_V4_OUTPUT_ROOT),
            "file_count": 3,
            "regular_file_apparent_bytes": 150396,
            "complete_root_projection_canonical_byte_count": 552,
            "complete_root_projection_sha256": (
                "c996f0f1242bc03756198d7eddfab69b6aeacf513150ffda13fdd567589a2766"
            ),
            "files_array_sha256": (
                "e9c8c9519a744a85b1cccc1746e7d7b473bcb67d999db820f5c469d486c49f20"
            ),
        },
        "material": {
            "path": str(DEFAULT_V4_MATERIAL_ROOT),
            "file_count": 258,
            "regular_file_apparent_bytes": 124565766,
            "complete_root_projection_canonical_byte_count": 35783,
            "complete_root_projection_sha256": (
                "2bd85555bc1e7f7e48039364273982925a0110921e130eee761f332896a02d76"
            ),
            "files_array_sha256": (
                "00346fd51dc42052143568e0b9bf26462182800eb1071cd8ec305f9772fc6cff"
            ),
        },
    }
    if (
        set(row) != expected_fields
        or row.get("schema")
        != (
            "physical_graph_edge_handoff_qualification_v4."
            "invalidated_partial_root_manifest_authority.v1"
        )
        or row.get("portable_complete_root_projection_fields")
        != ["path", "file_count", "regular_file_apparent_bytes", "files"]
        or row.get("portable_file_row_fields") != ["path", "bytes", "sha256"]
        or row.get("portable_files_sort") != "relative POSIX path lexicographic"
        or row.get("complete_root_projection_sha256_domain")
        != (
            "SHA-256 of canonical_json_bytes over the complete portable root "
            "projection, including the canonical terminal LF"
        )
        or row.get("files_array_sha256_domain")
        != (
            "SHA-256 of canonical_json_bytes over only the complete sorted "
            "files array, including the canonical terminal LF"
        )
        or row.get("roots") != expected_roots
        or row.get("qualification_material_pair_count") != 128
        or row.get("all_observed_leaves_ordinary_single_link_regular_files")
        is not True
        or row.get("symlinks_observed") != 0
        or row.get("observed_before_invalidated_roots_are_replaced") is not True
        or row.get("manifest_is_transparency_evidence_not_reusable_scientific_input")
        is not True
        or row.get("fresh_recomputation_may_legitimately_reproduce_identical_file_bytes")
        is not True
        or row.get("file_hash_inequality_is_nonreuse_proof") is not False
    ):
        raise RegenerationError(
            "V4 invalidated partial-root manifest authority drift"
        )
    return row


def _validate_pre_panel_engineering_correction_authority(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RegenerationError(
            "V4 pre-panel engineering correction authority is not an object"
        )
    row = json.loads(json.dumps(value))
    V1E._validate_content_digest(
        row, "V4 pre-panel engineering correction authority"
    )
    expected_fields = {
        "schema",
        "status",
        "invalidated_source_freeze_commit",
        "invalidated_runtime_contract_content_digest",
        "invalidated_scientific_contract_content_digest",
        "invalidated_partial_root_manifest_authority",
        "completed_pool_indices_at_detection",
        "qualification_material_pair_count_at_detection",
        "official_leaves_at_detection",
        "detected_before_qualification_ledger_or_panel_construction",
        "qualification_ledger_or_panel_adequacy_persisted",
        "downstream_outcomes_opened",
        "producer_runtime_numpy_version",
        "independent_reducer_numpy_version",
        "defect",
        "persisted_crossing_mismatch_pool_indices",
        "unmaterialized_compact_projection_mismatch_pool_indices",
        "affected_fields",
        "dependent_fields_recomputed_from_canonical_projection",
        "pure_projection_fields",
        "binary64_operation_order",
        "matched_frozen_producer_values_for_available_pool_count",
        "corrected_full_raw_reduction_cross_runtime_exact_for_available_pool_count",
        "corrected_full_raw_reduction_aggregate_sha256_domain",
        "corrected_full_raw_reduction_aggregate_sha256",
        "cross_runtime_exact_match_required",
        "existing_partial_material_reuse_authorized",
        "file_hash_inequality_is_nonreuse_proof",
        "nonreuse_proof",
        "restart_from_fresh_v4_roots_required",
        "qualification_disposition_or_criterion_changed",
        "mathematical_formula_or_tolerance_changed",
        "metric_gate_threshold_model_tuning_or_role_rule_changed",
        "v1_v2_v3_source_or_evidence_changed",
        "content_digest",
    }
    true_fields = {
        "detected_before_qualification_ledger_or_panel_construction",
        "cross_runtime_exact_match_required",
        "restart_from_fresh_v4_roots_required",
    }
    false_fields = {
        "qualification_ledger_or_panel_adequacy_persisted",
        "downstream_outcomes_opened",
        "existing_partial_material_reuse_authorized",
        "file_hash_inequality_is_nonreuse_proof",
        "qualification_disposition_or_criterion_changed",
        "mathematical_formula_or_tolerance_changed",
        "metric_gate_threshold_model_tuning_or_role_rule_changed",
        "v1_v2_v3_source_or_evidence_changed",
    }
    if (
        set(row) != expected_fields
        or row.get("schema")
        != (
            "physical_graph_edge_handoff_qualification_v4."
            "pre_panel_engineering_correction_authority.v1"
        )
        or row.get("status") != PRE_PANEL_ENGINEERING_CORRECTION_STATUS
        or row.get("invalidated_source_freeze_commit")
        != INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        or row.get("invalidated_runtime_contract_content_digest")
        != INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
        or row.get("invalidated_scientific_contract_content_digest")
        != INVALIDATED_V4_SCIENTIFIC_CONTRACT_CONTENT_DIGEST
        or row.get("completed_pool_indices_at_detection") != list(range(128))
        or row.get("qualification_material_pair_count_at_detection") != 128
        or row.get("official_leaves_at_detection")
        != [
            "contract.json",
            "scientific_invariance_receipt.json",
            "v1_v2_v3_custody_and_nonreuse.json",
        ]
        or row.get("producer_runtime_numpy_version") != "2.4.6"
        or row.get("independent_reducer_numpy_version") != "1.26.4"
        or row.get("defect")
        != (
            "two-element NumPy norm/matmul accumulation was not byte-stable "
            "across the frozen producer and independent reducer runtimes"
        )
        or row.get("persisted_crossing_mismatch_pool_indices")
        != list(_PERSISTED_CROSSING_MISMATCH_POOL_INDICES)
        or row.get("unmaterialized_compact_projection_mismatch_pool_indices")
        != list(_UNMATERIALIZED_COMPACT_PROJECTION_MISMATCH_POOL_INDICES)
        or row.get("affected_fields")
        != [
            "teacher.crossing.lateral_coordinate_m",
            "teacher.competing_crossing.lateral_coordinate_m",
            "teacher_trace_index.records[].crossing_lateral_fraction",
            "teacher_trace_index.records[].endpoint_lateral_error_m",
        ]
        or row.get("dependent_fields_recomputed_from_canonical_projection")
        != [
            "state_disposition.teacher_criteria.teacher_within_lateral_bounds",
            "panel_manifest.pool_qualification.teacher_within_lateral_bounds",
        ]
        or row.get("pure_projection_fields") != list(_BINARY64_PROJECTION_FIELDS)
        or row.get("binary64_operation_order") != _BINARY64_OPERATION_ORDER
        or row.get("matched_frozen_producer_values_for_available_pool_count") != 128
        or row.get(
            "corrected_full_raw_reduction_cross_runtime_exact_for_available_pool_count"
        )
        != 128
        or row.get("corrected_full_raw_reduction_aggregate_sha256_domain")
        != _CORRECTED_RAW_REDUCTION_AGGREGATE_DOMAIN
        or row.get("corrected_full_raw_reduction_aggregate_sha256")
        != _CORRECTED_RAW_REDUCTION_AGGREGATE_SHA256
        or row.get("nonreuse_proof")
        != (
            "every replacement terminal binds the corrected runtime contract "
            "content digest and corrected source freeze commit; all 256 bindings "
            "must equal the initialized runtime contract and must differ from "
            "the invalidated source/runtime identities"
        )
        or any(row.get(field) is not True for field in true_fields)
        or any(row.get(field) is not False for field in false_fields)
    ):
        raise RegenerationError("V4 pre-panel engineering correction authority drift")
    _validate_invalidated_partial_root_manifest_authority(
        row.get("invalidated_partial_root_manifest_authority")
    )
    return row


def _validate_reducer_authority(module: Any) -> dict[str, Any]:
    authority = _call(module, "reducer_authority")
    if not isinstance(authority, Mapping):
        raise RegenerationError("V4 reducer authority is not an object")
    row = json.loads(json.dumps(authority))
    V3E.V2E.V1._validate_content_digest(row, "V4 reducer authority")
    if (
        row.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.reducer_authority.v1"
        or row.get("experiment_id") != EXPERIMENT_ID
        or row.get("success_output_leaf_count") != 27
        or row.get("success_output_leaves") != list(V4_SUCCESS_FILES)
        or row.get("panel_inadequate_output_leaf_count") != 9
        or row.get("panel_inadequate_output_leaves")
        != list(V4_PANEL_INADEQUATE_FILES)
        or row.get("development_only") is not True
        or row.get("final_evaluation_eligible") is not False
    ):
        raise RegenerationError("V4 reducer authority identity/inventory drift")
    required = {
        "state_disposition_authority",
        "panel_adequacy_authority",
        "panel_teacher_subset_authority",
        "raw_teacher_reduction_authority",
        "pre_panel_engineering_correction_authority",
        "teacher_termination_derivation_authority",
        "teacher_selection_authority",
        "material_inventory_authority",
        "external_historical_custody_authority",
        "v1_v2_v3_custody_and_nonreuse_authority",
        "scientific_invariance_authority",
        "material_shard_validation_fields",
        "panel_inadequate_evidence_keys",
        "success_evidence_keys",
        "result_fields",
        "publication_projection_fields",
        "result_publication_authority",
        "persisted_array_hash_authority",
        "terminal_nonfinite_authority",
        "qualification_runtime_authority",
    }
    if not required.issubset(row):
        raise RegenerationError(
            f"V4 reducer authority lacks sections: {sorted(required-set(row))}"
        )
    inherited_v3_only = {
        "successful_output_leaf_count",
        "successful_output_leaves",
        "first_eight_reproduction_authority",
        "reproduction_mismatch_output_leaves",
        "new_documents",
        "v3_recompute_evidence_keys",
        "v3_additional_recompute_evidence_keys",
        "v1_custody_and_nonreuse_authority",
        "v1_v2_custody_and_nonreuse_authority",
        "external_historical_custody_receipt_authority",
        "historical_snapshot_deserializer_authority",
        "qualification_shard_augmentation_authority",
        "regression_gate_authority",
    }
    leaked = inherited_v3_only & set(row)
    if leaked:
        raise RegenerationError(
            f"V4 reducer authority leaks predecessor-only fields: {sorted(leaked)}"
        )
    publication = row["result_publication_authority"]
    if not isinstance(publication, Mapping):
        raise RegenerationError("V4 result-publication authority is not an object")
    publication = json.loads(json.dumps(publication))
    V3E.V2E.V1._validate_content_digest(
        publication, "V4 result-publication authority"
    )
    if (
        publication.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.result_publication_authority.v1"
        or publication.get("result_fields") != row["result_fields"]
        or publication.get("publication_projection_fields")
        != row["publication_projection_fields"]
        or publication.get("success_prepublication_binding_count") != 24
        or publication.get("panel_inadequate_prepublication_binding_count") != 6
        or publication.get("external_reducer_receipt_required_for_success_or_panel_inadequate")
        is not True
        or publication.get("development_only") is not True
        or publication.get("final_evaluation_eligible") is not False
        or publication.get(
            "qualification_runtime_environment_required_for_both_terminals"
        )
        is not True
        or publication.get(
            "successful_qualification_runtime_cross_bound_to_assembled_physical_runtime"
        )
        is not True
    ):
        raise RegenerationError("V4 result-publication authority drift")
    for name in (
        "panel_teacher_subset_authority",
        "raw_teacher_reduction_authority",
        "teacher_termination_derivation_authority",
        "teacher_selection_authority",
        "material_inventory_authority",
        "terminal_nonfinite_authority",
        "qualification_runtime_authority",
    ):
        section = row[name]
        if not isinstance(section, Mapping):
            raise RegenerationError(f"V4 {name} is not an object")
        V3E.V2E.V1._validate_content_digest(
            json.loads(json.dumps(section)), f"V4 {name}"
        )
    correction = _validate_pre_panel_engineering_correction_authority(
        row["pre_panel_engineering_correction_authority"]
    )
    correction_digest = correction["content_digest"]
    if row["raw_teacher_reduction_authority"].get(
        "deterministic_binary64_projection_authority_content_digest"
    ) != correction_digest:
        raise RegenerationError(
            "V4 raw teacher reduction lacks corrected binary64 authority"
        )
    terminal_binding = row["state_disposition_authority"].get(
        "terminal_source_runtime_binding"
    )
    expected_terminal_binding = {
        "metadata_fields": [
            "source_freeze_commit", "runtime_contract_content_digest",
        ],
        "source_freeze_commit": (
            "exact initialized runtime_contract.source_freeze_commit"
        ),
        "runtime_contract_content_digest": (
            "exact initialized runtime_contract.content_digest"
        ),
        "invalidated_source_freeze_commit_rejected": (
            INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        ),
        "invalidated_runtime_contract_content_digest_rejected": (
            INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
        ),
        "all_256_terminal_bindings_must_be_equal": True,
        "file_hash_inequality_is_nonreuse_proof": False,
    }
    runtime_authority = row["qualification_runtime_authority"]
    if (
        terminal_binding != expected_terminal_binding
        or runtime_authority.get("terminal_source_runtime_binding_fields")
        != ["source_freeze_commit", "runtime_contract_content_digest"]
        or runtime_authority.get(
            "terminal_source_freeze_commit_must_equal_initialized_runtime"
        )
        is not True
        or runtime_authority.get(
            "terminal_runtime_contract_digest_must_equal_initialized_runtime"
        )
        is not True
        or runtime_authority.get(
            "all_256_terminal_source_runtime_bindings_must_be_identical"
        )
        is not True
        or runtime_authority.get("invalidated_source_freeze_commit_rejected")
        != INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        or runtime_authority.get(
            "invalidated_runtime_contract_content_digest_rejected"
        )
        != INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
        or runtime_authority.get("file_hash_inequality_is_nonreuse_proof")
        is not False
    ):
        raise RegenerationError("V4 corrected terminal nonreuse authority drift")
    invariance_authority = row["scientific_invariance_authority"]
    expected_correction_projection = {
        "status": PRE_PANEL_ENGINEERING_CORRECTION_STATUS,
        "authority_content_digest": correction_digest,
        "existing_partial_material_reuse_authorized": False,
        "restart_from_fresh_v4_roots_required": True,
        "scientific_formula_or_decision_changed": False,
    }
    if (
        not isinstance(invariance_authority, Mapping)
        or invariance_authority.get("authorized_change_scope")
        != AUTHORIZED_SCIENTIFIC_CHANGE
        or invariance_authority.get("v3_terminal_interpretation")
        != V3_TERMINAL_INTERPRETATION
        or invariance_authority.get("pre_panel_engineering_correction")
        != expected_correction_projection
    ):
        raise RegenerationError("V4 scientific-invariance authority drift")
    internal_custody_authority = row[
        "v1_v2_v3_custody_and_nonreuse_authority"
    ]
    if not isinstance(internal_custody_authority, Mapping):
        raise RegenerationError("V4 internal custody authority is not an object")
    V1E._validate_content_digest(
        json.loads(json.dumps(internal_custody_authority)),
        "V4 internal custody authority",
    )
    if (
        internal_custody_authority.get("development_source_audit_disclosure")
        != DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
        or internal_custody_authority.get("zero_counter_scope")
        != SCIENTIFIC_ZERO_COUNTER_SCOPE
    ):
        raise RegenerationError("V4 internal custody counter scope drift")
    material_inventory = row["material_inventory_authority"]
    if (
        material_inventory.get("panel_inadequate_exact_file_count") != 514
        or material_inventory.get("success_exact_file_count") != 805
        or material_inventory.get("success_root_files")
        != [
            "material_contract.json",
            "prospective_pool.json",
            "teacher_selection.json",
            "panel_context.json",
            "encoding_receipt.json",
        ]
        or material_inventory.get("no_unlisted_material_files") is not True
    ):
        raise RegenerationError("V4 material-inventory authority drift")
    terminal_nonfinite = row["terminal_nonfinite_authority"]
    if (
        terminal_nonfinite.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.terminal_nonfinite_authority.v1"
        or terminal_nonfinite.get("scope")
        != "V4 terminal rejection evidence only"
        or terminal_nonfinite.get("initial_allowed_members")
        != [
            "base_pose_world",
            "base_twist_world",
            "joint_position",
            "joint_velocity",
        ]
        or terminal_nonfinite.get("trace_allowed_members")
        != [
            "base_pose_world",
            "base_twist_world",
            "joint_position",
            "joint_velocity",
        ]
        or terminal_nonfinite.get(
            "false_nan_flag_with_any_nonfinite_is_materialisation_corrupt"
        )
        is not True
        or terminal_nonfinite.get(
            "nonfinite_before_final_sample_is_materialisation_corrupt"
        )
        is not True
    ):
        raise RegenerationError("V4 terminal-nonfinite authority drift")
    qualification_runtime = row["qualification_runtime_authority"]
    if (
        qualification_runtime.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.qualification_runtime_authority.v1"
        or qualification_runtime.get("qualification_shard_count") != 256
        or qualification_runtime.get(
            "stage_runtime_must_be_identical_across_all_shards"
        )
        is not True
        or qualification_runtime.get(
            "backend_runtime_core_must_be_identical_across_all_shards"
        )
        is not True
        or qualification_runtime.get("backend_runtime_mode_rule")
        != (
            "teacher_executed=false carries exactly the common backend core; "
            "teacher_executed=true adds exactly "
            "snapshot_captured_before_teacher=true, "
            "teacher_restored_from_serialized_snapshot=true, and a teacher "
            "snapshot SHA equal to the terminal's initial artifact identity"
        )
        or qualification_runtime.get("production_fake_runtime") is not False
        or qualification_runtime.get("models_trained") != 0
        or qualification_runtime.get(
            "prohibited_components_trained_or_implemented"
        )
        != []
        or qualification_runtime.get("metrics_only_projection_no_additional_leaf")
        is not True
    ):
        raise RegenerationError("V4 qualification-runtime authority drift")
    return row


def _load_json_at(root_fd: int, leaf: str) -> tuple[bytes, dict[str, Any]]:
    return V3E.V2E.V1._load_json_at(root_fd, leaf)


def _load_ordinary_json_at(
    root_fd: int, leaf: str, *, label: str
) -> tuple[bytes, dict[str, Any]]:
    """Load canonical ordinary JSON whose schema expressly forbids self-digests."""

    raw = V3E.V2E.CUSTODY._read_regular_at(root_fd, leaf)
    if raw is None:
        raise RegenerationError(f"{label} is absent")
    value = parse_canonical_json(raw, label=label)
    if not isinstance(value, dict):
        raise RegenerationError(f"{label} must contain a JSON object")
    _validate_no_self_digest(value, label)
    return raw, value


def _load_jsonl_at(root_fd: int, leaf: str) -> tuple[bytes, list[dict[str, Any]]]:
    return V3E.V2E.V1._load_jsonl_at(root_fd, leaf)


def _validated_exact(
    supplied: Any, validated: Any, label: str
) -> Any:
    """Require a pure validator to preserve the exact public evidence."""

    if canonical_json_bytes(validated) != canonical_json_bytes(supplied):
        raise RegenerationError(f"V4 pure {label} validation projection drift")
    return validated


def _termination_flags(value: Any, label: str) -> dict[str, bool]:
    order = ("fall", "out_of_bounds", "tipped", "nan")
    if (
        not isinstance(value, Mapping)
        or set(value) != set(order)
        or any(type(value[name]) is not bool for name in order)
    ):
        raise RegenerationError(f"{label} termination-flag drift")
    return {name: value[name] for name in order}


def _terminal_nonfinite_group(
    arrays: Mapping[str, Any],
    flags: Mapping[str, Any],
    *,
    allowed_members: Sequence[str],
    require_trace_axis: bool,
    label: str,
) -> dict[str, Any]:
    """Independently enforce the narrow V4 terminal IEEE-value boundary."""

    import numpy as np

    exact_flags = _termination_flags(flags, label)
    allowed = set(allowed_members)
    masks: dict[str, Any] = {}
    for member, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "fc":
            continue
        mask = ~np.isfinite(array)
        if not bool(mask.any()):
            continue
        if member not in allowed:
            raise RegenerationError(f"{label}/{member} has off-authority nonfinite")
        if require_trace_axis and (array.ndim < 1 or bool(mask[:-1].any())):
            raise RegenerationError(
                f"{label}/{member} has preterminal nonfinite evidence"
            )
        masks[member] = np.ascontiguousarray(mask)
    base_mask = masks.get("base_pose_world")
    base_nonfinite = bool(base_mask is not None and base_mask.any())
    if exact_flags["nan"] is not base_nonfinite:
        raise RegenerationError(f"{label} nan flag/base-pose nonfinite drift")
    if masks and exact_flags["nan"] is not True:
        raise RegenerationError(f"{label} nonfinite evidence has a false nan flag")
    return masks


def _nonfinite_ieee_cells_equal(left: Any, right: Any) -> bool:
    import numpy as np

    a = np.ascontiguousarray(np.asarray(left))
    b = np.ascontiguousarray(np.asarray(right))
    if a.dtype != b.dtype or a.shape != b.shape or a.dtype.kind not in "fc":
        return False
    left_mask = ~np.isfinite(a)
    right_mask = ~np.isfinite(b)
    if not np.array_equal(left_mask, right_mask):
        return False
    if not bool(left_mask.any()):
        return True
    left_bits = a.view(np.uint8).reshape(a.shape + (a.dtype.itemsize,))
    right_bits = b.view(np.uint8).reshape(b.shape + (b.dtype.itemsize,))
    return bool(np.array_equal(left_bits[left_mask], right_bits[right_mask]))


def _validate_terminal_nonfinite_material(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck terminal nonfinite evidence directly from reopened NPZ arrays."""

    import numpy as np

    state = metadata.get("state_disposition")
    if not isinstance(state, Mapping):
        raise RegenerationError("V4 terminal material lacks state disposition")
    stage = state.get("stage_reached")
    processed: set[str] = set()
    group_rows: list[dict[str, Any]] = []

    if stage == "INITIAL_BOUNDARY":
        masks = _terminal_nonfinite_group(
            arrays,
            state.get("initial_termination_flags"),
            allowed_members=authority["initial_allowed_members"],
            require_trace_axis=False,
            label="V4 initial boundary",
        )
        processed.update(arrays)
        group_rows.append(
            {
                "group": "initial",
                "nonfinite_members": sorted(masks),
                "finite_prefix_sample_count": None,
            }
        )
    else:
        probe_flags = state.get("probe_trial_termination_flags")
        if not isinstance(probe_flags, list) or len(probe_flags) != 2:
            raise RegenerationError("V4 terminal material probe flags are absent")
        probe_groups: list[dict[str, Any]] = []
        for trial_index in range(2):
            prefix = f"probe__{trial_index}__"
            group = {
                name[len(prefix):]: value
                for name, value in arrays.items()
                if name.startswith(prefix)
            }
            if "base_pose_world" not in group:
                raise RegenerationError("V4 terminal material probe pose is absent")
            masks = _terminal_nonfinite_group(
                group,
                probe_flags[trial_index],
                allowed_members=authority["trace_allowed_members"],
                require_trace_axis=True,
                label=f"V4 probe trial {trial_index}",
            )
            processed.update(f"{prefix}{name}" for name in group)
            probe_groups.append(group)
            sample_count = int(np.asarray(group["base_pose_world"]).shape[0])
            group_rows.append(
                {
                    "group": f"probe-{trial_index}",
                    "nonfinite_members": sorted(masks),
                    "finite_prefix_sample_count": (
                        sample_count - 1 if masks else sample_count
                    ),
                }
            )
        if stage == "RESTORATION_PROBE":
            common_float_members = {
                name
                for name in probe_groups[0]
                if name in probe_groups[1]
                and np.asarray(probe_groups[0][name]).dtype.kind in "fc"
            }
            if any(
                not _nonfinite_ieee_cells_equal(
                    probe_groups[0][name], probe_groups[1][name]
                )
                for name in common_float_members
            ):
                raise RegenerationError(
                    "V4 probe pair nonfinite mask/IEEE payload drift"
                )

        teacher_prefix = "teacher__"
        teacher_group = {
            name[len(teacher_prefix):]: value
            for name, value in arrays.items()
            if name.startswith(teacher_prefix)
        }
        if teacher_group:
            if "base_pose_world" not in teacher_group:
                raise RegenerationError("V4 terminal material teacher pose is absent")
            teacher_flags = _termination_flags(
                state.get("teacher_termination_flags"), "V4 teacher trace"
            )
            masks = _terminal_nonfinite_group(
                teacher_group,
                teacher_flags,
                allowed_members=authority["trace_allowed_members"],
                require_trace_axis=True,
                label="V4 teacher trace",
            )
            processed.update(f"{teacher_prefix}{name}" for name in teacher_group)
            sample_count = int(np.asarray(teacher_group["base_pose_world"]).shape[0])
            teacher_summary = metadata.get("teacher")
            expected_contact_free = not bool(
                np.asarray(teacher_group["physics_contact"]).any()
            )
            if (
                not isinstance(teacher_summary, Mapping)
                or teacher_summary.get("sample_count") != sample_count
                or teacher_summary.get("termination_flags") != teacher_flags
                or teacher_summary.get("terminated_unsafe")
                is not any(teacher_flags.values())
                or teacher_summary.get("contact_free") is not expected_contact_free
            ):
                raise RegenerationError(
                    "V4 teacher terminal summary/raw-prefix cross-link drift"
                )
            group_rows.append(
                {
                    "group": "teacher",
                    "nonfinite_members": sorted(masks),
                    "finite_prefix_sample_count": (
                        sample_count - 1 if masks else sample_count
                    ),
                }
            )

    for name, value in arrays.items():
        if name in processed:
            continue
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not bool(np.isfinite(array).all()):
            raise RegenerationError(
                f"V4 material has nonfinite outside terminal group: {name}"
            )
    return {
        "groups": group_rows,
        "all_nonfinite_confined_to_authorized_terminal_cells": True,
        "probe_pair_nonfinite_masks_and_ieee_payloads_exact": True,
        "teacher_terminal_summary_raw_cross_links_exact": True,
    }


def _require_real_qualification_runtime(stage_runtime: Mapping[str, Any]) -> None:
    """The sole private seam a synthetic test may replace after proving reject."""

    if stage_runtime.get("fake_runtime") is not False:
        raise RegenerationError("production qualification contains fake runtime")


def _qualification_runtime_digests(
    metadata: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    authority: Mapping[str, Any],
    *,
    expected_pool_index: int,
) -> dict[str, Any]:
    """Independently bind one shard's full stage/backend runtime evidence."""

    runtime_authority = authority["qualification_runtime_authority"]
    stage = metadata.get("stage_runtime")
    backend = metadata.get("backend_runtime")
    stage_fields = runtime_authority.get("stage_runtime_fields")
    core_fields = runtime_authority.get("backend_runtime_core_fields")
    teacher_fields = runtime_authority.get("teacher_only_backend_runtime_fields")
    source_freeze_commit = metadata.get("source_freeze_commit")
    runtime_contract_content_digest = metadata.get(
        "runtime_contract_content_digest"
    )
    if (
        metadata.get("pool_index") != expected_pool_index
        or not isinstance(source_freeze_commit, str)
        or len(source_freeze_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_freeze_commit
        )
        or not isinstance(runtime_contract_content_digest, str)
        or len(runtime_contract_content_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in runtime_contract_content_digest
        )
        or source_freeze_commit == INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        or runtime_contract_content_digest
        == INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
        or source_freeze_commit != runtime_contract.get("source_freeze_commit")
        or runtime_contract_content_digest != runtime_contract.get("content_digest")
        or not isinstance(stage, Mapping)
        or not isinstance(backend, Mapping)
        or not isinstance(stage_fields, list)
        or not isinstance(core_fields, list)
        or not isinstance(teacher_fields, list)
        or set(stage) != set(stage_fields)
    ):
        raise RegenerationError(
            f"V4 qualification source/runtime field drift: {expected_pool_index}"
        )
    _require_real_qualification_runtime(stage)
    physical = runtime_contract.get("scientific_contract", {}).get(
        "runtime_environment_authority", {}
    ).get("physical")
    required_environment = runtime_contract.get("runtime_policy", {}).get(
        "required_environment_before_simulator_creation"
    )
    if not isinstance(physical, Mapping) or not isinstance(
        required_environment, Mapping
    ):
        raise RegenerationError("V4 runtime contract physical authority is absent")
    expected_stage = {
        "stage_id": physical.get("stage_id"),
        "python_executable": physical.get("real_python_executable"),
        "python_version": physical.get("python_version"),
        "torch_version": physical.get("torch_version"),
        "torch_hip_version": physical.get("torch_hip_version"),
        "genesis_version": physical.get("genesis_version"),
        "quadrants_version": physical.get("quadrants_version"),
        "visible_device_count": physical.get("visible_device_count"),
        "device": physical.get("device"),
        "backend": physical.get("backend"),
        "deterministic_environment": dict(required_environment),
        "fake_runtime": stage.get("fake_runtime"),
    }
    # Production cannot reach this branch with fake evidence because the
    # private guard above fails first.  A synthetic integration test may
    # replace only that guard after proving production rejection; retain the
    # truthful test backend/device while validating every other frozen field.
    if stage.get("fake_runtime") is True:
        expected_stage["backend"] = stage.get("backend")
        expected_stage["device"] = stage.get("device")
    if dict(stage) != expected_stage:
        raise RegenerationError(
            f"V4 qualification runtime differs from contract: {expected_pool_index}"
        )
    expected_backend_fields = set(core_fields)
    teacher_executed = metadata.get("teacher_executed")
    if type(teacher_executed) is not bool:
        raise RegenerationError("V4 qualification teacher runtime flag drift")
    if teacher_executed:
        expected_backend_fields.update(teacher_fields)
    if set(backend) != expected_backend_fields:
        raise RegenerationError(
            f"V4 qualification backend field drift: {expected_pool_index}"
        )
    backend_core = {field: backend[field] for field in core_fields}
    expected_backend_core = json.loads(
        json.dumps(runtime_authority.get("backend_runtime_core"))
    )
    if stage.get("fake_runtime") is True and isinstance(
        expected_backend_core, dict
    ):
        expected_backend_core["backend"] = stage.get("backend")
        expected_backend_core["policy_device"] = stage.get("device")
    if backend_core != expected_backend_core:
        raise RegenerationError(
            f"V4 qualification backend core drift: {expected_pool_index}"
        )
    if teacher_executed:
        identity = metadata.get("snapshot_identity")
        artifact_sha = (
            identity.get("artifact_file_sha256")
            if isinstance(identity, Mapping)
            else None
        )
        if (
            backend.get("snapshot_captured_before_teacher") is not True
            or backend.get("teacher_restored_from_serialized_snapshot") is not True
            or backend.get("teacher_snapshot_sha256") != artifact_sha
            or metadata.get("initial_decision_state_sha256") != artifact_sha
        ):
            raise RegenerationError(
                f"V4 qualification teacher/runtime snapshot drift: {expected_pool_index}"
            )
    stage_row = json.loads(json.dumps(stage))
    backend_row = json.loads(json.dumps(backend))
    backend_core_row = json.loads(json.dumps(backend_core))
    stage_sha = hashlib.sha256(canonical_json_bytes(stage_row)).hexdigest()
    return {
        "stage_runtime": stage_row,
        "stage_runtime_sha256": stage_sha,
        "backend_runtime": backend_row,
        "backend_runtime_sha256": hashlib.sha256(
            canonical_json_bytes(backend_row)
        ).hexdigest(),
        "backend_runtime_core": backend_core_row,
        "backend_runtime_core_sha256": hashlib.sha256(
            canonical_json_bytes(backend_core_row)
        ).hexdigest(),
    }


def _validate_probe_state_crosslinks(
    metadata: Mapping[str, Any], arrays: Mapping[str, Any], module: Any
) -> dict[str, Any] | None:
    """Recompute per-trial behavioural digests from reopened trace members."""

    if metadata.get("stage_reached") == "INITIAL_BOUNDARY":
        return None
    state = metadata.get("state_disposition")
    probe = metadata.get("behavioural_probe")
    identity = metadata.get("snapshot_identity")
    if (
        not isinstance(state, Mapping)
        or not isinstance(probe, Mapping)
        or not isinstance(identity, Mapping)
        or state.get("snapshot_identity") != identity
    ):
        raise RegenerationError("V4 probe/state identity cross-link drift")
    trials = probe.get("trials")
    state_flags = state.get("probe_trial_termination_flags")
    state_tips = state.get("probe_tip_sample_indices")
    if (
        not isinstance(trials, list)
        or len(trials) != 2
        or not isinstance(state_flags, list)
        or len(state_flags) != 2
        or not isinstance(state_tips, list)
        or len(state_tips) != 2
    ):
        raise RegenerationError("V4 probe/state trial cardinality drift")
    observed_digests: list[str | None] = []
    for trial_index, trial in enumerate(trials):
        if not isinstance(trial, Mapping) or trial.get("trial_index") != trial_index:
            raise RegenerationError("V4 probe trial identity drift")
        if (
            trial.get("termination_flags") != state_flags[trial_index]
            or trial.get("tip_sample_index") != state_tips[trial_index]
        ):
            raise RegenerationError("V4 probe/state terminal evidence drift")
        manifest = trial.get("trace_member_manifest")
        if not isinstance(manifest, list) or not manifest:
            raise RegenerationError("V4 probe trace manifest is absent")
        trace: dict[str, Any] = {}
        for member_row in manifest:
            if not isinstance(member_row, Mapping):
                raise RegenerationError("V4 probe trace manifest row drift")
            member = member_row.get("member")
            array = arrays.get(f"probe__{trial_index}__{member}")
            if not isinstance(member, str) or array is None:
                raise RegenerationError("V4 probe trace member is absent")
            trace[member] = array
        completed = trial.get("termination_reason") == "H3_COMPLETE"
        if completed:
            trace.update(
                {
                    "termination_reason": trial.get("termination_reason"),
                    "stuck": trial.get("stuck"),
                    "final_snapshot_semantic_digest_v1": trial.get(
                        "final_snapshot_semantic_digest_v1"
                    ),
                }
            )
        digest = (
            _call(module, "snapshot_behavioural_digest", trace)
            if completed
            else None
        )
        if trial.get("snapshot_behavioural_digest_v1") != digest:
            raise RegenerationError("V4 per-trial behavioural digest drift")
        observed_digests.append(digest)
    if (
        probe.get("completed") is True
        and identity.get("snapshot_behavioural_digest_v1") != observed_digests[0]
    ) or (
        probe.get("completed") is False
        and identity.get("snapshot_behavioural_digest_v1") is not None
    ):
        raise RegenerationError("V4 probe/state behavioural identity drift")
    return {
        "trial_snapshot_behavioural_digest_v1s": observed_digests,
        "state_snapshot_behavioural_digest_v1": identity.get(
            "snapshot_behavioural_digest_v1"
        ),
        "trial_terminal_evidence_exact": True,
    }


def _success_cardinality_projection(
    authority: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, int], dict[str, int], list[int]]:
    """Rebuild every successful-document count through its frozen container.

    This deliberately does not guess that every document uses ``records``.
    The frozen schemas use ``states``, ``assignments``, ``graphs``, ``rows``,
    two distinct development-selection containers, and a nested qualification
    ledger.  Exact root/row field checks make this projection a second,
    evaluator-local guard after the pure document validators.
    """

    document_authority = authority.get("documents")
    ledger_authority = authority.get("ledgers")
    if (
        not isinstance(document_authority, Mapping)
        or set(document_authority) != set(_DOCUMENT_LEAVES)
        or set(documents) != set(_DOCUMENT_LEAVES)
        or not isinstance(ledger_authority, Mapping)
        or set(ledger_authority) != set(_LEDGER_LEAVES)
        or set(ledgers) != set(_LEDGER_LEAVES)
    ):
        raise RegenerationError("V4 successful authority/evidence inventory drift")

    frozen_container_projection = {
        "panel_manifest": {
            "container": "states",
            "count": 64,
            "qualification_container": (
                "prospective_pool_selection.qualification_rows"
            ),
            "qualification_count": 256,
        },
        "split_manifest": {"container": "assignments", "count": 64},
        "graph_manifest": {"container": "graphs", "count": 64},
        "state_snapshot_index": {"container": "records", "count": 64},
        "teacher_trace_index": {
            "container": "records",
            "count": "actual teacher_executed state count in [64,256]",
            "minimum_count": 64,
            "maximum_count": 256,
        },
        "edge_port_index": {"container": "records", "count": 64},
        "waypoint_contracts": {"container": "rows", "count": 192},
        "pixel_index": {"container": "records", "count": 64},
        "latent_index": {
            "container": "records",
            "count": "unique_pixel_count",
        },
        "development_target_selection": {
            "state_target_container": "state_target_rows",
            "state_target_count": 144,
            "summary_container": "target_summaries",
            "summary_count": 3,
        },
    }
    for name, expected in frozen_container_projection.items():
        schema = document_authority.get(name)
        if not isinstance(schema, Mapping) or {
            key: schema.get(key) for key in expected
        } != expected:
            raise RegenerationError(
                f"V4 frozen document container authority drift: {name}"
            )

    def exact_root(name: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        document = documents[name]
        schema = document_authority[name]
        root_fields = schema.get("root")
        if (
            not isinstance(document, Mapping)
            or not isinstance(root_fields, list)
            or not all(isinstance(field, str) for field in root_fields)
            or set(document) != set(root_fields)
        ):
            raise RegenerationError(f"V4 successful document root-field drift: {name}")
        return document, schema

    def exact_rows(
        name: str,
        rows: Any,
        fields: Any,
        *,
        suffix: str = "",
    ) -> list[Mapping[str, Any]]:
        label = f"{name}{suffix}"
        if (
            not isinstance(rows, list)
            or not isinstance(fields, list)
            or not all(isinstance(field, str) for field in fields)
        ):
            raise RegenerationError(f"V4 successful row container drift: {label}")
        expected_fields = set(fields)
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != expected_fields:
                raise RegenerationError(
                    f"V4 successful row-field drift: {label}[{index}]"
                )
        return rows

    rows_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for name in (
        "panel_manifest",
        "split_manifest",
        "graph_manifest",
        "state_snapshot_index",
        "teacher_trace_index",
        "edge_port_index",
        "waypoint_contracts",
        "pixel_index",
        "latent_index",
    ):
        document, schema = exact_root(name)
        container = schema["container"]
        rows_by_name[name] = exact_rows(
            name, document.get(container), schema.get("row")
        )

    panel, panel_schema = exact_root("panel_manifest")
    qualification: Any = panel
    for component in str(panel_schema["qualification_container"]).split("."):
        qualification = (
            qualification.get(component)
            if isinstance(qualification, Mapping)
            else None
        )
    qualification_rows = exact_rows(
        "panel_manifest",
        qualification,
        panel_schema.get("qualification_row"),
        suffix=".qualification_rows",
    )

    selection, selection_schema = exact_root("development_target_selection")
    state_target_rows = exact_rows(
        "development_target_selection",
        selection.get(selection_schema["state_target_container"]),
        selection_schema.get("state_target_row"),
        suffix=".state_target_rows",
    )
    target_summaries = exact_rows(
        "development_target_selection",
        selection.get(selection_schema["summary_container"]),
        selection_schema.get("summary_row"),
        suffix=".target_summaries",
    )

    pixel_unique_count = documents["pixel_index"].get("unique_pixel_count")
    teacher_rows = rows_by_name["teacher_trace_index"]
    if (
        type(pixel_unique_count) is not int
        or not 1 <= pixel_unique_count <= 64
        or not 64 <= len(teacher_rows) <= 256
    ):
        raise RegenerationError("V4 dynamic document cardinality drift")

    document_rows = {
        "panel_manifest": len(rows_by_name["panel_manifest"]),
        "panel_qualification": len(qualification_rows),
        "split_manifest": len(rows_by_name["split_manifest"]),
        "graph_manifest": len(rows_by_name["graph_manifest"]),
        "state_snapshot_index": len(rows_by_name["state_snapshot_index"]),
        "teacher_trace_index": len(teacher_rows),
        "edge_port_index": len(rows_by_name["edge_port_index"]),
        "waypoint_contracts": len(rows_by_name["waypoint_contracts"]),
        "pixel_index": len(rows_by_name["pixel_index"]),
        "latent_index": len(rows_by_name["latent_index"]),
        "development_target_selection.state_target_rows": len(state_target_rows),
        "development_target_selection.target_summaries": len(target_summaries),
    }
    expected_document_rows = {
        "panel_manifest": 64,
        "panel_qualification": 256,
        "split_manifest": 64,
        "graph_manifest": 64,
        "state_snapshot_index": 64,
        "teacher_trace_index": len(teacher_rows),
        "edge_port_index": 64,
        "waypoint_contracts": 192,
        "pixel_index": 64,
        "latent_index": pixel_unique_count,
        "development_target_selection.state_target_rows": 144,
        "development_target_selection.target_summaries": 3,
    }
    if document_rows != expected_document_rows:
        raise RegenerationError("V4 successful document row count drift")

    ledger_rows: dict[str, int] = {}
    for name, values in ledgers.items():
        schema = ledger_authority[name]
        fields = schema.get("fields") if isinstance(schema, Mapping) else None
        rows = exact_rows(name, values, fields)
        count = schema.get("count") if isinstance(schema, Mapping) else None
        if type(count) is not int or len(rows) != count:
            raise RegenerationError(f"V4 successful ledger row count drift: {name}")
        ledger_rows[name] = len(rows)
    if ledger_rows != {
        "candidate_fanout": 768,
        "heldout_ranker_scores": 64,
        "repeated_execution": 64,
    }:
        raise RegenerationError("V4 successful ledger authority drift")

    teacher_pool_indices = [row.get("qualification_pool_index") for row in teacher_rows]
    expected_teacher_pools = [
        index
        for index, row in enumerate(qualification_rows)
        if row.get("teacher_trace_id") is not None
    ]
    if (
        any(type(index) is not int for index in teacher_pool_indices)
        or teacher_pool_indices != expected_teacher_pools
    ):
        raise RegenerationError("V4 compact teacher/pool identity mapping drift")
    return document_rows, ledger_rows, teacher_pool_indices


def _validate_success_documents_and_ledgers(
    module: Any,
    authority: Mapping[str, Any],
    documents: Mapping[str, Mapping[str, Any]],
    ledgers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Validate the native V4 sparse-teacher document graph.

    The predecessor reducer's generic authority table assumes one teacher
    trace for every one of the 256 prospective states.  V4 deliberately does
    not fabricate rows for states rejected before the teacher.  Validate the
    public graph through the V4 pure APIs in dependency order, then derive an
    independent compact identity/count projection from the returned bytes.
    """

    panel = _validated_exact(
        documents["panel_manifest"],
        _call(module, "validate_panel_manifest", documents["panel_manifest"]),
        "panel_manifest",
    )
    split = _validated_exact(
        documents["split_manifest"],
        _call(
            module,
            "validate_split_manifest",
            documents["split_manifest"],
            panel,
        ),
        "split_manifest",
    )
    graph = _validated_exact(
        documents["graph_manifest"],
        _call(
            module,
            "validate_graph_manifest",
            documents["graph_manifest"],
            panel,
        ),
        "graph_manifest",
    )
    snapshots = _validated_exact(
        documents["state_snapshot_index"],
        _call(
            module,
            "validate_state_snapshot_index",
            documents["state_snapshot_index"],
            panel,
        ),
        "state_snapshot_index",
    )
    teachers = _validated_exact(
        documents["teacher_trace_index"],
        _call(
            module,
            "validate_teacher_trace_index",
            documents["teacher_trace_index"],
            panel,
        ),
        "teacher_trace_index",
    )
    edge_ports = _validated_exact(
        documents["edge_port_index"],
        _call(
            module,
            "validate_edge_port_index",
            documents["edge_port_index"],
            panel,
            graph,
            teachers,
        ),
        "edge_port_index",
    )
    waypoints = _validated_exact(
        documents["waypoint_contracts"],
        _call(
            module,
            "validate_waypoint_contracts",
            documents["waypoint_contracts"],
            panel,
            graph,
            edge_ports,
        ),
        "waypoint_contracts",
    )
    pixels = _validated_exact(
        documents["pixel_index"],
        _call(
            module,
            "validate_pixel_index",
            documents["pixel_index"],
            panel,
        ),
        "pixel_index",
    )
    latents = _validated_exact(
        documents["latent_index"],
        _call(
            module,
            "validate_latent_index",
            documents["latent_index"],
            pixels,
        ),
        "latent_index",
    )
    fanout = _validated_exact(
        ledgers["candidate_fanout"],
        _call(
            module,
            "validate_candidate_fanout_rows",
            ledgers["candidate_fanout"],
            panel,
            snapshots,
        ),
        "candidate_fanout",
    )
    selection = _validated_exact(
        documents["development_target_selection"],
        _call(
            module,
            "validate_development_target_selection",
            documents["development_target_selection"],
            panel,
            fanout,
            waypoints,
        ),
        "development_target_selection",
    )
    scores = _validated_exact(
        ledgers["heldout_ranker_scores"],
        _call(
            module,
            "validate_heldout_ranker_score_rows",
            ledgers["heldout_ranker_scores"],
            panel,
            fanout,
            selection,
            waypoints,
            teachers,
        ),
        "heldout_ranker_scores",
    )
    repeats = _validated_exact(
        ledgers["repeated_execution"],
        _call(
            module,
            "validate_repeated_execution_rows",
            ledgers["repeated_execution"],
            panel,
            fanout,
            scores,
            snapshots,
        ),
        "repeated_execution",
    )

    validated_documents = {
        "panel_manifest": panel,
        "split_manifest": split,
        "graph_manifest": graph,
        "state_snapshot_index": snapshots,
        "teacher_trace_index": teachers,
        "edge_port_index": edge_ports,
        "waypoint_contracts": waypoints,
        "pixel_index": pixels,
        "latent_index": latents,
        "development_target_selection": selection,
    }
    validated_ledgers = {
        "candidate_fanout": fanout,
        "heldout_ranker_scores": scores,
        "repeated_execution": repeats,
    }
    document_rows, ledger_rows, teacher_pool_indices = (
        _success_cardinality_projection(
            authority, validated_documents, validated_ledgers
        )
    )
    document_validation = {
        "row_counts": document_rows,
        "all_root_and_row_field_authorities_exact": True,
        "all_frozen_container_names_exact": True,
        "teacher_pool_indices_sha256": hashlib.sha256(
            canonical_json_bytes(teacher_pool_indices)
        ).hexdigest(),
        "all_native_v4_validators_passed": True,
    }
    ledger_validation = {
        "row_counts": ledger_rows,
        "all_row_field_authorities_exact": True,
        "all_native_v4_validators_passed": True,
    }
    return document_validation, ledger_validation, document_rows[
        "teacher_trace_index"
    ]


def _material_binding_from_inventory(
    inventory: Mapping[str, Any], relative: str, *, label: str
) -> dict[str, Any]:
    row = _inventory_file_row(inventory, relative, label)
    return {"path": relative, "bytes": row["bytes"], "sha256": row["sha256"]}


def _independent_planar_projection(
    module: Any,
    point_world_xy: Any,
    opening_segment_world: Any,
    *,
    label: str,
) -> dict[str, Any]:
    """Require the pure projection to equal the evaluator's own arithmetic."""

    independent = _canonical_v1_planar_segment_projection(
        point_world_xy, opening_segment_world,
    )
    pure = _call(
        module,
        "canonical_v1_planar_segment_projection",
        point_world_xy,
        opening_segment_world,
    )
    if (
        not isinstance(pure, Mapping)
        or canonical_json_bytes(pure) != canonical_json_bytes(independent)
    ):
        raise RegenerationError(
            f"{label} differs from independent binary64 projection"
        )
    return independent


def _validate_teacher_planar_projections(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, Any],
    module: Any,
    *,
    teacher_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Independently bind every corrected planar teacher projection."""

    import numpy as np

    spec = metadata.get("candidate_spec")
    teacher = metadata.get("teacher")
    if not isinstance(spec, Mapping) or not isinstance(teacher, Mapping):
        raise RegenerationError("V4 teacher projection context is absent")
    geometry = spec.get("geometry")
    if not isinstance(geometry, Mapping):
        raise RegenerationError("V4 teacher projection geometry is absent")
    selected = geometry.get("selected_directed_edge")
    competitors = geometry.get("competing_directed_edges")
    if not isinstance(selected, Mapping) or not isinstance(competitors, list):
        raise RegenerationError("V4 teacher projection edge inventory drift")

    def project_crossing(
        value: Any, edge: Mapping[str, Any], label: str,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise RegenerationError(f"{label} is not an object")
        projection = _independent_planar_projection(
            module,
            value.get("point_world"),
            edge.get("opening_segment_world"),
            label=label,
        )
        if value.get("lateral_coordinate_m") != projection[
            "lateral_coordinate_m"
        ]:
            raise RegenerationError(
                f"{label} lateral coordinate differs from corrected projection"
            )
        return projection

    selected_projection = project_crossing(
        teacher.get("crossing"), selected, "V4 selected teacher crossing",
    )
    competing_value = teacher.get("competing_crossing")
    competing_projection = None
    if competing_value is not None:
        if not isinstance(competing_value, Mapping):
            raise RegenerationError("V4 competing teacher crossing is not an object")
        edge_id = competing_value.get("edge_id")
        matches = [
            edge for edge in competitors
            if isinstance(edge, Mapping) and edge.get("edge_id") == edge_id
        ]
        if len(matches) != 1:
            raise RegenerationError("V4 competing teacher edge identity drift")
        competing_projection = project_crossing(
            competing_value, matches[0], "V4 competing teacher crossing",
        )

    result = {
        "selected_crossing_checked": selected_projection is not None,
        "competing_crossing_checked": competing_projection is not None,
        "teacher_record_checked": teacher_record is not None,
    }
    if teacher_record is None:
        return result
    if not isinstance(teacher_record, Mapping):
        raise RegenerationError("V4 compact teacher record is not an object")
    expected_crossing_fraction = (
        None
        if selected_projection is None
        else selected_projection["lateral_coordinate_m"]
        / selected_projection["segment_width_m"]
        + 0.5
    )
    if teacher_record.get("crossing_lateral_fraction") != expected_crossing_fraction:
        raise RegenerationError(
            "V4 compact teacher crossing fraction differs from corrected projection"
        )

    poses = np.asarray(arrays.get("teacher__base_pose_world"))
    if poses.ndim != 2 or poses.shape[1:] != (7,) or len(poses) == 0:
        raise RegenerationError("V4 compact teacher endpoint pose is absent")
    state = metadata.get("state_disposition")
    flags = state.get("teacher_termination_flags") if isinstance(state, Mapping) else None
    if not isinstance(flags, Mapping) or type(flags.get("nan")) is not bool:
        raise RegenerationError("V4 compact teacher termination flags drift")
    if flags["nan"]:
        if len(poses) > 1:
            endpoint = poses[-2]
        else:
            endpoint = np.asarray(arrays.get("snapshot__base_pose_world"))
            if endpoint.shape != (7,):
                raise RegenerationError(
                    "V4 first-sample NaN teacher endpoint anchor is absent"
                )
    else:
        endpoint = poses[-1]
    endpoint_projection = _independent_planar_projection(
        module,
        endpoint[:2],
        selected.get("opening_segment_world"),
        label="V4 compact teacher endpoint",
    )
    expected_endpoint = abs(endpoint_projection["lateral_coordinate_m"])
    if teacher_record.get("endpoint_lateral_error_m") != expected_endpoint:
        raise RegenerationError(
            "V4 compact teacher endpoint differs from corrected projection"
        )
    return result


def _validate_qualification_material(
    material_root: Path,
    *,
    module: Any,
    runtime_contract: Mapping[str, Any],
    official_rows: Sequence[Mapping[str, Any]],
    panel_inadequate: bool,
    success_documents: Mapping[str, Mapping[str, Any]] | None = None,
    panel_adequacy: Mapping[str, Any] | None = None,
    panel_adequacy_binding: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Reopen and independently bind all 256 terminal metadata/NPZ pairs."""

    root = V3E.V2E._root_path(material_root, "V4 material root")
    inventory = _root_inventory(root, "V4 material root")
    authority = _validate_reducer_authority(module)
    try:
        archive_comment = authority["persisted_array_hash_authority"][
            "npz_archive_comment_utf8"
        ]
    except (KeyError, TypeError) as exc:
        raise RegenerationError("V4 material archive-comment authority drift") from exc
    if not isinstance(archive_comment, str) or not archive_comment:
        raise RegenerationError("V4 material archive-comment authority is invalid")
    if len(official_rows) != 256:
        raise RegenerationError("V4 official disposition row count drift")

    qualification_metadata = [
        item["path"]
        for item in inventory["files"]
        if item["path"].startswith("qualification/pool-")
        and item["path"].endswith("/metadata.json")
    ]
    expected_metadata = [
        f"qualification/pool-{index:03d}/metadata.json" for index in range(256)
    ]
    if qualification_metadata != expected_metadata:
        raise RegenerationError("V4 qualification metadata inventory drift")
    for index in range(256):
        payload = f"qualification/pool-{index:03d}/payload.npz"
        _inventory_file_row(inventory, payload, "V4 material root")

    expected_files = {
        "material_contract.json",
        "prospective_pool.json",
        *expected_metadata,
        *(f"qualification/pool-{index:03d}/payload.npz" for index in range(256)),
    }
    expected_directories = {
        ".",
        "qualification",
        "selected",
        "fanout",
        "repeat",
        "scenes",
        *(f"qualification/pool-{index:03d}" for index in range(256)),
    }
    selected_state_ids: list[str] = []
    heldout_state_ids: list[str] = []
    if panel_inadequate:
        if any(
            value is not None
            for value in (success_documents, panel_adequacy, panel_adequacy_binding)
        ):
            raise RegenerationError("panel-inadequate material received success context")
    else:
        if (
            not isinstance(success_documents, Mapping)
            or not isinstance(panel_adequacy, Mapping)
            or not isinstance(panel_adequacy_binding, Mapping)
        ):
            raise RegenerationError("successful V4 material context is absent")
        states = success_documents["panel_manifest"].get("states")
        if not isinstance(states, list) or len(states) != 64:
            raise RegenerationError("successful V4 panel states are absent")
        selected_state_ids = [str(row["state_id"]) for row in states]
        heldout_state_ids = [
            str(row["state_id"])
            for row in states
            if row.get("role") == "DEVELOPMENT_HELDOUT"
        ]
        if len(heldout_state_ids) != 16:
            raise RegenerationError("successful V4 heldout state count drift")
        for state_id in selected_state_ids:
            candidate = Path(state_id)
            if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != state_id:
                raise RegenerationError("unsafe V4 selected state identity")
        expected_files.update(
            {
                "teacher_selection.json",
                "panel_context.json",
                "encoding_receipt.json",
            }
        )
        for stage, state_ids in (
            ("selected", selected_state_ids),
            ("fanout", selected_state_ids),
            ("repeat", heldout_state_ids),
        ):
            for state_id in state_ids:
                expected_directories.add(f"{stage}/{state_id}")
                expected_files.add(f"{stage}/{state_id}/metadata.json")
                expected_files.add(f"{stage}/{state_id}/payload.npz")
    observed_files = {item["path"] for item in inventory["files"]}
    observed_directories = {item["path"] for item in inventory["directories"]}
    if observed_files != expected_files or observed_directories != expected_directories:
        mode = "panel-inadequate" if panel_inadequate else "successful"
        raise RegenerationError(
            f"{mode} V4 material contains missing, downstream, or unexpected paths"
        )

    validations: list[dict[str, Any]] = []
    rebuilt_rows: list[dict[str, Any]] = []
    candidate_specs: list[dict[str, Any]] = []
    raw_teacher_reductions: list[dict[str, Any]] = []
    validated_metadata_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    teacher_record_by_pool: dict[int, Mapping[str, Any]] = {}
    if not panel_inadequate:
        assert success_documents is not None
        teacher_records = success_documents["teacher_trace_index"].get("records")
        if not isinstance(teacher_records, list):
            raise RegenerationError("successful V4 teacher records are absent")
        for record in teacher_records:
            pool_index = record.get("qualification_pool_index")
            if (
                not isinstance(pool_index, int)
                or isinstance(pool_index, bool)
                or pool_index in teacher_record_by_pool
            ):
                raise RegenerationError("V4 teacher record pool identity drift")
            teacher_record_by_pool[pool_index] = record
    for index in range(256):
        prefix = f"qualification/pool-{index:03d}"
        metadata_relative = f"{prefix}/metadata.json"
        payload_relative = f"{prefix}/payload.npz"
        metadata_path = root / metadata_relative
        payload_path = root / payload_relative
        metadata_raw, metadata = _load_canonical_object(
            metadata_path, f"V4 pool-{index:03d} metadata"
        )
        V3E.V2E.V1._validate_content_digest(
            metadata, f"V4 pool-{index:03d} metadata"
        )
        metadata_binding = _material_binding_from_inventory(
            inventory, metadata_relative, label="V4 material root"
        )
        payload_binding = _material_binding_from_inventory(
            inventory, payload_relative, label="V4 material root"
        )
        if metadata_binding != {
            "path": metadata_relative,
            "bytes": len(metadata_raw),
            "sha256": hashlib.sha256(metadata_raw).hexdigest(),
        }:
            raise RegenerationError(f"V4 pool-{index:03d} metadata binding drift")
        state = metadata.get("state_disposition")
        persisted = metadata.get("persisted_array_evidence")
        if not isinstance(state, Mapping) or not isinstance(persisted, Mapping):
            raise RegenerationError(
                f"V4 pool-{index:03d} state/persisted evidence is absent"
            )
        validated_state = _call(
            module,
            "validate_state_disposition_record",
            state,
            expected_pool_index=index,
        )
        spec = metadata.get("candidate_spec")
        if (
            metadata.get("schema")
            != "physical_graph_edge_handoff_qualification_v4.teacher_pool_terminal.v1"
            or metadata.get("experiment_id") != EXPERIMENT_ID
            or metadata.get("pool_index") != index
            or not isinstance(spec, Mapping)
            or spec.get("candidate_spec_id") != validated_state["candidate_spec_id"]
            or spec.get("canonical_spec_sha256")
            != validated_state["canonical_spec_sha256"]
            or metadata.get("disposition") != validated_state["disposition"]
            or metadata.get("qualified") is not validated_state["qualified"]
            or metadata.get("stage_reached") != validated_state["stage_reached"]
            or metadata.get("executable_snapshot_exists")
            is not validated_state["executable_snapshot_exists"]
            or metadata.get("teacher_executed") is not validated_state["teacher_executed"]
            or metadata.get("reset_or_candidate_outcome_opened") is not False
        ):
            raise RegenerationError(f"V4 pool-{index:03d} terminal metadata drift")
        arrays, npz_validation = V3E._material_npz_arrays(
            payload_path,
            metadata,
            material_root=root,
            module=module,
            archive_comment=archive_comment,
            label=f"V4 pool-{index:03d} payload",
        )
        validated_material = _call(
            module,
            "validate_qualification_material_shard",
            metadata,
            reopened_arrays=arrays,
            expected_pool_index=index,
        )
        if (
            not isinstance(validated_material, Mapping)
            or canonical_json_bytes(validated_material.get("metadata"))
            != canonical_json_bytes(metadata)
        ):
            raise RegenerationError(
                f"V4 pool-{index:03d} outer material validation projection drift"
            )
        validated_state = validated_material["metadata"]["state_disposition"]
        persisted = validated_material["metadata"]["persisted_array_evidence"]
        validated_metadata = json.loads(
            json.dumps(validated_material["metadata"])
        )
        validated_metadata_rows.append(validated_metadata)
        runtime_row = _qualification_runtime_digests(
            validated_metadata,
            runtime_contract,
            authority,
            expected_pool_index=index,
        )
        runtime_rows.append(runtime_row)
        _validate_terminal_nonfinite_material(
            validated_metadata,
            arrays,
            authority["terminal_nonfinite_authority"],
        )
        _validate_probe_state_crosslinks(validated_metadata, arrays, module)
        if validated_state["teacher_executed"] is True:
            _validate_teacher_planar_projections(
                validated_metadata,
                arrays,
                module,
                teacher_record=teacher_record_by_pool.get(index),
            )
            raw_teacher_reductions.append(
                _call(
                    module,
                    "validate_qualification_teacher_raw_evidence",
                    metadata,
                    arrays,
                    expected_pool_index=index,
                    teacher_record=teacher_record_by_pool.get(index),
                )
            )
            if not panel_inadequate and index not in teacher_record_by_pool:
                raise RegenerationError(
                    f"V4 teacher-executed pool lacks compact record: {index}"
                )
        elif index in teacher_record_by_pool:
            raise RegenerationError(
                f"V4 pre-teacher rejection acquired a compact record: {index}"
            )
        persisted_sha = hashlib.sha256(canonical_json_bytes(persisted)).hexdigest()
        rebuilt = _call(
            module,
            "build_qualification_disposition_row",
            validated_state,
            material_metadata_binding=metadata_binding,
            material_payload_binding=payload_binding,
            persisted_array_evidence_sha256=persisted_sha,
        )
        if canonical_json_bytes(rebuilt) != canonical_json_bytes(official_rows[index]):
            raise RegenerationError(
                f"V4 pool-{index:03d} official disposition row differs from material"
            )
        state_projection = {
            key: official_rows[index][key]
            for key in validated_state
        }
        state_sha = hashlib.sha256(canonical_json_bytes(state_projection)).hexdigest()
        validation = {
            "pool_index": index,
            "metadata_binding": metadata_binding,
            "payload_binding": payload_binding,
            "persisted_array_evidence_sha256": persisted_sha,
            "state_disposition_sha256": state_sha,
            "stage_runtime_sha256": runtime_row["stage_runtime_sha256"],
            "backend_runtime_sha256": runtime_row["backend_runtime_sha256"],
            "backend_runtime_core_sha256": runtime_row[
                "backend_runtime_core_sha256"
            ],
            "metadata_payload_reopened": True,
            "persisted_arrays_valid": True,
            "pass": True,
        }
        expected_fields = set(authority["material_shard_validation_fields"])
        if set(validation) != expected_fields:
            raise RegenerationError("V4 material validation field authority drift")
        validations.append(validation)
        candidate_specs.append(json.loads(json.dumps(spec)))
        rebuilt_rows.append(
            {
                "pool_index": index,
                "disposition": validated_state["disposition"],
                "metadata_sha256": metadata_binding["sha256"],
                "payload_sha256": payload_binding["sha256"],
                "array_count": npz_validation["array_count"],
                "array_inventory_sha256": npz_validation[
                    "array_inventory_sha256"
                ],
            }
        )

    downstream_rows: list[dict[str, Any]] = []
    teacher_selection_validation: dict[str, Any] | None = None
    inherited_support_validation: dict[str, Any] | None = None
    success_support_documents: dict[str, Any] = {}
    if not panel_inadequate:
        assert success_documents is not None
        assert panel_adequacy is not None
        assert panel_adequacy_binding is not None
        selection_raw, selection = _load_canonical_object(
            root / "teacher_selection.json", "V4 teacher selection"
        )
        V1E._validate_content_digest(selection, "V4 teacher selection")
        panel = success_documents["panel_manifest"]
        teachers = success_documents["teacher_trace_index"]
        qualification = panel["prospective_pool_selection"]["qualification_rows"]
        selected_pool_indices = panel_adequacy.get("selected_pool_indices")
        if not isinstance(selected_pool_indices, list) or len(selected_pool_indices) != 64:
            raise RegenerationError("V4 selected pool-index evidence drift")
        expected_selected_specs = [candidate_specs[index] for index in selected_pool_indices]
        expected_rejection_reason_counts: dict[str, int] = {}
        for row in qualification:
            if row.get("qualified") is False:
                reason = row.get("rejection_reason")
                if not isinstance(reason, str) or not reason:
                    raise RegenerationError(
                        "V4 nonqualified row lacks its singular rejection reason"
                    )
                expected_rejection_reason_counts[reason] = (
                    expected_rejection_reason_counts.get(reason, 0) + 1
                )
        expected_selection_fields = {
            "schema",
            "experiment_id",
            "candidate_specs_sha256",
            "teacher_traces_file",
            "teacher_records",
            "qualification_rows",
            "qualification_projection_sha256",
            "selected_specs",
            "selected_pool_indices",
            "rejection_reason_counts",
            "panel_adequacy_sha256",
            "fanout_or_ranker_outcomes_opened",
            "content_digest",
        }
        if (
            set(selection) != expected_selection_fields
            or selection.get("schema")
            != "physical_graph_edge_handoff_qualification_v4.teacher_selection.v1"
            or selection.get("experiment_id") != EXPERIMENT_ID
            or selection.get("candidate_specs_sha256")
            != panel["prospective_pool_selection"]["candidate_specs_sha256"]
            or selection.get("teacher_traces_file") != teachers["traces_file"]
            or selection.get("teacher_records") != teachers["records"]
            or selection.get("qualification_rows") != qualification
            or selection.get("qualification_projection_sha256")
            != hashlib.sha256(canonical_json_bytes(qualification)).hexdigest()
            or selection.get("selected_specs") != expected_selected_specs
            or selection.get("selected_pool_indices") != selected_pool_indices
            or selection.get("rejection_reason_counts")
            != expected_rejection_reason_counts
            or selection.get("panel_adequacy_sha256")
            != panel_adequacy_binding.get("sha256")
            or selection.get("fanout_or_ranker_outcomes_opened") != 0
        ):
            raise RegenerationError("V4 teacher selection/material cross-link drift")
        teacher_selection_validation = {
            "path": "teacher_selection.json",
            "bytes": len(selection_raw),
            "sha256": hashlib.sha256(selection_raw).hexdigest(),
            "teacher_trace_count": len(teachers["records"]),
            "selected_pool_count": len(selected_pool_indices),
            "compact_teacher_mapping_exact": True,
            "panel_adequacy_binding_exact": True,
        }

        panel_context_raw, panel_context = _load_canonical_object(
            root / "panel_context.json", "V4 material panel context"
        )
        V1E._validate_content_digest(
            panel_context, "V4 material panel context"
        )
        panel_states = panel["states"]
        expected_roles = {
            row["state_id"]: row["role"] for row in panel_states
        }
        if (
            set(panel_context)
            != {
                "schema",
                "experiment_id",
                "selected_state_ids",
                "role_by_state",
                "panel_sha256",
                "split_sha256",
                "heldout_outcomes_opened",
                "content_digest",
            }
            or panel_context.get("schema")
            != "physical_graph_edge_handoff_qualification_v4.panel_context.v1"
            or panel_context.get("experiment_id") != EXPERIMENT_ID
            or panel_context.get("selected_state_ids") != selected_state_ids
            or panel_context.get("role_by_state") != expected_roles
            or panel_context.get("panel_sha256")
            != hashlib.sha256(canonical_document_bytes(panel)).hexdigest()
            or panel_context.get("split_sha256")
            != hashlib.sha256(
                canonical_document_bytes(success_documents["split_manifest"])
            ).hexdigest()
            or panel_context.get("heldout_outcomes_opened") != 0
        ):
            raise RegenerationError("V4 material panel-context drift")
        panel_context = _validated_exact(
            panel_context,
            _call(
                module,
                "validate_panel_context",
                panel_context,
                panel,
                success_documents["split_manifest"],
            ),
            "panel_context",
        )

        encoding_raw, encoding = _load_canonical_object(
            root / "encoding_receipt.json", "V4 material encoding receipt"
        )
        V1E._validate_content_digest(
            encoding, "V4 material encoding receipt"
        )
        latent = success_documents["latent_index"]
        latent_records = latent["records"]
        expected_encoding_records = [
            {
                "canonical_pixel_index": row["canonical_pixel_index"],
                "pixel_sha256": row["pixel_sha256"],
                "preprocessed_tensor_sha256": row[
                    "preprocessed_tensor_sha256"
                ],
                "raw_token_sha256": row["raw_token_sha256"],
                "spatial_descriptor_sha256": row[
                    "spatial_descriptor_sha256"
                ],
            }
            for row in latent_records
        ]
        if (
            set(encoding)
            != {
                "schema",
                "experiment_id",
                "singleton_count",
                "pixel_order",
                "records",
                "preprocessing_authority",
                "external_encoder_source",
                "encoder_runtime_environment",
                "fanout_outcomes_opened",
                "content_digest",
            }
            or encoding.get("schema")
            != "physical_graph_edge_handoff_qualification_v4.encoding_receipt.v1"
            or encoding.get("experiment_id") != EXPERIMENT_ID
            or encoding.get("singleton_count") != len(latent_records)
            or encoding.get("pixel_order")
            != [row["pixel_sha256"] for row in latent_records]
            or encoding.get("records") != expected_encoding_records
            or encoding.get("preprocessing_authority")
            != latent["preprocessing_authority"]
            or encoding.get("external_encoder_source")
            != latent["external_encoder_source"]
            or encoding.get("encoder_runtime_environment")
            != latent["encoder_runtime_environment"]
            or encoding.get("fanout_outcomes_opened") != 0
        ):
            raise RegenerationError("V4 material encoding-receipt drift")
        encoding = _validated_exact(
            encoding,
            _call(
                module,
                "validate_encoding_receipt",
                encoding,
                panel,
                success_documents["pixel_index"],
                latent,
            ),
            "encoding_receipt",
        )
        success_support_documents = {
            "panel_context": panel_context,
            "encoding_receipt": encoding,
        }
        inherited_support_validation = {
            "panel_context": {
                "path": "panel_context.json",
                "bytes": len(panel_context_raw),
                "sha256": hashlib.sha256(panel_context_raw).hexdigest(),
                "selected_state_count": len(selected_state_ids),
                "heldout_outcomes_opened": 0,
            },
            "encoding_receipt": {
                "path": "encoding_receipt.json",
                "bytes": len(encoding_raw),
                "sha256": hashlib.sha256(encoding_raw).hexdigest(),
                "singleton_count": len(latent_records),
                "fanout_outcomes_opened": 0,
            },
        }

        expected_stage_schemas = {
            "selected": (
                "physical_graph_edge_handoff_qualification_v4."
                "selected_state_shard.v1"
            ),
            "fanout": (
                "physical_graph_edge_handoff_qualification_v4."
                "fanout_state_shard.v1"
            ),
            "repeat": (
                "physical_graph_edge_handoff_qualification_v4."
                "repeat_state_shard.v1"
            ),
        }
        for stage, state_ids in (
            ("selected", selected_state_ids),
            ("fanout", selected_state_ids),
            ("repeat", heldout_state_ids),
        ):
            for state_id in state_ids:
                relative_metadata = f"{stage}/{state_id}/metadata.json"
                relative_payload = f"{stage}/{state_id}/payload.npz"
                metadata_raw, metadata = _load_canonical_object(
                    root / relative_metadata,
                    f"V4 {stage} material {state_id}",
                )
                V1E._validate_content_digest(
                    metadata, f"V4 {stage} material {state_id}"
                )
                if (
                    metadata.get("schema") != expected_stage_schemas[stage]
                    or metadata.get("experiment_id") != EXPERIMENT_ID
                    or metadata.get("state_id") != state_id
                ):
                    raise RegenerationError(
                        f"V4 {stage} material identity drift: {state_id}"
                    )
                arrays, validation = V3E._material_npz_arrays(
                    root / relative_payload,
                    metadata,
                    material_root=root,
                    module=module,
                    archive_comment=archive_comment,
                    label=f"V4 {stage} material {state_id}",
                )
                del arrays
                if validation.get("payload_path") != relative_payload:
                    raise RegenerationError(
                        f"V4 {stage} material payload path drift: {state_id}"
                    )
                downstream_rows.append(
                    {
                        "metadata_path": relative_metadata,
                        "metadata_bytes": len(metadata_raw),
                        "metadata_sha256": hashlib.sha256(metadata_raw).hexdigest(),
                        **validation,
                    }
                )
    if (
        len(validated_metadata_rows) != 256
        or len(runtime_rows) != 256
        or any(
            row["stage_runtime"] != runtime_rows[0]["stage_runtime"]
            for row in runtime_rows[1:]
        )
        or any(
            row["backend_runtime_core"]
            != runtime_rows[0]["backend_runtime_core"]
            for row in runtime_rows[1:]
        )
    ):
        raise RegenerationError("V4 qualification runtime population drift")
    qualification_runtime = _call(
        module,
        "build_qualification_runtime_environment",
        validated_metadata_rows,
        runtime_contract,
    )
    qualification_runtime = _validated_exact(
        qualification_runtime,
        _call(
            module,
            "validate_qualification_runtime_environment",
            qualification_runtime,
            runtime_contract,
            metadata_rows=validated_metadata_rows,
            material_shard_validations=validations,
        ),
        "qualification_runtime_environment",
    )
    qualification_runtime = _validated_exact(
        qualification_runtime,
        _call(
            module,
            "validate_qualification_runtime_environment",
            qualification_runtime,
            runtime_contract,
            material_shard_validations=validations,
        ),
        "qualification_runtime_environment/material validations",
    )
    publication_runtime_fields = authority["result_publication_authority"].get(
        "qualification_runtime_environment_fields"
    )
    if (
        not isinstance(publication_runtime_fields, list)
        or set(qualification_runtime) != set(publication_runtime_fields)
        or qualification_runtime.get("fake_runtime") is not False
        or qualification_runtime.get("models_trained") != 0
        or qualification_runtime.get(
            "prohibited_components_trained_or_implemented"
        )
        != []
        or qualification_runtime.get("qualification_stage_runtime_sha256s")
        != [row["stage_runtime_sha256"] for row in runtime_rows]
        or qualification_runtime.get("qualification_backend_runtime_sha256s")
        != [row["backend_runtime_sha256"] for row in runtime_rows]
    ):
        raise RegenerationError("V4 qualification runtime projection drift")
    after_inventory = _root_inventory(root, "V4 material root")
    _assert_inventory_equal(inventory, after_inventory, "V4 material root")
    return validations, {
        "material_file_count": inventory["file_count"],
        "material_directory_count": inventory["directory_count"],
        "qualification_pair_count": 256,
        "downstream_pair_count": len(downstream_rows),
        "total_material_shard_count": 256 + len(downstream_rows),
        "all_metadata_and_payload_files_regular_single_link": True,
        "all_npz_members_reopened": True,
        "all_dtype_shape_raw_c_byte_hashes_exact": True,
        "all_archive_comments_exact": True,
        "qualification_projection_sha256": hashlib.sha256(
            canonical_json_bytes(rebuilt_rows)
        ).hexdigest(),
        "raw_teacher_physics_validation": {
            "teacher_trace_count": len(raw_teacher_reductions),
            "pool_indices": [row["pool_index"] for row in raw_teacher_reductions],
            "qualified_teacher_trace_count": sum(
                row["disposition"] == "QUALIFIED"
                for row in raw_teacher_reductions
            ),
            "unsafe_teacher_trace_count": sum(
                any(row["termination_flags"].values())
                for row in raw_teacher_reductions
            ),
            "raw_projection_sha256": hashlib.sha256(
                canonical_json_bytes(raw_teacher_reductions)
            ).hexdigest(),
            "all_teacher_labels_regenerated_from_reopened_material": True,
        },
        "downstream_projection_sha256": hashlib.sha256(
            canonical_json_bytes(downstream_rows)
        ).hexdigest(),
        "teacher_selection_validation": teacher_selection_validation,
        "inherited_support_document_validation": inherited_support_validation,
        "qualification_runtime_environment": qualification_runtime,
        "root_inventory_manifest_sha256": inventory["manifest_sha256"],
    }, success_support_documents


def _repo_source_binding(relative: str, *, head: str) -> dict[str, Any]:
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() != relative
        or relative == "sealed_test.json"
        or "sealed" in candidate.parts
        or any(part.startswith("sealed_") for part in candidate.parts)
    ):
        raise RegenerationError(f"unsafe V4 source path: {relative!r}")
    path = REPO_ROOT / candidate
    info = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or path.resolve(strict=True) != path
    ):
        raise RegenerationError(f"V4 source is not a direct regular file: {relative}")
    raw = _read_regular(path, f"V4 source {relative}")
    if raw != _git_bytes("show", f"{head}:{relative}"):
        raise RegenerationError(f"live V4 source differs from freeze: {relative}")
    return {
        "path": relative,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _observe_source_freeze(
    runtime_contract: Mapping[str, Any], module: Any
) -> dict[str, Any]:
    head = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    if _git_bytes("status", "--porcelain=v1", "--untracked-files=all"):
        raise RegenerationError("V4 reducer requires a clean source-freeze worktree")
    if runtime_contract.get("source_freeze_commit") != head:
        raise RegenerationError("V4 runtime contract differs from HEAD")
    parent = _git_bytes("rev-parse", f"{head}^").decode("ascii").strip()
    subject = _git_bytes("show", "-s", "--format=%s", head).decode(
        "utf-8"
    ).strip()
    if parent != SOURCE_PARENT_COMMIT or subject != FREEZE_SUBJECT:
        raise RegenerationError("V4 source freeze parent/subject drift")
    scientific = runtime_contract.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise RegenerationError("V4 runtime contract lacks scientific contract")
    tracked = scientific.get("tracked_source_paths")
    closure = scientific.get("source_closure_paths")
    if (
        not isinstance(tracked, list)
        or not isinstance(closure, list)
        or len(tracked) < 8
        or len(closure) < 1
        or len(tracked) != len(set(tracked))
        or len(closure) != len(set(closure))
    ):
        raise RegenerationError("V4 source-path authority drift")
    forbidden = [
        item
        for item in [*tracked, *closure]
        if item == "sealed_test.json"
        or "sealed" in Path(item).parts
        or any(part.startswith("sealed_") for part in Path(item).parts)
    ]
    if forbidden:
        raise RegenerationError("V4 source closure contains sealed paths")
    tracked_bindings = [_repo_source_binding(path, head=head) for path in tracked]
    closure_candidates = [path for path in tracked if "source_closure" in path]
    if len(closure_candidates) != 1:
        raise RegenerationError("V4 source-closure document identity drift")
    closure_raw, closure_document = _load_canonical_object(
        REPO_ROOT / closure_candidates[0], "V4 source closure"
    )
    V3E.V2E.V1._validate_content_digest(closure_document, "V4 source closure")
    expected_rows = [_repo_source_binding(path, head=head) for path in closure]
    if (
        closure_document.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.source_closure.v1"
        or closure_document.get("parent_commit") != SOURCE_PARENT_COMMIT
        or closure_document.get("row_count") != len(expected_rows)
        or closure_document.get("rows") != expected_rows
    ):
        raise RegenerationError("V4 source closure differs from ordered live bytes")
    return {
        "head_commit": head,
        "parent_commit": parent,
        "freeze_subject": subject,
        "worktree_clean": True,
        "tracked_source_count": len(tracked_bindings),
        "tracked_sources_sha256": hashlib.sha256(
            canonical_json_bytes(tracked_bindings)
        ).hexdigest(),
        "source_closure_path": closure_candidates[0],
        "source_closure_bytes": len(closure_raw),
        "source_closure_sha256": hashlib.sha256(closure_raw).hexdigest(),
        "source_closure_row_count": len(expected_rows),
        "source_closure_live_bytes_exact": True,
        "sealed_path_accesses": 0,
        "ignore_bypasses": 0,
        "metrics_module": getattr(module, "__name__", type(module).__name__),
    }


def _validate_source_freeze_observation(
    value: Mapping[str, Any], *, expected_commit: str,
    expected_tracked_source_count: int,
    expected_source_closure_row_count: int,
    expected_metrics_module: str,
) -> dict[str, Any]:
    fields = {
        "head_commit",
        "parent_commit",
        "freeze_subject",
        "worktree_clean",
        "tracked_source_count",
        "tracked_sources_sha256",
        "source_closure_path",
        "source_closure_bytes",
        "source_closure_sha256",
        "source_closure_row_count",
        "source_closure_live_bytes_exact",
        "sealed_path_accesses",
        "ignore_bypasses",
        "metrics_module",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RegenerationError("V4 source-freeze observation field drift")
    row = dict(value)
    if (
        row["head_commit"] != expected_commit
        or row["parent_commit"] != SOURCE_PARENT_COMMIT
        or row["freeze_subject"] != FREEZE_SUBJECT
        or row["worktree_clean"] is not True
        or row["source_closure_live_bytes_exact"] is not True
        or row["sealed_path_accesses"] != 0
        or row["ignore_bypasses"] != 0
        or row["tracked_source_count"] != expected_tracked_source_count
        or row["source_closure_row_count"]
        != expected_source_closure_row_count
        or row["metrics_module"] != expected_metrics_module
    ):
        raise RegenerationError("V4 source-freeze observation failed")
    _hex64(row["tracked_sources_sha256"], "V4 tracked source projection")
    _hex64(row["source_closure_sha256"], "V4 source closure")
    return json.loads(json.dumps(row))


def _validate_runtime_contract(
    value: Mapping[str, Any], module: Any
) -> dict[str, Any]:
    validator = getattr(module, "validate_runtime_contract", None)
    if callable(validator):
        try:
            row = validator(value)
        except Exception as exc:
            raise RegenerationError(
                f"V4 pure runtime-contract validation failed: {exc}"
            ) from exc
    else:
        contract_module = getattr(module, "C", None)
        validator = getattr(contract_module, "validate_runtime_contract", None)
        if not callable(validator):
            raise RegenerationError("V4 pure runtime-contract validator is absent")
        try:
            row = validator(value)
        except Exception as exc:
            raise RegenerationError(
                f"V4 contract rejected runtime evidence: {exc}"
            ) from exc
    if not isinstance(row, Mapping):
        raise RegenerationError("V4 runtime-contract validator returned no object")
    if (
        row.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.runtime_contract.v1"
        or row.get("experiment_id") != EXPERIMENT_ID
        or row.get("source_parent_commit") != SOURCE_PARENT_COMMIT
        or row.get("source_baseline_commit") != SOURCE_BASELINE_COMMIT
    ):
        raise RegenerationError("V4 runtime contract identity drift")
    scientific = row.get("scientific_contract")
    obsolete_v3_runtime_fields = {
        "first_eight_reproduction_authority",
        "qualification_shard_augmentation_authority",
        "historical_snapshot_deserializer_authority",
        "v1_v2_custody_and_nonreuse_authority",
        "v3_runtime_policy",
        "v3_wrapper_dependency_paths",
    }
    if not isinstance(scientific, Mapping):
        raise RegenerationError("V4 runtime lacks its scientific contract")
    leaked = obsolete_v3_runtime_fields & set(scientific)
    if leaked:
        raise RegenerationError(
            f"V4 runtime retains obsolete V3 fields: {sorted(leaked)}"
        )
    v4_policy = row.get("v4_runtime_policy")
    correction = _validate_pre_panel_engineering_correction_authority(
        scientific.get("pre_panel_engineering_correction_authority")
    )
    expected_policy_correction = {
        "status": PRE_PANEL_ENGINEERING_CORRECTION_STATUS,
        "authority_content_digest": correction["content_digest"],
        "existing_partial_material_reuse_authorized": False,
        "restart_from_fresh_v4_roots_required": True,
    }
    if (
        not isinstance(v4_policy, Mapping)
        or v4_policy.get("development_source_audit_disclosure")
        != DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
        or v4_policy.get("zero_counter_scope")
        != SCIENTIFIC_ZERO_COUNTER_SCOPE
        or v4_policy.get("ignore_bypasses") != 0
        or v4_policy.get("pre_panel_engineering_correction")
        != expected_policy_correction
        or row.get("source_freeze_commit")
        == INVALIDATED_V4_SOURCE_FREEZE_COMMIT
        or row.get("content_digest")
        == INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
        or scientific.get("content_digest")
        == INVALIDATED_V4_SCIENTIFIC_CONTRACT_CONTENT_DIGEST
    ):
        raise RegenerationError(
            "V4 runtime source-audit/correction/restart authority drift"
        )
    return json.loads(json.dumps(row))


def _validate_v4_nonreuse(
    *,
    historical: Mapping[str, Any],
    output_root: Path,
    material_root: Path,
) -> dict[str, Any]:
    historical_roots = historical.get("roots")
    if not isinstance(historical_roots, Mapping):
        raise RegenerationError("historical root inventories are absent")
    v4_official = _root_inventory(output_root, "V4 official root")
    v4_material = _root_inventory(material_root, "V4 material root")
    historical_inodes = {
        (item["device"], item["inode"])
        for root in historical_roots.values()
        for item in root["files"]
    }
    v4_inodes = {
        (item["device"], item["inode"])
        for root in (v4_official, v4_material)
        for item in root["files"]
    }
    shared_inodes = historical_inodes & v4_inodes
    if shared_inodes:
        raise RegenerationError("V4 roots share historical file inodes")
    historical_file_hashes = {
        item["sha256"]
        for root in historical_roots.values()
        for item in root["files"]
    }
    v4_file_hashes = {
        item["sha256"]
        for root_inventory in (v4_official, v4_material)
        for item in root_inventory["files"]
    }
    copied_files = historical_file_hashes & v4_file_hashes
    if copied_files:
        raise RegenerationError("V4 roots contain a byte-identical historical file")
    return {
        "historical_shared_inode_count": 0,
        "historical_file_sha256_reuse_count": 0,
        "historical_payload_file_sha256_reuse_count": 0,
        "historical_runtime_artifact_or_shard_reused": False,
        "historical_snapshot_or_probe_rerun": False,
        "sealed_path_accesses": 0,
        "ignore_bypasses": 0,
    }


def _validate_publication_if_present(
    *,
    root: Path,
    root_fd: int,
    complete: bool,
    mode: str,
    scientific_bindings: Mapping[str, Mapping[str, Any]],
    recomputed_metrics: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
    expected_receipt_sha256: str,
    historical_binding: Mapping[str, Any],
    module: Any,
) -> dict[str, Any] | None:
    if not complete:
        return None
    result_raw, result = _load_json_at(root_fd, "result.json")
    report_raw = V3E.V2E.CUSTODY._read_regular_at(root_fd, "result.md")
    manifest_raw, manifest = _load_json_at(root_fd, "file_hashes.json")
    assert report_raw is not None
    try:
        report_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegenerationError("V4 result.md is not UTF-8") from exc
    if not report_raw or not report_raw.endswith(b"\n"):
        raise RegenerationError("V4 result.md must be nonempty and LF terminated")
    expected_rows = [dict(item) for item in scientific_bindings.values()]
    expected_rows.extend(
        (
            _official_binding("result.json", result_raw),
            _official_binding("result.md", report_raw),
        )
    )
    expected_rows.sort(key=lambda item: item["path"])
    expected_count = 8 if mode == "PANEL_INADEQUATE" else 26
    expected_leaf_count = 9 if mode == "PANEL_INADEQUATE" else 27
    if (
        manifest.get("schema")
        != "physical_graph_edge_handoff_qualification_v4.file_hashes.v1"
        or manifest.get("root") != str(root)
        or manifest.get("files") != expected_rows
        or manifest.get("file_count_excluding_self") != expected_count
        or manifest.get("bytes_excluding_self")
        != sum(int(item["bytes"]) for item in expected_rows)
        or manifest.get("file_hashes_self_sha256_excluded") is not True
    ):
        raise RegenerationError("V4 file_hashes.json differs from exact live bytes")
    V3E.V2E.V1._validate_content_digest(manifest, "V4 file hashes")
    projection = _call(
        module,
        "validate_result_publication_projection",
        result,
        recomputed_metrics=recomputed_metrics,
        scientific_bindings=scientific_bindings,
        runtime_contract=runtime_contract,
        independent_reducer_receipt_sha256=expected_receipt_sha256,
        historical_custody_receipt_binding=historical_binding,
    )
    expected_report = _call(module, "build_result_report_bytes", projection)
    if not isinstance(expected_report, bytes) or report_raw != expected_report:
        raise RegenerationError("V4 result.md differs from exact reduced report")
    if len(expected_rows) + 1 != expected_leaf_count:
        raise RegenerationError("V4 publication inventory arithmetic drift")
    return {
        "result_binding": _official_binding("result.json", result_raw),
        "report_binding": _official_binding("result.md", report_raw),
        "manifest_binding": _official_binding("file_hashes.json", manifest_raw),
        "result_projection_validated": True,
        "report_exact_byte_equal": True,
        "manifest_exact_live_bytes": True,
    }


def _receipt_counters() -> dict[str, int]:
    """Counters scoped to this V4 scientific reduction process only."""

    return {
        "checkpoint_deserializations": 0,
        "encoder_initializations": 0,
        "model_initializations": 0,
        "ranker_inference_calls": 0,
        "training_steps": 0,
        "simulator_initializations": 0,
        "simulator_steps": 0,
        "physics_steps": 0,
        "reset_calls": 0,
        "render_calls": 0,
        "runner_calls": 0,
        "historical_snapshot_deserializations": 0,
        "historical_probe_executions": 0,
        "sealed_path_accesses": 0,
        "ignore_bypasses": 0,
    }


def _canonicalise_mapping(value: Mapping[str, Any], label: str) -> tuple[bytes, dict[str, Any]]:
    raw = canonical_document_bytes(dict(value))
    parsed = parse_canonical_json(raw, label=label)
    if not isinstance(parsed, dict):
        raise RegenerationError(f"{label} canonical form is not an object")
    return raw, parsed


def _build_panel_inadequate_reduction(
    *,
    root: Path,
    root_fd: int,
    material_root: Path,
    module: Any,
    authority: Mapping[str, Any],
    contract_raw: bytes,
    contract: Mapping[str, Any],
    historical: Mapping[str, Any],
    historical_binding: Mapping[str, Any],
    internal_raw: bytes,
    internal: Mapping[str, Any],
    invariance_raw: bytes,
    invariance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    dispositions_raw, parsed_rows = _load_jsonl_at(
        root_fd, "qualification_state_dispositions.jsonl"
    )
    rows = _call(
        module,
        "validate_qualification_state_dispositions_jsonl",
        dispositions_raw,
    )
    if canonical_json_bytes(rows) != canonical_json_bytes(parsed_rows):
        raise RegenerationError("V4 disposition JSONL parser projection drift")
    panel_raw, panel = _load_json_at(root_fd, "panel_adequacy.json")
    panel = _call(module, "validate_panel_adequacy", panel, rows)
    if panel.get("adequate") is not False or panel.get("status") != PANEL_INADEQUATE_DISPOSITION:
        raise RegenerationError("V4 panel-inadequate inventory has an adequate panel")
    (
        material_validations,
        material_summary,
        support_documents,
    ) = _validate_qualification_material(
        material_root,
        module=module,
        runtime_contract=contract,
        official_rows=rows,
        panel_inadequate=True,
    )
    if support_documents:
        raise RegenerationError("panel-inadequate material acquired support documents")
    evidence = {
        "runtime_contract": dict(contract),
        "external_historical_custody_receipt": dict(historical),
        "v1_v2_v3_custody_and_nonreuse": dict(internal),
        "scientific_invariance_receipt": dict(invariance),
        "qualification_state_dispositions_jsonl": dispositions_raw,
        "panel_adequacy": dict(panel),
        "material_shard_validations": material_validations,
        "qualification_runtime_environment": material_summary[
            "qualification_runtime_environment"
        ],
    }
    if set(evidence) != set(authority["panel_inadequate_evidence_keys"]):
        raise RegenerationError("V4 panel-inadequate evidence key-set drift")
    recomputed = _call(module, "recompute_metrics", evidence)
    if not isinstance(recomputed, Mapping):
        raise RegenerationError("V4 recompute_metrics returned no object")
    recomputed_raw, recomputed = _canonicalise_mapping(
        recomputed, "V4 recomputed panel-inadequate metrics"
    )
    V3E.V2E.V1._validate_content_digest(
        recomputed, "V4 recomputed panel-inadequate metrics"
    )
    metrics_raw, _metrics = _load_json_at(root_fd, "metrics.json")
    if metrics_raw != recomputed_raw:
        raise RegenerationError(
            "V4 panel-inadequate metrics differ from exact independent rebuild"
        )
    bindings = {
        "contract.json": _official_binding("contract.json", contract_raw),
        "v1_v2_v3_custody_and_nonreuse.json": _official_binding(
            "v1_v2_v3_custody_and_nonreuse.json", internal_raw
        ),
        "scientific_invariance_receipt.json": _official_binding(
            "scientific_invariance_receipt.json", invariance_raw
        ),
        "qualification_state_dispositions.jsonl": _official_binding(
            "qualification_state_dispositions.jsonl", dispositions_raw
        ),
        "panel_adequacy.json": _official_binding("panel_adequacy.json", panel_raw),
        "metrics.json": _official_binding("metrics.json", metrics_raw),
    }
    if set(bindings) != set(V4_PANEL_INADEQUATE_SCIENTIFIC_FILES):
        raise RegenerationError("V4 panel terminal binding inventory drift")
    return recomputed, bindings, {
        "qualification_material": material_summary,
        "qualification_row_count": len(rows),
        "panel_status": panel["status"],
        "shortfall_stratum_count": panel["shortfall_stratum_count"],
        "shortfall_strata": panel["shortfall_strata"],
        "shortfall_causes": panel["shortfall_causes"],
        "offset_opening_supplies_any_qualified_state": panel[
            "offset_opening_supplies_any_qualified_state"
        ],
        "downstream_scientific_execution_authorized": False,
        "downstream_outcomes_opened": 0,
        "metrics_exact_byte_equal": True,
        "recomputed_metrics_sha256": hashlib.sha256(metrics_raw).hexdigest(),
    }


def _load_common_gates(
    *,
    root_fd: int,
    contract: Mapping[str, Any],
    module: Any,
    authority: Mapping[str, Any],
    historical_path: Path,
) -> tuple[
    dict[str, bytes],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Validate ordinary V4 gates and their exact frozen external bindings."""

    ordinary_raw: dict[str, bytes] = {}
    ordinary: dict[str, dict[str, Any]] = {}
    for leaf in (
        "v1_v2_v3_custody_and_nonreuse.json",
        "scientific_invariance_receipt.json",
    ):
        raw, value = _load_ordinary_json_at(
            root_fd, leaf, label=f"V4 {leaf}"
        )
        ordinary_raw[leaf] = raw
        ordinary[leaf] = value

    historical = validate_existing_historical_custody_receipt(
        historical_path, metrics_module=module
    )
    historical_binding = _binding(
        historical_path, "combined V1/V2/V3 historical custody receipt"
    )
    if contract.get("historical_custody_receipt_binding") != historical_binding:
        raise RegenerationError("V4 runtime historical custody binding drift")
    historical = _call(
        module,
        "validate_external_v1_v2_v3_custody_receipt",
        historical,
        expected_binding=historical_binding,
    )
    projection_sha = _call(
        module, "historical_custody_projection_sha256", historical
    )
    internal = _call(
        module,
        "validate_v1_v2_v3_custody_and_nonreuse",
        ordinary["v1_v2_v3_custody_and_nonreuse.json"],
        v4_source_freeze_commit=contract["source_freeze_commit"],
        external_receipt_binding=historical_binding,
        external_receipt_projection_sha256=projection_sha,
    )
    invariance = _call(
        module,
        "validate_scientific_invariance_receipt",
        ordinary["scientific_invariance_receipt.json"],
    )
    if internal.get("pass") is not True or invariance.get("pass") is not True:
        raise RegenerationError("V4 historical custody or invariance gate failed")
    correction = _validate_pre_panel_engineering_correction_authority(
        authority.get("pre_panel_engineering_correction_authority")
    )
    expected_correction_projection = {
        "status": PRE_PANEL_ENGINEERING_CORRECTION_STATUS,
        "authority_content_digest": correction["content_digest"],
        "existing_partial_material_reuse_authorized": False,
        "restart_from_fresh_v4_roots_required": True,
        "scientific_formula_or_decision_changed": False,
    }
    expected_interpretation = authority[
        "external_historical_custody_authority"
    ].get("v3_terminal_interpretation")
    if (
        expected_interpretation != V3_TERMINAL_INTERPRETATION
        or historical.get("v3_terminal_interpretation")
        != expected_interpretation
        or internal.get("v3_terminal_interpretation")
        != expected_interpretation
        or invariance.get("v3_terminal_interpretation")
        != expected_interpretation
        or invariance.get("authorized_change_scope")
        != AUTHORIZED_SCIENTIFIC_CHANGE
        or invariance.get("pre_panel_engineering_correction")
        != expected_correction_projection
    ):
        raise RegenerationError(
            "V3 terminal interpretation or V4 correction gate drift"
        )

    external_bindings = _call(module, "external_artifact_bindings", contract)
    if (
        not isinstance(external_bindings, list)
        or len(external_bindings) == 0
    ):
        raise RegenerationError("V4 external artifact projection is absent")
    external_validations = [
        V1E._stream_external_binding(binding, f"V4 external artifact[{index}]")
        for index, binding in enumerate(external_bindings)
    ]
    if [row["role"] for row in external_validations] != authority[
        "external_artifact_roles"
    ]:
        raise RegenerationError("V4 external artifact role order drift")
    predecessor = V1E._validate_v2_context(contract, module)
    return (
        ordinary_raw,
        internal,
        invariance,
        historical,
        historical_binding,
        external_validations,
        predecessor,
    )


def _build_success_reduction(
    *,
    root: Path,
    root_fd: int,
    material_root: Path,
    module: Any,
    authority: Mapping[str, Any],
    contract_raw: bytes,
    contract: Mapping[str, Any],
    historical: Mapping[str, Any],
    historical_binding: Mapping[str, Any],
    internal_raw: bytes,
    internal: Mapping[str, Any],
    invariance_raw: bytes,
    invariance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Recompute the adequate-panel inherited science from persisted bytes."""

    raw_documents: dict[str, bytes] = {}
    documents: dict[str, dict[str, Any]] = {}
    for name, leaf in _DOCUMENT_LEAVES.items():
        raw, value = _load_json_at(root_fd, leaf)
        raw_documents[leaf] = raw
        documents[name] = value
    raw_ledgers: dict[str, bytes] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for name, leaf in _LEDGER_LEAVES.items():
        raw, value = _load_jsonl_at(root_fd, leaf)
        raw_ledgers[leaf] = raw
        ledgers[name] = value

    identity_rows: list[dict[str, Any]] = []
    for name, value in documents.items():
        schema = value.get("schema")
        if (
            value.get("experiment_id") != EXPERIMENT_ID
            or not isinstance(schema, str)
            or "physical_graph_edge_handoff_qualification_v4" not in schema
        ):
            raise RegenerationError(
                f"V4 official document uses predecessor identity: {name}"
            )
        identity_rows.append({"kind": "document", "name": name, "schema": schema})
    for name, values in ledgers.items():
        ledger_authority = authority.get("ledgers", {}).get(name)
        if not isinstance(ledger_authority, Mapping):
            raise RegenerationError(f"V4 ledger authority is absent: {name}")
        expected_fields = set(ledger_authority.get("fields", ()))
        identity_order = ledger_authority.get("identity_order")
        if (
            not expected_fields
            or not isinstance(identity_order, list)
            or not identity_order
            or not set(identity_order).issubset(expected_fields)
            or {"schema", "experiment_id"} & expected_fields
        ):
            raise RegenerationError(f"V4 schema-less ledger authority drift: {name}")
        identities: list[list[Any]] = []
        for index, value in enumerate(values):
            if (
                not isinstance(value, Mapping)
                or set(value) != expected_fields
                or "schema" in value
                or "experiment_id" in value
            ):
                raise RegenerationError(
                    f"V4 schema-less ledger field authority drift: {name}[{index}]"
                )
            identities.append([value[field] for field in identity_order])
        if len({canonical_json_bytes(item) for item in identities}) != len(identities):
            raise RegenerationError(f"V4 ledger identity duplication: {name}")
        identity_rows.append(
            {
                "kind": "ledger",
                "name": name,
                "row_count": len(values),
                "field_authority_sha256": hashlib.sha256(
                    canonical_json_bytes(sorted(expected_fields))
                ).hexdigest(),
                "identity_order": identity_order,
                "identity_projection_sha256": hashlib.sha256(
                    canonical_json_bytes(identities)
                ).hexdigest(),
                "schema_and_experiment_fields_absent_by_frozen_authority": True,
            }
        )
    official_identity_validation = {
        "all_official_documents_use_v4_identity": True,
        "schema_less_ledgers_match_exact_frozen_field_authority": True,
        "predecessor_identity_used_only_inside_frozen_compatibility_projection": True,
        "projection_sha256": hashlib.sha256(
            canonical_json_bytes(identity_rows)
        ).hexdigest(),
    }

    (
        document_validation,
        ledger_validation,
        teacher_trace_count,
    ) = _validate_success_documents_and_ledgers(
        module, authority, documents, ledgers
    )
    runtime_environment_validation = V1E._validate_runtime_environments(
        module, documents, ledgers
    )
    encoder_source_validation = V1E._validate_external_encoder_source(
        documents["latent_index"]
    )

    symbols: dict[str, int] = {}
    captured_reset_slices: dict[str, list[bytes]] = {}
    captured_physical_slices: dict[str, dict[str, list[bytes]]] = {}
    inspections: dict[str, dict[str, Any]] = {}
    for leaf in _PAYLOAD_LEAVES:
        npz_authority = (
            _call(module, "teacher_trace_npz_authority", teacher_trace_count)
            if leaf == "teacher_traces.npz"
            else authority["npz_authorities"][leaf]
        )
        inspections[leaf] = V1E._inspect_npz_at(
            root_fd,
            leaf,
            npz_authority,
            symbols=symbols,
            captured_reset_slices=captured_reset_slices,
            captured_physical_slices=captured_physical_slices,
        )
    if symbols.get("U") != documents["pixel_index"].get("unique_pixel_count"):
        raise RegenerationError("V4 latent U dimension differs from pixel index")
    pure_npz = _call(
        module,
        "validate_npz_inspections",
        [inspections[leaf] for leaf in _PAYLOAD_LEAVES],
        teacher_trace_count=teacher_trace_count,
    )
    if not isinstance(pure_npz, Mapping) or set(pure_npz) != set(_PAYLOAD_LEAVES):
        raise RegenerationError("V4 pure NPZ inspection projection drift")
    previous_raw = V3E.V2E._validate_official_previous_command_raw_hashes(
        root=root,
        state_snapshot_index=documents["state_snapshot_index"],
        module=module,
    )
    # The corrected official V4 field is raw-C-byte SHA-256.  The frozen V1
    # helper names the earlier header+NUL canonical-array hash in that one
    # slot, so bridge only a private deep copy after the raw authority passes.
    bridge_documents = json.loads(json.dumps(documents))
    legacy_rows = inspections["state_snapshots.npz"]["members"][
        "previous_applied_command"
    ]["row_or_slice_sha256s"]
    snapshot_records = bridge_documents["state_snapshot_index"]["records"]
    if len(legacy_rows) != len(snapshot_records):
        raise RegenerationError("V4 legacy previous-command bridge drift")
    for record, legacy_sha in zip(snapshot_records, legacy_rows):
        record["previous_applied_command_sha256"] = legacy_sha
    npz_cross_links = V1E._validate_npz_cross_links(
        bridge_documents, ledgers, inspections
    )
    reset_pairs = V1E._validate_reset_trace_pairs(
        documents["state_snapshot_index"],
        inspections["candidate_traces.npz"],
        captured_reset_slices,
        authority,
    )
    physical = authority["physical_trace_reduction_authority"]
    raw_candidate = V1E._validate_raw_candidate_and_repeat_evidence(
        documents, ledgers, inspections, captured_physical_slices, physical
    )
    candidate_alignment = (
        V3E.V2E._validate_candidate_port_metric_alignment_from_raw(
            documents=documents,
            ledgers=ledgers,
            captured=captured_physical_slices,
            material_root=material_root,
            module=module,
        )
    )

    dispositions_raw, parsed_dispositions = _load_jsonl_at(
        root_fd, "qualification_state_dispositions.jsonl"
    )
    disposition_rows = _call(
        module, "validate_qualification_state_dispositions_jsonl", dispositions_raw
    )
    if canonical_json_bytes(disposition_rows) != canonical_json_bytes(
        parsed_dispositions
    ):
        raise RegenerationError("V4 disposition JSONL parser projection drift")
    panel_raw, panel = _load_json_at(root_fd, "panel_adequacy.json")
    panel = _call(module, "validate_panel_adequacy", panel, disposition_rows)
    if panel.get("adequate") is not True or panel.get("status") != "ADEQUATE":
        raise RegenerationError("V4 success inventory has an inadequate panel")
    (
        material_validations,
        material_summary,
        support_documents,
    ) = _validate_qualification_material(
        material_root,
        module=module,
        runtime_contract=contract,
        official_rows=disposition_rows,
        panel_inadequate=False,
        success_documents=documents,
        panel_adequacy=panel,
        panel_adequacy_binding=_official_binding("panel_adequacy.json", panel_raw),
    )
    _teacher_selection_raw, teacher_selection = _load_canonical_object(
        material_root / "teacher_selection.json", "V4 teacher selection"
    )
    teacher_selection = _validated_exact(
        teacher_selection,
        _call(
            module,
            "validate_teacher_selection",
            teacher_selection,
            documents["panel_manifest"],
            documents["teacher_trace_index"],
            disposition_rows,
            panel,
        ),
        "teacher_selection",
    )
    raw_teacher = material_summary["raw_teacher_physics_validation"]
    qualification_runtime = material_summary[
        "qualification_runtime_environment"
    ]
    assembled_physical = documents["panel_manifest"].get(
        "physical_runtime_environment"
    )
    if (
        raw_teacher.get("teacher_trace_count") != teacher_trace_count
        or raw_teacher.get("qualified_teacher_trace_count")
        != panel.get("qualified_count")
        or raw_teacher.get("pool_indices")
        != [
            row["qualification_pool_index"]
            for row in documents["teacher_trace_index"]["records"]
        ]
        or raw_teacher.get("all_teacher_labels_regenerated_from_reopened_material")
        is not True
        or not isinstance(assembled_physical, Mapping)
        or qualification_runtime.get("physical_runtime_core_sha256")
        != assembled_physical.get("runtime_core_sha256")
        or qualification_runtime.get("qualification_stage_runtime_sha256s")
        != assembled_physical.get("qualification_runtime_sha256s")
    ):
        raise RegenerationError(
            "V4 raw material/official teacher or qualification runtime drift"
        )

    evidence = {
        **documents,
        **ledgers,
        "npz_inspections": [inspections[leaf] for leaf in _PAYLOAD_LEAVES],
        "runtime_contract": dict(contract),
        "external_historical_custody_receipt": dict(historical),
        "v1_v2_v3_custody_and_nonreuse": dict(internal),
        "scientific_invariance_receipt": dict(invariance),
        "qualification_state_dispositions_jsonl": dispositions_raw,
        "panel_adequacy": dict(panel),
        "material_shard_validations": material_validations,
        "qualification_runtime_environment": qualification_runtime,
        "teacher_selection": teacher_selection,
        **support_documents,
    }
    if set(evidence) != set(authority["success_evidence_keys"]):
        raise RegenerationError("V4 success evidence key-set drift")
    recomputed = _call(module, "recompute_metrics", evidence)
    if not isinstance(recomputed, Mapping):
        raise RegenerationError("V4 recompute_metrics returned no object")
    recomputed_raw, recomputed = _canonicalise_mapping(
        recomputed, "V4 recomputed successful metrics"
    )
    V1E._validate_content_digest(recomputed, "V4 recomputed successful metrics")
    metrics_raw, _metrics = _load_json_at(root_fd, "metrics.json")
    if metrics_raw != recomputed_raw:
        raise RegenerationError(
            "V4 successful metrics differ from exact independent rebuild"
        )

    bindings: dict[str, dict[str, Any]] = {
        "contract.json": _official_binding("contract.json", contract_raw),
        "v1_v2_v3_custody_and_nonreuse.json": _official_binding(
            "v1_v2_v3_custody_and_nonreuse.json", internal_raw
        ),
        "scientific_invariance_receipt.json": _official_binding(
            "scientific_invariance_receipt.json", invariance_raw
        ),
        "qualification_state_dispositions.jsonl": _official_binding(
            "qualification_state_dispositions.jsonl", dispositions_raw
        ),
        "panel_adequacy.json": _official_binding("panel_adequacy.json", panel_raw),
        "metrics.json": _official_binding("metrics.json", metrics_raw),
    }
    bindings.update(
        {leaf: _official_binding(leaf, raw) for leaf, raw in raw_documents.items()}
    )
    bindings.update(
        {leaf: _official_binding(leaf, raw) for leaf, raw in raw_ledgers.items()}
    )
    bindings.update(
        {
            leaf: {
                field: inspections[leaf][field]
                for field in ("path", "bytes", "sha256")
            }
            for leaf in _PAYLOAD_LEAVES
        }
    )
    if set(bindings) != set(V4_SUCCESS_SCIENTIFIC_FILES):
        raise RegenerationError("V4 successful scientific binding inventory drift")
    return recomputed, bindings, {
        "qualification_material": material_summary,
        "qualification_row_count": len(disposition_rows),
        "panel_status": panel["status"],
        "selected_state_count": len(panel["selected_pool_indices"]),
        "official_identity_validation": official_identity_validation,
        "document_validation": document_validation,
        "ledger_validation": ledger_validation,
        "npz_validation": V1E._compact_npz_validation(inspections),
        "npz_symbols": {name: symbols[name] for name in sorted(symbols)},
        "pure_npz_validation_passed": True,
        "npz_cross_link_validation": npz_cross_links,
        "official_previous_command_raw_hash_validation": previous_raw,
        "reset_pair_validation": reset_pairs,
        "raw_teacher_physics_validation": raw_teacher,
        "raw_candidate_and_repeat_physics_validation": raw_candidate,
        "candidate_port_metric_alignment_validation": candidate_alignment,
        "runtime_environment_validation": runtime_environment_validation,
        "external_encoder_source_validation": encoder_source_validation,
        "metrics_exact_byte_equal": True,
        "recomputed_metrics_sha256": hashlib.sha256(metrics_raw).hexdigest(),
    }


def build_regeneration_receipt(
    output_root: Path | str = DEFAULT_V4_OUTPUT_ROOT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
    historical_custody_receipt: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
) -> dict[str, Any]:
    """Rebuild V4 raw custody and metrics without simulator/model execution."""

    module = _load_metrics_module() if metrics_module is None else metrics_module
    authority = _validate_reducer_authority(module)
    root, root_fd = _open_official_root(output_root)
    material = (
        Path(material_root)
        if material_root is not None
        else root.parent / f"{root.name}_material"
    )
    historical_path = Path(historical_custody_receipt)
    try:
        _names, mode, complete = _validate_official_inventory(root_fd)
        contract_raw, contract_raw_value = _load_json_at(root_fd, "contract.json")
        contract = _validate_runtime_contract(contract_raw_value, module)
        source_commit = contract.get("source_freeze_commit")
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
        ):
            raise RegenerationError("V4 runtime source-freeze binding drift")
        observed_source = (
            _observe_source_freeze(contract, module)
            if source_freeze_observation is None
            else dict(source_freeze_observation)
        )
        source_freeze = _validate_source_freeze_observation(
            observed_source,
            expected_commit=source_commit,
            expected_tracked_source_count=len(
                contract["scientific_contract"]["tracked_source_paths"]
            ),
            expected_source_closure_row_count=len(
                contract["scientific_contract"]["source_closure_paths"]
            ),
            expected_metrics_module=getattr(
                module, "__name__", type(module).__name__
            ),
        )
        (
            ordinary_raw,
            internal,
            invariance,
            historical,
            historical_binding,
            external_validations,
            predecessor_validation,
        ) = _load_common_gates(
            root_fd=root_fd,
            contract=contract,
            module=module,
            authority=authority,
            historical_path=historical_path,
        )

        if mode == "PANEL_INADEQUATE":
            recomputed, scientific_bindings, scientific_validation = (
                _build_panel_inadequate_reduction(
                    root=root,
                    root_fd=root_fd,
                    material_root=material,
                    module=module,
                    authority=authority,
                    contract_raw=contract_raw,
                    contract=contract,
                    historical=historical,
                    historical_binding=historical_binding,
                    internal_raw=ordinary_raw[
                        "v1_v2_v3_custody_and_nonreuse.json"
                    ],
                    internal=internal,
                    invariance_raw=ordinary_raw[
                        "scientific_invariance_receipt.json"
                    ],
                    invariance=invariance,
                )
            )
            terminal_disposition: str | None = PANEL_INADEQUATE_DISPOSITION
            receipt_mode = "INDEPENDENT_COMPLETE_PANEL_INADEQUACY_REDUCTION_ONLY"
            scientific_leaf_count = 6
            allowed_leaf_count = 9
        else:
            recomputed, scientific_bindings, scientific_validation = (
                _build_success_reduction(
                    root=root,
                    root_fd=root_fd,
                    material_root=material,
                    module=module,
                    authority=authority,
                    contract_raw=contract_raw,
                    contract=contract,
                    historical=historical,
                    historical_binding=historical_binding,
                    internal_raw=ordinary_raw[
                        "v1_v2_v3_custody_and_nonreuse.json"
                    ],
                    internal=internal,
                    invariance_raw=ordinary_raw[
                        "scientific_invariance_receipt.json"
                    ],
                    invariance=invariance,
                )
            )
            terminal_disposition = None
            receipt_mode = "INDEPENDENT_PERSISTED_PHYSICAL_EVIDENCE_REDUCTION_ONLY"
            scientific_leaf_count = 24
            allowed_leaf_count = 27

        historical_nonreuse = _validate_v4_nonreuse(
            historical=historical,
            output_root=root,
            material_root=material,
        )
        receipt = {
            "schema": REGENERATION_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "pass": True,
            "mode": receipt_mode,
            "technical_disposition": None,
            "terminal_disposition": terminal_disposition,
            "scientific_result_produced": True,
            "development_only": True,
            "final_evaluation_eligible": False,
            "reducer_authority_digest": authority["content_digest"],
            "source_freeze": source_freeze,
            "historical_custody_receipt_binding": historical_binding,
            "historical_custody_projection_sha256": _call(
                module, "historical_custody_projection_sha256", historical
            ),
            "scientific_invariance_receipt_validated": invariance["pass"] is True,
            "v1_v2_v3_custody_and_nonreuse_validated": internal["pass"] is True,
            "official_root_inventory": {
                "scientific_leaf_count": scientific_leaf_count,
                "allowed_complete_leaf_count": allowed_leaf_count,
                "allowed_leaf_names_validated": True,
                "publication_all_or_absent_validated": True,
            },
            "inputs": {
                name: scientific_bindings[name]
                for name in sorted(scientific_bindings)
            },
            "scientific_validation": scientific_validation,
            "external_artifact_validation": external_validations,
            "predecessor_context_validation": predecessor_validation,
            "historical_nonreuse_validation": historical_nonreuse,
            "metrics_exact_byte_equal": True,
            "recomputed_metrics_sha256": scientific_validation[
                "recomputed_metrics_sha256"
            ],
            "scientific_execution_counters": _receipt_counters(),
        }
        _validate_no_self_digest(receipt, "V4 external regeneration receipt")
        V1E._reject_nonfinite(receipt, label="V4 external regeneration receipt")
        receipt_sha = hashlib.sha256(canonical_document_bytes(receipt)).hexdigest()
        _validate_publication_if_present(
            root=root,
            root_fd=root_fd,
            complete=complete,
            mode=mode,
            scientific_bindings=scientific_bindings,
            recomputed_metrics=recomputed,
            runtime_contract=contract,
            expected_receipt_sha256=receipt_sha,
            historical_binding=historical_binding,
            module=module,
        )
        return receipt
    finally:
        os.close(root_fd)


def verify_and_emit(
    output_root: Path | str = DEFAULT_V4_OUTPUT_ROOT,
    output: Path | str = DEFAULT_EXTERNAL_RECEIPT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
    historical_custody_receipt: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
) -> dict[str, Any]:
    receipt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
        material_root=material_root,
        historical_custody_receipt=historical_custody_receipt,
    )
    root = Path(output_root)
    material = (
        Path(material_root)
        if material_root is not None
        else root.parent / f"{root.name}_material"
    )
    V3E.V2E._emit_external((root, material), output, receipt)
    return receipt


def validate_existing_regeneration_receipt(
    output_root: Path | str = DEFAULT_V4_OUTPUT_ROOT,
    output: Path | str = DEFAULT_EXTERNAL_RECEIPT,
    *,
    metrics_module: Any | None = None,
    source_freeze_observation: Mapping[str, Any] | None = None,
    material_root: Path | str | None = None,
    historical_custody_receipt: Path | str = DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
) -> dict[str, Any]:
    raw, supplied = _load_canonical_object(
        Path(output), "V4 external regeneration receipt"
    )
    _validate_no_self_digest(supplied, "V4 external regeneration receipt")
    rebuilt = build_regeneration_receipt(
        output_root,
        metrics_module=metrics_module,
        source_freeze_observation=source_freeze_observation,
        material_root=material_root,
        historical_custody_receipt=historical_custody_receipt,
    )
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError(
            "V4 external regeneration receipt differs from exact rebuild"
        )
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently validate PGEHQ V4 evidence and publication"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_V4_OUTPUT_ROOT)
    parser.add_argument(
        "--material-root", type=Path, default=DEFAULT_V4_MATERIAL_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXTERNAL_RECEIPT)
    parser.add_argument(
        "--historical-custody-receipt",
        type=Path,
        default=DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.output.exists() or arguments.output.is_symlink():
        receipt = validate_existing_regeneration_receipt(
            arguments.root,
            arguments.output,
            material_root=arguments.material_root,
            historical_custody_receipt=arguments.historical_custody_receipt,
        )
    else:
        receipt = verify_and_emit(
            arguments.root,
            arguments.output,
            material_root=arguments.material_root,
            historical_custody_receipt=arguments.historical_custody_receipt,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RegenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
