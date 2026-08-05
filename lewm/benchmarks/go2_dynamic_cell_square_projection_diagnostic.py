"""Pure contracts and all-cell scoring for dynamic cell-square geometry.

The module performs no file I/O and has no NumPy or torch dependency.  The
runner imports it only after validating the reviewed source allowlist.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from lewm.benchmarks.go2_dynamic_cell_square_projection import (
    CAMERA_NEAR_M,
    CAMERA_RPY_BODY_RAD,
    CAMERA_XYZ_BODY_M,
    CELL_SIZE_M,
    FORWARD_MIN_EDGE_M,
    GRID_SHAPE,
    HORIZONTAL_FOV_DEG,
    LEFT_MIN_EDGE_M,
    VERTICAL_ANCHOR_Z_M,
    VERTICAL_FOV_DEG,
    build_dynamic_cell_square_support_mask,
    cell_center,
    compose_yaw_aligned_camera,
    support_mask_sha256,
)


CANDIDATE_SCHEMA = "lewm_go2_dynamic_cell_square_projection_candidate_v1"
FINAL_SCHEMA = "lewm_go2_dynamic_cell_square_projection_diagnostic_v1"
ACCESS_LEDGER_SCHEMA = "lewm_go2_dynamic_projection_access_ledger_v1"
AUDIT_SCHEMA = "lewm_go2_n32_camera_frustum_observability_audit_result_v1"

BINDING_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_binding_2026-07-11.md"
)
BINDING_SHA256 = "211043ee3c3200d1fc93febbae73059341aea19560c83f53f3b3bb231bf06e66"
PREDECESSOR_REPORT_RELATIVE_PATH = (
    "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_result_2026-07-11.md"
)
PREDECESSOR_REPORT_SHA256 = (
    "8bfb4c9a8b69f67b3b9e4d6e3b21e9ff89ecaff89a2bab3eb83d759ca4fe6d22"
)
PREDECESSOR_RESULT_RELATIVE_PATH = (
    ".generated/go2_n32_camera_frustum_observability_audit/v2/result.json"
)
PREDECESSOR_RESULT_FILE_SHA256 = (
    "7725ecddf2fa77bb762733fd35df2efd2fb60d4f9aa8ab6fdf2bee660522909e"
)
PREDECESSOR_RESULT_CONTENT_SHA256 = (
    "11420607d2c4f8e79af9214d43bbc6259669ee84c9ccc0aaefd4167cc1d809a1"
)
DYNAMIC_GEOMETRY_RELATIVE_PATH = (
    "lewm/benchmarks/go2_dynamic_cell_square_projection.py"
)
DYNAMIC_GEOMETRY_SHA256 = (
    "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
)
HUMAN_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.md"
)
MACHINE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_n32_dynamic_cell_square_geometry_implementation_manifest_2026-07-11.json"
)
CANDIDATE_RELATIVE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/candidate.json"
)
FINAL_RESULT_RELATIVE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/result.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FAILURE_RESULT_RELATIVE_PATH = (
    ".generated/go2_dynamic_cell_square_projection_diagnostic/v1/"
    "failure_diagnostic.json"
)

EXPECTED_LABEL_SHARD_COUNT = 20
EXPECTED_LABEL_SHARD_MANIFEST_SHA256 = (
    "998ce5a768029c23c931fbbec730c1fe31b9ed1fe155494fc68f34a0c23d3d1b"
)
EXPECTED_FRAME_COUNT = 320
EXPECTED_TRANSITION_COUNT = 160
EXPECTED_SELECTED_TARGET_BYTE_COUNT = 1_310_720
EXPECTED_SELECTED_TARGET_SHA256 = (
    "6952c1f9604da1d9fd4c94a3f33deb142451836609b7059970ff6c459737ce05"
)
EXPECTED_FRAME_IDENTITY_SHA256 = (
    "1233e980e7136ad0900c2b4d2c978a3134b24af8f088a149ec8d36254adf7548"
)
EXPECTED_CLASS_TOTALS = {
    "unknown": 1_181_699,
    "free": 118_793,
    "occupied": 10_228,
    "all": 1_310_720,
}
EXPECTED_KNOWN_TOTAL = 129_021
EXPECTED_LEVEL_CENTER_SUPPORT_COUNT = 1_990
EXPECTED_LEVEL_CENTER_SUPPORT_SHA256 = (
    "026d7654864bea7ae0545bd6448f6def64519a3bedcbc7ea747e7b4b95f82b3a"
)
EXPECTED_LEVEL_CENTER_FREE_SUPPORTED = 118_792
EXPECTED_LEVEL_CENTER_OCCUPIED_SUPPORTED = 9_856
EXPECTED_CENTER_VIOLATION_COUNT = 373
EXPECTED_CENTER_VIOLATION_IDENTITIES_SHA256 = (
    "f85a9ece8f4a34fe0f175de900934780a750d076f70a7e672be8337cffb64bcc"
)
EXPECTED_LEVEL_CELL_SQUARE_SUPPORT_COUNT = 2_062
EXPECTED_LEVEL_CELL_SQUARE_SUPPORT_SHA256 = (
    "4ebbafb6d4dd5fb13b96df978abfa7b81bc2f879b2ba6dec2fcda38dec54e60b"
)
EXPECTED_STATIC_SUPPORTED_COUNT = 129_017
EXPECTED_STATIC_UNSUPPORTED_COUNT = 4
EXPECTED_STATIC_UNSUPPORTED_IDENTITIES_SHA256 = (
    "c574f35890ef68114fb36ebf701eec7552262d03c49cf4d1c07b47740fc505f0"
)
EMPTY_LIST_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)

FAMILY_ORDER = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
SIDE_ORDER = ("current", "next")
CLASS_ORDER = ("unknown", "free", "occupied")
KNOWN_CLASS_ORDER = ("free", "occupied")
CLASS_IDS = {"unknown": 0, "free": 1, "occupied": 2}
FORBIDDEN_ROLES = (
    "g2",
    "heldout",
    "image",
    "model_output",
    "physical_nontrain",
    "runtime_result",
    "sealed",
    "selection_calibration",
    "source_geometry",
)
SOURCE_MAP_CONTRACT = (
    ("dynamic_geometry", "lewm/benchmarks/go2_dynamic_cell_square_projection.py"),
    (
        "diagnostic_core",
        "lewm/benchmarks/go2_dynamic_cell_square_projection_diagnostic.py",
    ),
    ("preparation", "scripts/prepare_go2_dynamic_cell_square_projection.py"),
    ("runner", "scripts/diagnose_go2_dynamic_cell_square_projection.py"),
    ("finalizer", "scripts/finalize_go2_dynamic_cell_square_projection.py"),
    ("geometry_test", "lewm/tests/test_go2_dynamic_cell_square_projection.py"),
    (
        "diagnostic_test",
        "lewm/tests/test_go2_dynamic_cell_square_projection_diagnostic.py",
    ),
    (
        "preparation_test",
        "lewm/tests/test_prepare_go2_dynamic_cell_square_projection.py",
    ),
    (
        "finalizer_test",
        "lewm/tests/test_finalize_go2_dynamic_cell_square_projection.py",
    ),
)
DENIED_REASON_ORDER = (
    "path_alias_or_escape",
    "symlink_component",
    "outside_repository",
    "unallowlisted",
    "forbidden_role",
    "modality_mismatch",
    "hash_mismatch",
)
FRAME_KEY_FIELDS = (
    "family",
    "global_row",
    "image_sha256",
    "label_row",
    "label_shard_sha256",
    "scene_id",
    "side",
)
FRAME_IDENTITY_ENCODING_FIELDS = (
    "family",
    "scene_id",
    "global_row",
    "side",
    "image_sha256",
    "label_shard_sha256",
    "label_row",
)
LEDGER_KEYS = {
    "schema",
    "phase",
    "authorized_read_paths",
    "authorized_read_path_set_sha256",
    "authorized_write_paths",
    "authorized_write_path_set_sha256",
    "role_byte_open_counts",
    "label_shard_pre_hash_byte_opens",
    "label_shard_post_hash_byte_opens",
    "label_shard_npz_parses",
    "array_decompression_counts",
    "selected_label_rows_read",
    "unselected_rows_scored",
    "unselected_rows_retained",
    "metadata_only_shard_stats",
    "denied_attempt_records",
    "denied_reason_counts",
    "unexpected_path_attempts",
    "forbidden_role_open_counts",
    "all_counts_reconcile",
}
CANDIDATE_KEYS = {
    "schema",
    "created_at_utc",
    "execution_binding",
    "implementation_manifests",
    "inputs",
    "source_map",
    "scope",
    "preparation_access_ledger",
    "runner_access_ledger",
    "label_reconciliation",
    "support",
    "family_class_rows",
    "frame_summary_records_sha256",
    "scientific_core_sha256",
    "gates",
    "content_sha256",
}
GATE_KEYS = (
    "binding_and_source_hashes_pass",
    "predecessor_authority_pass",
    "label_manifest_and_bytes_pass",
    "label_count_reconciliation_pass",
    "level_center_parity_pass",
    "level_cell_square_frozen_pass",
    "static_all_known_scored_pass",
    "dynamic_all_known_scored_pass",
    "dynamic_zero_known_unsupported_pass",
    "access_reconciliation_pass",
    "independent_recomputation_pass",
    "all_passed",
)


class DiagnosticContractError(ValueError):
    """Raised when input evidence or a result violates the frozen contract."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiagnosticContractError("value is not canonical JSON") from exc


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(core: dict[str, Any]) -> dict[str, Any]:
    if type(core) is not dict or "content_sha256" in core:
        raise DiagnosticContractError("content-hashed core must be an exact object")
    return {**core, "content_sha256": canonical_json_sha256(core)}


def validate_content_sha256(payload: object) -> str:
    record = exact_dict(payload, name="content-hashed record")
    declared = exact_sha256(record.get("content_sha256"), name="content_sha256")
    core = dict(record)
    del core["content_sha256"]
    if canonical_json_sha256(core) != declared:
        raise DiagnosticContractError("embedded content_sha256 does not match")
    return declared


def exact_dict(value: object, *, name: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise DiagnosticContractError(f"{name} must be an exact string-keyed object")
    return value


def exact_list(value: object, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise DiagnosticContractError(f"{name} must be an exact array")
    return value


def exact_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise DiagnosticContractError(f"{name} must be an exact string")
    return value


def exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DiagnosticContractError(f"{name} must be an exact integer >= {minimum}")
    return value


def finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticContractError(f"{name} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise DiagnosticContractError(f"{name} must be finite")
    return result


def exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise DiagnosticContractError(f"{name} must be an exact bool")
    return value


def exact_sha256(value: object, *, name: str) -> str:
    text = exact_string(value, name=name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DiagnosticContractError(f"{name} must be a lowercase SHA-256")
    return text


def require_exact_keys(record: dict[str, Any], keys: set[str], *, name: str) -> None:
    if set(record) != keys:
        raise DiagnosticContractError(f"{name} has an unexpected key set")


def require_equal(actual: object, expected: object, *, name: str) -> None:
    if not type_exact_equal(actual, expected):
        raise DiagnosticContractError(f"{name} differs from the frozen contract")


def type_exact_equal(actual: object, expected: object) -> bool:
    """Recursively compare values without Python's bool/int/float coercions."""

    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        if set(actual) != set(expected):
            return False
        return all(
            type_exact_equal(actual[key], expected[key]) for key in actual
        )
    if type(actual) in (list, tuple):
        return len(actual) == len(expected) and all(
            type_exact_equal(left, right)
            for left, right in zip(actual, expected)
        )
    if type(actual) in (set, frozenset):
        unmatched = list(expected)
        for left in actual:
            for index, right in enumerate(unmatched):
                if type_exact_equal(left, right):
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    return actual == expected


def validate_utc_timestamp(value: object, *, name: str) -> str:
    text = exact_string(value, name=name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise DiagnosticContractError(f"{name} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise DiagnosticContractError(f"{name} must carry an explicit UTC offset")
    if parsed.isoformat() != text or not text.endswith("+00:00"):
        raise DiagnosticContractError(f"{name} is not canonical UTC ISO-8601")
    return text


def canonical_repository_path(value: object, *, name: str) -> str:
    text = exact_string(value, name=name)
    if (
        not text
        or "\\" in text
        or os.path.normpath(text) != text
        or Path(text).as_posix() != text
    ):
        raise DiagnosticContractError(f"{name} is not raw canonical path text")
    path = Path(text)
    if not path.is_absolute():
        raise DiagnosticContractError(f"{name} must be absolute after anchoring")
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise DiagnosticContractError(f"{name} is outside the repository") from exc
    return text


def _frame_identity(frame_key: object, *, name: str) -> tuple[object, ...]:
    key = exact_dict(frame_key, name=name)
    require_exact_keys(key, set(FRAME_KEY_FIELDS), name=name)
    family = exact_string(key["family"], name=f"{name}.family")
    scene_id = exact_string(key["scene_id"], name=f"{name}.scene_id")
    global_row = exact_int(key["global_row"], name=f"{name}.global_row")
    side = exact_string(key["side"], name=f"{name}.side")
    image_hash = exact_sha256(key["image_sha256"], name=f"{name}.image_sha256")
    shard_hash = exact_sha256(
        key["label_shard_sha256"], name=f"{name}.label_shard_sha256"
    )
    label_row = exact_int(key["label_row"], name=f"{name}.label_row")
    if family not in FAMILY_ORDER or side not in SIDE_ORDER:
        raise DiagnosticContractError(f"{name} has an unregistered family or side")
    return family, scene_id, global_row, side, image_hash, shard_hash, label_row


def validate_label_shard_manifest(value: object) -> list[dict[str, Any]]:
    manifest = exact_dict(value, name="label_shard_manifest")
    require_exact_keys(
        manifest, {"entries", "entry_count", "manifest_sha256"}, name="label manifest"
    )
    entries = exact_list(manifest["entries"], name="label manifest entries")
    require_equal(
        exact_int(manifest["entry_count"], name="label manifest entry_count"),
        EXPECTED_LABEL_SHARD_COUNT,
        name="label manifest entry_count",
    )
    if len(entries) != EXPECTED_LABEL_SHARD_COUNT:
        raise DiagnosticContractError("label manifest does not contain 20 entries")
    declared_hash = exact_sha256(
        manifest["manifest_sha256"], name="label manifest SHA-256"
    )
    if declared_hash != EXPECTED_LABEL_SHARD_MANIFEST_SHA256:
        raise DiagnosticContractError("label manifest hash is not frozen")
    if canonical_json_sha256(entries) != declared_hash:
        raise DiagnosticContractError("label manifest entries hash does not match")

    paths: set[str] = set()
    storage_rows: set[tuple[str, int, str]] = set()
    selected_count = 0
    for entry_index, entry_value in enumerate(entries):
        entry = exact_dict(entry_value, name=f"label manifest entry {entry_index}")
        require_exact_keys(
            entry,
            {"path", "sha256", "selected_tuples", "selected_row_count", "family_side_counts"},
            name=f"label manifest entry {entry_index}",
        )
        path = exact_string(entry["path"], name="label shard path")
        if path in paths:
            raise DiagnosticContractError("label shard path occurs more than once")
        paths.add(path)
        exact_sha256(entry["sha256"], name="label shard SHA-256")
        tuples = exact_list(entry["selected_tuples"], name="selected tuples")
        if exact_int(entry["selected_row_count"], name="selected row count") != len(tuples):
            raise DiagnosticContractError("selected row count does not reconcile")
        expected_counts = {
            family: {side: 0 for side in SIDE_ORDER} for family in FAMILY_ORDER
        }
        for tuple_index, selected_value in enumerate(tuples):
            selected = exact_list(selected_value, name=f"selected tuple {tuple_index}")
            if len(selected) != 5:
                raise DiagnosticContractError("selected tuple must have five fields")
            family = exact_string(selected[0], name="selected family")
            exact_string(selected[1], name="selected scene")
            exact_int(selected[2], name="selected global row")
            side = exact_string(selected[3], name="selected side")
            row = exact_int(selected[4], name="selected label row")
            if family not in FAMILY_ORDER or side not in SIDE_ORDER:
                raise DiagnosticContractError("selected tuple has unregistered family/side")
            storage_identity = path, row, side
            if storage_identity in storage_rows:
                raise DiagnosticContractError("selected shard/row/side is duplicated")
            storage_rows.add(storage_identity)
            expected_counts[family][side] += 1
        require_equal(
            entry["family_side_counts"], expected_counts, name="family-side counts"
        )
        selected_count += len(tuples)
    if selected_count != EXPECTED_FRAME_COUNT or len(storage_rows) != EXPECTED_FRAME_COUNT:
        raise DiagnosticContractError("selected tuple denominator is not 320")
    return entries


def validate_predecessor_result(value: object) -> dict[str, Any]:
    result = exact_dict(value, name="predecessor result")
    require_equal(result.get("schema"), AUDIT_SCHEMA, name="predecessor schema")
    if validate_content_sha256(result) != PREDECESSOR_RESULT_CONTENT_SHA256:
        raise DiagnosticContractError("predecessor content hash is not frozen")
    scope = exact_dict(result.get("scope"), name="predecessor scope")
    for key, expected in {
        "dataset_role": "train",
        "learning_performed": False,
        "frame_count": EXPECTED_FRAME_COUNT,
        "transition_count": EXPECTED_TRANSITION_COUNT,
        "families": list(FAMILY_ORDER),
        "endpoint_sides": list(SIDE_ORDER),
    }.items():
        require_equal(scope.get(key), expected, name=f"scope.{key}")
    for object_name, fields in {
        "provenance": ("passes", "source_hashes_pass"),
        "reconstruction": ("passes",),
        "camera_mount_composition": ("passes",),
        "two_phase_access_reconciliation": ("passes", "forbidden_counters_zero", "unexpected_paths_zero"),
    }.items():
        record = exact_dict(result.get(object_name), name=object_name)
        for field in fields:
            require_equal(record.get(field), True, name=f"{object_name}.{field}")
    validate_label_shard_manifest(result.get("label_shard_manifest"))
    selected = exact_dict(result.get("selected_label_bytes"), name="selected_label_bytes")
    require_exact_keys(
        selected, {"frame_count", "encoding", "byte_count", "sha256"}, name="selected bytes"
    )
    require_equal(selected["frame_count"], EXPECTED_FRAME_COUNT, name="selected frames")
    require_equal(
        selected["byte_count"], EXPECTED_SELECTED_TARGET_BYTE_COUNT, name="selected bytes"
    )
    require_equal(
        selected["sha256"], EXPECTED_SELECTED_TARGET_SHA256, name="selected target hash"
    )

    reports = exact_list(result.get("frame_reports"), name="frame reports")
    if len(reports) != EXPECTED_FRAME_COUNT:
        raise DiagnosticContractError("predecessor frame report count is not 320")
    frame_identity_values: list[list[Any]] = []
    seen: set[tuple[object, ...]] = set()
    for index, frame_value in enumerate(reports):
        frame = exact_dict(frame_value, name=f"frame report {index}")
        key = exact_dict(frame.get("record_key"), name=f"frame report {index} key")
        identity = _frame_identity(key, name=f"frame report {index} key")
        if identity in seen:
            raise DiagnosticContractError("frame identity is duplicated")
        seen.add(identity)
        frame_identity_values.append([key[field] for field in FRAME_IDENTITY_ENCODING_FIELDS])
        composition = exact_dict(
            frame.get("camera_mount_composition"), name=f"frame report {index} camera"
        )
        require_equal(composition.get("passes"), True, name="frame camera parity")
        quaternion = exact_list(
            composition.get("base_quat_world_xyzw"), name="frame quaternion"
        )
        if len(quaternion) != 4:
            raise DiagnosticContractError("frame quaternion must have four values")
        for component in quaternion:
            finite_number(component, name="frame quaternion component")
        finite_number(composition.get("stored_base_yaw_rad"), name="stored yaw")
    identity_record = exact_dict(result.get("frame_identity"), name="frame identity")
    require_equal(identity_record.get("count"), 320, name="frame identity count")
    require_equal(
        identity_record.get("encoding_fields"),
        list(FRAME_IDENTITY_ENCODING_FIELDS),
        name="frame identity encoding",
    )
    require_equal(
        identity_record.get("sha256"), EXPECTED_FRAME_IDENTITY_SHA256, name="frame identity hash"
    )
    if canonical_json_sha256(frame_identity_values) != EXPECTED_FRAME_IDENTITY_SHA256:
        raise DiagnosticContractError("ordered frame identities do not match")
    family_table = exact_dict(
        result.get("family_class_count_table"), name="family class count table"
    )
    require_exact_keys(
        family_table, {"family_order", "rows", "table_sha256"}, name="family class table"
    )
    require_equal(
        family_table["family_order"], list(FAMILY_ORDER), name="family class order"
    )
    family_rows = exact_list(family_table["rows"], name="family class rows")
    if canonical_json_sha256(family_rows) != family_table["table_sha256"]:
        raise DiagnosticContractError("family class count table hash mismatch")
    return result


def validate_source_map(value: object) -> dict[str, Any]:
    source_map = exact_dict(value, name="source map")
    require_exact_keys(
        source_map, {"entries", "entry_count", "source_map_sha256"}, name="source map"
    )
    entries = exact_list(source_map["entries"], name="source map entries")
    if exact_int(source_map["entry_count"], name="source map count") != len(
        SOURCE_MAP_CONTRACT
    ) or len(entries) != len(SOURCE_MAP_CONTRACT):
        raise DiagnosticContractError("source map must contain exactly nine entries")
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for index, (entry_value, expected) in enumerate(zip(entries, SOURCE_MAP_CONTRACT)):
        entry = exact_dict(entry_value, name=f"source map entry {index}")
        require_exact_keys(entry, {"path", "role", "sha256"}, name="source map entry")
        role, path = expected
        require_equal(entry["role"], role, name="source map role")
        require_equal(entry["path"], path, name="source map path")
        exact_sha256(entry["sha256"], name="source map hash")
        if entry["path"] in seen_paths or entry["role"] in seen_roles:
            raise DiagnosticContractError("source map role/path is duplicated")
        seen_paths.add(entry["path"])
        seen_roles.add(entry["role"])
    if entries[0]["sha256"] != DYNAMIC_GEOMETRY_SHA256:
        raise DiagnosticContractError("dynamic geometry source hash mismatch")
    declared = exact_sha256(source_map["source_map_sha256"], name="source map hash")
    if canonical_json_sha256(entries) != declared:
        raise DiagnosticContractError("source map entries hash mismatch")
    return source_map


def validate_implementation_manifests(value: object) -> dict[str, Any]:
    manifests = exact_dict(value, name="implementation manifests")
    require_exact_keys(manifests, {"human", "machine"}, name="implementation manifests")
    human = exact_dict(manifests["human"], name="human manifest")
    machine = exact_dict(manifests["machine"], name="machine manifest")
    require_exact_keys(human, {"path", "file_sha256"}, name="human manifest")
    require_exact_keys(
        machine, {"path", "file_sha256", "content_sha256"}, name="machine manifest"
    )
    require_equal(human["path"], HUMAN_MANIFEST_RELATIVE_PATH, name="human manifest path")
    require_equal(
        machine["path"], MACHINE_MANIFEST_RELATIVE_PATH, name="machine manifest path"
    )
    exact_sha256(human["file_sha256"], name="human manifest hash")
    exact_sha256(machine["file_sha256"], name="machine manifest hash")
    exact_sha256(machine["content_sha256"], name="machine manifest content hash")
    return manifests


def build_independent_level_center_support_mask() -> tuple[tuple[bool, ...], ...]:
    tan_h = math.tan(math.radians(HORIZONTAL_FOV_DEG) * 0.5)
    tan_v = math.tan(math.radians(VERTICAL_FOV_DEG) * 0.5)
    camera_forward, camera_left, camera_up = CAMERA_XYZ_BODY_M
    rows: list[tuple[bool, ...]] = []
    for row in range(GRID_SHAPE[0]):
        values: list[bool] = []
        for column in range(GRID_SHAPE[1]):
            point_forward, point_left = cell_center(row, column)
            forward = point_forward - camera_forward
            left = point_left - camera_left
            values.append(
                any(
                    forward >= CAMERA_NEAR_M
                    and -forward * tan_h <= left <= forward * tan_h
                    and -forward * tan_v <= point_up - camera_up <= forward * tan_v
                    for point_up in VERTICAL_ANCHOR_Z_M
                )
            )
        rows.append(tuple(values))
    return tuple(rows)


def validate_ordered_target_bytes(value: object) -> bytes:
    if type(value) is not bytes:
        raise DiagnosticContractError("ordered targets must be exact bytes")
    if len(value) != EXPECTED_SELECTED_TARGET_BYTE_COUNT:
        raise DiagnosticContractError("selected target byte count is not frozen")
    if hashlib.sha256(value).hexdigest() != EXPECTED_SELECTED_TARGET_SHA256:
        raise DiagnosticContractError("selected target byte hash is not frozen")
    if not set(value).issubset({0, 1, 2}):
        raise DiagnosticContractError("selected targets contain an invalid class")
    return value


def _known_identity(
    key: dict[str, Any], class_name: str, row: int, column: int
) -> dict[str, Any]:
    return {
        "class_id": CLASS_IDS[class_name],
        "class_name": class_name,
        "column": column,
        "frame_key": key,
        "row": row,
    }


def _family_class_rows(
    counts: Counter[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    return [
        {
            "family": family,
            "class_id": CLASS_IDS[class_name],
            "class_name": class_name,
            "total": counts[(family, class_name, "total")],
            "level_center_supported": counts[(family, class_name, "center")],
            "static_cell_square_supported": counts[(family, class_name, "static")],
            "dynamic_cell_square_supported": counts[(family, class_name, "dynamic")],
        }
        for family in FAMILY_ORDER
        for class_name in KNOWN_CLASS_ORDER
    ]


def compute_scientific_evidence(
    predecessor: object,
    ordered_target_bytes: object,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Scan every selected target cell and return persisted plus ephemeral evidence."""

    result = validate_predecessor_result(predecessor)
    targets = validate_ordered_target_bytes(ordered_target_bytes)
    center_mask = build_independent_level_center_support_mask()
    center_count = sum(sum(row) for row in center_mask)
    center_hash = support_mask_sha256(center_mask)
    static_mask = build_dynamic_cell_square_support_mask((0.0, 0.0, 0.0, 1.0), 0.0)
    static_mask_count = sum(sum(row) for row in static_mask)
    static_mask_hash = support_mask_sha256(static_mask)

    reports = result["frame_reports"]
    counts: Counter[tuple[str, str, str]] = Counter()
    class_totals = Counter({name: 0 for name in CLASS_ORDER})
    family_class_totals: Counter[tuple[str, str]] = Counter()
    family_frame_counts: Counter[str] = Counter()
    remaining: dict[str, list[dict[str, Any]]] = {
        "center": [],
        "static": [],
        "dynamic": [],
    }
    per_frame_totals: list[dict[str, int]] = []
    frame_summaries: list[dict[str, int]] = []
    frame_bytes = GRID_SHAPE[0] * GRID_SHAPE[1]
    for frame_rank, frame in enumerate(reports):
        key = frame["record_key"]
        family = key["family"]
        family_rank = FAMILY_ORDER.index(family)
        family_frame_counts[family] += 1
        composition = frame["camera_mount_composition"]
        dynamic_mask = build_dynamic_cell_square_support_mask(
            composition["base_quat_world_xyzw"], composition["stored_base_yaw_rad"]
        )
        frame_target = targets[frame_rank * frame_bytes : (frame_rank + 1) * frame_bytes]
        frame_counts = Counter({name: 0 for name in CLASS_ORDER})
        frame_supported: Counter[tuple[str, str]] = Counter()
        for flat_index, class_id in enumerate(frame_target):
            class_name = CLASS_ORDER[class_id]
            frame_counts[class_name] += 1
            class_totals[class_name] += 1
            family_class_totals[(family, class_name)] += 1
            if class_id == 0:
                continue
            row, column = divmod(flat_index, GRID_SHAPE[1])
            support_values = {
                "center": center_mask[row][column],
                "static": static_mask[row][column],
                "dynamic": dynamic_mask[row][column],
            }
            counts[(family, class_name, "total")] += 1
            for support_name, supported in support_values.items():
                counts[(family, class_name, support_name)] += int(supported)
                frame_supported[(support_name, class_name)] += int(supported)
                if not supported:
                    remaining[support_name].append(
                        _known_identity(key, class_name, row, column)
                    )
        per_frame_totals.append(
            {
                "frame_rank": frame_rank,
                "unknown": frame_counts["unknown"],
                "free": frame_counts["free"],
                "occupied": frame_counts["occupied"],
                "all": sum(frame_counts.values()),
            }
        )
        frame_summaries.append(
            {
                "family_rank": family_rank,
                "frame_rank": frame_rank,
                "unknown_total": frame_counts["unknown"],
                "free_total": frame_counts["free"],
                "occupied_total": frame_counts["occupied"],
                "level_center_free_supported": frame_supported[("center", "free")],
                "level_center_occupied_supported": frame_supported[("center", "occupied")],
                "static_free_supported": frame_supported[("static", "free")],
                "static_occupied_supported": frame_supported[("static", "occupied")],
                "dynamic_free_supported": frame_supported[("dynamic", "free")],
                "dynamic_occupied_supported": frame_supported[("dynamic", "occupied")],
            }
        )

    class_totals_record = {
        "unknown": class_totals["unknown"],
        "free": class_totals["free"],
        "occupied": class_totals["occupied"],
        "all": sum(class_totals.values()),
    }
    known_total = class_totals["free"] + class_totals["occupied"]
    family_count_rows = [
        {
            "frame_count": len(reports),
            "free": class_totals["free"],
            "occupied": class_totals["occupied"],
            "scope": "aggregate",
            "unknown": class_totals["unknown"],
        },
        *[
            {
                "frame_count": family_frame_counts[family],
                "free": family_class_totals[(family, "free")],
                "occupied": family_class_totals[(family, "occupied")],
                "scope": family,
                "unknown": family_class_totals[(family, "unknown")],
            }
            for family in FAMILY_ORDER
        ],
    ]
    predecessor_family_rows = result["family_class_count_table"]["rows"]
    family_rows = _family_class_rows(counts)
    center_free = sum(
        counts[(family, "free", "center")] for family in FAMILY_ORDER
    )
    center_occupied = sum(
        counts[(family, "occupied", "center")] for family in FAMILY_ORDER
    )

    def known_support_record(name: str) -> dict[str, Any]:
        unsupported = remaining[name]
        unsupported_free = sum(item["class_name"] == "free" for item in unsupported)
        unsupported_occupied = len(unsupported) - unsupported_free
        return {
            "known_total": known_total,
            "supported_count": known_total - len(unsupported),
            "unsupported_count": len(unsupported),
            "unsupported_free_count": unsupported_free,
            "unsupported_occupied_count": unsupported_occupied,
            "unsupported_frame_count": len(
                {tuple(_frame_identity(item["frame_key"], name="remaining key")) for item in unsupported}
            ),
            "unsupported_identities_sha256": canonical_json_sha256(unsupported),
        }

    label_reconciliation = {
        "byte_count": len(targets),
        "byte_sha256": hashlib.sha256(targets).hexdigest(),
        "class_totals": class_totals_record,
        "known_total": known_total,
        "per_frame_cell_count": frame_bytes,
        "per_frame_count": len(per_frame_totals),
        "per_frame_totals_sha256": canonical_json_sha256(per_frame_totals),
        "all_counts_reconcile": (
            class_totals_record == EXPECTED_CLASS_TOTALS
            and known_total == EXPECTED_KNOWN_TOTAL
            and all(row["all"] == frame_bytes for row in per_frame_totals)
            and family_count_rows == predecessor_family_rows
        ),
    }
    support = {
        "level_center": {
            "support_cell_count": center_count,
            "support_mask_sha256": center_hash,
            "free_total": class_totals["free"],
            "free_supported": center_free,
            "occupied_total": class_totals["occupied"],
            "occupied_supported": center_occupied,
            "known_violation_count": len(remaining["center"]),
            "known_violation_identities_sha256": canonical_json_sha256(
                remaining["center"]
            ),
        },
        "level_cell_square": {
            "support_cell_count": static_mask_count,
            "support_mask_sha256": static_mask_hash,
        },
        "static_cell_square_known": known_support_record("static"),
        "dynamic_cell_square_known": known_support_record("dynamic"),
    }
    persisted = {
        "label_reconciliation": label_reconciliation,
        "support": support,
        "family_class_rows": family_rows,
        "frame_summary_records_sha256": canonical_json_sha256(frame_summaries),
    }
    return persisted, remaining


def scientific_gates(
    scientific: object,
    *,
    access_reconciliation_pass: object,
    independent_recomputation_pass: bool,
) -> dict[str, bool]:
    evidence = validate_scientific_evidence(scientific)
    label = exact_dict(evidence.get("label_reconciliation"), name="label reconciliation")
    support = exact_dict(evidence.get("support"), name="support")
    center = exact_dict(support.get("level_center"), name="level center")
    square = exact_dict(support.get("level_cell_square"), name="level square")
    static = exact_dict(support.get("static_cell_square_known"), name="static known")
    dynamic = exact_dict(support.get("dynamic_cell_square_known"), name="dynamic known")
    gates = {
        "binding_and_source_hashes_pass": True,
        "predecessor_authority_pass": True,
        "label_manifest_and_bytes_pass": (
            label.get("byte_count") == EXPECTED_SELECTED_TARGET_BYTE_COUNT
            and label.get("byte_sha256") == EXPECTED_SELECTED_TARGET_SHA256
        ),
        "label_count_reconciliation_pass": label.get("all_counts_reconcile") is True,
        "level_center_parity_pass": type_exact_equal(center, {
            "support_cell_count": EXPECTED_LEVEL_CENTER_SUPPORT_COUNT,
            "support_mask_sha256": EXPECTED_LEVEL_CENTER_SUPPORT_SHA256,
            "free_total": EXPECTED_CLASS_TOTALS["free"],
            "free_supported": EXPECTED_LEVEL_CENTER_FREE_SUPPORTED,
            "occupied_total": EXPECTED_CLASS_TOTALS["occupied"],
            "occupied_supported": EXPECTED_LEVEL_CENTER_OCCUPIED_SUPPORTED,
            "known_violation_count": EXPECTED_CENTER_VIOLATION_COUNT,
            "known_violation_identities_sha256": EXPECTED_CENTER_VIOLATION_IDENTITIES_SHA256,
        }),
        "level_cell_square_frozen_pass": type_exact_equal(square, {
            "support_cell_count": EXPECTED_LEVEL_CELL_SQUARE_SUPPORT_COUNT,
            "support_mask_sha256": EXPECTED_LEVEL_CELL_SQUARE_SUPPORT_SHA256,
        }),
        "static_all_known_scored_pass": type_exact_equal(static, {
            "known_total": EXPECTED_KNOWN_TOTAL,
            "supported_count": EXPECTED_STATIC_SUPPORTED_COUNT,
            "unsupported_count": EXPECTED_STATIC_UNSUPPORTED_COUNT,
            "unsupported_free_count": 0,
            "unsupported_occupied_count": 4,
            "unsupported_frame_count": 4,
            "unsupported_identities_sha256": EXPECTED_STATIC_UNSUPPORTED_IDENTITIES_SHA256,
        }),
        "dynamic_all_known_scored_pass": dynamic.get("known_total") == EXPECTED_KNOWN_TOTAL,
        "dynamic_zero_known_unsupported_pass": type_exact_equal(dynamic, {
            "known_total": EXPECTED_KNOWN_TOTAL,
            "supported_count": EXPECTED_KNOWN_TOTAL,
            "unsupported_count": 0,
            "unsupported_free_count": 0,
            "unsupported_occupied_count": 0,
            "unsupported_frame_count": 0,
            "unsupported_identities_sha256": EMPTY_LIST_SHA256,
        }),
        "access_reconciliation_pass": exact_bool(
            access_reconciliation_pass, name="access reconciliation pass"
        ),
        "independent_recomputation_pass": exact_bool(
            independent_recomputation_pass, name="independent recomputation pass"
        ),
        "all_passed": False,
    }
    gates["all_passed"] = all(
        gates[key] for key in GATE_KEYS if key != "all_passed"
    )
    return gates


def validate_scientific_evidence(value: object) -> dict[str, Any]:
    evidence = exact_dict(value, name="scientific evidence")
    require_exact_keys(
        evidence,
        {
            "label_reconciliation",
            "support",
            "family_class_rows",
            "frame_summary_records_sha256",
        },
        name="scientific evidence",
    )
    label = exact_dict(evidence["label_reconciliation"], name="label reconciliation")
    require_exact_keys(
        label,
        {
            "byte_count",
            "byte_sha256",
            "class_totals",
            "known_total",
            "per_frame_cell_count",
            "per_frame_count",
            "per_frame_totals_sha256",
            "all_counts_reconcile",
        },
        name="label reconciliation",
    )
    for field in ("byte_count", "known_total", "per_frame_cell_count", "per_frame_count"):
        exact_int(label[field], name=f"label reconciliation {field}")
    exact_sha256(label["byte_sha256"], name="target byte hash")
    exact_sha256(label["per_frame_totals_sha256"], name="per-frame totals hash")
    exact_bool(label["all_counts_reconcile"], name="label reconciliation pass")
    class_totals = exact_dict(label["class_totals"], name="class totals")
    require_exact_keys(class_totals, {"unknown", "free", "occupied", "all"}, name="class totals")
    for count in class_totals.values():
        exact_int(count, name="class total")

    support = exact_dict(evidence["support"], name="support")
    require_exact_keys(
        support,
        {
            "level_center",
            "level_cell_square",
            "static_cell_square_known",
            "dynamic_cell_square_known",
        },
        name="support",
    )
    center = exact_dict(support["level_center"], name="level center")
    require_exact_keys(
        center,
        {
            "support_cell_count",
            "support_mask_sha256",
            "free_total",
            "free_supported",
            "occupied_total",
            "occupied_supported",
            "known_violation_count",
            "known_violation_identities_sha256",
        },
        name="level center",
    )
    square = exact_dict(support["level_cell_square"], name="level square")
    require_exact_keys(
        square, {"support_cell_count", "support_mask_sha256"}, name="level square"
    )
    known_keys = {
        "known_total",
        "supported_count",
        "unsupported_count",
        "unsupported_free_count",
        "unsupported_occupied_count",
        "unsupported_frame_count",
        "unsupported_identities_sha256",
    }
    for record_name in ("static_cell_square_known", "dynamic_cell_square_known"):
        record = exact_dict(support[record_name], name=record_name)
        require_exact_keys(record, known_keys, name=record_name)
        for field in known_keys - {"unsupported_identities_sha256"}:
            exact_int(record[field], name=f"{record_name}.{field}")
        exact_sha256(
            record["unsupported_identities_sha256"],
            name=f"{record_name} identity hash",
        )
        if record["supported_count"] + record["unsupported_count"] != record["known_total"]:
            raise DiagnosticContractError(f"{record_name} totals do not reconcile")
    for record in (center, square):
        for field, item in record.items():
            if field.endswith("sha256"):
                exact_sha256(item, name=f"support {field}")
            else:
                exact_int(item, name=f"support {field}")

    rows = exact_list(evidence["family_class_rows"], name="family class rows")
    if len(rows) != len(FAMILY_ORDER) * len(KNOWN_CLASS_ORDER):
        raise DiagnosticContractError("family-class support rows must contain ten rows")
    expected_pairs = [
        (family, class_name)
        for family in FAMILY_ORDER
        for class_name in KNOWN_CLASS_ORDER
    ]
    row_keys = {
        "family",
        "class_id",
        "class_name",
        "total",
        "level_center_supported",
        "static_cell_square_supported",
        "dynamic_cell_square_supported",
    }
    for row_value, (family, class_name) in zip(rows, expected_pairs):
        row = exact_dict(row_value, name="family class row")
        require_exact_keys(row, row_keys, name="family class row")
        require_equal(row["family"], family, name="family row family")
        require_equal(row["class_name"], class_name, name="family row class")
        require_equal(row["class_id"], CLASS_IDS[class_name], name="family row class id")
        for field in row_keys - {"family", "class_name"}:
            exact_int(row[field], name=f"family row {field}")
        for supported_field in (
            "level_center_supported",
            "static_cell_square_supported",
            "dynamic_cell_square_supported",
        ):
            if row[supported_field] > row["total"]:
                raise DiagnosticContractError("family support exceeds its denominator")
    for class_name in KNOWN_CLASS_ORDER:
        if sum(row["total"] for row in rows if row["class_name"] == class_name) != class_totals[class_name]:
            raise DiagnosticContractError("family-class denominators do not reconcile")
    exact_sha256(
        evidence["frame_summary_records_sha256"], name="frame summary records hash"
    )
    return evidence


def validate_access_ledger(value: object, *, expected_phase: str) -> dict[str, Any]:
    ledger = exact_dict(value, name=f"{expected_phase} access ledger")
    require_exact_keys(ledger, LEDGER_KEYS, name=f"{expected_phase} access ledger")
    require_equal(ledger["schema"], ACCESS_LEDGER_SCHEMA, name="ledger schema")
    require_equal(ledger["phase"], expected_phase, name="ledger phase")
    for collection_name in ("authorized_read_paths", "authorized_write_paths"):
        records = exact_list(ledger[collection_name], name=collection_name)
        if records != sorted(records, key=lambda item: (item["path"], item["role"])):
            raise DiagnosticContractError(f"{collection_name} is not sorted")
        seen_paths: set[str] = set()
        seen_roles: set[str] = set()
        for item_value in records:
            item = exact_dict(item_value, name=f"{collection_name} record")
            require_exact_keys(item, {"path", "role", "sha256"}, name=collection_name)
            path = canonical_repository_path(
                item["path"], name="authorized path"
            )
            role = exact_string(item["role"], name="authorized role")
            if path in seen_paths or role in seen_roles:
                # Label shards intentionally share one semantic role.
                if role != "label_shard" or path in seen_paths:
                    raise DiagnosticContractError("authorized paths/roles are not unique")
            seen_paths.add(path)
            seen_roles.add(role)
            if collection_name == "authorized_read_paths":
                exact_sha256(item["sha256"], name="authorized read hash")
            elif item["sha256"] is not None:
                raise DiagnosticContractError("write allowlist hash must be null")
        declared_hash = exact_sha256(
            ledger[f"{collection_name[:-1]}_set_sha256"],
            name=f"{collection_name} hash",
        )
        if canonical_json_sha256(records) != declared_hash:
            raise DiagnosticContractError(f"{collection_name} hash mismatch")
    role_counts = exact_dict(ledger["role_byte_open_counts"], name="role counts")
    for role, count in role_counts.items():
        exact_string(role, name="role count key")
        exact_int(count, name="role count")
    for key in (
        "label_shard_pre_hash_byte_opens",
        "label_shard_post_hash_byte_opens",
        "label_shard_npz_parses",
        "selected_label_rows_read",
        "unselected_rows_scored",
        "unselected_rows_retained",
        "metadata_only_shard_stats",
        "unexpected_path_attempts",
    ):
        exact_int(ledger[key], name=f"ledger.{key}")
    decompressions = exact_dict(
        ledger["array_decompression_counts"], name="array decompression counts"
    )
    for key, count in decompressions.items():
        exact_string(key, name="array name")
        exact_int(count, name="array count")
    denied = exact_list(ledger["denied_attempt_records"], name="denied records")
    for index, denied_value in enumerate(denied):
        record = exact_dict(denied_value, name=f"denied record {index}")
        require_exact_keys(
            record,
            {
                "requested_role",
                "declared_role",
                "modality",
                "lexical_path",
                "resolved_path",
                "primary_reason",
            },
            name="denied record",
        )
        for field in (
            "requested_role",
            "declared_role",
            "modality",
            "lexical_path",
            "primary_reason",
        ):
            exact_string(record[field], name=f"denied record {field}")
        if record["resolved_path"] is not None:
            exact_string(record["resolved_path"], name="denied resolved path")
        if record["primary_reason"] not in DENIED_REASON_ORDER:
            raise DiagnosticContractError("denied primary reason is unregistered")
    denied_counts = exact_dict(ledger["denied_reason_counts"], name="denied counts")
    require_equal(
        set(denied_counts), set(DENIED_REASON_ORDER), name="denied reason key set"
    )
    for count in denied_counts.values():
        exact_int(count, name="denied reason count")
    if sum(denied_counts.values()) != len(denied):
        raise DiagnosticContractError("denied record counts do not reconcile")
    forbidden = exact_dict(
        ledger["forbidden_role_open_counts"], name="forbidden role counts"
    )
    require_equal(
        set(forbidden), set(FORBIDDEN_ROLES), name="forbidden role key set"
    )
    if any(exact_int(value, name="forbidden role count") for value in forbidden.values()):
        raise DiagnosticContractError("a forbidden role was opened")
    exact_bool(ledger["all_counts_reconcile"], name="ledger reconciliation")
    return ledger


def validate_phase_ledger(value: object, *, expected_phase: str) -> dict[str, Any]:
    ledger = validate_access_ledger(value, expected_phase=expected_phase)
    reads = ledger["authorized_read_paths"]
    writes = ledger["authorized_write_paths"]
    label_reads = [record for record in reads if record["role"] == "label_shard"]
    if len(label_reads) != EXPECTED_LABEL_SHARD_COUNT:
        raise DiagnosticContractError("phase allowlist must contain 20 label shards")
    roles = {record["role"] for record in reads}
    if set(ledger["role_byte_open_counts"]) != roles:
        raise DiagnosticContractError("phase role-open map differs from read roles")
    expected_nonshard = {
        role: 1 for role in roles if role != "label_shard"
    }
    expected_nonshard["label_shard"] = 0 if expected_phase == "preparation" else 40
    if ledger["role_byte_open_counts"] != expected_nonshard:
        raise DiagnosticContractError("phase role-open counts are not exact")
    expected_numeric = (
        {
            "label_shard_pre_hash_byte_opens": 0,
            "label_shard_post_hash_byte_opens": 0,
            "label_shard_npz_parses": 0,
            "selected_label_rows_read": 0,
            "metadata_only_shard_stats": 20,
        }
        if expected_phase == "preparation"
        else {
            "label_shard_pre_hash_byte_opens": 20,
            "label_shard_post_hash_byte_opens": 20,
            "label_shard_npz_parses": 20,
            "selected_label_rows_read": 320,
            "metadata_only_shard_stats": 0,
        }
    )
    if any(ledger[key] != expected for key, expected in expected_numeric.items()):
        raise DiagnosticContractError("phase numeric label-access ledger is not exact")
    expected_arrays = (
        {}
        if expected_phase == "preparation"
        else {"current_labels": 20, "next_labels": 20}
    )
    if ledger["array_decompression_counts"] != expected_arrays:
        raise DiagnosticContractError("phase array decompression ledger is not exact")
    if (
        ledger["unselected_rows_scored"] != 0
        or ledger["unselected_rows_retained"] != 0
        or ledger["denied_attempt_records"]
        or any(ledger["denied_reason_counts"].values())
        or ledger["unexpected_path_attempts"] != 0
        or any(ledger["forbidden_role_open_counts"].values())
        or ledger["all_counts_reconcile"] is not True
    ):
        raise DiagnosticContractError("passing phase ledger contains denied/forbidden access")
    expected_output = (
        MACHINE_MANIFEST_RELATIVE_PATH
        if expected_phase == "preparation"
        else CANDIDATE_RELATIVE_PATH
    )
    expected_write = {
        "path": str(REPOSITORY_ROOT / expected_output),
        "role": (
            "machine_manifest_output"
            if expected_phase == "preparation"
            else "runner_output"
        ),
        "sha256": None,
    }
    if len(writes) != 1 or writes[0] != expected_write:
        raise DiagnosticContractError("phase write allowlist is not the fixed output")
    return ledger


def _absolute_record(path: str, role: str, digest: str) -> dict[str, str]:
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = REPOSITORY_ROOT / absolute
    canonical_repository_path(str(absolute), name=f"{role} path")
    return {"path": str(absolute), "role": role, "sha256": digest}


def _exact_shard_read_records(entries: object) -> list[dict[str, str]]:
    values = exact_list(entries, name="label shard entries")
    if len(values) != EXPECTED_LABEL_SHARD_COUNT:
        raise DiagnosticContractError("exact shard graph must contain 20 entries")
    records: list[dict[str, str]] = []
    for index, value in enumerate(values):
        entry = exact_dict(value, name=f"label shard entry {index}")
        path = canonical_repository_path(
            entry.get("path"), name=f"label shard entry {index} path"
        )
        digest = exact_sha256(
            entry.get("sha256"), name=f"label shard entry {index} hash"
        )
        records.append({"path": path, "role": "label_shard", "sha256": digest})
    records.sort(key=lambda item: (item["path"], item["role"]))
    if len({item["path"] for item in records}) != EXPECTED_LABEL_SHARD_COUNT:
        raise DiagnosticContractError("exact shard graph contains duplicate paths")
    return records


def _source_records_by_role(source_map: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {entry["role"]: entry for entry in source_map["entries"]}


def validate_exact_preparation_ledger(
    value: object,
    *,
    source_map: dict[str, Any],
    human_manifest: dict[str, Any],
    label_shard_entries: object,
) -> dict[str, Any]:
    ledger = validate_phase_ledger(value, expected_phase="preparation")
    sources = _source_records_by_role(source_map)
    expected = [
        _absolute_record(BINDING_RELATIVE_PATH, "binding", BINDING_SHA256),
        _absolute_record(
            PREDECESSOR_REPORT_RELATIVE_PATH,
            "predecessor_report",
            PREDECESSOR_REPORT_SHA256,
        ),
        _absolute_record(
            PREDECESSOR_RESULT_RELATIVE_PATH,
            "predecessor_result",
            PREDECESSOR_RESULT_FILE_SHA256,
        ),
        _absolute_record(
            human_manifest["path"],
            "human_manifest",
            human_manifest["file_sha256"],
        ),
        *[
            _absolute_record(entry["path"], role, entry["sha256"])
            for role, entry in (
                (role, sources[role]) for role, _path in SOURCE_MAP_CONTRACT
            )
        ],
        *_exact_shard_read_records(label_shard_entries),
    ]
    expected.sort(key=lambda item: (item["path"], item["role"]))
    require_equal(
        ledger["authorized_read_paths"],
        expected,
        name="preparation exact read graph",
    )
    require_equal(
        ledger["authorized_read_path_set_sha256"],
        canonical_json_sha256(expected),
        name="preparation read graph hash",
    )
    return ledger


def validate_exact_runner_ledger(
    value: object,
    *,
    source_map: dict[str, Any],
    manifests: dict[str, Any],
    label_shard_entries: object,
) -> dict[str, Any]:
    ledger = validate_phase_ledger(value, expected_phase="runner")
    sources = _source_records_by_role(source_map)
    human = manifests["human"]
    machine = manifests["machine"]
    expected = [
        _absolute_record(BINDING_RELATIVE_PATH, "binding", BINDING_SHA256),
        _absolute_record(
            human["path"], "human_manifest", human["file_sha256"]
        ),
        _absolute_record(
            machine["path"], "machine_manifest", machine["file_sha256"]
        ),
        _absolute_record(
            PREDECESSOR_RESULT_RELATIVE_PATH,
            "predecessor_result",
            PREDECESSOR_RESULT_FILE_SHA256,
        ),
        *[
            _absolute_record(sources[role]["path"], role, sources[role]["sha256"])
            for role in (
                "dynamic_geometry",
                "diagnostic_core",
                "runner",
                "geometry_test",
                "diagnostic_test",
                "preparation_test",
                "finalizer_test",
            )
        ],
        *_exact_shard_read_records(label_shard_entries),
    ]
    expected.sort(key=lambda item: (item["path"], item["role"]))
    require_equal(
        ledger["authorized_read_paths"], expected, name="runner exact read graph"
    )
    require_equal(
        ledger["authorized_read_path_set_sha256"],
        canonical_json_sha256(expected),
        name="runner read graph hash",
    )
    return ledger


def build_candidate(
    *,
    created_at_utc: str,
    implementation_manifests: object,
    source_map: object,
    preparation_access_ledger: object,
    runner_access_ledger: object,
    scientific: object,
    label_shard_entries: object,
) -> dict[str, Any]:
    manifests = validate_implementation_manifests(implementation_manifests)
    sources = validate_source_map(source_map)
    preparation = validate_exact_preparation_ledger(
        preparation_access_ledger,
        source_map=sources,
        human_manifest=manifests["human"],
        label_shard_entries=label_shard_entries,
    )
    runner = validate_exact_runner_ledger(
        runner_access_ledger,
        source_map=sources,
        manifests=manifests,
        label_shard_entries=label_shard_entries,
    )
    evidence = exact_dict(scientific, name="scientific evidence")
    scientific_core = {
        "label_reconciliation": evidence["label_reconciliation"],
        "support": evidence["support"],
        "family_class_rows": evidence["family_class_rows"],
        "frame_summary_records_sha256": evidence["frame_summary_records_sha256"],
    }
    gates = scientific_gates(
        scientific_core,
        access_reconciliation_pass=(
            preparation["all_counts_reconcile"] and runner["all_counts_reconcile"]
        ),
        independent_recomputation_pass=False,
    )
    core = {
        "schema": CANDIDATE_SCHEMA,
        "created_at_utc": validate_utc_timestamp(
            created_at_utc, name="created_at_utc"
        ),
        "execution_binding": {
            "path": BINDING_RELATIVE_PATH,
            "file_sha256": BINDING_SHA256,
        },
        "implementation_manifests": manifests,
        "inputs": {
            "predecessor_report": {
                "path": PREDECESSOR_REPORT_RELATIVE_PATH,
                "file_sha256": PREDECESSOR_REPORT_SHA256,
            },
            "predecessor_result": {
                "path": PREDECESSOR_RESULT_RELATIVE_PATH,
                "file_sha256": PREDECESSOR_RESULT_FILE_SHA256,
                "content_sha256": PREDECESSOR_RESULT_CONTENT_SHA256,
            },
            "dynamic_geometry": {
                "path": DYNAMIC_GEOMETRY_RELATIVE_PATH,
                "file_sha256": DYNAMIC_GEOMETRY_SHA256,
            },
            "label_shard_manifest": {
                "entry_count": EXPECTED_LABEL_SHARD_COUNT,
                "manifest_sha256": EXPECTED_LABEL_SHARD_MANIFEST_SHA256,
            },
            "selected_targets": {
                "frame_count": EXPECTED_FRAME_COUNT,
                "byte_count": EXPECTED_SELECTED_TARGET_BYTE_COUNT,
                "sha256": EXPECTED_SELECTED_TARGET_SHA256,
            },
        },
        "source_map": sources,
        "scope": {
            "dataset_role": "train",
            "learning_performed": False,
            "frame_count": EXPECTED_FRAME_COUNT,
            "transition_count": EXPECTED_TRANSITION_COUNT,
            "families": list(FAMILY_ORDER),
            "endpoint_sides": list(SIDE_ORDER),
            "class_order": list(CLASS_ORDER),
            "forbidden_roles": list(FORBIDDEN_ROLES),
        },
        "preparation_access_ledger": preparation,
        "runner_access_ledger": runner,
        **scientific_core,
        "scientific_core_sha256": canonical_json_sha256(scientific_core),
        "gates": gates,
    }
    result = with_content_sha256(core)
    require_exact_keys(result, CANDIDATE_KEYS, name="candidate")
    return result
