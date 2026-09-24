"""Non-authoritative standard-library reference for camera-frustum fit labels.

This module deliberately has no NumPy, Torch, dataset, model, or runner
dependency.  It decodes the four registered arrays in one fit-label NPZ,
extracts only explicitly selected endpoint rows, and reproduces the frozen
camera mapping and per-frame observability diagnostics with scalar Python.

The decoder is intentionally narrow.  It accepts NPY major versions 1, 2,
and 3, but only C-order ``[N,64,64]`` uint8 label arrays and bool supervision
arrays.  That keeps this path independent from both ``numpy.load`` and the
primary audit implementation it is intended to check.

This file is synthetic-test support, not part of the frozen authoritative
source graph.  Authoritative validation must inline and bind independently
reviewed logic rather than importing this reference.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
import math
import struct
from typing import Any, Mapping, Sequence
import zipfile


NPZ_EVIDENCE_SCHEMA = "lewm_go2_n32_fit_label_npz_stdlib_evidence_v1"
FRAME_ANALYSIS_SCHEMA = "lewm_go2_n32_camera_frustum_frame_analysis_v1"
LABEL_SUPPORT_SCHEMA = "lewm_go2_n32_camera_frustum_label_support_v1"
RAY_SEQUENCE_SCHEMA = "lewm_go2_n32_camera_frustum_ray_sequences_v1"

FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
ENDPOINT_SIDES = ("current", "next")
UNKNOWN_CLASS = 0
FREE_CLASS = 1
OCCUPIED_CLASS = 2
CLASS_NAMES = ("unknown", "free", "occupied")
CLASS_IDS = (UNKNOWN_CLASS, FREE_CLASS, OCCUPIED_CLASS)

CARTESIAN_ROWS = 64
CARTESIAN_COLUMNS = 64
CARTESIAN_CELL_COUNT = CARTESIAN_ROWS * CARTESIAN_COLUMNS
CARTESIAN_CELL_SIZE_M = 0.10
CARTESIAN_FORWARD_MIN_EDGE_M = -1.0
CARTESIAN_LEFT_MIN_EDGE_M = -3.2
CAMERA_FORWARD_BODY_M = 0.326
CAMERA_NEAR_M = 0.05
HORIZONTAL_FOV_DEG = 78.323
HALF_FOV_RAD = math.radians(HORIZONTAL_FOV_DEG / 2.0)
RANGE_BIN_COUNT = 64
RANGE_BIN_SIZE_M = 0.10
RANGE_LIMIT_M = RANGE_BIN_COUNT * RANGE_BIN_SIZE_M
ANGULAR_BIN_COUNT = 256

_DECODED_ARRAY_KINDS = {
    "current_labels.npy": "uint8",
    "current_supervision_mask.npy": "bool",
    "next_labels.npy": "uint8",
    "next_supervision_mask.npy": "bool",
}
_AUXILIARY_MEMBER_NAMES = (
    "current_image_path.npy",
    "current_image_sha256.npy",
    "current_observed_mask.npy",
    "next_image_path.npy",
    "next_image_sha256.npy",
    "next_observed_mask.npy",
    "primitive.npy",
    "relative_se2_current_frame.npy",
)
_EXPECTED_MEMBER_NAMES = frozenset(
    (*_DECODED_ARRAY_KINDS, *_AUXILIARY_MEMBER_NAMES)
)
_DIRECTED_TRANSITIONS = (
    (UNKNOWN_CLASS, FREE_CLASS),
    (UNKNOWN_CLASS, OCCUPIED_CLASS),
    (FREE_CLASS, UNKNOWN_CLASS),
    (FREE_CLASS, OCCUPIED_CLASS),
    (OCCUPIED_CLASS, UNKNOWN_CLASS),
    (OCCUPIED_CLASS, FREE_CLASS),
)
_TRANSITION_NAMES = tuple(
    f"{CLASS_NAMES[source]}_to_{CLASS_NAMES[destination]}"
    for source, destination in _DIRECTED_TRANSITIONS
)


def canonical_json_sha256(value: object) -> str:
    """Hash strict compact canonical JSON."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_copy(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty mapping")
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        result = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain strict JSON values") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must encode a JSON object")
    return result


@dataclass(frozen=True)
class DecodedNpyArray:
    """One validated one-byte, three-dimensional NPY array."""

    name: str
    version: tuple[int, int]
    dtype: str
    shape: tuple[int, int, int]
    data: bytes

    @property
    def storage_row_count(self) -> int:
        return self.shape[0]

    def row_bytes(self, row: int) -> bytes:
        if isinstance(row, bool) or not isinstance(row, int):
            raise TypeError("selected NPY row must be an integer")
        if not 0 <= row < self.storage_row_count:
            raise IndexError("selected NPY row is outside storage")
        start = row * CARTESIAN_CELL_COUNT
        return self.data[start : start + CARTESIAN_CELL_COUNT]

    def storage_metadata(self) -> NpyArrayStorageMetadata:
        return NpyArrayStorageMetadata(
            name=self.name,
            version=self.version,
            dtype=self.dtype,
            shape=self.shape,
        )


@dataclass(frozen=True)
class NpyArrayStorageMetadata:
    """Array-level evidence that contains no retained array values."""

    name: str
    version: tuple[int, int]
    dtype: str
    shape: tuple[int, int, int]

    @property
    def storage_row_count(self) -> int:
        return self.shape[0]

    def evidence(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "npy_version": list(self.version),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "c_order": True,
            "storage_row_count": self.storage_row_count,
        }


@dataclass(frozen=True)
class SelectedFitLabelRow:
    """An independent contiguous copy of one selected endpoint label row."""

    side: str
    row: int
    target: bytes
    supervision: bytes

    def evidence(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "row": self.row,
            "target_sha256": hashlib.sha256(self.target).hexdigest(),
            "supervision_sha256": hashlib.sha256(self.supervision).hexdigest(),
            "target_byte_count": len(self.target),
            "supervision_byte_count": len(self.supervision),
        }


@dataclass(frozen=True)
class FitLabelNpzEvidence:
    """Validated NPZ storage metadata and the requested selected rows."""

    arrays: tuple[NpyArrayStorageMetadata, ...]
    selected_rows: tuple[SelectedFitLabelRow, ...]
    npz_file_sha256: str

    @property
    def storage_row_counts(self) -> dict[str, int]:
        return {array.name: array.storage_row_count for array in self.arrays}

    def evidence(self) -> dict[str, Any]:
        arrays = [array.evidence() for array in self.arrays]
        selected = [row.evidence() for row in self.selected_rows]
        counts = self.storage_row_counts
        materialized_label_rows = sum(
            counts[f"{side}_labels.npy"] for side in ENDPOINT_SIDES
        )
        materialized_supervision_rows = sum(
            counts[f"{side}_supervision_mask.npy"] for side in ENDPOINT_SIDES
        )
        result = {
            "schema": NPZ_EVIDENCE_SCHEMA,
            "npz_file_sha256": self.npz_file_sha256,
            "archive_member_count": len(_EXPECTED_MEMBER_NAMES),
            "decoded_array_count": len(arrays),
            "arrays_decompressed": len(arrays),
            "arrays": arrays,
            "auxiliary_member_names": list(_AUXILIARY_MEMBER_NAMES),
            "auxiliary_members_decompressed": False,
            "storage_row_counts": counts,
            "materialized_label_rows": materialized_label_rows,
            "materialized_supervision_rows": materialized_supervision_rows,
            "materialized_row_totals_agree": (
                materialized_label_rows == materialized_supervision_rows
            ),
            "selected_row_count": len(selected),
            "selected_label_rows": len(selected),
            "selected_supervision_rows": len(selected),
            "selected_rows": selected,
        }
        result["evidence_sha256"] = canonical_json_sha256(result)
        return result


def _strict_npy_header(header: str, *, name: str) -> dict[str, Any]:
    """Parse a literal NPY header while preserving duplicate-key detection."""

    if not header.startswith("{") or not header.isascii():
        raise ValueError(f"{name} NPY header must be a single ASCII dictionary")
    if any(character in header for character in "\r\n\t\v\f\x00"):
        raise ValueError(f"{name} NPY header contains illegal whitespace")
    try:
        expression = ast.parse(header, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{name} has a malformed NPY header") from exc
    if not isinstance(expression.body, ast.Dict):
        raise ValueError(f"{name} NPY header must be a dictionary literal")
    body_end = expression.body.end_col_offset
    if body_end is None or any(character != " " for character in header[body_end:]):
        raise ValueError(f"{name} NPY header has malformed space padding")
    keys: list[str] = []
    for key_node in expression.body.keys:
        try:
            key = ast.literal_eval(key_node)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"{name} NPY header has a nonliteral key") from exc
        if not isinstance(key, str):
            raise ValueError(f"{name} NPY header keys must be strings")
        if key in keys:
            raise ValueError(f"{name} NPY header repeats key {key!r}")
        keys.append(key)
    try:
        value = ast.literal_eval(expression.body)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"{name} has a nonliteral NPY header") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} NPY header must decode to a dictionary")
    required = {"descr", "fortran_order", "shape"}
    if set(value) != required:
        raise ValueError(f"{name} NPY header must contain exactly {sorted(required)!r}")
    return value


def _dtype_name(descr: object, *, name: str) -> str:
    if not isinstance(descr, str):
        raise ValueError(f"{name} NPY dtype descriptor must be a string")
    if descr in {"|u1", "<u1"}:
        return "uint8"
    if descr == "|b1":
        return "bool"
    raise ValueError(
        f"{name} NPY dtype must use frozen uint8 or bool descriptors"
    )


def decode_npy(
    payload: bytes,
    *,
    expected_dtype: str,
    name: str = "array.npy",
) -> DecodedNpyArray:
    """Strictly decode one registered NPY v1/v2/v3 array without NumPy."""

    if not isinstance(payload, bytes):
        raise TypeError("NPY payload must be immutable bytes")
    if expected_dtype not in {"uint8", "bool"}:
        raise ValueError("expected NPY dtype must be uint8 or bool")
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        raise ValueError(f"{name} lacks the NPY magic")
    version = (payload[6], payload[7])
    if version == (1, 0):
        length_size = 2
        header_encoding = "latin1"
    elif version in {(2, 0), (3, 0)}:
        length_size = 4
        header_encoding = "latin1" if version[0] == 2 else "utf-8"
    else:
        raise ValueError(f"{name} uses unsupported NPY version {version!r}")
    length_start = 8
    length_end = length_start + length_size
    if len(payload) < length_end:
        raise ValueError(f"{name} truncates the NPY header length")
    header_length = int.from_bytes(payload[length_start:length_end], "little")
    header_end = length_end + header_length
    if header_length == 0 or header_end > len(payload):
        raise ValueError(f"{name} truncates the NPY header")
    header_bytes = payload[length_end:header_end]
    if not header_bytes.endswith(b"\n") or b"\n" in header_bytes[:-1]:
        raise ValueError(f"{name} NPY header must have exactly one final newline")
    if header_end % 64 != 0:
        raise ValueError(f"{name} NPY header is not aligned to 64 bytes")
    try:
        header = header_bytes[:-1].decode(header_encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} NPY header has invalid {header_encoding} text") from exc
    fields = _strict_npy_header(header, name=name)
    dtype = _dtype_name(fields["descr"], name=name)
    if dtype != expected_dtype:
        raise ValueError(
            f"{name} NPY dtype is {dtype}, expected {expected_dtype}"
        )
    if fields["fortran_order"] is not False:
        raise ValueError(f"{name} NPY array must be C-order")
    shape = fields["shape"]
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError(f"{name} NPY shape must be [N,64,64]")
    if any(isinstance(dimension, bool) or not isinstance(dimension, int) for dimension in shape):
        raise ValueError(f"{name} NPY shape dimensions must be integers")
    if shape[0] <= 0 or shape[1:] != (CARTESIAN_ROWS, CARTESIAN_COLUMNS):
        raise ValueError(f"{name} NPY shape must be [N,64,64] with N positive")
    data = payload[header_end:]
    expected_data_bytes = shape[0] * CARTESIAN_CELL_COUNT
    if len(data) != expected_data_bytes:
        raise ValueError(
            f"{name} NPY data length is {len(data)}, expected {expected_data_bytes}"
        )
    return DecodedNpyArray(
        name=name,
        version=version,
        dtype=dtype,
        shape=(shape[0], shape[1], shape[2]),
        data=bytes(data),
    )


def _normalize_selection(selection: object) -> tuple[str, int]:
    if (
        not isinstance(selection, tuple)
        or len(selection) != 2
        or not isinstance(selection[0], str)
    ):
        raise TypeError("selected row must be a (side, row) tuple")
    side, row = selection
    if side not in ENDPOINT_SIDES:
        raise ValueError("selected row side must be current or next")
    if isinstance(row, bool) or not isinstance(row, int):
        raise TypeError("selected row index must be an integer")
    if row < 0:
        raise IndexError("selected row index must be nonnegative")
    return side, row


def _validate_local_zip_header(payload: bytes, info: zipfile.ZipInfo) -> None:
    """Reconcile one central entry with its local header without opening data."""

    offset = info.header_offset
    if offset < 0 or offset + 30 > len(payload):
        raise ValueError(f"fit label NPZ member {info.filename!r} has a bad local offset")
    if payload[offset : offset + 4] != b"PK\x03\x04":
        raise ValueError(
            f"fit label NPZ member {info.filename!r} lacks its local header"
        )
    local_flags = struct.unpack_from("<H", payload, offset + 6)[0]
    local_compression = struct.unpack_from("<H", payload, offset + 8)[0]
    name_length = struct.unpack_from("<H", payload, offset + 26)[0]
    extra_length = struct.unpack_from("<H", payload, offset + 28)[0]
    name_start = offset + 30
    header_end = name_start + name_length + extra_length
    if header_end > len(payload):
        raise ValueError(
            f"fit label NPZ member {info.filename!r} truncates its local header"
        )
    encoding = "utf-8" if local_flags & 0x800 else "cp437"
    try:
        local_name = payload[name_start : name_start + name_length].decode(encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"fit label NPZ member {info.filename!r} has an invalid local name"
        ) from exc
    if local_name != info.filename:
        raise ValueError(
            f"fit label NPZ member {info.filename!r} disagrees with local name "
            f"{local_name!r}"
        )
    local_encrypted = bool(local_flags & 0x1)
    central_encrypted = bool(info.flag_bits & 0x1)
    if local_encrypted != central_encrypted or local_encrypted:
        raise ValueError(
            f"fit label NPZ member {info.filename!r} has inconsistent encryption flags"
        )
    if local_compression != info.compress_type:
        raise ValueError(
            f"fit label NPZ member {info.filename!r} has inconsistent compression"
        )


def decode_fit_label_npz(
    payload: bytes,
    *,
    selected_rows: Sequence[tuple[str, int]],
) -> FitLabelNpzEvidence:
    """Decode exactly four fit-label arrays and extract selected endpoint rows."""

    if not isinstance(payload, bytes):
        raise TypeError("NPZ payload must be immutable bytes")
    selections = tuple(_normalize_selection(selection) for selection in selected_rows)
    if not selections:
        raise ValueError("fit-label extraction requires at least one selected row")
    if len(set(selections)) != len(selections):
        raise ValueError("fit-label extraction repeats a selected side/row")

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), mode="r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("fit label payload is not a valid NPZ") from exc
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(f"fit label NPZ repeats array members {duplicates!r}")
        expected_names = set(_EXPECTED_MEMBER_NAMES)
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise ValueError(
                f"fit label NPZ array members changed; missing={missing!r} extra={extra!r}"
            )
        for info in infos:
            _validate_local_zip_header(payload, info)
            if info.is_dir() or info.flag_bits & 0x1:
                raise ValueError(
                    f"fit label NPZ member {info.filename!r} is not a plain unencrypted file"
                )
        arrays_by_name: dict[str, DecodedNpyArray] = {}
        # The auxiliary arrays are identity commitments only in this evidence
        # path.  Their exact names are checked above, but their compressed
        # payloads are deliberately never opened or decompressed.
        for name in sorted(_DECODED_ARRAY_KINDS):
            info = archive.getinfo(name)
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError(f"fit label NPZ member {name!r} uses unsupported compression")
            try:
                member_payload = archive.read(info)
            except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
                raise ValueError(f"fit label NPZ member {name!r} failed integrity checks") from exc
            arrays_by_name[name] = decode_npy(
                member_payload,
                expected_dtype=_DECODED_ARRAY_KINDS[name],
                name=name,
            )

    for side in ENDPOINT_SIDES:
        label_count = arrays_by_name[f"{side}_labels.npy"].storage_row_count
        mask_count = arrays_by_name[
            f"{side}_supervision_mask.npy"
        ].storage_row_count
        if label_count != mask_count:
            raise ValueError(f"{side} label and supervision storage row counts differ")

    selected: list[SelectedFitLabelRow] = []
    for side, row in selections:
        labels = arrays_by_name[f"{side}_labels.npy"]
        masks = arrays_by_name[f"{side}_supervision_mask.npy"]
        target = labels.row_bytes(row)
        supervision = masks.row_bytes(row)
        if any(value > OCCUPIED_CLASS for value in target):
            raise ValueError("selected fit target contains a class outside 0/1/2")
        if any(value not in (0, 1) for value in supervision):
            raise ValueError("selected fit bool supervision has a noncanonical byte")
        if any(value != 1 for value in supervision):
            raise ValueError("selected fit supervision is not the full bool grid")
        selected.append(
            SelectedFitLabelRow(
                side=side,
                row=row,
                target=bytes(target),
                supervision=bytes(supervision),
            )
        )
    ordered_arrays = tuple(
        arrays_by_name[name].storage_metadata() for name in sorted(arrays_by_name)
    )
    result = FitLabelNpzEvidence(
        arrays=ordered_arrays,
        selected_rows=tuple(selected),
        npz_file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    del arrays_by_name, labels, masks, target, supervision
    return result


def _cell_center(row: int, column: int) -> tuple[float, float]:
    forward_m = CARTESIAN_FORWARD_MIN_EDGE_M + (
        row + 0.5
    ) * CARTESIAN_CELL_SIZE_M
    left_m = CARTESIAN_LEFT_MIN_EDGE_M + (
        column + 0.5
    ) * CARTESIAN_CELL_SIZE_M
    return forward_m, left_m


def _camera_point_to_bin(
    forward_body_m: float, left_body_m: float
) -> tuple[int, int] | None:
    forward_camera = forward_body_m - CAMERA_FORWARD_BODY_M
    left_camera = left_body_m
    range_m = math.hypot(forward_camera, left_camera)
    bearing_rad = math.atan2(left_camera, forward_camera)
    if forward_camera < CAMERA_NEAR_M:
        return None
    if not 0.0 <= range_m < RANGE_LIMIT_M:
        return None
    if not -HALF_FOV_RAD <= bearing_rad <= HALF_FOV_RAD:
        return None
    range_bin = int(math.floor(range_m / RANGE_BIN_SIZE_M))
    angular_fraction = (bearing_rad + HALF_FOV_RAD) / (2.0 * HALF_FOV_RAD)
    angular_bin = min(
        ANGULAR_BIN_COUNT - 1,
        int(math.floor(angular_fraction * ANGULAR_BIN_COUNT)),
    )
    return range_bin, angular_bin


def build_camera_centered_mapping() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Build the literal frozen ``[64][64]`` mapping with ``(-1,-1)`` sentinel."""

    rows = []
    for row in range(CARTESIAN_ROWS):
        cells = []
        for column in range(CARTESIAN_COLUMNS):
            polar_bin = _camera_point_to_bin(*_cell_center(row, column))
            cells.append((-1, -1) if polar_bin is None else polar_bin)
        rows.append(tuple(cells))
    return tuple(rows)


def camera_centered_support_mask() -> tuple[tuple[bool, ...], ...]:
    mapping = build_camera_centered_mapping()
    return tuple(
        tuple(range_bin >= 0 and angular_bin >= 0 for range_bin, angular_bin in row)
        for row in mapping
    )


def mapping_sha256() -> str:
    digest = hashlib.sha256()
    for row in build_camera_centered_mapping():
        for range_bin, angular_bin in row:
            digest.update(struct.pack("<hh", range_bin, angular_bin))
    return digest.hexdigest()


def support_mask_sha256() -> str:
    payload = bytes(
        int(supported)
        for row in camera_centered_support_mask()
        for supported in row
    )
    return hashlib.sha256(payload).hexdigest()


def _validated_frame_bytes(
    values: bytes, *, name: str, allowed_values: tuple[int, ...]
) -> bytes:
    if not isinstance(values, bytes):
        raise TypeError(f"{name} must be immutable row-major bytes")
    if len(values) != CARTESIAN_CELL_COUNT:
        raise ValueError(f"{name} must contain exactly 64 x 64 bytes")
    if any(value not in allowed_values for value in values):
        raise ValueError(f"{name} contains values outside its registered domain")
    return values


def _collapse_classes(classes: Sequence[int]) -> list[int]:
    result: list[int] = []
    for class_id in classes:
        if not result or class_id != result[-1]:
            result.append(class_id)
    return result


def _ray_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    length_histogram = Counter(str(int(record["length"])) for record in records)
    transition_histogram = Counter(
        str(int(record["transition_count"])) for record in records
    )
    transition_counts = Counter({name: 0 for name in _TRANSITION_NAMES})
    for record in records:
        transition_counts.update(record["directed_unequal_transition_counts"])
    sequence_count = len(records)
    eligible = sum(int(record["length"]) >= 2 for record in records)
    transition_event_count = sum(
        int(record["transition_count"]) for record in records
    )
    return {
        "sequence_count": sequence_count,
        "length_histogram": dict(
            sorted(length_histogram.items(), key=lambda item: int(item[0]))
        ),
        "sequences_with_fewer_than_two_cells_count": sequence_count - eligible,
        "transition_rate_eligible_sequence_count": eligible,
        "class_transition_histogram": dict(
            sorted(transition_histogram.items(), key=lambda item: int(item[0]))
        ),
        "maximum_transitions_per_sequence": max(
            (int(record["transition_count"]) for record in records), default=0
        ),
        "directed_unequal_transition_counts": {
            name: int(transition_counts[name]) for name in _TRANSITION_NAMES
        },
        "transition_bucket_counts": {
            "0": sum(int(record["transition_count"]) == 0 for record in records),
            "1": sum(int(record["transition_count"]) == 1 for record in records),
            "2": sum(int(record["transition_count"]) == 2 for record in records),
            "3_plus": sum(
                int(record["transition_count"]) >= 3 for record in records
            ),
        },
        "transition_event_count": transition_event_count,
        "transition_events_per_eligible_sequence": (
            float(transition_event_count / eligible) if eligible else None
        ),
        "contains_known_after_unknown_count": sum(
            bool(record["contains_known_after_unknown"]) for record in records
        ),
        "contains_free_after_occupied_count": sum(
            bool(record["contains_free_after_occupied"]) for record in records
        ),
        "scalar_first_hit_irregular_count": sum(
            not bool(record["scalar_first_hit_regular"]) for record in records
        ),
        "scalar_first_hit_regular_count": sum(
            bool(record["scalar_first_hit_regular"]) for record in records
        ),
    }


def _ray_records(
    target: bytes,
    mapping: tuple[tuple[tuple[int, int], ...], ...],
    *,
    frame_key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    locations_by_angle: list[list[tuple[int, int, int]]] = [
        [] for _ in range(ANGULAR_BIN_COUNT)
    ]
    for row in range(CARTESIAN_ROWS):
        for column in range(CARTESIAN_COLUMNS):
            range_bin, angular_bin = mapping[row][column]
            if angular_bin >= 0:
                locations_by_angle[angular_bin].append((range_bin, row, column))
    records = []
    regular_ranks = {FREE_CLASS: 0, OCCUPIED_CLASS: 1, UNKNOWN_CLASS: 2}
    for angular_bin, locations in enumerate(locations_by_angle):
        ordered = sorted(locations)
        range_bins = [record[0] for record in ordered]
        if len(range_bins) != len(set(range_bins)):
            raise ValueError("ray sequence contains a range-bin tie")
        classes = [target[row * CARTESIAN_COLUMNS + column] for _, row, column in ordered]
        collapsed = _collapse_classes(classes)
        directed = Counter({name: 0 for name in _TRANSITION_NAMES})
        for source, destination in zip(collapsed, collapsed[1:]):
            directed[f"{CLASS_NAMES[source]}_to_{CLASS_NAMES[destination]}"] += 1
        unknown_positions = [
            index for index, value in enumerate(classes) if value == UNKNOWN_CLASS
        ]
        known_positions = [
            index for index, value in enumerate(classes) if value != UNKNOWN_CLASS
        ]
        occupied_positions = [
            index for index, value in enumerate(classes) if value == OCCUPIED_CLASS
        ]
        free_positions = [
            index for index, value in enumerate(classes) if value == FREE_CLASS
        ]
        scalar_regular = all(
            regular_ranks[source] <= regular_ranks[destination]
            for source, destination in zip(classes, classes[1:])
        )
        records.append(
            {
                "frame_key": dict(frame_key),
                "angular_bin": angular_bin,
                "length": len(classes),
                "range_bins": range_bins,
                "class_sequence": classes,
                "collapsed_class_sequence": collapsed,
                "transition_count": max(0, len(collapsed) - 1),
                "directed_unequal_transition_counts": {
                    name: int(directed[name]) for name in _TRANSITION_NAMES
                },
                "contains_known_after_unknown": bool(
                    unknown_positions
                    and known_positions
                    and min(unknown_positions) < max(known_positions)
                ),
                "contains_free_after_occupied": bool(
                    occupied_positions
                    and free_positions
                    and min(occupied_positions) < max(free_positions)
                ),
                "scalar_first_hit_regular": scalar_regular,
            }
        )
    return records


def analyze_frame_labels(
    target: bytes,
    supervision: bytes,
    *,
    frame_key: Mapping[str, Any],
    family: str,
    endpoint_side: str,
) -> dict[str, Any]:
    """Reproduce one primary per-frame observability report independently."""

    if family not in FAMILIES:
        raise ValueError("frame family is not registered")
    if endpoint_side not in ENDPOINT_SIDES:
        raise ValueError("frame endpoint side is not registered")
    key = _canonical_json_copy(frame_key, name="frame_key")
    labels = _validated_frame_bytes(
        target, name="target", allowed_values=CLASS_IDS
    )
    mask = _validated_frame_bytes(
        supervision, name="supervision", allowed_values=(0, 1)
    )
    if any(value != 1 for value in mask):
        raise ValueError("supervision must cover the full 64 x 64 grid")

    mapping = build_camera_centered_mapping()
    support = camera_centered_support_mask()
    by_class = {
        name: {"total": 0, "supported": 0, "unsupported": 0}
        for name in CLASS_NAMES
    }
    violations = []
    supported_count = 0
    for row in range(CARTESIAN_ROWS):
        for column in range(CARTESIAN_COLUMNS):
            index = row * CARTESIAN_COLUMNS + column
            class_id = labels[index]
            class_name = CLASS_NAMES[class_id]
            supported = support[row][column]
            by_class[class_name]["total"] += 1
            by_class[class_name]["supported" if supported else "unsupported"] += 1
            supported_count += int(supported)
            if not supported and class_id in (FREE_CLASS, OCCUPIED_CLASS):
                violations.append(
                    {
                        "frame_key": dict(key),
                        "row": row,
                        "column": column,
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )
    unsupported_count = CARTESIAN_CELL_COUNT - supported_count
    unsupported_free = by_class["free"]["unsupported"]
    unsupported_occupied = by_class["occupied"]["unsupported"]
    unsupported_unknown = by_class["unknown"]["unsupported"]
    support_passes = bool(
        unsupported_free == 0
        and unsupported_occupied == 0
        and unsupported_unknown == unsupported_count
    )
    label_support = {
        "schema": LABEL_SUPPORT_SCHEMA,
        "total_supervised_label_count": CARTESIAN_CELL_COUNT,
        "supported_label_count": supported_count,
        "unsupported_label_count": unsupported_count,
        "class_counts": {
            name: by_class[name]["total"] for name in CLASS_NAMES
        },
        "by_class": by_class,
        "unsupported_free_count": unsupported_free,
        "unsupported_occupied_count": unsupported_occupied,
        "unsupported_unknown_count": unsupported_unknown,
        "unsupported_targets_are_all_unknown": (
            unsupported_unknown == unsupported_count
        ),
        "violations": violations,
        "passes": support_passes,
    }
    ray_records = _ray_records(labels, mapping, frame_key=key)
    transition_table = _ray_summary(ray_records)
    return {
        "schema": FRAME_ANALYSIS_SCHEMA,
        "frame_key": key,
        "family": family,
        "endpoint_side": endpoint_side,
        "label_support": label_support,
        "ray_sequences": {
            "schema": RAY_SEQUENCE_SCHEMA,
            "records": ray_records,
            "summary": transition_table,
            "sequence_summary_records_sha256": canonical_json_sha256(ray_records),
            "transition_table_sha256": canonical_json_sha256(transition_table),
        },
    }


__all__ = [
    "ANGULAR_BIN_COUNT",
    "CARTESIAN_CELL_COUNT",
    "DecodedNpyArray",
    "ENDPOINT_SIDES",
    "FAMILIES",
    "FitLabelNpzEvidence",
    "FREE_CLASS",
    "NPZ_EVIDENCE_SCHEMA",
    "NpyArrayStorageMetadata",
    "OCCUPIED_CLASS",
    "SelectedFitLabelRow",
    "UNKNOWN_CLASS",
    "analyze_frame_labels",
    "build_camera_centered_mapping",
    "camera_centered_support_mask",
    "canonical_json_sha256",
    "decode_fit_label_npz",
    "decode_npy",
    "mapping_sha256",
    "support_mask_sha256",
]
