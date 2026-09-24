from __future__ import annotations

import io
from pathlib import Path
import subprocess
import struct
import sys
import warnings
import zipfile

import numpy as np
import pytest

from lewm.benchmarks import go2_n32_camera_frustum_fit_evidence_stdlib as stdlib
from lewm.benchmarks import go2_n32_camera_frustum_observability as primary


EXPECTED_MAPPING_SHA256 = (
    "2b8cfb9dcf2deeebe7304d64a4a79b1631eb658991108eb3c3149cccf7a7dd4e"
)
EXPECTED_SUPPORT_SHA256 = (
    "026d7654864bea7ae0545bd6448f6def64519a3bedcbc7ea747e7b4b95f82b3a"
)
AUXILIARY_MEMBERS = (
    "current_image_path.npy",
    "current_image_sha256.npy",
    "current_observed_mask.npy",
    "next_image_path.npy",
    "next_image_sha256.npy",
    "next_observed_mask.npy",
    "primitive.npy",
    "relative_se2_current_frame.npy",
)


def _npy_bytes(array: np.ndarray, version: tuple[int, int] = (1, 0)) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        array,
        version=version,
        allow_pickle=False,
    )
    return stream.getvalue()


def _handcrafted_npy(
    header_literal: str,
    *,
    version: tuple[int, int] = (1, 0),
    data: bytes = b"",
    aligned: bool = True,
    newline_count: int = 1,
    padding_byte: bytes = b" ",
) -> bytes:
    if len(padding_byte) != 1:
        raise ValueError("padding byte must be singular")
    length_size = 2 if version == (1, 0) else 4
    prefix_size = 8 + length_size
    literal = header_literal.encode("ascii")
    newline = b"\n" * newline_count
    padding_count = (
        (-(prefix_size + len(literal) + len(newline))) % 64 if aligned else 0
    )
    header = literal + padding_byte * padding_count + newline
    length = (
        struct.pack("<H", len(header))
        if length_size == 2
        else struct.pack("<I", len(header))
    )
    return b"\x93NUMPY" + bytes(version) + length + header + data


def _mutate_npy_data(payload: bytes, *, flat_index: int, value: int) -> bytes:
    result = bytearray(payload)
    version = (result[6], result[7])
    length_size = 2 if version == (1, 0) else 4
    header_length = int.from_bytes(result[8 : 8 + length_size], "little")
    offset = 8 + length_size + header_length + flat_index
    result[offset] = value
    return bytes(result)


def _mark_zip_member_encrypted(
    payload: bytes,
    member_name: str,
    *,
    local: bool = True,
    central: bool = True,
) -> bytes:
    """Set the metadata encryption bit in local and central ZIP records."""

    result = bytearray(payload)
    encoded_name = member_name.encode("utf-8")
    found_local = False
    position = 0
    while True:
        position = result.find(b"PK\x03\x04", position)
        if position < 0:
            break
        name_length = struct.unpack_from("<H", result, position + 26)[0]
        extra_length = struct.unpack_from("<H", result, position + 28)[0]
        name_start = position + 30
        if bytes(result[name_start : name_start + name_length]) == encoded_name:
            if local:
                flags = struct.unpack_from("<H", result, position + 6)[0]
                struct.pack_into("<H", result, position + 6, flags | 0x1)
            found_local = True
        compressed_size = struct.unpack_from("<I", result, position + 18)[0]
        position = name_start + name_length + extra_length + compressed_size

    found_central = False
    position = 0
    while True:
        position = result.find(b"PK\x01\x02", position)
        if position < 0:
            break
        name_length = struct.unpack_from("<H", result, position + 28)[0]
        extra_length = struct.unpack_from("<H", result, position + 30)[0]
        comment_length = struct.unpack_from("<H", result, position + 32)[0]
        name_start = position + 46
        if bytes(result[name_start : name_start + name_length]) == encoded_name:
            if central:
                flags = struct.unpack_from("<H", result, position + 8)[0]
                struct.pack_into("<H", result, position + 8, flags | 0x1)
            found_central = True
        position = name_start + name_length + extra_length + comment_length
    if not found_local or not found_central:
        raise AssertionError("synthetic ZIP member was not found")
    return bytes(result)


def _replace_zip_local_member_name(
    payload: bytes, member_name: str, replacement: str
) -> bytes:
    result = bytearray(payload)
    encoded_name = member_name.encode("utf-8")
    encoded_replacement = replacement.encode("utf-8")
    if len(encoded_name) != len(encoded_replacement):
        raise ValueError("replacement ZIP name must preserve its byte length")
    position = 0
    while True:
        position = result.find(b"PK\x03\x04", position)
        if position < 0:
            break
        name_length = struct.unpack_from("<H", result, position + 26)[0]
        extra_length = struct.unpack_from("<H", result, position + 28)[0]
        name_start = position + 30
        if bytes(result[name_start : name_start + name_length]) == encoded_name:
            result[name_start : name_start + name_length] = encoded_replacement
            return bytes(result)
        compressed_size = struct.unpack_from("<I", result, position + 18)[0]
        position = name_start + name_length + extra_length + compressed_size
    raise AssertionError("synthetic ZIP local member was not found")


def _npz_bytes(
    *,
    current_labels: np.ndarray,
    current_mask: np.ndarray,
    next_labels: np.ndarray,
    next_mask: np.ndarray,
    versions: tuple[tuple[int, int], ...] = ((1, 0),) * 4,
    omit: str | None = None,
    extra: tuple[str, bytes] | None = None,
    duplicate: str | None = None,
    member_overrides: dict[str, bytes] | None = None,
) -> bytes:
    arrays = (
        ("current_labels.npy", current_labels),
        ("current_supervision_mask.npy", current_mask),
        ("next_labels.npy", next_labels),
        ("next_supervision_mask.npy", next_mask),
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for (name, array), version in zip(arrays, versions):
            if name != omit:
                member_payload = (
                    member_overrides[name]
                    if member_overrides is not None and name in member_overrides
                    else _npy_bytes(array, version)
                )
                archive.writestr(name, member_payload)
        for name in AUXILIARY_MEMBERS:
            if name != omit:
                # The independent reader must bind these exact names without
                # opening or attempting to decode their payloads.
                archive.writestr(name, b"deliberately-not-an-npy-payload")
        if extra is not None:
            archive.writestr(extra[0], extra[1])
        if duplicate is not None:
            array = dict(arrays)[duplicate]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(duplicate, _npy_bytes(array))
    return stream.getvalue()


def _synthetic_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = np.zeros((3, 64, 64), dtype=np.uint8)
    current[1, 20:25, 30:35] = primary.FREE_CLASS
    current[2, 30:35, 20:25] = primary.OCCUPIED_CLASS
    next_labels = np.zeros((2, 64, 64), dtype=np.uint8)
    next_labels[1, 10:18, 28:36] = primary.FREE_CLASS
    return (
        current,
        np.ones_like(current, dtype=np.bool_),
        next_labels,
        np.ones_like(next_labels, dtype=np.bool_),
    )


@pytest.mark.parametrize("version", ((1, 0), (2, 0), (3, 0)))
@pytest.mark.parametrize(
    ("dtype", "expected"),
    ((np.dtype(np.uint8), "uint8"), (np.dtype(np.bool_), "bool")),
)
def test_npy_v1_v2_v3_decode_exact_one_byte_c_order(
    version: tuple[int, int], dtype: np.dtype, expected: str
) -> None:
    array = np.arange(2 * 64 * 64, dtype=np.uint64).reshape(2, 64, 64)
    if expected == "bool":
        array = (array % 3 == 0).astype(dtype)
    else:
        array = (array % 251).astype(dtype)
    payload = _npy_bytes(array, version)

    decoded = stdlib.decode_npy(
        payload,
        expected_dtype=expected,
        name="synthetic.npy",
    )

    assert decoded.version == version
    assert decoded.dtype == expected
    assert decoded.shape == (2, 64, 64)
    assert decoded.storage_row_count == 2
    assert decoded.data == array.tobytes(order="C")
    assert decoded.row_bytes(1) == array[1].tobytes(order="C")
    metadata = decoded.storage_metadata().evidence()
    assert metadata["shape"] == [2, 64, 64]
    assert metadata["c_order"] is True
    assert "contiguous_data_sha256" not in metadata
    assert "npy_file_sha256" not in metadata


def test_npz_extracts_exact_sides_rows_and_reports_all_storage_counts() -> None:
    current, current_mask, next_labels, next_mask = _synthetic_arrays()
    payload = _npz_bytes(
        current_labels=current,
        current_mask=current_mask,
        next_labels=next_labels,
        next_mask=next_mask,
        versions=((1, 0), (2, 0), (3, 0), (1, 0)),
    )

    decoded = stdlib.decode_fit_label_npz(
        payload,
        selected_rows=(("current", 2), ("next", 1)),
    )

    assert decoded.storage_row_counts == {
        "current_labels.npy": 3,
        "current_supervision_mask.npy": 3,
        "next_labels.npy": 2,
        "next_supervision_mask.npy": 2,
    }
    assert [(row.side, row.row) for row in decoded.selected_rows] == [
        ("current", 2),
        ("next", 1),
    ]
    assert decoded.selected_rows[0].target == current[2].tobytes(order="C")
    assert decoded.selected_rows[0].supervision == current_mask[2].tobytes(
        order="C"
    )
    assert decoded.selected_rows[1].target == next_labels[1].tobytes(order="C")
    evidence = decoded.evidence()
    assert evidence["schema"] == stdlib.NPZ_EVIDENCE_SCHEMA
    assert evidence["archive_member_count"] == 12
    assert evidence["decoded_array_count"] == 4
    assert evidence["arrays_decompressed"] == 4
    assert evidence["auxiliary_member_names"] == list(AUXILIARY_MEMBERS)
    assert evidence["auxiliary_members_decompressed"] is False
    assert evidence["materialized_label_rows"] == 5
    assert evidence["materialized_supervision_rows"] == 5
    assert evidence["materialized_row_totals_agree"] is True
    assert evidence["selected_row_count"] == 2
    assert evidence["selected_label_rows"] == 2
    assert evidence["selected_supervision_rows"] == 2
    assert len(evidence["evidence_sha256"]) == 64
    assert evidence == decoded.evidence()
    assert all(not hasattr(array, "data") for array in decoded.arrays)
    assert all(
        "contiguous_data_sha256" not in array
        and "npy_file_sha256" not in array
        for array in evidence["arrays"]
    )


def test_npz_does_not_inspect_or_retain_unselected_row_values() -> None:
    current, current_mask, next_labels, next_mask = _synthetic_arrays()
    current[0, 0, 0] = 3
    raw_current_mask = _mutate_npy_data(
        _npy_bytes(current_mask), flat_index=0, value=2
    )
    payload = _npz_bytes(
        current_labels=current,
        current_mask=current_mask,
        next_labels=next_labels,
        next_mask=next_mask,
        member_overrides={
            "current_supervision_mask.npy": raw_current_mask,
        },
    )

    decoded = stdlib.decode_fit_label_npz(
        payload,
        selected_rows=(("current", 1),),
    )

    assert decoded.selected_rows[0].target == current[1].tobytes(order="C")
    assert decoded.selected_rows[0].supervision == current_mask[1].tobytes(
        order="C"
    )
    assert all(not hasattr(array, "data") for array in decoded.arrays)


def test_npz_validates_auxiliary_names_without_decompressing_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, current_mask, next_labels, next_mask = _synthetic_arrays()
    payload = _npz_bytes(
        current_labels=current,
        current_mask=current_mask,
        next_labels=next_labels,
        next_mask=next_mask,
    )
    opened: list[str] = []
    original_read = zipfile.ZipFile.read

    def recording_read(
        archive: zipfile.ZipFile,
        member: str | zipfile.ZipInfo,
        pwd: bytes | None = None,
    ) -> bytes:
        opened.append(member.filename if isinstance(member, zipfile.ZipInfo) else member)
        return original_read(archive, member, pwd=pwd)

    monkeypatch.setattr(zipfile.ZipFile, "read", recording_read)
    stdlib.decode_fit_label_npz(
        payload,
        selected_rows=(("current", 0), ("next", 0)),
    )

    assert sorted(opened) == [
        "current_labels.npy",
        "current_supervision_mask.npy",
        "next_labels.npy",
        "next_supervision_mask.npy",
    ]
    assert not set(opened).intersection(AUXILIARY_MEMBERS)


@pytest.mark.parametrize("local_only", (False, True))
def test_npz_rejects_encrypted_auxiliary_metadata_without_opening_it(
    local_only: bool,
) -> None:
    current, current_mask, next_labels, next_mask = _synthetic_arrays()
    payload = _npz_bytes(
        current_labels=current,
        current_mask=current_mask,
        next_labels=next_labels,
        next_mask=next_mask,
    )
    payload = _mark_zip_member_encrypted(
        payload,
        "current_image_path.npy",
        local=True,
        central=not local_only,
    )

    with pytest.raises(ValueError, match="encryption flags"):
        stdlib.decode_fit_label_npz(
            payload,
            selected_rows=(("current", 0),),
        )


def test_npz_rejects_auxiliary_central_local_name_disagreement() -> None:
    current, current_mask, next_labels, next_mask = _synthetic_arrays()
    payload = _npz_bytes(
        current_labels=current,
        current_mask=current_mask,
        next_labels=next_labels,
        next_mask=next_mask,
    )
    payload = _replace_zip_local_member_name(
        payload,
        "current_image_path.npy",
        "a/current_image_path.n",
    )

    with pytest.raises(ValueError, match="disagrees with local name"):
        stdlib.decode_fit_label_npz(
            payload,
            selected_rows=(("current", 0),),
        )


def test_scalar_mapping_and_support_are_byte_identical_to_primary() -> None:
    scalar_mapping = np.asarray(stdlib.build_camera_centered_mapping(), dtype="<i2")
    scalar_support = np.asarray(stdlib.camera_centered_support_mask(), dtype=np.bool_)
    primary_mapping = primary.build_camera_centered_mapping()
    primary_support = primary.camera_centered_support_mask(primary_mapping)

    np.testing.assert_array_equal(scalar_mapping, primary_mapping)
    np.testing.assert_array_equal(scalar_support, primary_support)
    assert int(np.count_nonzero(scalar_support)) == 1990
    assert stdlib.mapping_sha256() == EXPECTED_MAPPING_SHA256
    assert stdlib.support_mask_sha256() == EXPECTED_SUPPORT_SHA256


def test_scalar_frame_support_rays_summaries_and_hashes_equal_primary() -> None:
    mapping = primary.build_camera_centered_mapping()
    support = primary.camera_centered_support_mask(mapping)
    target = np.zeros((64, 64), dtype=np.uint8)
    supported = np.argwhere(support)
    unsupported = np.argwhere(~support)
    target[tuple(supported[0])] = primary.FREE_CLASS
    target[tuple(unsupported[0])] = primary.OCCUPIED_CLASS

    counts = np.bincount(
        mapping[..., 1][mapping[..., 1] >= 0],
        minlength=primary.ANGULAR_BIN_COUNT,
    )
    angular_bin = int(np.argmax(counts))
    locations = np.argwhere(mapping[..., 1] == angular_bin)
    ordered = sorted(
        (int(mapping[row, column, 0]), int(row), int(column))
        for row, column in locations
    )
    sequence = (0, 1, 0, 2, 0, 2, 1)
    for (_, row, column), class_id in zip(ordered, sequence):
        target[row, column] = class_id

    mask = np.ones((64, 64), dtype=np.bool_)
    frame_key = {
        "family": primary.FAMILIES[2],
        "scene_id": "synthetic_independent",
        "global_row": 17,
        "side": "next",
    }
    expected = primary.analyze_frame_labels(
        target,
        mask,
        frame_key=frame_key,
        family=primary.FAMILIES[2],
        endpoint_side="next",
    )
    actual = stdlib.analyze_frame_labels(
        target.tobytes(order="C"),
        mask.tobytes(order="C"),
        frame_key=frame_key,
        family=primary.FAMILIES[2],
        endpoint_side="next",
    )

    assert actual == expected
    assert len(actual["ray_sequences"]["records"]) == 256
    assert actual["ray_sequences"]["sequence_summary_records_sha256"] == (
        expected["ray_sequences"]["sequence_summary_records_sha256"]
    )
    assert actual["ray_sequences"]["transition_table_sha256"] == (
        expected["ray_sequences"]["transition_table_sha256"]
    )


@pytest.mark.parametrize(
    ("array", "expected_dtype", "message"),
    (
        (
            np.zeros((2, 64, 64), dtype=np.int16),
            "uint8",
            "frozen uint8 or bool descriptors",
        ),
        (
            np.zeros((2, 32, 128), dtype=np.uint8),
            "uint8",
            r"shape must be \[N,64,64\]",
        ),
        (
            np.asfortranarray(np.zeros((2, 64, 64), dtype=np.uint8)),
            "uint8",
            "must be C-order",
        ),
    ),
)
def test_npy_rejects_wrong_dtype_shape_and_storage_order(
    array: np.ndarray, expected_dtype: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        stdlib.decode_npy(
            _npy_bytes(array),
            expected_dtype=expected_dtype,
            name="bad.npy",
        )


def test_npy_rejects_trailing_data_and_wrong_expected_kind_without_scanning_values() -> None:
    labels = np.zeros((1, 64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="data length"):
        stdlib.decode_npy(
            _npy_bytes(labels) + b"x",
            expected_dtype="uint8",
            name="trailing.npy",
        )
    with pytest.raises(ValueError, match="is uint8, expected bool"):
        stdlib.decode_npy(
            _npy_bytes(labels),
            expected_dtype="bool",
            name="wrong.npy",
        )

    mask_payload = bytearray(_npy_bytes(np.ones((1, 64, 64), dtype=np.bool_)))
    mask_payload[-1] = 2
    decoded = stdlib.decode_npy(
        bytes(mask_payload),
        expected_dtype="bool",
        name="mask.npy",
    )
    assert decoded.data[-1] == 2


def test_npy_rejects_duplicate_header_keys_before_materialization() -> None:
    payload = _handcrafted_npy(
        "{'descr': '|u1', 'descr': '|u1', 'fortran_order': False, "
        "'shape': (1, 64, 64), }",
        data=bytes(64 * 64),
    )

    with pytest.raises(ValueError, match="repeats key 'descr'"):
        stdlib.decode_npy(
            payload,
            expected_dtype="uint8",
            name="duplicate-header.npy",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            _handcrafted_npy(
                "{'descr': '|O', 'fortran_order': False, "
                "'shape': (1, 64, 64), }",
            ),
            "frozen uint8 or bool descriptors",
        ),
        (
            _handcrafted_npy(
                "{'descr': '>b1', 'fortran_order': False, "
                "'shape': (1, 64, 64), }",
            ),
            "frozen uint8 or bool descriptors",
        ),
        (
            _handcrafted_npy(
                "{'descr': '|u1', 'fortran_order': False, "
                "'shape': (1, 64, 64), }",
                aligned=False,
            ),
            "not aligned to 64 bytes",
        ),
        (
            _handcrafted_npy(
                "{'descr': '|u1', 'fortran_order': False, "
                "'shape': (1, 64, 64), }",
                newline_count=2,
            ),
            "exactly one final newline",
        ),
        (
            _handcrafted_npy(
                "{'descr': '|u1', 'fortran_order': False, "
                "'shape': (1, 64, 64), }",
                padding_byte=b"\t",
            ),
            "illegal whitespace",
        ),
    ),
)
def test_npy_rejects_object_noncanonical_dtype_alignment_and_padding(
    payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        stdlib.decode_npy(
            payload,
            expected_dtype="uint8",
            name="malformed.npy",
        )


def test_npy_accepts_only_frozen_positive_descriptor_boundary() -> None:
    payload = _handcrafted_npy(
        "{'descr': '<u1', 'fortran_order': False, "
        "'shape': (1, 64, 64), }",
        data=bytes(64 * 64),
    )
    decoded = stdlib.decode_npy(
        payload,
        expected_dtype="uint8",
        name="little-endian-u1.npy",
    )
    assert decoded.dtype == "uint8"


def test_npy_rejects_unsupported_version_and_truncated_header() -> None:
    with pytest.raises(ValueError, match="unsupported NPY version"):
        stdlib.decode_npy(
            b"\x93NUMPY" + bytes((4, 0)) + b"\x00\x00",
            expected_dtype="uint8",
            name="version.npy",
        )
    with pytest.raises(ValueError, match="truncates the NPY header"):
        stdlib.decode_npy(
            b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", 64) + b"{}",
            expected_dtype="uint8",
            name="truncated.npy",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "missing=.*next_labels.npy"),
        ("extra", "extra=.*unregistered.npy"),
        ("duplicate", "repeats array members"),
    ),
)
def test_npz_rejects_missing_extra_and_duplicate_arrays(
    mutation: str, message: str
) -> None:
    current, current_mask, next_labels, next_mask = _synthetic_arrays()
    kwargs: dict[str, object] = {}
    if mutation == "missing":
        kwargs["omit"] = "next_labels.npy"
    elif mutation == "extra":
        kwargs["extra"] = ("unregistered.npy", _npy_bytes(current))
    else:
        kwargs["duplicate"] = "current_labels.npy"
    payload = _npz_bytes(
        current_labels=current,
        current_mask=current_mask,
        next_labels=next_labels,
        next_mask=next_mask,
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        stdlib.decode_fit_label_npz(
            payload,
            selected_rows=(("current", 0),),
        )


def test_npz_rejects_dtype_pair_count_selection_and_selected_value_mutations() -> None:
    current, current_mask, next_labels, next_mask = _synthetic_arrays()
    with pytest.raises(ValueError, match="expected bool"):
        stdlib.decode_fit_label_npz(
            _npz_bytes(
                current_labels=current,
                current_mask=current_mask.astype(np.uint8),
                next_labels=next_labels,
                next_mask=next_mask,
            ),
            selected_rows=(("current", 0),),
        )

    with pytest.raises(ValueError, match="storage row counts differ"):
        stdlib.decode_fit_label_npz(
            _npz_bytes(
                current_labels=current,
                current_mask=current_mask[:2],
                next_labels=next_labels,
                next_mask=next_mask,
            ),
            selected_rows=(("current", 0),),
        )

    malformed_labels = current.copy()
    malformed_labels[2, 0, 0] = 3
    with pytest.raises(ValueError, match="class outside 0/1/2"):
        stdlib.decode_fit_label_npz(
            _npz_bytes(
                current_labels=malformed_labels,
                current_mask=current_mask,
                next_labels=next_labels,
                next_mask=next_mask,
            ),
            selected_rows=(("current", 2),),
        )

    partial_mask = current_mask.copy()
    partial_mask[1, 0, 0] = False
    with pytest.raises(ValueError, match="not the full bool grid"):
        stdlib.decode_fit_label_npz(
            _npz_bytes(
                current_labels=current,
                current_mask=partial_mask,
                next_labels=next_labels,
                next_mask=next_mask,
            ),
            selected_rows=(("current", 1),),
        )

    noncanonical_mask_payload = bytearray(_npy_bytes(current_mask))
    noncanonical_mask_payload[-1] = 2
    with pytest.raises(ValueError, match="noncanonical byte"):
        stdlib.decode_fit_label_npz(
            _npz_bytes(
                current_labels=current,
                current_mask=current_mask,
                next_labels=next_labels,
                next_mask=next_mask,
                member_overrides={
                    "current_supervision_mask.npy": bytes(
                        noncanonical_mask_payload
                    )
                },
            ),
            selected_rows=(("current", 2),),
        )

    valid_payload = _npz_bytes(
        current_labels=current,
        current_mask=current_mask,
        next_labels=next_labels,
        next_mask=next_mask,
    )
    with pytest.raises(IndexError, match="outside storage"):
        stdlib.decode_fit_label_npz(
            valid_payload,
            selected_rows=(("next", 2),),
        )
    with pytest.raises(ValueError, match="repeats"):
        stdlib.decode_fit_label_npz(
            valid_payload,
            selected_rows=(("next", 1), ("next", 1)),
        )


def test_clean_system_python_imports_reference_without_numpy_or_torch() -> None:
    root = Path(__file__).resolve().parents[2]
    module_path = (
        root
        / "lewm/benchmarks/go2_n32_camera_frustum_fit_evidence_stdlib.py"
    )
    script = (
        "import importlib.util,sys; "
        f"p={str(module_path)!r}; "
        "s=importlib.util.spec_from_file_location('stdlib_fit_evidence',p); "
        "m=importlib.util.module_from_spec(s); "
        "sys.modules[s.name]=m; "
        "s.loader.exec_module(m); "
        "assert 'numpy' not in sys.modules; "
        "assert 'torch' not in sys.modules; "
        "assert m.mapping_sha256() == "
        f"{EXPECTED_MAPPING_SHA256!r}"
    )
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
