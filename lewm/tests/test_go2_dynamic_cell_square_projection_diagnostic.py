from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from lewm.benchmarks import go2_dynamic_cell_square_projection_diagnostic as core
from lewm.benchmarks.go2_dynamic_cell_square_projection import support_mask_sha256
from scripts import diagnose_go2_dynamic_cell_square_projection as runner


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path_record(path: str, role: str, digest: str | None) -> dict:
    return {"path": path, "role": role, "sha256": digest}


def _ledger(phase: str) -> dict:
    source_map = _source_map()
    sources = {entry["role"]: entry for entry in source_map["entries"]}
    manifests = _manifests()
    shards = [
        _path_record(entry["path"], "label_shard", entry["sha256"])
        for entry in _shard_entries()
    ]
    if phase == "preparation":
        reads = [
            _path_record(
                str(core.REPOSITORY_ROOT / core.BINDING_RELATIVE_PATH),
                "binding",
                core.BINDING_SHA256,
            ),
            _path_record(
                str(core.REPOSITORY_ROOT / core.PREDECESSOR_REPORT_RELATIVE_PATH),
                "predecessor_report",
                core.PREDECESSOR_REPORT_SHA256,
            ),
            _path_record(
                str(core.REPOSITORY_ROOT / core.PREDECESSOR_RESULT_RELATIVE_PATH),
                "predecessor_result",
                core.PREDECESSOR_RESULT_FILE_SHA256,
            ),
            _path_record(
                str(core.REPOSITORY_ROOT / manifests["human"]["path"]),
                "human_manifest",
                manifests["human"]["file_sha256"],
            ),
            *[
                _path_record(
                    str(core.REPOSITORY_ROOT / entry["path"]),
                    role,
                    entry["sha256"],
                )
                for role, entry in sources.items()
            ],
            *shards,
        ]
    else:
        reads = [
            _path_record(
                str(core.REPOSITORY_ROOT / core.BINDING_RELATIVE_PATH),
                "binding",
                core.BINDING_SHA256,
            ),
            _path_record(
                str(core.REPOSITORY_ROOT / manifests["human"]["path"]),
                "human_manifest",
                manifests["human"]["file_sha256"],
            ),
            _path_record(
                str(core.REPOSITORY_ROOT / manifests["machine"]["path"]),
                "machine_manifest",
                manifests["machine"]["file_sha256"],
            ),
            _path_record(
                str(core.REPOSITORY_ROOT / core.PREDECESSOR_RESULT_RELATIVE_PATH),
                "predecessor_result",
                core.PREDECESSOR_RESULT_FILE_SHA256,
            ),
            *[
                _path_record(
                    str(core.REPOSITORY_ROOT / sources[role]["path"]),
                    role,
                    sources[role]["sha256"],
                )
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
            *shards,
        ]
    reads.sort(key=lambda item: (item["path"], item["role"]))
    output = (
        core.MACHINE_MANIFEST_RELATIVE_PATH
        if phase == "preparation"
        else core.CANDIDATE_RELATIVE_PATH
    )
    writes = [
        _path_record(
            str(core.REPOSITORY_ROOT / output),
            (
                "machine_manifest_output"
                if phase == "preparation"
                else "runner_output"
            ),
            None,
        )
    ]
    label_opens = 0 if phase == "preparation" else 20
    roles = {record["role"] for record in reads}
    return {
        "schema": core.ACCESS_LEDGER_SCHEMA,
        "phase": phase,
        "authorized_read_paths": reads,
        "authorized_read_path_set_sha256": core.canonical_json_sha256(reads),
        "authorized_write_paths": writes,
        "authorized_write_path_set_sha256": core.canonical_json_sha256(writes),
        "role_byte_open_counts": {
            role: (
                0 if role == "label_shard" and phase == "preparation"
                else 40 if role == "label_shard"
                else 1
            )
            for role in roles
        },
        "label_shard_pre_hash_byte_opens": label_opens,
        "label_shard_post_hash_byte_opens": label_opens,
        "label_shard_npz_parses": label_opens,
        "array_decompression_counts": (
            {}
            if phase == "preparation"
            else {"current_labels": 20, "next_labels": 20}
        ),
        "selected_label_rows_read": 0 if phase == "preparation" else 320,
        "unselected_rows_scored": 0,
        "unselected_rows_retained": 0,
        "metadata_only_shard_stats": 20 if phase == "preparation" else 0,
        "denied_attempt_records": [],
        "denied_reason_counts": {
            reason: 0 for reason in core.DENIED_REASON_ORDER
        },
        "unexpected_path_attempts": 0,
        "forbidden_role_open_counts": {
            role: 0 for role in core.FORBIDDEN_ROLES
        },
        "all_counts_reconcile": True,
    }


def _source_map() -> dict:
    entries = [
        {
            "role": role,
            "path": path,
            "sha256": (
                core.DYNAMIC_GEOMETRY_SHA256
                if role == "dynamic_geometry"
                else _sha(role)
            ),
        }
        for role, path in core.SOURCE_MAP_CONTRACT
    ]
    return {
        "entries": entries,
        "entry_count": len(entries),
        "source_map_sha256": core.canonical_json_sha256(entries),
    }


def _manifests() -> dict:
    return {
        "human": {
            "path": core.HUMAN_MANIFEST_RELATIVE_PATH,
            "file_sha256": _sha("human"),
        },
        "machine": {
            "path": core.MACHINE_MANIFEST_RELATIVE_PATH,
            "file_sha256": _sha("machine"),
            "content_sha256": _sha("machine-content"),
        },
    }


def _shard_entries() -> list[dict]:
    return [
        {
            "path": str(
                core.REPOSITORY_ROOT / f".synthetic/shard_{index:02d}.npz"
            ),
            "sha256": _sha(f"shard-{index}"),
        }
        for index in range(20)
    ]


def _passing_scientific() -> dict:
    label = {
        "byte_count": core.EXPECTED_SELECTED_TARGET_BYTE_COUNT,
        "byte_sha256": core.EXPECTED_SELECTED_TARGET_SHA256,
        "class_totals": dict(core.EXPECTED_CLASS_TOTALS),
        "known_total": core.EXPECTED_KNOWN_TOTAL,
        "per_frame_cell_count": 4096,
        "per_frame_count": 320,
        "per_frame_totals_sha256": _sha("per-frame"),
        "all_counts_reconcile": True,
    }
    support = {
        "level_center": {
            "support_cell_count": core.EXPECTED_LEVEL_CENTER_SUPPORT_COUNT,
            "support_mask_sha256": core.EXPECTED_LEVEL_CENTER_SUPPORT_SHA256,
            "free_total": core.EXPECTED_CLASS_TOTALS["free"],
            "free_supported": core.EXPECTED_LEVEL_CENTER_FREE_SUPPORTED,
            "occupied_total": core.EXPECTED_CLASS_TOTALS["occupied"],
            "occupied_supported": core.EXPECTED_LEVEL_CENTER_OCCUPIED_SUPPORTED,
            "known_violation_count": core.EXPECTED_CENTER_VIOLATION_COUNT,
            "known_violation_identities_sha256": (
                core.EXPECTED_CENTER_VIOLATION_IDENTITIES_SHA256
            ),
        },
        "level_cell_square": {
            "support_cell_count": core.EXPECTED_LEVEL_CELL_SQUARE_SUPPORT_COUNT,
            "support_mask_sha256": core.EXPECTED_LEVEL_CELL_SQUARE_SUPPORT_SHA256,
        },
        "static_cell_square_known": {
            "known_total": core.EXPECTED_KNOWN_TOTAL,
            "supported_count": core.EXPECTED_STATIC_SUPPORTED_COUNT,
            "unsupported_count": 4,
            "unsupported_free_count": 0,
            "unsupported_occupied_count": 4,
            "unsupported_frame_count": 4,
            "unsupported_identities_sha256": (
                core.EXPECTED_STATIC_UNSUPPORTED_IDENTITIES_SHA256
            ),
        },
        "dynamic_cell_square_known": {
            "known_total": core.EXPECTED_KNOWN_TOTAL,
            "supported_count": core.EXPECTED_KNOWN_TOTAL,
            "unsupported_count": 0,
            "unsupported_free_count": 0,
            "unsupported_occupied_count": 0,
            "unsupported_frame_count": 0,
            "unsupported_identities_sha256": core.EMPTY_LIST_SHA256,
        },
    }
    rows = [
        {
            "family": family,
            "class_id": core.CLASS_IDS[class_name],
            "class_name": class_name,
            "total": 0,
            "level_center_supported": 0,
            "static_cell_square_supported": 0,
            "dynamic_cell_square_supported": 0,
        }
        for family in core.FAMILY_ORDER
        for class_name in core.KNOWN_CLASS_ORDER
    ]
    rows[0]["total"] = core.EXPECTED_CLASS_TOTALS["free"]
    rows[1]["total"] = core.EXPECTED_CLASS_TOTALS["occupied"]
    return {
        "label_reconciliation": label,
        "support": support,
        "family_class_rows": rows,
        "frame_summary_records_sha256": _sha("frame-summary"),
    }


def _synthetic_npz(
    *,
    rows: int = 2,
    dtype: np.dtype = np.dtype("uint8"),
    invalid_class: bool = False,
    extra_member: bool = False,
    current_value: int = 0,
    next_value: int = 0,
) -> bytes:
    labels = np.full((rows, 64, 64), current_value, dtype=dtype)
    if invalid_class:
        labels[0, 0, 0] = 3
    arrays: dict[str, np.ndarray] = {
        "current_labels": labels,
        "current_supervision_mask": np.ones_like(labels, dtype=bool),
        "next_labels": np.full((rows, 64, 64), next_value, dtype=dtype),
        "next_supervision_mask": np.ones_like(labels, dtype=bool),
        "current_observed_mask": np.ones_like(labels, dtype=bool),
        "next_observed_mask": np.ones_like(labels, dtype=bool),
        "relative_se2_current_frame": np.zeros((rows, 3), dtype=np.float32),
        "primitive": np.zeros((rows,), dtype=np.int64),
        "current_image_path": np.array(["a"] * rows),
        "next_image_path": np.array(["b"] * rows),
        "current_image_sha256": np.array([_sha("a")] * rows),
        "next_image_sha256": np.array([_sha("b")] * rows),
    }
    if extra_member:
        arrays["extra"] = np.zeros((1,), dtype=np.uint8)
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _synthetic_entry() -> dict:
    return {
        "path": "/repo/shard.npz",
        "sha256": _sha("shard"),
        "selected_tuples": [
            ["open_obstacle_field", "synthetic_scene", 7, "current", 0],
            ["open_obstacle_field", "synthetic_scene", 7, "next", 0],
        ],
        "selected_row_count": 2,
        "family_side_counts": {},
    }


def _decode_state(entry: dict) -> tuple[dict[tuple[object, ...], int], bytearray, bytearray]:
    ranks = {
        (*selected, entry["sha256"]): rank
        for rank, selected in enumerate(entry["selected_tuples"])
    }
    return (
        ranks,
        bytearray(runner.EXPECTED_TARGET_BYTES),
        bytearray(runner.EXPECTED_TARGET_ROWS),
    )


def test_independent_level_center_support_reproduces_predecessor_geometry() -> None:
    mask = core.build_independent_level_center_support_mask()
    assert sum(sum(row) for row in mask) == 1990
    assert support_mask_sha256(mask) == core.EXPECTED_LEVEL_CENTER_SUPPORT_SHA256


def test_candidate_gates_require_independent_finalization() -> None:
    scientific = _passing_scientific()
    gates = core.scientific_gates(
        scientific,
        access_reconciliation_pass=True,
        independent_recomputation_pass=False,
    )
    assert all(
        value
        for key, value in gates.items()
        if key not in {"independent_recomputation_pass", "all_passed"}
    )
    assert gates["independent_recomputation_pass"] is False
    assert gates["all_passed"] is False


def test_candidate_exact_schema_round_trips_without_identity_leakage() -> None:
    candidate = core.build_candidate(
        created_at_utc="2026-07-11T00:00:00+00:00",
        implementation_manifests=_manifests(),
        source_map=_source_map(),
        preparation_access_ledger=_ledger("preparation"),
        runner_access_ledger=_ledger("runner"),
        scientific=_passing_scientific(),
        label_shard_entries=_shard_entries(),
    )
    reparsed = json.loads(core.canonical_json_bytes(candidate))
    assert core.validate_content_sha256(reparsed) == candidate["content_sha256"]
    core.validate_access_ledger(
        reparsed["preparation_access_ledger"], expected_phase="preparation"
    )
    serialized = core.canonical_json_bytes(reparsed)
    for forbidden in (
        b'"scene_id":',
        b'"global_row":',
        b'"label_row":',
        b'"image_sha256":',
        b'"base_quat_world_xyzw":',
        b'"stored_base_yaw_rad":',
        b'"remaining_identities":',
    ):
        assert forbidden not in serialized


def test_ledger_round_trip_uses_reason_sets_not_object_insertion_order() -> None:
    ledger = _ledger("runner")
    ledger["denied_reason_counts"] = {
        key: ledger["denied_reason_counts"][key]
        for key in reversed(core.DENIED_REASON_ORDER)
    }
    ledger["forbidden_role_open_counts"] = {
        key: 0 for key in reversed(core.FORBIDDEN_ROLES)
    }
    reparsed = json.loads(core.canonical_json_bytes(ledger))
    assert core.validate_access_ledger(reparsed, expected_phase="runner") == reparsed


def test_ledger_rejects_noncanonical_scalars_and_denial_mismatch() -> None:
    ledger = _ledger("runner")
    ledger["selected_label_rows_read"] = True
    with pytest.raises(core.DiagnosticContractError, match="exact integer"):
        core.validate_access_ledger(ledger, expected_phase="runner")

    ledger = _ledger("runner")
    ledger["denied_reason_counts"]["unallowlisted"] = 1
    with pytest.raises(core.DiagnosticContractError, match="do not reconcile"):
        core.validate_access_ledger(ledger, expected_phase="runner")


def test_content_hash_and_exact_container_types_fail_closed() -> None:
    payload = core.with_content_sha256({"schema": "synthetic"})
    assert core.validate_content_sha256(payload) == payload["content_sha256"]
    payload["schema"] = "changed"
    with pytest.raises(core.DiagnosticContractError, match="does not match"):
        core.validate_content_sha256(payload)

    class DictSubclass(dict):
        pass

    with pytest.raises(core.DiagnosticContractError, match="exact"):
        core.validate_content_sha256(DictSubclass(payload))


def test_recursive_type_exact_equality_rejects_nested_scalar_coercions() -> None:
    expected = {"family_side_counts": {"family": {"current": 1}}}
    assert core.type_exact_equal(expected, copy.deepcopy(expected))
    assert not core.type_exact_equal(
        {"family_side_counts": {"family": {"current": True}}}, expected
    )
    assert not core.type_exact_equal(
        {"family_side_counts": {"family": {"current": 1.0}}}, expected
    )


@pytest.mark.parametrize("replacement", (False, 0.0))
def test_label_manifest_rejects_nested_family_count_scalar_coercion(
    replacement: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = []
    for shard_index in range(20):
        counts = {
            family: {side: 0 for side in core.SIDE_ORDER}
            for family in core.FAMILY_ORDER
        }
        counts["open_obstacle_field"]["current"] = 16
        entries.append(
            {
                "path": str(
                    core.REPOSITORY_ROOT
                    / f".synthetic/type_shard_{shard_index:02d}.npz"
                ),
                "sha256": _sha(f"type-shard-{shard_index}"),
                "selected_tuples": [
                    [
                        "open_obstacle_field",
                        f"scene-{shard_index}",
                        shard_index * 16 + row,
                        "current",
                        row,
                    ]
                    for row in range(16)
                ],
                "selected_row_count": 16,
                "family_side_counts": counts,
            }
        )
    entries[0]["family_side_counts"]["large_enclosed_maze"][
        "next"
    ] = replacement
    digest = core.canonical_json_sha256(entries)
    monkeypatch.setattr(core, "EXPECTED_LABEL_SHARD_MANIFEST_SHA256", digest)
    with pytest.raises(core.DiagnosticContractError, match="family-side"):
        core.validate_label_shard_manifest(
            {
                "entries": entries,
                "entry_count": 20,
                "manifest_sha256": digest,
            }
        )


@pytest.mark.parametrize(
    "path",
    (
        "docs//synthetic.json",
        "docs/./synthetic.json",
        "docs/other/../synthetic.json",
        "docs/synthetic.json/",
        "/etc/passwd",
        "/etc/shadow",
    ),
)
def test_runner_rejects_raw_path_aliases_and_outside_paths(path: str) -> None:
    with pytest.raises(PermissionError):
        runner._lexically_safe_absolute(path)


def test_runner_denial_telemetry_uses_frozen_reason_precedence() -> None:
    assert runner.primary_denial_reason(
        ["hash_mismatch", "forbidden_role", "unallowlisted"]
    ) == "unallowlisted"
    ledger = _ledger("runner")
    primary = runner.record_denial(
        ledger,
        requested_role="heldout",
        declared_role="model_output",
        lexical_path="docs//alias",
        resolved_path=None,
        reasons=["modality_mismatch", "forbidden_role", "path_alias_or_escape"],
    )
    assert primary == "path_alias_or_escape"
    assert ledger["denied_attempt_records"][-1]["primary_reason"] == primary
    assert ledger["denied_reason_counts"][primary] == 1
    assert ledger["unexpected_path_attempts"] == 1


def test_runner_records_unallowlisted_denial_before_byte_open() -> None:
    ledger = _ledger("runner")
    missing = runner.ROOT / ".synthetic/not_authorized.json"
    with pytest.raises(PermissionError, match="unallowlisted"):
        runner._read_authorized(
            missing,
            role="runner",
            allowlist={},
            ledger=ledger,
        )
    assert ledger["role_byte_open_counts"]["runner"] == 1
    assert ledger["denied_attempt_records"][-1]["primary_reason"] == "unallowlisted"


def test_runner_reads_the_validated_anchored_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "scripts/diagnose_go2_dynamic_cell_square_projection.py"
    absolute = runner.ROOT / relative
    payload = absolute.read_bytes()
    record = {
        "path": str(absolute),
        "role": "runner",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.chdir(tmp_path)
    assert runner._read_authorized(
        Path(relative),
        role="runner",
        allowlist={str(absolute): record},
        ledger=_ledger("runner"),
    ) == payload


@pytest.mark.parametrize(
    "suffix",
    (
        "docs//synthetic.json",
        "docs/./synthetic.json",
        "docs/other/../synthetic.json",
        "docs/synthetic.json/",
    ),
)
def test_core_rejects_raw_absolute_path_aliases(suffix: str) -> None:
    with pytest.raises(core.DiagnosticContractError, match="canonical"):
        core.canonical_repository_path(
            f"{core.REPOSITORY_ROOT}/{suffix}", name="synthetic"
        )
    with pytest.raises(core.DiagnosticContractError, match="outside"):
        core.canonical_repository_path("/etc/passwd", name="synthetic")


@pytest.mark.parametrize(
    "timestamp",
    (True, 1.0, "2026-07-11T00:00:00", "2026-07-11T00:00:00Z"),
)
def test_candidate_builder_rejects_noncanonical_timestamp_types(
    timestamp: object,
) -> None:
    with pytest.raises(core.DiagnosticContractError):
        core.build_candidate(
            created_at_utc=timestamp,  # type: ignore[arg-type]
            implementation_manifests=_manifests(),
            source_map=_source_map(),
            preparation_access_ledger=_ledger("preparation"),
            runner_access_ledger=_ledger("runner"),
            scientific=_passing_scientific(),
            label_shard_entries=_shard_entries(),
        )


def test_strict_json_rejects_duplicates_nonfinite_and_nonobject() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        runner.strict_json_bytes(b'{"a":1,"a":2}', name="synthetic")
    with pytest.raises(ValueError, match="nonfinite"):
        runner.strict_json_bytes(b'{"a":NaN}', name="synthetic")
    with pytest.raises(ValueError, match="root"):
        runner.strict_json_bytes(b"[]", name="synthetic")


def test_manifest_self_template_substitution_is_single_and_postverification() -> None:
    template = [
        {"path": "a", "role": "a", "sha256": _sha("a")},
        dict(runner.SELF_TEMPLATE_ENTRY),
    ]
    digest = _sha("manifest")
    with pytest.raises(ValueError, match="verified before"):
        runner.instantiate_read_template(
            template,
            verified_manifest_sha256=digest,
            manifest_bytes_verified=False,
        )
    actual = runner.instantiate_read_template(
        template,
        verified_manifest_sha256=digest,
        manifest_bytes_verified=True,
    )
    assert actual[-1] == {
        "path": runner.MACHINE_MANIFEST_RELATIVE_PATH,
        "role": "machine_manifest",
        "sha256": digest,
    }

    for mutation in (
        [*template, dict(runner.SELF_TEMPLATE_ENTRY)],
        [
            template[0],
            {**runner.SELF_TEMPLATE_ENTRY, "role": "wrong"},
        ],
        [
            template[0],
            {
                "path": runner.MACHINE_MANIFEST_RELATIVE_PATH,
                "role": "machine_manifest",
                "sha256": digest,
            },
        ],
    ):
        with pytest.raises(ValueError, match="placeholder"):
            runner.instantiate_read_template(
                mutation,
                verified_manifest_sha256=digest,
                manifest_bytes_verified=True,
            )


def test_synthetic_npz_decodes_only_registered_label_arrays() -> None:
    entry = _synthetic_entry()
    ranks, target_buffer, rank_filled = _decode_state(entry)
    lifetime: list[dict] = []
    selected_count, decompressions = runner.decode_selected_label_rows(
        _synthetic_npz(),
        entry=entry,
        np=np,
        rank_by_identity=ranks,
        target_buffer=target_buffer,
        rank_filled=rank_filled,
        lifetime_events=lifetime,
    )
    assert selected_count == 2
    assert rank_filled[:2] == b"\x01\x01"
    assert decompressions == {"current_labels": 1, "next_labels": 1}
    assert [event["event"] for event in lifetime] == [
        "array_released",
        "array_released",
        "archive_released",
    ]
    assert all(event["selected_row_copies_retained"] == 0 for event in lifetime)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"dtype": np.dtype("int16")}, "dtype"),
        ({"invalid_class": True}, "classes"),
        ({"extra_member": True}, "inventory"),
    ),
)
def test_synthetic_npz_malformed_labels_fail_closed(
    kwargs: dict, message: str
) -> None:
    entry = _synthetic_entry()
    ranks, target_buffer, rank_filled = _decode_state(entry)
    with pytest.raises(ValueError, match=message):
        runner.decode_selected_label_rows(
            _synthetic_npz(**kwargs),
            entry=entry,
            np=np,
            rank_by_identity=ranks,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
        )


def test_synthetic_npz_rejects_duplicate_zip_member() -> None:
    raw = _synthetic_npz()
    source = zipfile.ZipFile(io.BytesIO(raw), "r")
    stream = io.BytesIO()
    with source, zipfile.ZipFile(stream, "w") as output:
        for info in source.infolist():
            output.writestr(info.filename, source.read(info.filename))
        with pytest.warns(UserWarning, match="Duplicate name"):
            output.writestr("current_labels.npy", source.read("current_labels.npy"))
    with pytest.raises(ValueError, match="inventory"):
        runner.validate_npz_inventory(stream.getvalue(), name="synthetic")


def test_selected_tuple_duplicate_and_out_of_range_fail_closed() -> None:
    entry = _synthetic_entry()
    entry["selected_tuples"].append(copy.deepcopy(entry["selected_tuples"][0]))
    ranks, target_buffer, rank_filled = _decode_state(entry)
    with pytest.raises(ValueError, match="duplicated"):
        runner.decode_selected_label_rows(
            _synthetic_npz(),
            entry=entry,
            np=np,
            rank_by_identity=ranks,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
        )

    entry = _synthetic_entry()
    entry["selected_tuples"][0][-1] = 99
    ranks, target_buffer, rank_filled = _decode_state(entry)
    with pytest.raises(ValueError, match="outside"):
        runner.decode_selected_label_rows(
            _synthetic_npz(),
            entry=entry,
            np=np,
            rank_by_identity=ranks,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
        )


def test_runner_releases_each_shard_before_opening_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        {
            "path": str(runner.ROOT / f".synthetic/lifetime_{index}.npz"),
            "sha256": _sha(f"lifetime-{index}"),
            "selected_tuples": [
                ["open_obstacle_field", f"scene-{index}", index, "current", 0],
                ["open_obstacle_field", f"scene-{index}", index, "next", 0],
            ],
        }
        for index in range(2)
    ]
    identities = [
        (*selected, entry["sha256"])
        for entry in entries
        for selected in entry["selected_tuples"]
    ]
    canonical_order = [identities[3], identities[0], identities[2], identities[1]]
    predecessor = {
        "frame_reports": [
            {
                "record_key": {
                    "family": identity[0],
                    "scene_id": identity[1],
                    "global_row": identity[2],
                    "side": identity[3],
                    "label_row": identity[4],
                    "label_shard_sha256": identity[5],
                }
            }
            for identity in canonical_order
        ]
    }
    row = runner.TARGET_ROW_BYTES
    expected = bytes([1]) * row + bytes([1]) * row + bytes([2]) * row + bytes([2]) * row
    monkeypatch.setattr(runner, "EXPECTED_TARGET_ROWS", 4)
    monkeypatch.setattr(runner, "EXPECTED_TARGET_BYTES", len(expected))
    monkeypatch.setattr(runner, "EXPECTED_TARGET_SHA256", hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(runner, "_manifest_entries", lambda _predecessor: entries)
    raw_by_path = {
        entries[0]["path"]: _synthetic_npz(current_value=1, next_value=2),
        entries[1]["path"]: _synthetic_npz(current_value=2, next_value=1),
    }

    def fake_read(
        path: Path,
        *,
        role: str,
        allowlist: dict,
        ledger: dict,
    ) -> bytes:
        assert role == "label_shard"
        ledger["role_byte_open_counts"][role] += 1
        return raw_by_path[str(path)]

    monkeypatch.setattr(runner, "_read_authorized", fake_read)
    allowlist = {
        entry["path"]: {
            "path": entry["path"],
            "role": "label_shard",
            "sha256": entry["sha256"],
        }
        for entry in entries
    }
    ledger = {
        "role_byte_open_counts": {"label_shard": 0},
        "label_shard_pre_hash_byte_opens": 0,
        "label_shard_post_hash_byte_opens": 0,
        "label_shard_npz_parses": 0,
        "array_decompression_counts": {"current_labels": 0, "next_labels": 0},
        "selected_label_rows_read": 0,
    }
    lifetime: list[dict] = []
    assert runner.load_ordered_targets(
        predecessor,
        np=np,
        allowlist=allowlist,
        ledger=ledger,
        lifetime_events=lifetime,
    ) == expected
    assert [event["event"] for event in lifetime] == [
        "shard_open",
        "array_released",
        "array_released",
        "archive_released",
        "shard_released",
        "shard_open",
        "array_released",
        "array_released",
        "archive_released",
        "shard_released",
    ]
    assert all(
        event.get("selected_row_copies_retained", 0) == 0
        for event in lifetime
    )

def test_runner_cli_has_only_the_manifest_hash_argument() -> None:
    args = runner._parse_args(
        ["--implementation-manifest-sha256", _sha("manifest")]
    )
    assert vars(args) == {"implementation_manifest_sha256": _sha("manifest")}
    with pytest.raises(SystemExit):
        runner._parse_args([])
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--implementation-manifest-sha256",
                _sha("manifest"),
                "--output",
                "alternate",
            ]
        )


def test_exclusive_writer_never_replaces_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "candidate.json"
    runner._write_exclusive(output, b"first\n")
    with pytest.raises(FileExistsError):
        runner._write_exclusive(output, b"second\n")
    assert output.read_bytes() == b"first\n"
