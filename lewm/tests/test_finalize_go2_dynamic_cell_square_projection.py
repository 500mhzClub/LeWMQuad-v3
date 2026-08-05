from __future__ import annotations

import copy
import ast
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from lewm.benchmarks import go2_dynamic_cell_square_projection_diagnostic as core
from lewm.tests import test_go2_dynamic_cell_square_projection_diagnostic as diagnostic_suite
from scripts import finalize_go2_dynamic_cell_square_projection as finalizer
from scripts import prepare_go2_dynamic_cell_square_projection as preparation


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _npz(
    *,
    dtype: np.dtype = np.dtype("uint8"),
    invalid: bool = False,
    current_value: int = 0,
    next_value: int = 0,
) -> bytes:
    labels = np.full((2, 64, 64), current_value, dtype=dtype)
    if invalid:
        labels[0, 0, 0] = 3
    arrays = {
        "current_labels": labels,
        "current_supervision_mask": np.ones_like(labels, dtype=bool),
        "next_labels": np.full((2, 64, 64), next_value, dtype=dtype),
        "next_supervision_mask": np.ones_like(labels, dtype=bool),
        "current_observed_mask": np.ones_like(labels, dtype=bool),
        "next_observed_mask": np.ones_like(labels, dtype=bool),
        "relative_se2_current_frame": np.zeros((2, 3), dtype=np.float32),
        "primitive": np.zeros((2,), dtype=np.int64),
        "current_image_path": np.array(["a", "b"]),
        "next_image_path": np.array(["c", "d"]),
        "current_image_sha256": np.array([_sha("a"), _sha("b")]),
        "next_image_sha256": np.array([_sha("c"), _sha("d")]),
    }
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _entry() -> dict:
    return {
        "sha256": _sha("synthetic-shard"),
        "selected_tuples": [
            ["open_obstacle_field", "synthetic", 1, "current", 0],
            ["open_obstacle_field", "synthetic", 1, "next", 0],
        ]
    }


def _decode_state(entry: dict) -> tuple[dict[tuple[object, ...], int], bytearray, bytearray]:
    ranks = {
        (*selected, entry["sha256"]): rank
        for rank, selected in enumerate(entry["selected_tuples"])
    }
    return (
        ranks,
        bytearray(finalizer.EXPECTED_TARGET_BYTES),
        bytearray(finalizer.EXPECTED_TARGET_ROWS),
    )


def _machine_manifest() -> dict:
    human_hash = _sha("human")
    sources = [
        {
            "role": role,
            "path": path,
            "sha256": (
                finalizer.DYNAMIC_GEOMETRY_SHA256
                if role == "dynamic_geometry"
                else _sha(role)
            ),
        }
        for role, path in preparation.SOURCE_MAP
    ]
    shards = [
        {
            "path": str(finalizer.ROOT / f".synthetic/shard_{index:02d}.npz"),
            "sha256": _sha(f"shard-{index}"),
            "selected_tuples": [],
        }
        for index in range(20)
    ]
    runner_template = preparation._read_template(
        phase="runner",
        source_entries=sources,
        shard_entries=shards,
        human_sha256=human_hash,
    )
    finalizer_template = preparation._read_template(
        phase="finalizer",
        source_entries=sources,
        shard_entries=shards,
        human_sha256=human_hash,
    )
    source_map = {
        "entries": sources,
        "entry_count": 9,
        "source_map_sha256": preparation.canonical_json_sha256(sources),
    }
    core = {
        "schema": finalizer.MACHINE_SCHEMA,
        "created_at_utc": "2026-07-11T00:00:00+00:00",
        "execution_binding": {
            "path": finalizer.BINDING_PATH,
            "file_sha256": finalizer.BINDING_SHA256,
        },
        "human_manifest": {
            "path": preparation.HUMAN_MANIFEST_RELATIVE_PATH,
            "file_sha256": human_hash,
        },
        "inputs": {
            "predecessor_report": {
                "path": finalizer.PREDECESSOR_REPORT_PATH,
                "file_sha256": finalizer.PREDECESSOR_REPORT_SHA256,
            },
            "predecessor_result": {
                "path": finalizer.PREDECESSOR_RESULT_PATH,
                "file_sha256": finalizer.PREDECESSOR_FILE_SHA256,
                "content_sha256": finalizer.PREDECESSOR_CONTENT_SHA256,
            },
            "label_shard_manifest": {
                "entry_count": 20,
                "manifest_sha256": finalizer.EXPECTED_LABEL_MANIFEST_SHA256,
            },
            "selected_targets": {
                "frame_count": 320,
                "byte_count": finalizer.EXPECTED_TARGET_BYTES,
                "sha256": finalizer.EXPECTED_TARGET_SHA256,
            },
        },
        "source_map": source_map,
        "phase_contracts": {
            "runner": preparation._phase_contract(
                phase="runner",
                template=runner_template,
                output_relative_path=preparation.CANDIDATE_RELATIVE_PATH,
            ),
            "finalizer": preparation._phase_contract(
                phase="finalizer",
                template=finalizer_template,
                output_relative_path=preparation.FINAL_RESULT_RELATIVE_PATH,
            ),
        },
        "preparation_access_ledger": diagnostic_suite._ledger("preparation"),
        "output_absence": {
            "paths": [
                {
                    "path": str(finalizer.ROOT / relative),
                    "exists": False,
                }
                for relative in (
                    finalizer.CANDIDATE_PATH,
                    finalizer.RESULT_PATH,
                    finalizer.FAILURE_PATH,
                )
            ],
            "all_absent": True,
        },
        "runtime_environment": {
            "python_implementation": sys.implementation.name,
            "python_version": list(sys.version_info[:3]),
            "numpy_version": importlib.metadata.version("numpy"),
        },
    }
    return {**core, "content_sha256": finalizer.canonical_hash(core)}


def _rehash(record: dict) -> None:
    record.pop("content_sha256", None)
    record["content_sha256"] = finalizer.canonical_hash(record)


def _runtime_runner_ledger(records: list[dict[str, str]]) -> dict:
    anchored, _lookup = finalizer.anchor_records(records)
    writes = [
        {
            "path": str(finalizer.safe_path(finalizer.CANDIDATE_PATH)),
            "role": "runner_output",
            "sha256": None,
        }
    ]
    roles = {item["role"] for item in anchored}
    return {
        "schema": finalizer.LEDGER_SCHEMA,
        "phase": "runner",
        "authorized_read_paths": anchored,
        "authorized_read_path_set_sha256": finalizer.canonical_hash(anchored),
        "authorized_write_paths": writes,
        "authorized_write_path_set_sha256": finalizer.canonical_hash(writes),
        "role_byte_open_counts": {
            role: (40 if role == "label_shard" else 1) for role in roles
        },
        "label_shard_pre_hash_byte_opens": 20,
        "label_shard_post_hash_byte_opens": 20,
        "label_shard_npz_parses": 20,
        "array_decompression_counts": {
            "current_labels": 20,
            "next_labels": 20,
        },
        "selected_label_rows_read": 320,
        "unselected_rows_scored": 0,
        "unselected_rows_retained": 0,
        "metadata_only_shard_stats": 0,
        "denied_attempt_records": [],
        "denied_reason_counts": {
            reason: 0 for reason in finalizer.DENIED_REASONS
        },
        "unexpected_path_attempts": 0,
        "forbidden_role_open_counts": {
            role: 0 for role in finalizer.FORBIDDEN_ROLES
        },
        "all_counts_reconcile": True,
    }


def _candidate_bundle() -> tuple[dict, dict, list[dict[str, str]]]:
    manifest = _machine_manifest()
    preparation_ledger = diagnostic_suite._ledger("preparation")
    manifest["preparation_access_ledger"] = preparation_ledger
    _rehash(manifest)
    _phase, _final_records, _runner_phase, runner_records = (
        finalizer.validate_machine_manifest(
            manifest,
            manifest_sha256=_sha("manifest"),
            candidate_sha256=_sha("candidate"),
        )
    )
    candidate = core.build_candidate(
        created_at_utc="2026-07-11T00:00:00+00:00",
        implementation_manifests={
            "human": {
                "path": manifest["human_manifest"]["path"],
                "file_sha256": manifest["human_manifest"]["file_sha256"],
            },
            "machine": {
                "path": finalizer.MACHINE_PATH,
                "file_sha256": _sha("manifest"),
                "content_sha256": manifest["content_sha256"],
            },
        },
        source_map=copy.deepcopy(manifest["source_map"]),
        preparation_access_ledger=copy.deepcopy(preparation_ledger),
        runner_access_ledger=_runtime_runner_ledger(runner_records),
        scientific=diagnostic_suite._passing_scientific(),
        label_shard_entries=[
            {
                "path": record["path"],
                "sha256": record["sha256"],
            }
            for record in runner_records
            if record["role"] == "label_shard"
        ],
    )
    return candidate, manifest, runner_records


def test_finalizer_import_is_independent_of_runner_and_diagnostic_core() -> None:
    source = Path(finalizer.__file__).read_text()
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "scripts.diagnose_go2_dynamic_cell_square_projection" not in imported_modules
    assert (
        "lewm.benchmarks.go2_dynamic_cell_square_projection_diagnostic"
        not in imported_modules
    )


def test_finalizer_requires_exact_two_verified_placeholders() -> None:
    template = [
        {"path": "a", "role": "a", "sha256": _sha("a")},
        dict(finalizer.SELF_TEMPLATE_ENTRY),
        dict(finalizer.CANDIDATE_TEMPLATE_ENTRY),
    ]
    with pytest.raises(ValueError, match="verified before"):
        finalizer.instantiate_template(
            template,
            manifest_sha256=_sha("manifest"),
            candidate_sha256=_sha("candidate"),
            manifest_verified=False,
            candidate_verified=True,
        )
    actual = finalizer.instantiate_template(
        template,
        manifest_sha256=_sha("manifest"),
        candidate_sha256=_sha("candidate"),
        manifest_verified=True,
        candidate_verified=True,
    )
    assert {item["role"] for item in actual} == {"a", "machine_manifest", "candidate"}

    mutations = (
        template[:-1],
        [*template, {"path": "x", "role": "x", "sha256_source": "third"}],
        [template[0], {**finalizer.SELF_TEMPLATE_ENTRY, "role": "wrong"}, template[2]],
        [template[0], template[1], {**finalizer.CANDIDATE_TEMPLATE_ENTRY, "path": "wrong"}],
    )
    for changed in mutations:
        with pytest.raises(ValueError, match="placeholders"):
            finalizer.instantiate_template(
                changed,
                manifest_sha256=_sha("manifest"),
                candidate_sha256=_sha("candidate"),
                manifest_verified=True,
                candidate_verified=True,
            )


def test_finalizer_strict_machine_validator_is_live_and_exact() -> None:
    manifest = _machine_manifest()
    phase, final_records, runner_phase, runner_records = (
        finalizer.validate_machine_manifest(
            manifest,
            manifest_sha256=_sha("manifest"),
            candidate_sha256=_sha("candidate"),
        )
    )
    assert phase is manifest["phase_contracts"]["finalizer"]
    assert runner_phase is manifest["phase_contracts"]["runner"]
    assert len([item for item in final_records if item["role"] == "label_shard"]) == 20
    assert len([item for item in runner_records if item["role"] == "label_shard"]) == 20
    source = Path(finalizer.__file__).read_text()
    assert "phase, instantiated, runner_phase, runner_instantiated = validate_machine_manifest(" in source


@pytest.mark.parametrize(
    "mutation",
    (
        "binding",
        "label_count",
        "label_count_float",
        "label_count_bool",
        "label_access",
        "write_path",
        "source_digest",
        "extra_role",
        "timestamp_bool",
        "timestamp_nonutc",
        "runtime_bool",
        "runtime_float",
        "output_absence_bool",
        "prep_etc_passwd",
        "prep_etc_shadow",
        "prep_source_digest",
        "source_map_count_bool",
        "source_map_count_float",
    ),
)
def test_finalizer_machine_manifest_mutations_fail_closed(mutation: str) -> None:
    manifest = _machine_manifest()
    if mutation == "binding":
        manifest["execution_binding"]["file_sha256"] = _sha("wrong")
    elif mutation == "label_count":
        manifest["phase_contracts"]["finalizer"][
            "expected_role_byte_open_counts"
        ]["label_shard"] = 39
    elif mutation == "label_count_float":
        manifest["phase_contracts"]["finalizer"][
            "expected_role_byte_open_counts"
        ]["label_shard"] = 40.0
    elif mutation == "label_count_bool":
        manifest["phase_contracts"]["finalizer"]["expected_label_access"][
            "metadata_only_shard_stats"
        ] = False
    elif mutation == "label_access":
        manifest["phase_contracts"]["finalizer"]["expected_label_access"][
            "selected_label_rows_read"
        ] = 319
    elif mutation == "write_path":
        manifest["phase_contracts"]["finalizer"]["authorized_write_paths"][0][
            "path"
        ] = "alternate.json"
        manifest["phase_contracts"]["finalizer"][
            "authorized_write_path_set_sha256"
        ] = finalizer.canonical_hash(
            manifest["phase_contracts"]["finalizer"]["authorized_write_paths"]
        )
    elif mutation == "source_digest":
        template = manifest["phase_contracts"]["finalizer"][
            "authorized_read_path_template"
        ]
        next(item for item in template if item["role"] == "finalizer")[
            "sha256"
        ] = _sha("substituted")
        manifest["phase_contracts"]["finalizer"][
            "authorized_read_path_template_sha256"
        ] = finalizer.canonical_hash(template)
    elif mutation == "extra_role":
        template = manifest["phase_contracts"]["finalizer"][
            "authorized_read_path_template"
        ]
        template.append(
            {"path": "extra.py", "role": "extra", "sha256": _sha("extra")}
        )
        template.sort(key=lambda item: (item["path"], item["role"]))
        manifest["phase_contracts"]["finalizer"][
            "authorized_read_path_template_sha256"
        ] = finalizer.canonical_hash(template)
        manifest["phase_contracts"]["finalizer"]["expected_roles"].append("extra")
        manifest["phase_contracts"]["finalizer"][
            "expected_role_byte_open_counts"
        ]["extra"] = 1
    elif mutation == "timestamp_bool":
        manifest["created_at_utc"] = True
    elif mutation == "timestamp_nonutc":
        manifest["created_at_utc"] = "2026-07-11T00:00:00"
    elif mutation == "runtime_bool":
        manifest["runtime_environment"]["python_version"][0] = True
    elif mutation == "runtime_float":
        manifest["runtime_environment"]["python_version"][0] = float(
            manifest["runtime_environment"]["python_version"][0]
        )
    elif mutation == "output_absence_bool":
        manifest["output_absence"]["paths"][0]["exists"] = 0
    elif mutation.startswith("prep_etc_"):
        reads = manifest["preparation_access_ledger"]["authorized_read_paths"]
        next(item for item in reads if item["role"] == "binding")["path"] = (
            "/etc/passwd" if mutation.endswith("passwd") else "/etc/shadow"
        )
        reads.sort(key=lambda item: (item["path"], item["role"]))
        manifest["preparation_access_ledger"][
            "authorized_read_path_set_sha256"
        ] = finalizer.canonical_hash(reads)
    elif mutation == "prep_source_digest":
        reads = manifest["preparation_access_ledger"]["authorized_read_paths"]
        next(item for item in reads if item["role"] == "runner")["sha256"] = (
            _sha("wrong-preparation-source")
        )
        manifest["preparation_access_ledger"][
            "authorized_read_path_set_sha256"
        ] = finalizer.canonical_hash(reads)
    elif mutation == "source_map_count_bool":
        manifest["source_map"]["entry_count"] = True
    elif mutation == "source_map_count_float":
        manifest["source_map"]["entry_count"] = 9.0
    _rehash(manifest)
    with pytest.raises(ValueError):
        finalizer.validate_machine_manifest(
            manifest,
            manifest_sha256=_sha("manifest"),
            candidate_sha256=_sha("candidate"),
        )


def test_finalizer_shard_records_must_equal_predecessor_commitment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        {
            "path": str(finalizer.ROOT / f".synthetic/shard_{index:02d}.npz"),
            "sha256": _sha(f"shard-{index}"),
        }
        for index in range(20)
    ]
    manifest_hash = finalizer.canonical_hash(entries)
    monkeypatch.setattr(finalizer, "EXPECTED_LABEL_MANIFEST_SHA256", manifest_hash)
    predecessor = {
        "label_shard_manifest": {
            "entry_count": 20,
            "entries": entries,
            "manifest_sha256": manifest_hash,
        }
    }
    records = sorted(
        [
            {"path": item["path"], "role": "label_shard", "sha256": item["sha256"]}
            for item in entries
        ],
        key=lambda item: (item["path"], item["role"]),
    )
    finalizer.validate_template_shards(records, predecessor)
    for replacement in (True, 20.0):
        changed_predecessor = copy.deepcopy(predecessor)
        changed_predecessor["label_shard_manifest"]["entry_count"] = replacement
        with pytest.raises(ValueError, match="commitment"):
            finalizer.manifest_entries(changed_predecessor)
    changed = [dict(item) for item in records]
    changed[-1]["sha256"] = _sha("wrong")
    with pytest.raises(ValueError, match="differs"):
        finalizer.validate_template_shards(changed, predecessor)


def test_finalizer_strict_candidate_validator_accepts_exact_candidate() -> None:
    candidate, manifest, runner_records = _candidate_bundle()
    scientific = finalizer.validate_candidate(
        candidate,
        candidate_hash=_sha("candidate"),
        manifest_hash=_sha("manifest"),
        manifest=manifest,
        runner_instantiated=runner_records,
    )
    assert scientific["support"]["dynamic_cell_square_known"][
        "unsupported_count"
    ] == 0


@pytest.mark.parametrize(
    "field",
    (
        "execution_binding",
        "implementation_manifests",
        "inputs",
        "source_map",
        "scope",
        "preparation_access_ledger",
        "runner_access_ledger",
        "runner_write_path",
        "runner_float_count",
        "runner_bool_count",
        "gates",
        "timestamp_bool",
        "timestamp_nonutc",
    ),
)
def test_finalizer_candidate_copied_field_mutations_fail_closed(field: str) -> None:
    candidate, manifest, runner_records = _candidate_bundle()
    if field == "execution_binding":
        candidate[field]["file_sha256"] = _sha("wrong")
    elif field == "implementation_manifests":
        candidate[field]["human"]["file_sha256"] = _sha("wrong")
    elif field == "inputs":
        candidate[field]["selected_targets"]["byte_count"] -= 1
    elif field == "source_map":
        candidate[field]["entries"][1]["sha256"] = _sha("wrong")
        candidate[field]["source_map_sha256"] = finalizer.canonical_hash(
            candidate[field]["entries"]
        )
    elif field == "scope":
        candidate[field]["dataset_role"] = "g2"
    elif field == "preparation_access_ledger":
        candidate[field]["metadata_only_shard_stats"] = 19
    elif field == "runner_access_ledger":
        candidate[field]["selected_label_rows_read"] = 319
    elif field == "runner_write_path":
        writes = candidate["runner_access_ledger"]["authorized_write_paths"]
        writes[0]["path"] = "/tmp/alternate/candidate.json"
        candidate["runner_access_ledger"][
            "authorized_write_path_set_sha256"
        ] = finalizer.canonical_hash(writes)
    elif field == "runner_float_count":
        candidate["runner_access_ledger"]["selected_label_rows_read"] = 320.0
    elif field == "runner_bool_count":
        candidate["runner_access_ledger"][
            "label_shard_pre_hash_byte_opens"
        ] = True
    elif field == "gates":
        candidate[field]["dynamic_zero_known_unsupported_pass"] = False
    elif field == "timestamp_bool":
        candidate["created_at_utc"] = True
    elif field == "timestamp_nonutc":
        candidate["created_at_utc"] = "2026-07-11T00:00:00"
    _rehash(candidate)
    with pytest.raises(ValueError):
        finalizer.validate_candidate(
            candidate,
            candidate_hash=_sha("candidate"),
            manifest_hash=_sha("manifest"),
            manifest=manifest,
            runner_instantiated=runner_records,
        )


def test_finalizer_synthetic_npz_decoder_is_independent_and_label_only() -> None:
    entry = _entry()
    ranks, target_buffer, rank_filled = _decode_state(entry)
    lifetime: list[dict] = []
    selected_count, counts = finalizer.decode_rows(
        _npz(),
        entry,
        np,
        rank_by_identity=ranks,
        target_buffer=target_buffer,
        rank_filled=rank_filled,
        lifetime_events=lifetime,
    )
    assert selected_count == 2
    assert rank_filled[:2] == b"\x01\x01"
    assert counts == {"current_labels": 1, "next_labels": 1}
    assert [event["event"] for event in lifetime] == [
        "array_released",
        "array_released",
        "archive_released",
    ]
    assert all(event["selected_row_copies_retained"] == 0 for event in lifetime)

    ranks, target_buffer, rank_filled = _decode_state(entry)
    with pytest.raises(ValueError, match="contract"):
        finalizer.decode_rows(
            _npz(dtype=np.dtype("int16")),
            entry,
            np,
            rank_by_identity=ranks,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
        )
    ranks, target_buffer, rank_filled = _decode_state(entry)
    with pytest.raises(ValueError, match="contract"):
        finalizer.decode_rows(
            _npz(invalid=True),
            entry,
            np,
            rank_by_identity=ranks,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
        )


def test_finalizer_selected_storage_identity_duplicates_fail_closed() -> None:
    entry = _entry()
    entry["selected_tuples"].append(copy.deepcopy(entry["selected_tuples"][0]))
    ranks, target_buffer, rank_filled = _decode_state(entry)
    with pytest.raises(ValueError, match="duplicated"):
        finalizer.decode_rows(
            _npz(),
            entry,
            np,
            rank_by_identity=ranks,
            target_buffer=target_buffer,
            rank_filled=rank_filled,
        )


def test_finalizer_releases_each_shard_before_opening_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        {
            "path": str(finalizer.ROOT / f".synthetic/final_lifetime_{index}.npz"),
            "sha256": _sha(f"final-lifetime-{index}"),
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
    row = finalizer.TARGET_ROW_BYTES
    expected = (
        bytes([1]) * row
        + bytes([1]) * row
        + bytes([2]) * row
        + bytes([2]) * row
    )
    monkeypatch.setattr(finalizer, "EXPECTED_TARGET_ROWS", 4)
    monkeypatch.setattr(finalizer, "EXPECTED_TARGET_BYTES", len(expected))
    monkeypatch.setattr(
        finalizer, "EXPECTED_TARGET_SHA256", hashlib.sha256(expected).hexdigest()
    )
    monkeypatch.setattr(finalizer, "manifest_entries", lambda _predecessor: entries)
    raw_by_path = {
        entries[0]["path"]: _npz(current_value=1, next_value=2),
        entries[1]["path"]: _npz(current_value=2, next_value=1),
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

    monkeypatch.setattr(finalizer, "read_allowed", fake_read)
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
    assert finalizer.load_targets(
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


def test_finalizer_strict_json_rejects_duplicates_nonfinite_and_nonobject() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        finalizer.strict_json(b'{"a":1,"a":2}', name="synthetic")
    with pytest.raises(ValueError, match="nonfinite"):
        finalizer.strict_json(b'{"a":-Infinity}', name="synthetic")
    with pytest.raises(ValueError, match="root"):
        finalizer.strict_json(b"[]", name="synthetic")


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
def test_finalizer_rejects_raw_path_aliases_and_outside_paths(path: str) -> None:
    with pytest.raises(PermissionError):
        finalizer.safe_path(path)


def test_finalizer_denial_telemetry_uses_frozen_reason_precedence() -> None:
    assert finalizer.primary_denial_reason(
        ["hash_mismatch", "forbidden_role", "unallowlisted"]
    ) == "unallowlisted"
    ledger = _runtime_runner_ledger([])
    primary = finalizer.record_denial(
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


def test_finalizer_reads_the_validated_anchored_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = "scripts/finalize_go2_dynamic_cell_square_projection.py"
    absolute = finalizer.ROOT / relative
    payload = absolute.read_bytes()
    source_record = {
        "path": relative,
        "role": "finalizer",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    anchored, allowlist = finalizer.anchor_records([source_record])
    ledger = _runtime_runner_ledger([source_record])
    assert anchored[0]["path"] == str(absolute)
    monkeypatch.chdir(tmp_path)
    assert finalizer.read_allowed(
        Path(relative),
        role="finalizer",
        allowlist=allowlist,
        ledger=ledger,
    ) == payload


def test_candidate_stat_guard_detects_change_before_publication(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(b"first")
    initial = candidate.lstat()
    assert finalizer.candidate_unchanged(candidate, initial)
    candidate.write_bytes(b"different")
    assert not finalizer.candidate_unchanged(candidate, initial)


def test_finalizer_cli_requires_ordered_hash_only_arguments() -> None:
    args = finalizer.parse_args(
        [
            "--implementation-manifest-sha256",
            _sha("manifest"),
            "--candidate-sha256",
            _sha("candidate"),
        ]
    )
    assert vars(args) == {
        "implementation_manifest_sha256": _sha("manifest"),
        "candidate_sha256": _sha("candidate"),
    }
    with pytest.raises(SystemExit):
        finalizer.parse_args(
            ["--implementation-manifest-sha256", _sha("manifest")]
        )
    with pytest.raises(SystemExit):
        finalizer.parse_args(
            [
                "--implementation-manifest-sha256",
                _sha("manifest"),
                "--candidate-sha256",
                _sha("candidate"),
                "--output",
                "alternate",
            ]
        )
    with pytest.raises(SystemExit):
        finalizer.parse_args(
            [
                "--candidate-sha256",
                _sha("candidate"),
                "--implementation-manifest-sha256",
                _sha("manifest"),
            ]
        )


def test_finalizer_writer_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    finalizer.write_exclusive(output, b"first\n")
    with pytest.raises(FileExistsError):
        finalizer.write_exclusive(output, b"second\n")
    assert output.read_bytes() == b"first\n"
