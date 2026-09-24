from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from lewm.safety import physical_graph_edge_handoff_qualification_v3_metrics as M
from lewm.safety import physical_graph_edge_handoff_snapshot_semantics_v1 as S
from scripts import evaluate_physical_graph_edge_handoff_qualification_v3 as E


def _frame(tag: bytes, payload: bytes) -> bytes:
    return bytes([len(tag)]) + tag + len(payload).to_bytes(8, "big") + payload


def _u64(value: int) -> bytes:
    return value.to_bytes(8, "big")


def _shape(*values: int) -> bytes:
    return _u64(len(values)) + b"".join(_u64(value) for value in values)


def _ivec(*values: int) -> bytes:
    return _u64(len(values)) + b"".join(
        _frame(b"int", b"\x00" if value == 0 else b"\x01" + bytes([value]))
        for value in values
    )


def _torch_semantic_stream(*, device_class: str = "cpu", type_name: str | None = None) -> bytes:
    body = _frame(b"storage-id", _u64(0))
    body += _frame(b"storage-offset-bytes", _u64(0))
    body += _frame(b"dtype", b"torch.uint8")
    body += _frame(b"shape", _shape(2))
    body += _frame(b"logical-stride", _ivec(1))
    body += _frame(
        b"device-rule",
        b"CPU_OR_ACCELERATOR_CLASS_WITH_PATH_ASSOCIATION_V1",
    )
    body += _frame(b"device-class", device_class.encode("ascii"))
    body += _frame(b"logical-cpu-c-bytes", b"\x03\x04")
    value = _frame(b"DEF", _u64(0) + _frame(b"torch", body))
    if type_name is not None:
        structured = _frame(b"type", type_name.encode("utf-8")) + _u64(0)
        value = _frame(b"DEF", _u64(0) + _frame(b"structured", structured))
    return E.SEMANTIC_MAGIC + value


def _semantic_fixture() -> tuple[bytes, dict[str, object]]:
    branch_fields = (
        "solver_state", "step_index", "last_actions", "harness", "rng",
        "counters", "goal", "identity", "boundary", "digest",
    )
    episode_fields = (
        "scene_id", "episode_id", "reset_count", "episode_step",
        "scene_family", "split", "manifest_sha256",
    )
    branch_type = E._worker_stub(
        "BranchSnapshot", "scripts.run_go2_oracle_branch_pilot_v1", branch_fields
    )
    episode_type = E._worker_stub(
        "EpisodeState", "lewm_genesis.lewm_contract", episode_fields
    )
    episode = episode_type()
    for name, value in zip(
        episode_fields,
        ("scene", "episode", 1, 2, "family", "DEVELOPMENT", "a" * 64),
        strict=True,
    ):
        object.__setattr__(episode, name, value)
    shared = np.arange(8, dtype=np.float32)
    cycle: list[object] = []
    cycle.append(cycle)
    snapshot = branch_type()
    values = (
        {"array": shared, "view": shared[2:6]},
        3,
        [shared, shared],
        cycle,
        {"cpu": np.array([1, 2], dtype=np.uint8)},
        {"count": 4},
        (1.0, 2.0),
        episode,
        set(),
        b"digest",
    )
    for name, value in zip(branch_fields, values, strict=True):
        object.__setattr__(snapshot, name, value)
    canonical = S.canonical_semantic_snapshot(snapshot)
    evidence = S.semantic_snapshot_evidence(snapshot)
    return canonical, evidence


def test_portable_semantic_parser_matches_reference_alias_storage_and_types() -> None:
    canonical, evidence = _semantic_fixture()
    inspected = E.inspect_semantic_snapshot(
        canonical, M.semantic_snapshot_serializer_authority()
    )
    assert inspected["sha256"] == hashlib.sha256(canonical).hexdigest()
    for field in (
        "referenceable_object_count",
        "reference_alias_edge_count",
        "reference_cycle_edge_count",
        "reference_manifest",
        "reference_edge_manifest",
        "storage_manifest",
        "structured_type_inventory",
        "nonfinite_sentinel_inventory",
        "type_inventory",
    ):
        assert inspected[field] == evidence[field]
    assert evidence["reference_alias_edge_count"] >= 2
    assert evidence["reference_cycle_edge_count"] == 1
    assert any(len(row["member_object_ids"]) == 2 for row in evidence["storage_manifest"])


def test_portable_semantic_parser_rejects_unknown_type_and_device_class() -> None:
    authority = M.semantic_snapshot_serializer_authority()
    valid = _torch_semantic_stream()
    inspected = E.inspect_semantic_snapshot(valid, authority)
    assert inspected["torch_tensor_count"] == 1
    with pytest.raises(E.RegenerationError, match="device authority"):
        E.inspect_semantic_snapshot(
            _torch_semantic_stream(device_class="gpu"), authority
        )
    with pytest.raises(E.RegenerationError, match="structured type authority"):
        E.inspect_semantic_snapshot(
            _torch_semantic_stream(type_name="unknown.module.Type"), authority
        )


def test_semantic_parser_rejects_byte_tamper_and_trailing_data() -> None:
    canonical, _evidence = _semantic_fixture()
    with pytest.raises(E.RegenerationError):
        E.inspect_semantic_snapshot(canonical[:-1], M.semantic_snapshot_serializer_authority())
    with pytest.raises(E.RegenerationError, match="trailing"):
        E.inspect_semantic_snapshot(
            canonical + b"x", M.semantic_snapshot_serializer_authority()
        )


def test_historical_receipt_treats_raw_drift_as_descriptive_but_semantic_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_evidence = {
        "serializer_schema": M.C.SEMANTIC_SERIALIZER_AUTHORITY["serializer_schema"],
        "snapshot_semantic_digest_v1": "1" * 64,
        "canonical_semantic_byte_count": 100,
    }
    # This boundary is tested without fabricating the large live custody
    # receipt: the pair projection must derive raw and semantic booleans from
    # the two version identities, never conflate them.
    v1 = {
        "artifact_file_sha256": "2" * 64,
        "snapshot_semantic_digest_v1": "1" * 64,
        "semantic_evidence": semantic_evidence,
    }
    v2 = copy.deepcopy(v1)
    v2["artifact_file_sha256"] = "3" * 64
    assert v1["artifact_file_sha256"] != v2["artifact_file_sha256"]
    assert v1["snapshot_semantic_digest_v1"] == v2["snapshot_semantic_digest_v1"]
    v2["snapshot_semantic_digest_v1"] = "4" * 64
    assert v1["snapshot_semantic_digest_v1"] != v2["snapshot_semantic_digest_v1"]


@pytest.mark.skipif(
    not E.DEFAULT_HISTORICAL_CUSTODY_RECEIPT.exists(),
    reason="immutable combined V1/V2 custody receipt is unavailable",
)
def test_emitted_historical_receipt_rejects_raw_and_semantic_boundary_tampering() -> None:
    supplied = E.parse_canonical_json(
        E.DEFAULT_HISTORICAL_CUSTODY_RECEIPT.read_bytes(),
        label="combined historical custody receipt fixture",
    )
    validated = E.validate_historical_custody_receipt_document(
        supplied, metrics=M
    )
    assert validated["nonreuse"][
        "v1_v2_artifact_file_sha256_equal_count"
    ] == 0
    assert validated["nonreuse"][
        "v1_v2_snapshot_semantic_digest_v1_equal_count"
    ] == 8

    raw_equality_tamper = copy.deepcopy(supplied)
    pair = raw_equality_tamper["first_eight_pairs"][0]
    pair["v2"]["artifact_file_sha256"] = pair["v1"][
        "artifact_file_sha256"
    ]
    pair["artifact_file_sha256_equal"] = True
    with pytest.raises(E.RegenerationError):
        E.validate_historical_custody_receipt_document(
            raw_equality_tamper, metrics=M
        )

    semantic_tamper = copy.deepcopy(supplied)
    semantic_tamper["first_eight_pairs"][0]["v2"][
        "snapshot_semantic_digest_v1"
    ] = "0" * 64
    with pytest.raises(E.RegenerationError):
        E.validate_historical_custody_receipt_document(
            semantic_tamper, metrics=M
        )


def test_outer_pickle_preflight_rejects_forbidden_global_and_opcode() -> None:
    authority = M.historical_snapshot_deserializer_authority()
    safe = b"\x80\x04N."
    E._preflight_outer_pickle(safe, authority)
    # GLOBAL is forbidden even if it names an otherwise benign constructor.
    forbidden = b"\x80\x04cbuiltins\nlist\n."
    with pytest.raises(E.RegenerationError, match="forbidden outer opcode"):
        E._preflight_outer_pickle(forbidden, authority)


def _probe_trace_fixture() -> dict[str, object]:
    trace: dict[str, object] = {}
    for member, authority in M.behavioural_probe_authority()[
        "npz_members"
    ].items():
        if member in {
            "trace_offsets", "version_code", "pool_index", "trial_index",
            "stuck", "termination_code", "final_snapshot_semantic_digest_bytes",
        }:
            continue
        shape = [750, *authority["shape"][1:]]
        trace[member] = np.zeros(shape, dtype=np.dtype(authority["descr"]))
    trace["requested_command"][:] = np.asarray(
        M.C.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND,
        dtype=np.float64,
    )
    trace["post_slew_applied_command"][:, 0] = 0.2
    trace["timestamp_s"] = (
        np.arange(1, 751, dtype=np.float64) * 0.002
    )
    trace["base_pose_world"][:, 6] = 1.0
    trace["termination_reason"] = "H3_COMPLETE"
    trace["stuck"] = True
    trace["final_snapshot_semantic_digest_v1"] = "a" * 64
    return trace


def test_independent_probe_comparison_enforces_exact_and_controller_boundaries() -> None:
    authority = M.behavioural_probe_authority()
    left = _probe_trace_fixture()
    right = copy.deepcopy(left)
    independent = E._compare_probe_traces(
        left, right, behavioural_authority=authority
    )
    pure = M.compare_behavioural_probe_traces(left, right)
    assert independent == pure
    assert independent["pass"] is True

    controller_drift = copy.deepcopy(right)
    controller_drift["controller_observation"][0, 0] = 2.0e-9
    assert E._compare_probe_traces(
        left, controller_drift, behavioural_authority=authority
    )["pass"] is False

    command_drift = copy.deepcopy(right)
    command_drift["requested_command"][0, 0] = np.nextafter(
        command_drift["requested_command"][0, 0], 1.0
    )
    with pytest.raises(E.RegenerationError, match="requested-command"):
        E._compare_probe_traces(
            left, command_drift, behavioural_authority=authority
        )

    invalid_requested = copy.deepcopy(right)
    invalid_requested["requested_command"][:] = np.asarray(
        M.C.BEHAVIOURAL_PROBE_COMMAND,
        dtype=np.float64,
    )
    with pytest.raises(E.RegenerationError, match="requested-command"):
        E._compare_probe_traces(
            left, invalid_requested, behavioural_authority=authority
        )

    cadence_drift = copy.deepcopy(right)
    cadence_drift["timestamp_s"][500] += 1.0e-8
    with pytest.raises(E.RegenerationError, match="timestamp cadence"):
        E._compare_probe_traces(
            left, cadence_drift, behavioural_authority=authority
        )

    # Final semantic identity remains descriptive after the probe and cannot
    # silently strengthen the frozen physical comparison.
    final_digest_drift = copy.deepcopy(right)
    final_digest_drift["final_snapshot_semantic_digest_v1"] = "b" * 64
    comparison = E._compare_probe_traces(
        left, final_digest_drift, behavioural_authority=authority
    )
    assert comparison["final_snapshot_semantic_digest_equal"] is False
    assert comparison["pass"] is True


def test_cli_imports_under_system_python_without_torch() -> None:
    source = Path(E.__file__).resolve()
    completed = E.subprocess.run(
        ["/usr/bin/python3", str(source), "--help"],
        cwd=E.REPO_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
        },
        stdout=E.subprocess.PIPE,
        stderr=E.subprocess.PIPE,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert b"PGEHQ V3" in completed.stdout


def test_publication_validator_uses_exact_projection_and_report_byte_apis(
    tmp_path: Path,
) -> None:
    metric_raw = E.canonical_document_bytes({"metric": True})
    scientific_bindings: dict[str, dict[str, object]] = {
        "metrics.json": E._official_binding("metrics.json", metric_raw)
    }
    for index in range(24):
        name = f"scientific-{index:02d}.bin"
        scientific_bindings[name] = {
            "path": name,
            "bytes": 1,
            "sha256": hashlib.sha256(bytes([index])).hexdigest(),
        }
    scientific_storage_bytes = sum(
        int(binding["bytes"]) for binding in scientific_bindings.values()
    )
    result_document = M.C.attach_content_digest(
        {"scientific_storage_bytes": scientific_storage_bytes}
    )
    result_raw = E.canonical_document_bytes(result_document)
    report_raw = b"exact report\n"
    expected_rows = [dict(binding) for binding in scientific_bindings.values()]
    expected_rows.extend(
        (
            E._official_binding("result.json", result_raw),
            E._official_binding("result.md", report_raw),
        )
    )
    expected_rows.sort(key=lambda row: row["path"])
    manifest = M.C.attach_content_digest(
        {
            "schema": (
                "physical_graph_edge_handoff_qualification_v3."
                "file_hashes.v1"
            ),
            "root": str(tmp_path),
            "files": expected_rows,
            "file_count_excluding_self": 27,
            "bytes_excluding_self": sum(
                int(row["bytes"]) for row in expected_rows
            ),
            "file_hashes_self_sha256_excluded": True,
        }
    )
    (tmp_path / "result.json").write_bytes(result_raw)
    (tmp_path / "result.md").write_bytes(report_raw)
    (tmp_path / "file_hashes.json").write_bytes(
        E.canonical_document_bytes(manifest)
    )

    class ExactPublicationModule:
        @staticmethod
        def validate_result_publication_projection(value, **kwargs):
            assert value == result_document
            assert kwargs["scientific_bindings"] == scientific_bindings
            assert kwargs["snapshot_equivalence"] == {"semantic": "exact"}
            assert kwargs["historical_custody_receipt_binding"] == {
                "path": "/historical.json",
                "bytes": 7,
                "sha256": "a" * 64,
            }
            assert kwargs["independent_reducer_receipt_sha256"] == "b" * 64
            return {"projection": "exact"}

        @staticmethod
        def build_result_report_bytes(value):
            assert value == {"projection": "exact"}
            return report_raw

    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        validated = E._validate_publication_if_present(
            root=tmp_path,
            root_fd=root_fd,
            complete=True,
            scientific_bindings=scientific_bindings,
            recomputed={"metrics": "exact"},
            runtime_contract={"runtime": "exact"},
            expected_receipt_sha256="b" * 64,
            semantic_validation={"semantic": "exact"},
            historical_binding={
                "path": "/historical.json",
                "bytes": 7,
                "sha256": "a" * 64,
            },
            module=ExactPublicationModule,
        )
    finally:
        os.close(root_fd)
    assert validated is not None
    assert validated["result_projection_validated"] is True
    assert validated["report_exact_byte_equal"] is True
    assert validated["manifest_exact_live_bytes"] is True


def test_first_probe_gate_is_mode_aware_without_relaxing_success() -> None:
    success = {
        "pass": True,
        "status": "PASS",
        "technical_disposition": None,
        "full_collection_authorized": True,
        "semantic_all_pass": True,
        "behavioural_all_pass": True,
    }
    E._validate_first_probe_gate_disposition(
        success,
        {
            "all_within_version_pass": True,
            "all_historical_cross_version_pass": True,
        },
    )
    with pytest.raises(E.RegenerationError, match="successful raw"):
        E._validate_first_probe_gate_disposition(
            success,
            {
                "all_within_version_pass": False,
                "all_historical_cross_version_pass": True,
            },
        )

    terminal = {
        "pass": False,
        "status": E.MISMATCH_DISPOSITION,
        "technical_disposition": E.MISMATCH_DISPOSITION,
        "full_collection_authorized": False,
        "semantic_all_pass": True,
        "behavioural_all_pass": False,
    }
    E._validate_first_probe_gate_disposition(
        terminal,
        {
            "all_within_version_pass": False,
            "all_historical_cross_version_pass": False,
        },
    )
    # A semantic-only terminal may legitimately have fully passing raw probe
    # comparisons; the pure row/array cross-link proves the semantic failure.
    semantic_terminal = dict(terminal)
    semantic_terminal["semantic_all_pass"] = False
    semantic_terminal["behavioural_all_pass"] = True
    E._validate_first_probe_gate_disposition(
        semantic_terminal,
        {
            "all_within_version_pass": True,
            "all_historical_cross_version_pass": True,
        },
    )
    invalid_terminal = dict(terminal)
    invalid_terminal["semantic_all_pass"] = True
    invalid_terminal["behavioural_all_pass"] = True
    with pytest.raises(E.RegenerationError, match="terminal gate"):
        E._validate_first_probe_gate_disposition(
            invalid_terminal,
            {
                "all_within_version_pass": True,
                "all_historical_cross_version_pass": True,
            },
        )


def test_terminal_verify_and_emit_never_touches_external_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "terminal_regeneration_receipt.json"
    assert not output.exists()

    monkeypatch.setattr(
        E,
        "build_regeneration_receipt",
        lambda *args, **kwargs: {
            "scientific_result_produced": False,
            "technical_disposition": E.MISMATCH_DISPOSITION,
        },
    )

    def forbidden_emit(*args, **kwargs):
        raise AssertionError("terminal validation attempted external emission")

    monkeypatch.setattr(E.V2E, "_emit_external", forbidden_emit)
    with pytest.raises(E.RegenerationError, match="validation-only"):
        E.verify_and_emit(
            tmp_path / "official",
            output,
            material_root=tmp_path / "material",
            historical_custody_receipt=tmp_path / "historical.json",
        )
    assert not output.exists()


def test_scientific_npz_inventory_includes_behavioural_probe_payload() -> None:
    assert E._SCIENTIFIC_NPZ_LEAVES == (
        "state_snapshots.npz",
        "teacher_traces.npz",
        "rgb_observations.npz",
        "canonical_latents.npz",
        "candidate_traces.npz",
        "snapshot_behavioural_probes.npz",
    )
    authority = M.reducer_authority()
    assert set(authority["npz_authorities"]) == set(E._PAYLOAD_LEAVES)
    probe = M.behavioural_probe_authority()["npz_members"]
    assert probe
    assert set(probe) == set(M.C.BEHAVIOURAL_PROBE_NPZ_AUTHORITY)


def _six_npz_inspection_fixture() -> list[dict[str, object]]:
    authorities = {
        **M.reducer_authority()["npz_authorities"],
        "snapshot_behavioural_probes.npz": M.behavioural_probe_authority()[
            "npz_members"
        ],
    }
    symbol_values = {"B": 64, "T": 256, "U": 1, "P": 720_000}
    rows: list[dict[str, object]] = []
    for path in E._SCIENTIFIC_NPZ_LEAVES:
        authority = authorities[path]
        offsets_by_member: dict[str, list[int]] = {}
        for member, spec in authority.items():
            if not member.endswith("offsets"):
                continue
            if path == "candidate_traces.npz":
                offsets = list(range(0, 720_001, 750))
            elif path == "teacher_traces.npz":
                offsets = list(range(257))
            elif path == "state_snapshots.npz":
                offsets = list(range(65))
            else:
                offsets = list(
                    M.behavioural_probe_authority()["trace_offsets"]
                )
            offsets_by_member[member] = offsets
        members: dict[str, dict[str, object]] = {}
        for member, spec in authority.items():
            shape = [
                symbol_values.get(dimension, dimension)
                if isinstance(dimension, str)
                else dimension
                for dimension in spec["shape"]
            ]
            mode = spec["hash_mode"]
            digest_count = (
                1
                if mode == "whole"
                else shape[0]
                if mode == "rows_axis0"
                else len(offsets_by_member[spec["offsets_member"]]) - 1
            )
            members[member] = {
                "descr": spec["descr"],
                "digest_dtype": spec["digest_dtype"],
                "shape": shape,
                "c_contiguous": True,
                "object_dtype": False,
                "member_sha256": "a" * 64,
                "row_or_slice_sha256s": ["b" * 64] * digest_count,
                "offset_values": offsets_by_member.get(member),
            }
        rows.append(
            {
                "path": path,
                "bytes": 1,
                "sha256": "c" * 64,
                "members": members,
            }
        )
    return rows


def test_six_npz_projection_rejects_coverage_layout_and_hash_drift() -> None:
    rows = _six_npz_inspection_fixture()
    assert list(M.validate_npz_inspections(rows)) == list(
        E._SCIENTIFIC_NPZ_LEAVES
    )
    inherited = [
        copy.deepcopy(row)
        for row in rows
        if row["path"] in E._PAYLOAD_LEAVES
    ]
    assert set(M.V2M.validate_npz_inspections(inherited)) == set(
        E._PAYLOAD_LEAVES
    )

    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.validate_npz_inspections(rows[:-1])
    duplicate = copy.deepcopy(rows)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.validate_npz_inspections(duplicate)
    layout = copy.deepcopy(rows)
    layout[-1]["members"]["timestamp_s"]["descr"] = "<f4"
    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.validate_npz_inspections(layout)
    digest = copy.deepcopy(rows)
    digest[-1]["members"]["timestamp_s"][
        "row_or_slice_sha256s"
    ][0] = "not-a-sha256"
    with pytest.raises(M.PhysicalGraphEdgeHandoffV3MetricsError):
        M.validate_npz_inspections(digest)
