from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "evaluate_physical_graph_edge_handoff_qualification_v4.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("pgehq_v4_evaluator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_evaluator_is_system_python_safe_and_has_exact_inventories() -> None:
    evaluator = _load_evaluator()
    assert evaluator.EXPERIMENT_ID == "PHYSICAL_GRAPH_EDGE_HANDOFF_QUALIFICATION_V4"
    assert evaluator.PANEL_INADEQUATE_DISPOSITION == "PHYSICAL_HANDOFF_PANEL_INADEQUATE"
    assert evaluator.NEXT_GENERATOR_DECISION == (
        "REVISE_GENERATOR_FOR_SHORTFALL_FAMILIES_KEEP_TEACHER_CONTRACT_FROZEN"
    )
    assert len(evaluator.V4_PANEL_INADEQUATE_FILES) == 9
    assert len(set(evaluator.V4_PANEL_INADEQUATE_FILES)) == 9
    assert len(evaluator.V4_SUCCESS_FILES) == 27
    assert len(set(evaluator.V4_SUCCESS_FILES)) == 27

    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imports.isdisjoint(
        {"torch", "genesis", "lewm_genesis", "cv2", "PIL", "torchvision"}
    )
    counters = evaluator._receipt_counters()
    assert counters
    assert all(type(value) is int and value == 0 for value in counters.values())


def test_v4_evaluator_help_works_under_stripped_system_python() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PGEHQ V4" in completed.stdout


def test_immutable_reader_covers_every_block_and_rejects_links(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator()
    raw = b"a" * (1 << 20) + b"b" * (1 << 20) + b"tail"
    path = tmp_path / "large.bin"
    path.write_bytes(raw)
    assert evaluator._read_regular(path, "large fixture") == raw
    link = tmp_path / "linked.bin"
    link.symlink_to(path)
    with pytest.raises(evaluator.RegenerationError, match="single-link regular"):
        evaluator._read_regular(link, "linked fixture")


def test_ordinary_gate_loader_accepts_only_canonical_non_self_digested_json(
    tmp_path: Path,
) -> None:
    evaluator = _load_evaluator()
    value = {"schema": "ordinary.fixture.v1", "pass": True}
    (tmp_path / "ordinary.json").write_bytes(
        evaluator.canonical_document_bytes(value)
    )
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        raw, reopened = evaluator._load_ordinary_json_at(
            descriptor, "ordinary.json", label="ordinary fixture"
        )
        assert raw == evaluator.canonical_document_bytes(value)
        assert reopened == value

        self_digested = {
            **value,
            "content_digest": hashlib.sha256(b"not authoritative").hexdigest(),
        }
        (tmp_path / "self-digested.json").write_bytes(
            evaluator.canonical_document_bytes(self_digested)
        )
        with pytest.raises(evaluator.RegenerationError, match="self digest"):
            evaluator._load_ordinary_json_at(
                descriptor, "self-digested.json", label="self-digested fixture"
            )

        (tmp_path / "noncanonical.json").write_bytes(b'{"pass": true}\n')
        with pytest.raises(evaluator.RegenerationError):
            evaluator._load_ordinary_json_at(
                descriptor, "noncanonical.json", label="noncanonical fixture"
            )
    finally:
        os.close(descriptor)


def _terminated_probe_trace(*, flag: str = "fall", samples: int = 7):
    import numpy as np

    from lewm.safety import physical_graph_edge_handoff_qualification_v4_contract as C

    trace = {}
    for member, authority in C.V4_PROBE_TRACE_MEMBER_AUTHORITY.items():
        shape = (samples, *authority["shape"][1:])
        trace[member] = np.zeros(shape, dtype=np.dtype(authority["descr"]))
    trace["timestamp_s"] = (np.arange(samples, dtype=np.float64) + 1) * 0.002
    trace["requested_command"][:] = np.asarray(
        C.V3.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND, dtype=np.float64
    )
    trace["post_slew_applied_command"][:] = trace["requested_command"]
    trace["base_pose_world"][:, 2] = 0.35
    trace["base_pose_world"][:, 6] = 1.0
    flags = {name: name == flag for name in C.TERMINATION_FLAG_ORDER}
    reasons = {
        "fall": "FALL",
        "out_of_bounds": "OUT_OF_BOUNDS",
        "tipped": "TIPPED",
        "nan": "NAN",
    }
    return {
        **trace,
        "termination_reason": reasons[flag],
        "termination_flags": flags,
        "tip_sample_index": samples - 1,
        "stuck": False,
    }


def test_evaluator_uses_general_partial_probe_comparison_authority() -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    left = _terminated_probe_trace(flag="fall")
    right = copy.deepcopy(left)
    comparison = evaluator._call(
        metrics, "compare_terminated_behavioural_probe_traces", left, right
    )
    assert comparison["pass"] is True

    wrong_flags = _terminated_probe_trace(flag="tipped")
    mismatch = evaluator._call(
        metrics,
        "compare_terminated_behavioural_probe_traces",
        left,
        wrong_flags,
    )
    assert mismatch["termination_flags_equal"] is False
    assert mismatch["pass"] is False

    trace_tamper = copy.deepcopy(right)
    trace_tamper["physics_contact"][-1] = 1
    mismatch = evaluator._call(
        metrics,
        "compare_terminated_behavioural_probe_traces",
        left,
        trace_tamper,
    )
    assert mismatch["exact_member_equal"]["physics_contact"] is False
    assert mismatch["pass"] is False


def test_evaluator_independently_enforces_corrected_binary64_projection() -> None:
    import numpy as np

    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    segment = [
        [0.3887954107122013, -0.148793657899766],
        [-0.06509439358982835, 0.4482691910120259],
    ]
    point = [0.23387508547129157, 0.05499406737182661]
    independent = evaluator._canonical_v1_planar_segment_projection(
        point, segment
    )
    assert independent["lateral_coordinate_m"].hex() == (
        "-0x1.e7795b3662ac9p-4"
    )
    assert evaluator._independent_planar_projection(
        metrics, point, segment, label="selected fixture"
    ) == independent

    endpoint = [0.2846214771270752, 0.177300825715065]
    endpoint_projection = evaluator._canonical_v1_planar_segment_projection(
        endpoint, segment
    )
    metadata = {
        "candidate_spec": {
            "geometry": {
                "selected_directed_edge": {
                    "edge_id": "selected-edge",
                    "opening_segment_world": segment,
                },
                "competing_directed_edges": [],
            }
        },
        "teacher": {
            "crossing": {
                "edge_id": "selected-edge",
                "point_world": point,
                "lateral_coordinate_m": independent["lateral_coordinate_m"],
            },
            "competing_crossing": None,
        },
        "state_disposition": {
            "teacher_termination_flags": {
                "fall": False,
                "out_of_bounds": False,
                "tipped": False,
                "nan": False,
            }
        },
    }
    poses = np.zeros((2, 7), dtype=np.float64)
    poses[-1, :2] = endpoint
    record = {
        "crossing_lateral_fraction": (
            independent["lateral_coordinate_m"]
            / independent["segment_width_m"]
            + 0.5
        ),
        "endpoint_lateral_error_m": abs(
            endpoint_projection["lateral_coordinate_m"]
        ),
    }
    assert evaluator._validate_teacher_planar_projections(
        metadata,
        {"teacher__base_pose_world": poses},
        metrics,
        teacher_record=record,
    ) == {
        "selected_crossing_checked": True,
        "competing_crossing_checked": False,
        "teacher_record_checked": True,
    }

    crossing_tamper = copy.deepcopy(metadata)
    crossing_tamper["teacher"]["crossing"]["lateral_coordinate_m"] = (
        np.nextafter(
            independent["lateral_coordinate_m"], np.float64(np.inf)
        ).item()
    )
    with pytest.raises(evaluator.RegenerationError, match="lateral coordinate"):
        evaluator._validate_teacher_planar_projections(
            crossing_tamper,
            {"teacher__base_pose_world": poses},
            metrics,
            teacher_record=record,
        )

    record_tamper = copy.deepcopy(record)
    record_tamper["endpoint_lateral_error_m"] = np.nextafter(
        record["endpoint_lateral_error_m"], np.float64(np.inf)
    ).item()
    with pytest.raises(evaluator.RegenerationError, match="teacher endpoint"):
        evaluator._validate_teacher_planar_projections(
            metadata,
            {"teacher__base_pose_world": poses},
            metrics,
            teacher_record=record_tamper,
        )

    pure_drift = SimpleNamespace(
        canonical_v1_planar_segment_projection=lambda *_args: {
            **independent,
            "lateral_coordinate_m": np.nextafter(
                independent["lateral_coordinate_m"], np.float64(np.inf)
            ).item(),
        }
    )
    with pytest.raises(evaluator.RegenerationError, match="independent binary64"):
        evaluator._independent_planar_projection(
            pure_drift, point, segment, label="drifting pure fixture"
        )


def _prefix_trace(prefix: str, trace):
    return {
        f"{prefix}{name}": value
        for name, value in trace.items()
        if hasattr(value, "dtype")
    }


def test_evaluator_independently_enforces_terminal_nonfinite_boundary() -> None:
    import numpy as np

    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    authority = metrics.reducer_authority()["terminal_nonfinite_authority"]
    zero_flags = {
        "fall": False,
        "out_of_bounds": False,
        "tipped": False,
        "nan": False,
    }
    nan_flags = {**zero_flags, "nan": True}
    left = _terminated_probe_trace(flag="nan")
    left["base_pose_world"][-1, 0] = np.asarray(
        [0x7FF8000000000001], dtype=np.uint64
    ).view(np.float64)[0]
    right = copy.deepcopy(left)
    arrays = {
        **_prefix_trace("probe__0__", left),
        **_prefix_trace("probe__1__", right),
    }
    metadata = {
        "state_disposition": {
            "stage_reached": "RESTORATION_PROBE",
            "probe_trial_termination_flags": [nan_flags, nan_flags],
        }
    }
    projection = evaluator._validate_terminal_nonfinite_material(
        metadata, arrays, authority
    )
    assert projection[
        "all_nonfinite_confined_to_authorized_terminal_cells"
    ] is True
    assert projection[
        "probe_pair_nonfinite_masks_and_ieee_payloads_exact"
    ] is True

    false_flag = copy.deepcopy(metadata)
    false_flag["state_disposition"]["probe_trial_termination_flags"] = [
        zero_flags,
        nan_flags,
    ]
    with pytest.raises(evaluator.RegenerationError, match="nan flag"):
        evaluator._validate_terminal_nonfinite_material(
            false_flag, arrays, authority
        )

    preterminal = copy.deepcopy(arrays)
    preterminal["probe__0__base_pose_world"][-2, 0] = np.nan
    with pytest.raises(evaluator.RegenerationError, match="preterminal"):
        evaluator._validate_terminal_nonfinite_material(
            metadata, preterminal, authority
        )

    off_list = copy.deepcopy(arrays)
    off_list["probe__0__requested_command"][-1, 0] = np.inf
    with pytest.raises(evaluator.RegenerationError, match="off-authority"):
        evaluator._validate_terminal_nonfinite_material(
            metadata, off_list, authority
        )

    different_payload = copy.deepcopy(arrays)
    different_payload["probe__1__base_pose_world"][-1, 0] = np.asarray(
        [0x7FF8000000000002], dtype=np.uint64
    ).view(np.float64)[0]
    with pytest.raises(evaluator.RegenerationError, match="IEEE payload"):
        evaluator._validate_terminal_nonfinite_material(
            metadata, different_payload, authority
        )

    finite_probe = _terminated_probe_trace(flag="fall")
    finite_probe["termination_flags"] = zero_flags
    teacher_pose = finite_probe["base_pose_world"].copy()
    teacher_pose[-1, 0] = np.nan
    teacher_arrays = {
        **_prefix_trace("probe__0__", finite_probe),
        **_prefix_trace("probe__1__", finite_probe),
        "teacher__base_pose_world": teacher_pose,
        "teacher__base_twist_world": finite_probe["base_twist_world"],
        "teacher__joint_position": finite_probe["joint_position"],
        "teacher__joint_velocity": finite_probe["joint_velocity"],
        "teacher__physics_contact": finite_probe["physics_contact"],
    }
    teacher_metadata = {
        "state_disposition": {
            "stage_reached": "TEACHER_EXECUTION",
            "probe_trial_termination_flags": [zero_flags, zero_flags],
            "teacher_termination_flags": nan_flags,
        },
        "teacher": {
            "sample_count": len(teacher_pose),
            "termination_flags": nan_flags,
            "terminated_unsafe": True,
            "contact_free": True,
        },
    }
    teacher_projection = evaluator._validate_terminal_nonfinite_material(
        teacher_metadata, teacher_arrays, authority
    )
    assert teacher_projection["groups"][-1] == {
        "group": "teacher",
        "nonfinite_members": ["base_pose_world"],
        "finite_prefix_sample_count": len(teacher_pose) - 1,
    }
    stale_summary = copy.deepcopy(teacher_metadata)
    stale_summary["teacher"]["sample_count"] -= 1
    with pytest.raises(evaluator.RegenerationError, match="summary/raw-prefix"):
        evaluator._validate_terminal_nonfinite_material(
            stale_summary, teacher_arrays, authority
        )


def test_evaluator_raw_previous_command_bridge_rejects_legacy_hash() -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    raw_sha = "1" * 64
    legacy_sha = "2" * 64
    records = [
        {"previous_applied_command_sha256": raw_sha} for _ in range(64)
    ]
    index = {
        "schema": "physical_graph_edge_handoff_qualification_v4.state_snapshot_index.v1",
        "experiment_id": evaluator.EXPERIMENT_ID,
        "records": records,
    }
    assert evaluator._call(
        metrics,
        "validate_state_snapshot_previous_applied_command_hashes",
        index,
        [raw_sha] * 64,
    ) == [raw_sha] * 64
    tampered = copy.deepcopy(index)
    tampered["records"][0]["previous_applied_command_sha256"] = legacy_sha
    with pytest.raises(evaluator.RegenerationError, match="raw-byte"):
        evaluator._call(
            metrics,
            "validate_state_snapshot_previous_applied_command_hashes",
            tampered,
            [raw_sha] * 64,
        )


def test_v4_nonreuse_rejects_any_historical_copy_not_only_npz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator()
    copied_sha = "a" * 64
    historical = {
        "roots": {
            "v1_official_root": {
                "files": [
                    {"path": "contract.json", "sha256": copied_sha,
                     "device": 1, "inode": 10}
                ]
            }
        }
    }
    inventories = iter(
        [
            {
                "files": [
                    {"path": "new.json", "sha256": copied_sha,
                     "device": 2, "inode": 20}
                ]
            },
            {"files": []},
        ]
    )
    monkeypatch.setattr(evaluator, "_root_inventory", lambda *_args: next(inventories))
    with pytest.raises(evaluator.RegenerationError, match="byte-identical"):
        evaluator._validate_v4_nonreuse(
            historical=historical,
            output_root=Path("/unused/official"),
            material_root=Path("/unused/material"),
        )


def test_reducer_authority_rejects_v3_surface_and_publication_tamper() -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    authority = metrics.reducer_authority()
    assert evaluator._validate_reducer_authority(metrics) == authority

    predecessor_leak = copy.deepcopy(authority)
    predecessor_leak.pop("content_digest")
    predecessor_leak["first_eight_reproduction_authority"] = {}
    predecessor_leak = metrics.C.attach_content_digest(predecessor_leak)
    with pytest.raises(evaluator.RegenerationError, match="predecessor-only"):
        evaluator._validate_reducer_authority(
            SimpleNamespace(reducer_authority=lambda: predecessor_leak)
        )

    publication_drift = copy.deepcopy(authority)
    publication_drift.pop("content_digest")
    publication = copy.deepcopy(publication_drift["result_publication_authority"])
    publication.pop("content_digest")
    publication["success_prepublication_binding_count"] = 23
    publication_drift["result_publication_authority"] = (
        metrics.C.attach_content_digest(publication)
    )
    publication_drift = metrics.C.attach_content_digest(publication_drift)
    with pytest.raises(evaluator.RegenerationError, match="publication authority drift"):
        evaluator._validate_reducer_authority(
            SimpleNamespace(reducer_authority=lambda: publication_drift)
        )

    terminal_drift = copy.deepcopy(authority)
    terminal_drift.pop("content_digest")
    terminal = copy.deepcopy(terminal_drift["terminal_nonfinite_authority"])
    terminal.pop("content_digest")
    terminal["false_nan_flag_with_any_nonfinite_is_materialisation_corrupt"] = False
    terminal_drift["terminal_nonfinite_authority"] = metrics.C.attach_content_digest(
        terminal
    )
    terminal_drift = metrics.C.attach_content_digest(terminal_drift)
    with pytest.raises(evaluator.RegenerationError, match="terminal-nonfinite"):
        evaluator._validate_reducer_authority(
            SimpleNamespace(reducer_authority=lambda: terminal_drift)
        )

    runtime_drift = copy.deepcopy(authority)
    runtime_drift.pop("content_digest")
    runtime = copy.deepcopy(runtime_drift["qualification_runtime_authority"])
    runtime.pop("content_digest")
    runtime["backend_runtime_mode_rule"] = "weakened"
    runtime_drift["qualification_runtime_authority"] = (
        metrics.C.attach_content_digest(runtime)
    )
    runtime_drift = metrics.C.attach_content_digest(runtime_drift)
    with pytest.raises(evaluator.RegenerationError, match="runtime authority"):
        evaluator._validate_reducer_authority(
            SimpleNamespace(reducer_authority=lambda: runtime_drift)
        )

    custody_scope_drift = copy.deepcopy(authority)
    custody_scope_drift.pop("content_digest")
    custody = copy.deepcopy(
        custody_scope_drift["v1_v2_v3_custody_and_nonreuse_authority"]
    )
    custody.pop("content_digest")
    custody["zero_counter_scope"] = "ENTIRE_DEVELOPMENT_SESSION"
    custody_scope_drift["v1_v2_v3_custody_and_nonreuse_authority"] = (
        metrics.C.attach_content_digest(custody)
    )
    custody_scope_drift = metrics.C.attach_content_digest(custody_scope_drift)
    with pytest.raises(evaluator.RegenerationError, match="counter scope"):
        evaluator._validate_reducer_authority(
            SimpleNamespace(reducer_authority=lambda: custody_scope_drift)
        )


def test_pre_panel_correction_authority_restart_and_manifest_are_fail_closed() -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    authority = metrics.reducer_authority()
    correction = authority["pre_panel_engineering_correction_authority"]
    assert evaluator._validate_pre_panel_engineering_correction_authority(
        correction
    ) == correction
    assert correction["existing_partial_material_reuse_authorized"] is False
    assert correction["restart_from_fresh_v4_roots_required"] is True
    assert correction["invalidated_source_freeze_commit"] == (
        evaluator.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
    )

    reuse = copy.deepcopy(authority)
    reuse.pop("content_digest")
    weakened = copy.deepcopy(reuse["pre_panel_engineering_correction_authority"])
    weakened.pop("content_digest")
    weakened["existing_partial_material_reuse_authorized"] = True
    reuse["pre_panel_engineering_correction_authority"] = (
        metrics.C.attach_content_digest(weakened)
    )
    reuse = metrics.C.attach_content_digest(reuse)
    with pytest.raises(evaluator.RegenerationError, match="correction authority"):
        evaluator._validate_reducer_authority(
            SimpleNamespace(reducer_authority=lambda: reuse)
        )

    manifest_drift = copy.deepcopy(correction)
    manifest_drift.pop("content_digest")
    manifest = copy.deepcopy(
        manifest_drift["invalidated_partial_root_manifest_authority"]
    )
    manifest.pop("content_digest")
    manifest["roots"]["material"]["file_count"] = 257
    manifest_drift["invalidated_partial_root_manifest_authority"] = (
        metrics.C.attach_content_digest(manifest)
    )
    manifest_drift = metrics.C.attach_content_digest(manifest_drift)
    with pytest.raises(evaluator.RegenerationError, match="partial-root manifest"):
        evaluator._validate_pre_panel_engineering_correction_authority(
            manifest_drift
        )

    binding_drift = copy.deepcopy(authority)
    binding_drift.pop("content_digest")
    state = copy.deepcopy(binding_drift["state_disposition_authority"])
    state.pop("content_digest")
    state["terminal_source_runtime_binding"][
        "all_256_terminal_bindings_must_be_equal"
    ] = False
    binding_drift["state_disposition_authority"] = (
        metrics.C.attach_content_digest(state)
    )
    binding_drift = metrics.C.attach_content_digest(binding_drift)
    with pytest.raises(evaluator.RegenerationError, match="terminal nonreuse"):
        evaluator._validate_reducer_authority(
            SimpleNamespace(reducer_authority=lambda: binding_drift)
        )


def test_development_source_audit_disclosure_and_process_scopes_are_exact() -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    authority = evaluator._validate_historical_receipt_authority(metrics)
    assert authority["development_source_audit_disclosure"] == (
        evaluator.DEVELOPMENT_SOURCE_AUDIT_DISCLOSURE
    )
    assert authority["zero_counter_scope"] == (
        "EXTERNAL_HISTORICAL_CUSTODY_RECEIPT_BUILD_AND_EMISSION_PROCESS_ONLY"
    )
    assert authority["development_source_audit_disclosure"][
        "matched_file_opens_or_reads"
    ] == 0
    assert authority["development_source_audit_disclosure"][
        "scientific_outcomes_contaminated"
    ] is False

    tampered = copy.deepcopy(authority)
    tampered.pop("content_digest")
    tampered["zero_counter_scope"] = "ENTIRE_DEVELOPMENT_SESSION"
    tampered = metrics.C.attach_content_digest(tampered)
    with pytest.raises(evaluator.RegenerationError, match="identity drift"):
        evaluator._validate_historical_receipt_authority(
            SimpleNamespace(historical_custody_authority=lambda: tampered)
        )

    binding = copy.deepcopy(metrics.C.HISTORICAL_CUSTODY_RECEIPT_BINDING)
    runtime = metrics.C.build_runtime_contract("b" * 40, binding)
    assert evaluator._validate_runtime_contract(runtime, metrics) == runtime
    assert runtime["source_freeze_commit"] != (
        evaluator.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
    )
    assert runtime["v4_runtime_policy"]["pre_panel_engineering_correction"] == {
        "status": evaluator.PRE_PANEL_ENGINEERING_CORRECTION_STATUS,
        "authority_content_digest": metrics.C.PRE_PANEL_ENGINEERING_CORRECTION_AUTHORITY[
            "content_digest"
        ],
        "existing_partial_material_reuse_authorized": False,
        "restart_from_fresh_v4_roots_required": True,
    }
    weakened_runtime = copy.deepcopy(runtime)
    weakened_runtime["v4_runtime_policy"]["zero_counter_scope"] = (
        "ENTIRE_DEVELOPMENT_SESSION"
    )
    with pytest.raises(evaluator.RegenerationError, match="source-audit"):
        evaluator._validate_runtime_contract(
            weakened_runtime,
            SimpleNamespace(validate_runtime_contract=lambda value: value),
        )

    invalidated_runtime = copy.deepcopy(runtime)
    invalidated_runtime["source_freeze_commit"] = (
        evaluator.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
    )
    with pytest.raises(evaluator.RegenerationError, match="correction/restart"):
        evaluator._validate_runtime_contract(
            invalidated_runtime,
            SimpleNamespace(validate_runtime_contract=lambda value: value),
        )

    reuse_runtime = copy.deepcopy(runtime)
    reuse_runtime["v4_runtime_policy"]["pre_panel_engineering_correction"][
        "existing_partial_material_reuse_authorized"
    ] = True
    with pytest.raises(evaluator.RegenerationError, match="correction/restart"):
        evaluator._validate_runtime_contract(
            reuse_runtime,
            SimpleNamespace(validate_runtime_contract=lambda value: value),
        )


def _qualification_runtime_fixture(evaluator, metrics, *, fake: bool = False):
    contract = metrics.C
    binding = copy.deepcopy(contract.HISTORICAL_CUSTODY_RECEIPT_BINDING)
    runtime_contract = contract.build_runtime_contract("b" * 40, binding)
    frozen = contract.RUNTIME_ENVIRONMENT_AUTHORITY["physical"]
    stage = {
        "stage_id": frozen["stage_id"],
        "python_executable": frozen["real_python_executable"],
        "python_version": frozen["python_version"],
        "torch_version": frozen["torch_version"],
        "torch_hip_version": frozen["torch_hip_version"],
        "genesis_version": frozen["genesis_version"],
        "quadrants_version": frozen["quadrants_version"],
        "visible_device_count": frozen["visible_device_count"],
        "device": frozen["device"],
        "backend": frozen["backend"],
        "deterministic_environment": copy.deepcopy(
            contract.DIRECT_RUNTIME_POLICY[
                "required_environment_before_simulator_creation"
            ]
        ),
        "fake_runtime": fake,
    }
    metadata = {
        "schema": (
            "physical_graph_edge_handoff_qualification_v4."
            "teacher_pool_terminal.v1"
        ),
        "experiment_id": evaluator.EXPERIMENT_ID,
        "pool_index": 0,
        "source_freeze_commit": runtime_contract["source_freeze_commit"],
        "runtime_contract_content_digest": runtime_contract["content_digest"],
        "stage_runtime": stage,
        "backend_runtime": copy.deepcopy(
            contract.QUALIFICATION_BACKEND_RUNTIME_AUTHORITY
        ),
        "teacher_executed": False,
    }
    return runtime_contract, metadata


def test_qualification_runtime_is_independently_bound_and_production_fake_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    authority = metrics.reducer_authority()
    runtime_contract, metadata = _qualification_runtime_fixture(
        evaluator, metrics
    )
    observed = evaluator._qualification_runtime_digests(
        metadata,
        runtime_contract,
        authority,
        expected_pool_index=0,
    )
    assert observed["stage_runtime"] == metadata["stage_runtime"]
    assert observed["backend_runtime"] == metadata["backend_runtime"]
    assert observed["stage_runtime_sha256"] == hashlib.sha256(
        evaluator.canonical_json_bytes(metadata["stage_runtime"])
    ).hexdigest()

    illegal_teacher_fields = copy.deepcopy(metadata)
    illegal_teacher_fields["backend_runtime"].update(
        {
            "snapshot_captured_before_teacher": True,
            "teacher_restored_from_serialized_snapshot": True,
            "teacher_snapshot_sha256": "c" * 64,
        }
    )
    with pytest.raises(evaluator.RegenerationError, match="backend field"):
        evaluator._qualification_runtime_digests(
            illegal_teacher_fields,
            runtime_contract,
            authority,
            expected_pool_index=0,
        )

    invalidated_source = copy.deepcopy(metadata)
    invalidated_source["source_freeze_commit"] = (
        evaluator.INVALIDATED_V4_SOURCE_FREEZE_COMMIT
    )
    with pytest.raises(evaluator.RegenerationError, match="source/runtime field"):
        evaluator._qualification_runtime_digests(
            invalidated_source,
            runtime_contract,
            authority,
            expected_pool_index=0,
        )

    stale_runtime = copy.deepcopy(metadata)
    stale_runtime["runtime_contract_content_digest"] = (
        evaluator.INVALIDATED_V4_RUNTIME_CONTRACT_CONTENT_DIGEST
    )
    with pytest.raises(evaluator.RegenerationError, match="source/runtime field"):
        evaluator._qualification_runtime_digests(
            stale_runtime,
            runtime_contract,
            authority,
            expected_pool_index=0,
        )

    teacher = copy.deepcopy(metadata)
    teacher["teacher_executed"] = True
    teacher["snapshot_identity"] = {
        "artifact_file_sha256": "c" * 64,
        "snapshot_semantic_digest_v1": "d" * 64,
        "snapshot_behavioural_digest_v1": "e" * 64,
    }
    teacher["initial_decision_state_sha256"] = "c" * 64
    teacher["backend_runtime"].update(
        {
            "snapshot_captured_before_teacher": True,
            "teacher_restored_from_serialized_snapshot": True,
            "teacher_snapshot_sha256": "c" * 64,
        }
    )
    assert evaluator._qualification_runtime_digests(
        teacher,
        runtime_contract,
        authority,
        expected_pool_index=0,
    )["backend_runtime"]["teacher_snapshot_sha256"] == "c" * 64
    teacher["backend_runtime"]["teacher_snapshot_sha256"] = "f" * 64
    with pytest.raises(evaluator.RegenerationError, match="snapshot drift"):
        evaluator._qualification_runtime_digests(
            teacher,
            runtime_contract,
            authority,
            expected_pool_index=0,
        )

    _runtime, fake_metadata = _qualification_runtime_fixture(
        evaluator, metrics, fake=True
    )
    all_fake = []
    for index in range(256):
        row = copy.deepcopy(fake_metadata)
        row["pool_index"] = index
        all_fake.append(row)
    with pytest.raises(evaluator.RegenerationError, match="fake runtime"):
        evaluator._qualification_runtime_digests(
            fake_metadata,
            runtime_contract,
            authority,
            expected_pool_index=0,
        )
    with pytest.raises(evaluator.RegenerationError, match="fake runtime"):
        evaluator._call(
            metrics,
            "build_qualification_runtime_environment",
            all_fake,
            runtime_contract,
        )

    # A test may replace exactly this private early guard after first proving
    # the production rejection.  No public evaluator or CLI option exposes
    # the pure module's explicit synthetic-fixture allowance.
    monkeypatch.setattr(
        evaluator, "_require_real_qualification_runtime", lambda _stage: None
    )
    assert evaluator._qualification_runtime_digests(
        fake_metadata,
        runtime_contract,
        authority,
        expected_pool_index=0,
    )["stage_runtime"]["fake_runtime"] is True
    for name in (
        "build_regeneration_receipt",
        "verify_and_emit",
        "validate_existing_regeneration_receipt",
    ):
        assert "allow_fake_runtime" not in inspect.signature(
            getattr(evaluator, name)
        ).parameters
    assert "--allow-fake" not in SCRIPT.read_text(encoding="utf-8")


def test_probe_trial_behavioural_digest_and_state_links_are_raw_derived() -> None:
    import numpy as np

    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    contract = metrics.C
    trace = {}
    for member, spec in contract.V4_PROBE_TRACE_MEMBER_AUTHORITY.items():
        trace[member] = np.zeros(
            [750, *spec["shape"][1:]], dtype=np.dtype(spec["descr"])
        )
    trace["timestamp_s"][:] = np.arange(1, 751, dtype=np.float64) * 0.002
    trace["base_pose_world"][:, 6] = 1.0
    trace["requested_command"][:] = np.asarray(
        contract.V3.BEHAVIOURAL_PROBE_REQUESTED_LEDGER_COMMAND,
        dtype=np.float64,
    )
    trace["post_slew_applied_command"][:, 0] = 0.2
    final_digest = "a" * 64
    digest_input = {
        **trace,
        "termination_reason": "H3_COMPLETE",
        "stuck": True,
        "final_snapshot_semantic_digest_v1": final_digest,
    }
    behavioural_digest = metrics.snapshot_behavioural_digest(digest_input)
    zero_flags = {name: False for name in contract.TERMINATION_FLAG_ORDER}
    trial = {
        "trial_index": 0,
        "termination_reason": "H3_COMPLETE",
        "termination_flags": zero_flags,
        "tipped": False,
        "contact": False,
        "stuck": True,
        "tip_sample_index": None,
        "final_executable_snapshot_exists": True,
        "final_snapshot_semantic_digest_v1": final_digest,
        "snapshot_behavioural_digest_v1": behavioural_digest,
        "trace_member_manifest": [
            {"member": member}
            for member in contract.V4_PROBE_TRACE_MEMBER_AUTHORITY
        ],
    }
    identity = {
        "artifact_file_sha256": "b" * 64,
        "snapshot_semantic_digest_v1": "c" * 64,
        "snapshot_behavioural_digest_v1": behavioural_digest,
    }
    metadata = {
        "stage_reached": "RESTORATION_PROBE",
        "snapshot_identity": identity,
        "state_disposition": {
            "snapshot_identity": identity,
            "probe_trial_termination_flags": [zero_flags, zero_flags],
            "probe_tip_sample_indices": [None, None],
        },
        "behavioural_probe": {
            "completed": True,
            "trials": [trial, {**copy.deepcopy(trial), "trial_index": 1}],
        },
    }
    arrays = {
        **_prefix_trace("probe__0__", trace),
        **_prefix_trace("probe__1__", trace),
    }
    projection = evaluator._validate_probe_state_crosslinks(
        metadata, arrays, metrics
    )
    assert projection["trial_snapshot_behavioural_digest_v1s"] == [
        behavioural_digest,
        behavioural_digest,
    ]

    raw_tamper = copy.deepcopy(arrays)
    # Preserve the frozen ten-physics-sample policy-act repetition while
    # changing the raw trial digest.
    raw_tamper["probe__1__controller_observation"][10:20, 0] = 1.0
    with pytest.raises(evaluator.RegenerationError, match="behavioural digest"):
        evaluator._validate_probe_state_crosslinks(
            metadata, raw_tamper, metrics
        )
    state_tamper = copy.deepcopy(metadata)
    state_tamper["state_disposition"]["probe_tip_sample_indices"][0] = 7
    with pytest.raises(evaluator.RegenerationError, match="terminal evidence"):
        evaluator._validate_probe_state_crosslinks(
            state_tamper, arrays, metrics
        )


def test_source_freeze_injection_requires_exact_counts_and_metrics_identity() -> None:
    evaluator = _load_evaluator()
    row = {
        "head_commit": "a" * 40,
        "parent_commit": evaluator.SOURCE_PARENT_COMMIT,
        "freeze_subject": evaluator.FREEZE_SUBJECT,
        "worktree_clean": True,
        "tracked_source_count": 15,
        "tracked_sources_sha256": hashlib.sha256(b"tracked").hexdigest(),
        "source_closure_path": "docs/v4_source_closure.json",
        "source_closure_bytes": 123,
        "source_closure_sha256": hashlib.sha256(b"closure").hexdigest(),
        "source_closure_row_count": 89,
        "source_closure_live_bytes_exact": True,
        "sealed_path_accesses": 0,
        "ignore_bypasses": 0,
        "metrics_module": evaluator.METRICS_MODULE,
    }
    assert evaluator._validate_source_freeze_observation(
        row,
        expected_commit="a" * 40,
        expected_tracked_source_count=15,
        expected_source_closure_row_count=89,
        expected_metrics_module=evaluator.METRICS_MODULE,
    ) == row
    for field, value in (
        ("tracked_source_count", 14),
        ("source_closure_row_count", 88),
        ("metrics_module", "wrong.module"),
    ):
        tampered = {**row, field: value}
        with pytest.raises(evaluator.RegenerationError, match="observation failed"):
            evaluator._validate_source_freeze_observation(
                tampered,
                expected_commit="a" * 40,
                expected_tracked_source_count=15,
                expected_source_closure_row_count=89,
                expected_metrics_module=evaluator.METRICS_MODULE,
            )


def test_compact_digest_domain_does_not_drop_canonical_closing_brace() -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    value = {"nested": {"value": 1}, "rows": [1, 2, 3]}
    compact = evaluator.canonical_json_bytes(value)
    assert compact.endswith(b"}")
    assert compact == metrics.C.canonical_json_bytes(value)[:-1]
    assert hashlib.sha256(compact).hexdigest() == hashlib.sha256(
        metrics.C.canonical_json_bytes(value)[:-1]
    ).hexdigest()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "canonical_json_bytes(persisted)[:-1]" not in source
    assert "canonical_json_bytes(state_projection)[:-1]" not in source
    assert "canonical_json_bytes(qualification)[:-1]" not in source


def _synthetic_success_cardinality_evidence(evaluator, authority):
    def row(fields):
        return {field: None for field in fields}

    documents = {}
    for name, schema in authority["documents"].items():
        documents[name] = {field: None for field in schema["root"]}
        if name == "development_target_selection":
            documents[name][schema["state_target_container"]] = [
                row(schema["state_target_row"])
                for _ in range(schema["state_target_count"])
            ]
            documents[name][schema["summary_container"]] = [
                row(schema["summary_row"])
                for _ in range(schema["summary_count"])
            ]
            continue
        count = schema["count"]
        if name == "teacher_trace_index":
            count = schema["minimum_count"]
        elif name == "latent_index":
            count = 64
        documents[name][schema["container"]] = [
            row(schema["row"]) for _ in range(count)
        ]

    panel_schema = authority["documents"]["panel_manifest"]
    qualification = [
        row(panel_schema["qualification_row"])
        for _ in range(panel_schema["qualification_count"])
    ]
    for index, item in enumerate(qualification):
        item["teacher_trace_id"] = (
            f"teacher-{index:03d}" if index < 64 else None
        )
    documents["panel_manifest"]["prospective_pool_selection"] = {
        "qualification_rows": qualification
    }
    for index, item in enumerate(documents["teacher_trace_index"]["records"]):
        item["qualification_pool_index"] = index
    documents["pixel_index"]["unique_pixel_count"] = 64

    ledgers = {
        name: [row(schema["fields"]) for _ in range(schema["count"])]
        for name, schema in authority["ledgers"].items()
    }
    return documents, ledgers


def test_success_cardinality_projection_uses_every_frozen_container_and_count() -> None:
    evaluator = _load_evaluator()
    metrics = evaluator._load_metrics_module()
    authority = metrics.reducer_authority()
    documents, ledgers = _synthetic_success_cardinality_evidence(
        evaluator, authority
    )

    document_rows, ledger_rows, teacher_pool_indices = (
        evaluator._success_cardinality_projection(
            authority, documents, ledgers
        )
    )
    assert document_rows == {
        "panel_manifest": 64,
        "panel_qualification": 256,
        "split_manifest": 64,
        "graph_manifest": 64,
        "state_snapshot_index": 64,
        "teacher_trace_index": 64,
        "edge_port_index": 64,
        "waypoint_contracts": 192,
        "pixel_index": 64,
        "latent_index": 64,
        "development_target_selection.state_target_rows": 144,
        "development_target_selection.target_summaries": 3,
    }
    assert ledger_rows == {
        "candidate_fanout": 768,
        "heldout_ranker_scores": 64,
        "repeated_execution": 64,
    }
    assert teacher_pool_indices == list(range(64))

    # These are the adjacent container-name families that previously exposed
    # stale generic ``records`` assumptions.  A synthetic success summary must
    # fail closed if any one is projected through that wrong container.
    for name, correct_container in (
        ("panel_manifest", "states"),
        ("split_manifest", "assignments"),
        ("graph_manifest", "graphs"),
        ("waypoint_contracts", "rows"),
    ):
        tampered_documents = copy.deepcopy(documents)
        values = tampered_documents[name].pop(correct_container)
        tampered_documents[name]["records"] = values
        with pytest.raises(
            evaluator.RegenerationError, match="root-field drift"
        ):
            evaluator._success_cardinality_projection(
                authority, tampered_documents, ledgers
            )

    missing_selection_summary = copy.deepcopy(documents)
    missing_selection_summary["development_target_selection"][
        "target_summaries"
    ].pop()
    with pytest.raises(evaluator.RegenerationError, match="document row count"):
        evaluator._success_cardinality_projection(
            authority, missing_selection_summary, ledgers
        )

    stale_authority = copy.deepcopy(authority)
    stale_authority["documents"]["graph_manifest"]["container"] = "records"
    with pytest.raises(evaluator.RegenerationError, match="container authority"):
        evaluator._success_cardinality_projection(
            stale_authority, documents, ledgers
        )


def test_historical_receipt_rebuild_is_read_only_and_deterministic_when_present() -> None:
    evaluator = _load_evaluator()
    required = [
        path
        for _name, path in evaluator._historical_root_paths()
    ] + [
        evaluator.DEFAULT_V1_CUSTODY_RECEIPT,
        evaluator.DEFAULT_V1_V2_CUSTODY_RECEIPT,
    ]
    if not all(path.exists() for path in required):
        pytest.skip("frozen PGEHQ historical custody roots are unavailable")
    receipt_existed = evaluator.DEFAULT_HISTORICAL_CUSTODY_RECEIPT.exists()
    receipt_before = (
        evaluator._read_regular(
            evaluator.DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
            "existing V1/V2/V3 custody receipt",
        )
        if receipt_existed
        else None
    )
    before = {
        name: evaluator._root_inventory(path, name)
        for name, path in evaluator._historical_root_paths()
    }
    require_v4_absent = not any(
        path.exists() or path.is_symlink()
        for path in (
            evaluator.DEFAULT_V4_OUTPUT_ROOT,
            evaluator.DEFAULT_V4_MATERIAL_ROOT,
            evaluator.DEFAULT_EXTERNAL_RECEIPT,
        )
    )
    first = evaluator.build_historical_custody_receipt(
        require_v4_absent=require_v4_absent
    )
    second = evaluator.build_historical_custody_receipt(
        require_v4_absent=require_v4_absent
    )
    canonical = evaluator.canonical_document_bytes(first)
    assert canonical == evaluator.canonical_document_bytes(second)
    reparsed = evaluator.parse_canonical_json(
        canonical, label="reopened V1/V2/V3 custody receipt"
    )
    assert evaluator.validate_historical_custody_receipt_document(reparsed) == reparsed
    reordered = copy.deepcopy(first)
    reordered["roots"] = {
        key: reordered["roots"][key] for key in reversed(tuple(reordered["roots"]))
    }
    assert evaluator.validate_historical_custody_receipt_document(reordered) == first
    manifest_tamper = copy.deepcopy(reparsed)
    manifest_tamper["roots"]["v3_material_root"]["files"][0:2] = reversed(
        manifest_tamper["roots"]["v3_material_root"]["files"][0:2]
    )
    with pytest.raises(evaluator.RegenerationError):
        evaluator.validate_historical_custody_receipt_document(manifest_tamper)
    assert first["generated_before_v4_simulator_creation"] is True
    assert first["repository"]["v4_development_only"] is True
    assert first["repository"][
        "v4_permanently_ineligible_for_final_evaluation"
    ] is True
    assert first["repository"]["sealed_paths_accessed"] == 0
    assert first["repository"]["tracked_ignore_bypass_used"] is False
    assert first["nonreuse"]["historical_snapshot_deserializations"] == 0
    assert first["v3_terminal_interpretation"] == (
        evaluator.V3_TERMINAL_INTERPRETATION
    )
    assert first["v3_terminal_interpretation"]["diagnosis"] == (
        "TIPPED_BOUNDARY_STATE_DISPOSITION_UNSPECIFIED"
    )
    interpretation_tamper = copy.deepcopy(first)
    interpretation_tamper["v3_terminal_interpretation"][
        "snapshot_behavioural_equivalence_passed"
    ] = False
    with pytest.raises(evaluator.RegenerationError):
        evaluator.validate_historical_custody_receipt_document(
            interpretation_tamper
        )
    after = {
        name: evaluator._root_inventory(path, name)
        for name, path in evaluator._historical_root_paths()
    }
    assert before == after
    if receipt_existed:
        assert evaluator._read_regular(
            evaluator.DEFAULT_HISTORICAL_CUSTODY_RECEIPT,
            "existing V1/V2/V3 custody receipt",
        ) == receipt_before
    else:
        assert not evaluator.DEFAULT_HISTORICAL_CUSTODY_RECEIPT.exists()
