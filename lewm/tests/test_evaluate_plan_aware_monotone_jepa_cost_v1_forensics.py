from __future__ import annotations

import ast
import copy
import json
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_plan_aware_monotone_jepa_cost_v1 as evaluator
from scripts import (
    run_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_child as wrapper,
)


def test_archive_identity_projection_excludes_validator_status_fields() -> None:
    minimal_runtime = [
        {
            "archive_path": "/technical/archive-one",
            "source_freeze_commit": "c" * 40,
            "inventory": {"files": 2, "bytes": 17, "manifest_sha256": "a" * 64},
            "failure_receipt": {"path": "receipts/failure.json"},
            "files_reused": 0,
            "partial_artifacts_reusable": False,
        },
        {
            "archive_path": "/technical/archive-two",
            "source_freeze_commit": "d" * 40,
            "inventory": {"files": 3, "bytes": 23, "manifest_sha256": "b" * 64},
            "failure_receipt": {"path": "receipts/failure.json"},
            "files_reused": 0,
            "partial_artifacts_reusable": False,
        },
    ]
    detailed_validator = copy.deepcopy(minimal_runtime)
    detailed_validator[0].pop("partial_artifacts_reusable")
    detailed_validator[0].update(
        {
            "source_closure_snapshot": {"path": "receipts/source_closure.json"},
            "stage_b_gate_receipt": {"path": "receipts/stage_b_gate.json"},
            "nothing_running": True,
            "pass": True,
        }
    )
    detailed_validator[1].update(
        {
            "persistence_receipt": {"path": "persistence.json"},
            "stage_c_executed": False,
            "full_inventory_verified": False,
            "pass": True,
        }
    )

    # The exact frozen defect compares minimal runtime custody with detailed
    # validator records. Proof strength is one extra detail, not the sole
    # schema mismatch.
    assert minimal_runtime != detailed_validator
    assert all(
        set(left) != set(right)
        for left, right in zip(minimal_runtime, detailed_validator)
    )
    assert evaluator._correction_2_failed_archive_identity_matches(
        minimal_runtime, detailed_validator
    )

    changed_identity = copy.deepcopy(detailed_validator)
    changed_identity[1]["inventory"]["bytes"] += 1
    assert not evaluator._correction_2_failed_archive_identity_matches(
        minimal_runtime, changed_identity
    )


def test_exact_incompatible_archive_schema_fixture_projects_stable_identity() -> None:
    evidence = evaluator._synthetic_archive_validation_projection_evidence()
    assert evidence["raw_record_equality"] is False
    assert evidence["identity_projection_equality"] is True
    assert evidence["mismatch_mechanism"] == (
        "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
    )
    assert all(
        minimal != detailed
        for minimal, detailed in zip(
            evidence["minimal_key_sets"], evidence["detailed_key_sets"]
        )
    )
    assert (
        evaluator._forensic_contract().validate_archive_schema_mismatch_evidence(
            evidence
        )
        == evidence
    )


def test_preexisting_scientific_functions_remain_ast_identical_to_source() -> None:
    correction = evaluator._forensic_correction_1_contract()
    proof = correction.build_correction_overlay_ast_proof(evaluator.ROOT)
    assert proof["pass"] is True
    assert proof["scientific_design_unchanged"] is True
    assert proof["all_unlisted_legacy_nodes_unchanged"] is True
    assert proof["removed_legacy_nodes"] == []
    evaluator_proof = proof["files"][
        "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
    ]
    assert set(evaluator_proof["changed_legacy_function_ast_sha256"]) == set(
        correction.CORRECTION_OVERLAY_SEMANTIC_ALLOW_LIST[
            "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
        ]["changed_legacy_functions"]
    )


def test_preexisting_top_level_constants_and_classes_are_ast_identical() -> None:
    relative = "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
    committed = subprocess.run(
        ["git", "show", f"1c18af8c3c45f9b14992362f7e50a35b651c6997:{relative}"],
        cwd=evaluator.ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    before = ast.parse(committed)
    after = ast.parse((evaluator.ROOT / relative).read_text(encoding="utf-8"))

    def assignments(tree: ast.Module) -> dict[str, str]:
        output: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) for target in node.targets
            ):
                for target in node.targets:
                    output[target.id] = ast.dump(node.value, include_attributes=False)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                output[node.target.id] = ast.dump(
                    node.value, include_attributes=False
                )
        return output

    before_assignments = assignments(before)
    after_assignments = assignments(after)
    assert set(before_assignments).issubset(after_assignments)
    assert {
        name: after_assignments[name] for name in before_assignments
    } == before_assignments
    before_classes = {
        node.name: ast.dump(node, include_attributes=False)
        for node in before.body
        if isinstance(node, ast.ClassDef)
    }
    after_classes = {
        node.name: ast.dump(node, include_attributes=False)
        for node in after.body
        if isinstance(node, ast.ClassDef)
    }
    assert {name: after_classes[name] for name in before_classes} == before_classes


def test_forensic_overlay_closure_never_hashes_scientific_payload_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forensic = evaluator._forensic_contract()
    opened: list[Path] = []
    original = forensic._read_no_follow_bound_file

    def recording_read(
        root: Path,
        relative: str | Path,
        *,
        expected_stat_rows: object,
    ) -> bytes:
        opened.append((Path(root) / relative).resolve(strict=False))
        return original(
            root,
            relative,
            expected_stat_rows=expected_stat_rows,
        )

    monkeypatch.setattr(forensic, "_read_no_follow_bound_file", recording_read)
    closure = forensic.build_forensic_source_closure(
        evaluator.ROOT, require_complete=False
    )
    declared = {
        (evaluator.ROOT / relative).resolve()
        for relative in forensic.FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS
    }
    assert opened
    assert set(opened).issubset(declared)
    forbidden = {
        Path(path).resolve(strict=False)
        for path in forensic.scientific_forbidden_bindings(
            evaluator.ROOT
        ).values()
    }
    assert set(opened).isdisjoint(forbidden)
    assert closure["base_execution_correction_2_source_closure"] == (
        forensic.BASE_EXECUTION_CORRECTION_2_SOURCE_CLOSURE_BINDING
    )


def test_correction_closure_loader_uses_public_no_follow_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"schema": "synthetic-correction-closure"}
    roots: list[Path] = []

    def load(repo_root: Path) -> dict[str, str]:
        roots.append(Path(repo_root))
        return expected

    def reject_direct_read(_path: Path) -> bytes:
        raise AssertionError("correction closure must not use Path.read_bytes")

    monkeypatch.setattr(Path, "read_bytes", reject_direct_read)
    correction = SimpleNamespace(
        load_and_validate_forensic_correction_source_closure=load
    )
    assert evaluator._load_forensic_correction_source_closure(correction) is expected
    assert roots == [evaluator.ROOT]

    correction.load_and_validate_forensic_correction_source_closure = (
        lambda _repo_root: (_ for _ in ()).throw(
            RuntimeError("correction source closure symlink drift")
        )
    )
    with pytest.raises(evaluator.QualificationError, match="symlink drift"):
        evaluator._load_forensic_correction_source_closure(correction)


def test_correction_wrapper_reuses_real_child_path_without_synthetic_mode() -> None:
    root = evaluator.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
    args = SimpleNamespace(
        mode=wrapper.CORRECTION_REAL_MODE,
        launcher_pid=17,
        launcher_start_time_ticks=23,
        fixture_id=None,
        diagnostic_root=root,
    )
    assert wrapper._inner_argv(args) == [
        str(wrapper.EVALUATOR),
        evaluator.PREEXECUTION_DIAGNOSTIC_CORRECTION_CHILD_SUBCOMMAND,
        "--launcher-pid",
        "17",
        "--launcher-start-time-ticks",
        "23",
        "--diagnostic-root",
        str(root),
    ]
    assert not hasattr(wrapper, "CORRECTION_SYNTHETIC_MODE")


def test_correction_terminal_phase_argv_roles_are_exact_and_distinct() -> None:
    correction = evaluator._validate_forensic_correction_1_constant_alignment()
    executable = str(evaluator.FORENSIC_INTERPRETER.resolve())
    cases = (
        (
            correction.expected_correction_finalizer_outer_argv(),
            "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_FINALIZER",
            correction.expected_correction_finalizer_inner_argv(),
        ),
        (
            correction.expected_correction_checker_outer_argv(),
            "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_CHECKER",
            correction.expected_correction_checker_inner_argv(),
        ),
    )
    for outer, role, inner in cases:
        assert evaluator._classify_forensic_argv(
            outer, executable=executable
        ) == role
        assert evaluator._classify_forensic_argv(
            [*outer, "--extra"], executable=executable
        ) == "INVALID_FORENSIC_ARGV"
        assert evaluator._classify_forensic_argv(
            [str(evaluator.FORENSIC_INTERPRETER), *inner],
            executable=executable,
        ) == "INVALID_FORENSIC_ARGV"
        parsed = wrapper._correction_phase_arguments(list(outer[5:]))
        assert parsed.mode == outer[6]
        assert parsed.diagnostic_root == correction.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
    assert cases[0][0] != cases[1][0]
    assert cases[0][2] != cases[1][2]
    assert not any("-fd" in value for outer, _role, _inner in cases for value in outer)


def test_correction_forensic_scanner_fails_visible_when_authority_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    correction = evaluator._validate_forensic_correction_1_constant_alignment()
    correction_argv = (
        correction.expected_correction_launcher_argv(),
        correction.expected_correction_finalizer_outer_argv(),
        correction.expected_correction_checker_outer_argv(),
    )

    def unavailable() -> object:
        raise evaluator.QualificationError("synthetic correction import failure")

    monkeypatch.setattr(
        evaluator,
        "_validate_forensic_correction_1_constant_alignment",
        unavailable,
    )
    for argv in correction_argv:
        assert evaluator._classify_forensic_argv(
            argv,
            executable=str(evaluator.FORENSIC_INTERPRETER.resolve()),
        ) == "INVALID_FORENSIC_ARGV"


def test_correction_phase_wrapper_emits_one_valid_success_envelope(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    correction = evaluator._validate_forensic_correction_1_constant_alignment()
    phase = "FINALIZER"
    phase_identity = {
        "pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "start_time_ticks": 17,
        "argv": correction.expected_correction_finalizer_outer_argv(),
        "argv_sha256": correction.BASE.canonical_json_sha256(
            correction.expected_correction_finalizer_outer_argv()
        ),
        "executable": str(correction.BASE.FORENSIC_INTERPRETER_RESOLVED),
        "role": correction.CORRECTION_FINALIZER_ROLE,
    }
    coordinator = {
        "pid": os.getpid() + 100_000,
        "process_group_id": os.getpgrp(),
        "start_time_ticks": 23,
        "argv": correction.expected_correction_terminal_coordinator_argv(),
        "argv_sha256": correction.BASE.canonical_json_sha256(
            correction.expected_correction_terminal_coordinator_argv()
        ),
        "executable": str(correction.BASE.FORENSIC_INTERPRETER_RESOLVED),
        "role": "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_ROOT_COORDINATOR",
    }
    receipt = correction.BASE.attach_self_digest(
        {"schema": "synthetic-finalizer-receipt.v1", "pass": True}
    )
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("PYTHONFAULTHANDLER", "1")
    monkeypatch.setenv(
        "VIRTUAL_ENV", str(correction.BASE.FORENSIC_INTERPRETER.parent.parent)
    )
    monkeypatch.setenv(
        "PATH",
        str(correction.BASE.FORENSIC_INTERPRETER.parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
    )
    monkeypatch.setattr(wrapper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrapper, "_correction_phase_contract", lambda: correction)
    monkeypatch.setattr(
        wrapper,
        "_correction_phase_process_identity",
        lambda _correction, *, phase: phase_identity,
    )
    monkeypatch.setattr(
        wrapper,
        "_correction_phase_coordinator",
        lambda _correction: coordinator,
    )
    monkeypatch.setattr(
        wrapper,
        "_correction_phase_inner_receipt",
        lambda _correction, *, phase: receipt,
    )
    argv = [
        "--mode",
        wrapper.CORRECTION_FINALIZER_MODE,
        "--diagnostic-root",
        str(wrapper.CORRECTION_DIAGNOSTIC_ROOT),
    ]
    assert wrapper._run_correction_phase_wrapper(argv) == 0
    stdout, stderr = capfd.readouterr()
    assert stderr == ""
    envelope = json.loads(stdout)
    assert stdout.encode("utf-8") == correction._authority_bytes(envelope)
    assert correction.validate_correction_phase_wrapper_envelope(
        envelope,
        phase=phase,
        expected_process_identity=phase_identity,
        expected_coordinator_identity=coordinator,
        expected_environment=dict(os.environ),
    ) == envelope
    assert envelope["phase_receipt"] == receipt
    assert envelope["phase_success"] is True


def test_correction_phase_wrapper_preserves_utf8_failure_envelope(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    correction = evaluator._validate_forensic_correction_1_constant_alignment()
    phase = "CHECKER"
    phase_identity = {
        "pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "start_time_ticks": 29,
        "argv": correction.expected_correction_checker_outer_argv(),
        "argv_sha256": correction.BASE.canonical_json_sha256(
            correction.expected_correction_checker_outer_argv()
        ),
        "executable": str(correction.BASE.FORENSIC_INTERPRETER_RESOLVED),
        "role": correction.CORRECTION_CHECKER_ROLE,
    }
    coordinator = {
        "pid": os.getpid() + 100_001,
        "process_group_id": os.getpgrp(),
        "start_time_ticks": 31,
        "argv": correction.expected_correction_terminal_coordinator_argv(),
        "argv_sha256": correction.BASE.canonical_json_sha256(
            correction.expected_correction_terminal_coordinator_argv()
        ),
        "executable": str(correction.BASE.FORENSIC_INTERPRETER_RESOLVED),
        "role": "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_ROOT_COORDINATOR",
    }
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setenv("PYTHONFAULTHANDLER", "1")
    monkeypatch.setenv(
        "VIRTUAL_ENV", str(correction.BASE.FORENSIC_INTERPRETER.parent.parent)
    )
    monkeypatch.setenv(
        "PATH",
        str(correction.BASE.FORENSIC_INTERPRETER.parent)
        + os.pathsep
        + os.environ.get("PATH", ""),
    )
    monkeypatch.setattr(wrapper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrapper, "_correction_phase_contract", lambda: correction)
    monkeypatch.setattr(
        wrapper,
        "_correction_phase_process_identity",
        lambda _correction, *, phase: phase_identity,
    )
    monkeypatch.setattr(
        wrapper,
        "_correction_phase_coordinator",
        lambda _correction: coordinator,
    )

    def fail_inner(_correction: object, *, phase: str) -> dict[str, object]:
        raise RuntimeError(f"{phase} H1–H4 synthetic failure")

    monkeypatch.setattr(wrapper, "_correction_phase_inner_receipt", fail_inner)
    argv = [
        "--mode",
        wrapper.CORRECTION_CHECKER_MODE,
        "--diagnostic-root",
        str(wrapper.CORRECTION_DIAGNOSTIC_ROOT),
    ]
    assert wrapper._run_correction_phase_wrapper(argv) == 1
    stdout, stderr = capfd.readouterr()
    assert "H1–H4 synthetic failure" in stderr
    envelope = json.loads(stdout)
    assert stdout.encode("utf-8") == correction._authority_bytes(envelope)
    assert correction.validate_correction_phase_wrapper_envelope(
        envelope,
        phase=phase,
        expected_process_identity=phase_identity,
        expected_coordinator_identity=coordinator,
        expected_environment=dict(os.environ),
    ) == envelope
    assert envelope["phase_receipt"] is None
    assert envelope["phase_success"] is False
    assert envelope["structured_exception"]["exception_type"] == "RuntimeError"
    assert envelope["last_stage_marker"]["event"] == "FAILED"


def test_correction_terminal_evaluator_entrypoints_bind_outer_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    finalizer_identity = {"role": "FINALIZER", "pid": 41}
    checker_identity = {"role": "CHECKER", "pid": 43}
    coordinator = {"role": "COORDINATOR", "pid": 47}
    calls: list[tuple[str, object, object, Path]] = []

    def build_finalizer(
        *, repo_root: Path, finalizer_process_identity: object,
        coordinator_process_identity: object,
    ) -> dict[str, object]:
        calls.append(
            (
                "FINALIZER",
                finalizer_process_identity,
                coordinator_process_identity,
                repo_root,
            )
        )
        return {"phase": "FINALIZER", "pass": True}

    def build_checker(
        *, repo_root: Path, checker_process_identity: object,
        coordinator_process_identity: object,
    ) -> dict[str, object]:
        calls.append(
            (
                "CHECKER",
                checker_process_identity,
                coordinator_process_identity,
                repo_root,
            )
        )
        return {"phase": "CHECKER", "pass": True}

    correction = SimpleNamespace(
        CORRECTION_FINALIZER_ROLE="FINALIZER",
        CORRECTION_CHECKER_ROLE="CHECKER",
        build_correction_finalizer_prospective_receipt=build_finalizer,
        build_correction_external_checker_receipt=build_checker,
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_forensic_correction_1_constant_alignment",
        lambda: correction,
    )
    monkeypatch.setattr(
        evaluator,
        "_correction_terminal_coordinator_from_environment",
        lambda _correction: coordinator,
    )

    def identity(_pid: int, *, require_role: str) -> dict[str, object]:
        return finalizer_identity if require_role == "FINALIZER" else checker_identity

    monkeypatch.setattr(evaluator, "_forensic_process_identity", identity)
    assert evaluator.finalize_preexecution_forensic_correction_1() == {
        "phase": "FINALIZER",
        "pass": True,
    }
    assert evaluator.check_preexecution_forensic_correction_1() == {
        "phase": "CHECKER",
        "pass": True,
    }
    assert calls == [
        ("FINALIZER", finalizer_identity, coordinator, evaluator.ROOT),
        ("CHECKER", checker_identity, coordinator, evaluator.ROOT),
    ]


def test_correction_active_freeze_adapter_returns_only_contract_evidence() -> None:
    freeze = {"content_digest": "a" * 64}
    expected = {
        "forensic_freeze_custody": freeze,
        "base_authorities_read_only_validated": True,
        "archive_custody_metadata_only_validated": True,
        "pass": True,
    }
    calls: list[tuple[Path, object, Path]] = []

    def validate(
        repo_root: Path,
        *,
        forensic_freeze_custody: object,
        diagnostic_root: Path,
    ) -> dict[str, object]:
        calls.append(
            (repo_root, forensic_freeze_custody, diagnostic_root)
        )
        return expected

    correction = SimpleNamespace(
        validate_active_forensic_correction_freeze_custody=validate
    )
    assert evaluator._validate_preexecution_active_freeze(
        root=evaluator.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
        freeze_receipt=freeze,
        correction=correction,
    ) is expected
    assert calls == [
        (
            evaluator.ROOT,
            freeze,
            evaluator.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT,
        )
    ]


def test_authority_read_guard_manifest_loads_in_stdlib_wrapper(
    tmp_path: Path,
) -> None:
    manifest = evaluator._build_preexecution_read_guard_manifest()
    path = tmp_path / "read_guard_manifest.json"
    path.write_bytes(evaluator.canonical_bytes(manifest))
    assert wrapper._load_read_guard_manifest(path) == manifest


def test_read_guard_admission_is_exact_and_read_only() -> None:
    manifest = evaluator._build_preexecution_read_guard_manifest()
    forbidden = tuple(
        Path(value).resolve(strict=False)
        for value in manifest["forbidden_path_prefixes"]
    )
    admitted = {
        Path(value).resolve(strict=False)
        for value in manifest["admitted_exact_read_only_technical_receipts"]
    }
    admitted_path = sorted(admitted, key=str)[0]
    assert wrapper._read_guard_disposition(
        forbidden=forbidden,
        admitted=admitted,
        path=admitted_path,
        mode="rb",
        flags=os.O_RDONLY,
    ) == "ALLOW_ADMITTED_READ"
    assert wrapper._read_guard_disposition(
        forbidden=forbidden,
        admitted=admitted,
        path=admitted_path,
        mode="wb",
        flags=os.O_WRONLY | os.O_TRUNC,
    ) == "DENY"
    assert wrapper._read_guard_disposition(
        forbidden=forbidden,
        admitted=admitted,
        path=admitted_path.with_name("unbound-sibling.json"),
        mode="rb",
        flags=os.O_RDONLY,
    ) == "DENY"
    assert wrapper._read_guard_disposition(
        forbidden=forbidden,
        admitted=admitted,
        path=admitted_path.parent.parent,
        mode="rb",
        flags=os.O_RDONLY,
    ) == "DENY"
    for inner_root in evaluator._diagnostic_forbidden_scientific_roots():
        assert any(
            inner_root == outer_root
            or inner_root.is_relative_to(outer_root)
            for outer_root in forbidden
        )
    for label, raw_path in manifest["scientific_forbidden_bindings"].items():
        assert label
        path = Path(raw_path).resolve(strict=False)
        assert any(
            path == outer_root or path.is_relative_to(outer_root)
            for outer_root in forbidden
        ), label


def test_stdlib_wrapper_guard_denies_before_evaluator_runpy(tmp_path: Path) -> None:
    protected_root = tmp_path / "protected"
    protected_root.mkdir()
    protected = protected_root / "scientific-input.bin"
    protected.write_bytes(b"must-not-be-read")
    manifest = evaluator._forensic_contract().attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1.preexecution_read_guard.v1"
            ),
            "experiment_id": "SYNTHETIC_WRAPPER_GUARD_TEST",
            "forbidden_path_prefixes": [str(protected_root)],
            "scientific_forbidden_bindings": {
                "synthetic_forbidden_input": str(protected)
            },
            "admitted_exact_read_only_technical_receipts": [],
            "admission_does_not_apply_to_parent_or_sibling_paths": True,
            "admission_precedence": (
                "EXACT_ADMITTED_PATH_CHECK_BEFORE_FORBIDDEN_PREFIX_CHECK"
            ),
            "admitted_open_modes": ["r", "rb"],
            "admitted_write_create_truncate_append_or_update": False,
            "installed_by_stdlib_wrapper_before_evaluator_runpy": True,
            "forbidden_open_event_action": (
                "APPEND_CANONICAL_EVENT_FSYNC_THEN_RAISE_PERMISSION_ERROR"
            ),
            "expected_event_rows": 0,
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(evaluator.canonical_bytes(manifest))
    events_path = tmp_path / "events.jsonl"
    code = "\n".join(
        (
            "import importlib.util, os, pathlib, sys",
            f"p={str(evaluator.PREEXECUTION_DIAGNOSTIC_WRAPPER_SCRIPT)!r}",
            "s=importlib.util.spec_from_file_location('forensic_wrapper',p)",
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)",
            "fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)",
            "v=m._load_read_guard_manifest(pathlib.Path(sys.argv[1]))",
            "m._install_read_guard(v,fd)",
            "try:",
            " open(sys.argv[3],'rb').read()",
            "except PermissionError:",
            " os.fsync(fd);os.close(fd);raise SystemExit(0)",
            "raise SystemExit(9)",
        )
    )
    completed = subprocess.run(
        [
            str(evaluator.FORENSIC_INTERPRETER),
            "-E",
            "-s",
            "-u",
            "-c",
            code,
            str(manifest_path),
            str(events_path),
            str(protected),
        ],
        cwd=evaluator.ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "FORBIDDEN_OPEN_ATTEMPT"
    assert rows[0]["path"] == str(protected)


def test_wrapper_argparse_failure_preserves_external_structured_custody(
    tmp_path: Path,
) -> None:
    root = tmp_path / "argparse-failure"
    (root / "streams").mkdir(parents=True)
    (root / "receipts").mkdir()
    paths = evaluator._forensic_runtime_paths(root)
    manifest = evaluator._build_preexecution_read_guard_manifest()
    paths["read_guard_manifest"].write_bytes(evaluator.canonical_bytes(manifest))
    last_stage_path = paths["last_stage_marker"]
    evaluator._atomic_preexecution_last_stage(
        last_stage_path,
        producer_role="PREEXECUTION_DIAGNOSTIC_LAUNCHER",
        stage_id="LAUNCHER_PRESPAWN",
        event="COMPLETED",
    )
    with (
        paths["child_stdout"].open("xb", buffering=0) as stdout_handle,
        paths["child_stderr"].open("xb", buffering=0) as stderr_handle,
        paths["child_traceback"].open("xb", buffering=0) as traceback_handle,
        paths["child_exception"].open("xb", buffering=0) as exception_handle,
        paths["heartbeat"].open("xb", buffering=0) as heartbeat_handle,
        paths["read_guard_events"].open("xb", buffering=0) as guard_handle,
    ):
        # All custody FDs are present, but --mode is deliberately invalid so
        # the full parser fails before evaluator runpy/import handoff.
        argv = [
            str(evaluator.FORENSIC_INTERPRETER),
            "-E",
            "-s",
            "-u",
            str(evaluator.PREEXECUTION_DIAGNOSTIC_WRAPPER_SCRIPT),
            "--mode",
            "INVALID",
            "--diagnostic-root",
            str(root),
            "--traceback-fd",
            str(traceback_handle.fileno()),
            "--exception-fd",
            str(exception_handle.fileno()),
            "--heartbeat-fd",
            str(heartbeat_handle.fileno()),
            "--read-guard-manifest",
            str(paths["read_guard_manifest"]),
            "--read-guard-events-fd",
            str(guard_handle.fileno()),
        ]
        completed = subprocess.run(
            argv,
            cwd=evaluator.ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            pass_fds=(
                traceback_handle.fileno(),
                exception_handle.fileno(),
                heartbeat_handle.fileno(),
                guard_handle.fileno(),
            ),
            check=False,
        )
    assert completed.returncode == 2
    exception = json.loads(paths["child_exception"].read_text())
    assert exception["mode"] == "WRAPPER_BOOTSTRAP"
    assert exception["exception_type"] == "SystemExit"
    assert exception["structured_capture_available"] is True
    assert paths["child_traceback"].stat().st_size > 0
    assert paths["child_stderr"].stat().st_size > 0
    heartbeat = paths["heartbeat"].read_text()
    assert "WRAPPER_BOOTSTRAP_STARTED" in heartbeat
    assert "WRAPPER_ARGPARSE_EXCEPTION" in heartbeat
    assert paths["read_guard_events"].stat().st_size == 0
    last_stage = json.loads(last_stage_path.read_text())
    assert last_stage["stage_id"] == "WRAPPER_ARGPARSE_EXCEPTION"
    assert last_stage["event"] == "COMPLETED"


def test_wrapper_logging_setup_failure_uses_preopened_fallback_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "logging-setup-failure"
    (root / "streams").mkdir(parents=True)
    (root / "receipts").mkdir()
    paths = evaluator._forensic_runtime_paths(root)
    paths["read_guard_manifest"].write_bytes(
        evaluator.canonical_bytes(
            evaluator._build_preexecution_read_guard_manifest()
        )
    )
    evaluator._atomic_preexecution_last_stage(
        paths["last_stage_marker"],
        producer_role="PREEXECUTION_DIAGNOSTIC_LAUNCHER",
        stage_id="LAUNCHER_PRESPAWN",
        event="COMPLETED",
    )

    def fail_setup(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic faulthandler setup failure")

    monkeypatch.setattr(wrapper.faulthandler, "enable", fail_setup)
    with (
        paths["child_traceback"].open("xb", buffering=0) as traceback_handle,
        paths["child_exception"].open("xb", buffering=0) as exception_handle,
        paths["heartbeat"].open("xb", buffering=0) as heartbeat_handle,
        paths["read_guard_events"].open("xb", buffering=0) as guard_handle,
    ):
        argv = [
            "--mode",
            wrapper.SYNTHETIC_MODE,
            "--fixture-id",
            "PASS",
            "--diagnostic-root",
            str(root),
            "--traceback-fd",
            str(traceback_handle.fileno()),
            "--exception-fd",
            str(exception_handle.fileno()),
            "--heartbeat-fd",
            str(heartbeat_handle.fileno()),
            "--read-guard-manifest",
            str(paths["read_guard_manifest"]),
            "--read-guard-events-fd",
            str(guard_handle.fileno()),
        ]
        with pytest.raises(RuntimeError, match="faulthandler setup failure"):
            wrapper.main(argv)

    exception = json.loads(paths["child_exception"].read_text())
    assert exception["mode"] == "WRAPPER_BOOTSTRAP"
    assert exception["exception_type"] == "RuntimeError"
    assert paths["child_traceback"].stat().st_size > 0
    assert paths["heartbeat"].stat().st_size > 0
    marker = evaluator._load_preexecution_last_stage_marker(
        paths["last_stage_marker"]
    )
    assert (marker["producer_role"], marker["stage_id"], marker["event"]) == (
        "PREEXECUTION_DIAGNOSTIC_WRAPPER",
        "WRAPPER_ARGPARSE_EXCEPTION",
        "COMPLETED",
    )


def test_forensic_wrapper_argv_is_exact_and_malformed_forms_fail_visible(
    tmp_path: Path,
) -> None:
    forensic = evaluator._validate_forensic_constant_alignment()
    root = tmp_path / "synthetic"
    argv = evaluator._expected_preexecution_diagnostic_synthetic_child_argv(
        fixture_id="PASS",
        diagnostic_root=root,
        traceback_fd=7,
        exception_fd=8,
        heartbeat_fd=9,
        read_guard_events_fd=10,
    )
    assert argv[:5] == [
        str(evaluator.FORENSIC_INTERPRETER),
        "-E",
        "-s",
        "-u",
        str(evaluator.PREEXECUTION_DIAGNOSTIC_WRAPPER_SCRIPT),
    ]
    assert evaluator._classify_forensic_argv(
        argv, executable=str(evaluator.FORENSIC_INTERPRETER.resolve())
    ) == "PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD"
    for changed in (
        argv[1:],
        [argv[0], "-s", "-E", *argv[3:]],
        [*argv, "--extra"],
        [*argv[:12], "0", *argv[13:]],
    ):
        assert evaluator._classify_forensic_argv(
            changed, executable=str(evaluator.FORENSIC_INTERPRETER.resolve())
        ) == "INVALID_FORENSIC_ARGV"
    assert tuple(forensic.PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES) == (
        "PASS",
        "RAISE",
        "EXIT_NONZERO",
        "SIGTERM",
        "MISSING_PATH",
        "UNICODE_RAISE",
    )


def test_forensic_freeze_argv_and_main_dispatch_never_enter_legacy_freeze(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    expected = evaluator._expected_preexecution_forensic_freeze_argv()
    assert expected == [
        str(evaluator.FORENSIC_INTERPRETER),
        str(evaluator.EVALUATOR_SCRIPT),
        "freeze-preexecution-forensic",
    ]
    assert evaluator._classify_forensic_argv(
        expected,
        executable=str(evaluator.FORENSIC_INTERPRETER.resolve()),
    ) == "PREEXECUTION_FORENSIC_FREEZE"
    for malformed in (
        expected[1:],
        [*expected, "--extra"],
        [expected[0], "-E", *expected[1:]],
    ):
        assert evaluator._classify_forensic_argv(
            malformed,
            executable=str(evaluator.FORENSIC_INTERPRETER.resolve()),
        ) == "INVALID_FORENSIC_ARGV"

    terminal = {"freeze_mode": "FORENSIC_ONLY", "pass": True}
    monkeypatch.setattr(
        evaluator,
        "freeze_preexecution_forensic_contract",
        lambda: terminal,
    )

    def forbidden_legacy_freeze() -> dict[str, object]:
        raise AssertionError("legacy scientific freeze path was called")

    monkeypatch.setattr(evaluator, "freeze_contract", forbidden_legacy_freeze)
    monkeypatch.setattr(
        evaluator.sys,
        "argv",
        [str(evaluator.EVALUATOR_SCRIPT), "freeze-preexecution-forensic"],
    )
    assert evaluator.main() == 0
    assert json.loads(capsys.readouterr().out) == terminal


def test_forensic_freeze_public_entry_validates_exact_head_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forensic = SimpleNamespace(SOURCE_COMMIT="f" * 40)
    expected = evaluator._expected_preexecution_forensic_freeze_argv()
    monkeypatch.setattr(
        evaluator,
        "_validate_forensic_constant_alignment",
        lambda: forensic,
    )
    monkeypatch.setattr(
        evaluator,
        "_forensic_process_identity",
        lambda _pid, require_role=None: {
            "argv": expected,
            "role": require_role,
        },
    )
    monkeypatch.setattr(
        evaluator,
        "git_output",
        lambda *args: forensic.SOURCE_COMMIT
        if args == ("rev-parse", "HEAD")
        else pytest.fail(f"unexpected git command: {args}"),
    )
    delegated = {"freeze_mode": "FORENSIC_ONLY", "pass": True}
    monkeypatch.setattr(
        evaluator,
        "_freeze_preexecution_forensic_authorities",
        lambda observed: delegated if observed is forensic else pytest.fail(),
    )
    assert evaluator.freeze_preexecution_forensic_contract() is delegated


def test_forensic_freeze_helper_uses_only_public_zero_input_authority_apis() -> None:
    calls: list[str] = []
    preparation = {"content_digest": "a" * 64}
    written = {
        "authorities": {"authority": {"sha256": "b" * 64}},
        "source_closure": {
            "content_digest": "c" * 64,
            "rows": 12,
        },
    }
    post = {
        "base_head": "d" * 40,
        "changed_paths": ["forensic.py"],
        "authorities": copy.deepcopy(written["authorities"]),
        "source_closure": copy.deepcopy(written["source_closure"]),
        "authority_custody": {"pass": True},
        "source_closure_rows": 12,
        "pass": True,
    }

    class FakeForensic:
        FORENSIC_FREEZE_COMMIT_SUBJECT = "Freeze technical forensics"

        def validate_forensic_freeze_preparation(self, root: Path) -> dict[str, object]:
            assert root == evaluator.ROOT
            calls.append("preflight")
            return preparation

        def write_forensic_authorities(self, root: Path) -> dict[str, object]:
            assert root == evaluator.ROOT
            calls.append("write")
            return written

        def validate_forensic_authority_write(
            self,
            root: Path,
            *,
            preparation_receipt: dict[str, object],
        ) -> dict[str, object]:
            assert root == evaluator.ROOT
            assert preparation_receipt is preparation
            calls.append("postflight")
            return post

    observed = evaluator._freeze_preexecution_forensic_authorities(
        FakeForensic()
    )
    assert calls == ["preflight", "write", "postflight"]
    assert observed == {
        "freeze_mode": "PREEXECUTION_FORENSIC_AUTHORITY_PREPARATION",
        "base_head": "d" * 40,
        "changed_paths": ["forensic.py"],
        "authorities": written["authorities"],
        "source_closure": written["source_closure"],
        "authority_custody": {"pass": True},
        "source_closure_rows": 12,
        "scientific_inputs_opened": 0,
        "scientific_archive_payload_files_opened": 0,
        "files_reused": 0,
        "required_enclosing_commit_subject": "Freeze technical forensics",
        "pass": True,
    }


@pytest.mark.parametrize(
    "postflight_failure",
    (RuntimeError("injected postflight failure"), KeyboardInterrupt()),
    ids=("runtime-error", "keyboard-interrupt"),
)
def test_forensic_freeze_postflight_failure_rolls_back_all_authorities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    postflight_failure: BaseException,
) -> None:
    forensic_contract = evaluator._forensic_contract()
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(
        forensic_contract,
        "_git_text",
        lambda _root, args: forensic_contract.SOURCE_COMMIT
        if tuple(args) == ("rev-parse", "HEAD")
        else pytest.fail(f"unexpected git command: {args}"),
    )

    class FakeForensic:
        FORENSIC_FREEZE_COMMIT_SUBJECT = "Freeze technical forensics"

        def validate_forensic_freeze_preparation(self, root: Path) -> dict[str, object]:
            assert root == tmp_path
            return {"content_digest": "a" * 64}

        def write_forensic_authorities(self, root: Path) -> dict[str, object]:
            assert root == tmp_path
            for relative in forensic_contract.FORENSIC_GENERATED_AUTHORITY_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"technical-authority\n")
            return {
                "authorities": {},
                "source_closure": {"content_digest": "b" * 64},
            }

        def validate_forensic_authority_write(
            self,
            root: Path,
            *,
            preparation_receipt: dict[str, object],
        ) -> dict[str, object]:
            assert root == tmp_path
            assert preparation_receipt["content_digest"] == "a" * 64
            raise postflight_failure

        def rollback_forensic_authorities(self, root: Path) -> dict[str, object]:
            return forensic_contract.rollback_forensic_authorities(root)

    expected_exception = (
        KeyboardInterrupt
        if isinstance(postflight_failure, KeyboardInterrupt)
        else evaluator.QualificationError
    )
    with pytest.raises(expected_exception) as captured:
        evaluator._freeze_preexecution_forensic_authorities(FakeForensic())
    if isinstance(postflight_failure, RuntimeError):
        assert "injected postflight failure" in str(captured.value)
    assert all(
        not (tmp_path / relative).exists()
        and not (tmp_path / relative).is_symlink()
        for relative in forensic_contract.FORENSIC_GENERATED_AUTHORITY_PATHS
    )


def test_forensic_runner_rejects_broken_symlink_reserved_path(
    tmp_path: Path,
) -> None:
    paths = {
        name: tmp_path / name
        for name in (
            "stdout",
            "stderr",
            "traceback",
            "exception",
            "heartbeat",
            "last-stage",
            "read-guard",
            "read-guard-events",
            "environment",
            "command",
        )
    }
    paths["stdout"].symlink_to(tmp_path / "absent-target")
    with pytest.raises(
        evaluator.QualificationError,
        match="forensic stream destination is stale",
    ):
        evaluator._run_forensic_child_with_external_stream_custody(
            lambda _fds: [],
            internal_argv_factory=lambda _fds: [],
            expected_role="PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD",
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            traceback_path=paths["traceback"],
            exception_path=paths["exception"],
            heartbeat_path=paths["heartbeat"],
            last_stage_path=paths["last-stage"],
            read_guard_manifest_path=paths["read-guard"],
            read_guard_events_path=paths["read-guard-events"],
            environment_path=paths["environment"],
            command_path=paths["command"],
        )


def test_forensic_interpreter_is_separate_from_frozen_scientific_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scientific = tmp_path / "scientific-python"
    monkeypatch.setattr(evaluator, "EVALUATOR_INTERPRETER", scientific)
    launcher = {"pid": 17, "start_time_ticks": 23}
    assert evaluator._expected_launcher_argv()[0] == str(scientific)
    assert evaluator._expected_scientific_argv(launcher)[0] == str(scientific)
    assert evaluator._expected_finalizer_argv(
        attempt=evaluator.CONTRACT.OUTPUT_ROOT.parent
        / f".{evaluator.CONTRACT.OUTPUT_ROOT.name}.attempt-test",
        launcher_identity=launcher,
        scientific_exit_receipt=tmp_path / "scientific-exit.json",
    )[0] == str(scientific)
    assert evaluator._expected_preexecution_diagnostic_launcher_argv()[0] == str(
        evaluator.FORENSIC_INTERPRETER
    )
    assert evaluator._expected_preexecution_diagnostic_synthetic_child_argv(
        fixture_id="PASS",
        diagnostic_root=tmp_path / "diagnostic",
        traceback_fd=7,
        exception_fd=8,
        heartbeat_fd=9,
        read_guard_events_fd=10,
    )[0] == str(evaluator.FORENSIC_INTERPRETER)


def test_current_runtime_context_round_trips_without_device_opens() -> None:
    forensic = evaluator._forensic_contract()
    value = evaluator._technical_runtime_context()
    assert forensic.validate_current_runtime_context(value) == value
    assert value["role"] == (
        "CURRENT_TECHNICAL_DIAGNOSTIC_ONLY_NOT_HISTORICAL_CHILD_CONTEXT"
    )
    assert value["gpu"]["device_files_opened"] == 0


def test_forensic_timeout_uses_bounded_term_then_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {
        "wall_clock_timeout_s": 30.0,
        "term_grace_s": 5.0,
        "kill_grace_s": 5.0,
    }
    monkeypatch.setattr(
        evaluator,
        "_validate_forensic_constant_alignment",
        lambda: SimpleNamespace(PREEXECUTION_DIAGNOSTIC_TIMEOUT_POLICY=policy),
    )

    class TimedOutProcess:
        pid = 4321

        def __init__(self) -> None:
            self.wait_timeouts: list[float] = []

        def wait(self, *, timeout: float) -> int:
            self.wait_timeouts.append(timeout)
            if len(self.wait_timeouts) < 3:
                raise subprocess.TimeoutExpired(["technical-child"], timeout)
            return -int(signal.SIGKILL)

    process = TimedOutProcess()
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        evaluator.os,
        "killpg",
        lambda pid, sig: signals.append((pid, signal.Signals(sig))),
    )
    monkeypatch.setattr(evaluator.time, "monotonic_ns", lambda: 3_000_000_000)
    returncode, timed_out = evaluator._wait_for_forensic_child_with_timeout(
        process, spawned_monotonic_ns=1_000_000_000
    )
    assert (returncode, timed_out) == (-int(signal.SIGKILL), True)
    assert process.wait_timeouts == [28.0, 5.0, 5.0]
    assert signals == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]


def test_synthetic_diagnostic_scopes_umask_before_artifacts_and_restores(
    tmp_path: Path,
) -> None:
    inherited = os.umask(0o002)
    try:
        root = tmp_path / "umask-pass"
        observed = evaluator._run_synthetic_preexecution_diagnostic_fixture(
            fixture_id="PASS", diagnostic_root=root
        )
        restored = os.umask(0o002)
        os.umask(restored)
        assert restored == 0o002
        assert observed["execution"]["technical_runtime_context"]["umask"] == {
            "value": 0o022,
            "sampled_and_restored": True,
        }
        for path in root.rglob("*"):
            mode = path.stat().st_mode & 0o777
            assert mode == (0o755 if path.is_dir() else 0o644), path
    finally:
        os.umask(inherited)


def test_active_diagnostic_namespace_modes_are_exact_from_inherited_0002(
    tmp_path: Path,
) -> None:
    forensic = evaluator._forensic_contract()
    inherited = os.umask(0o002)
    try:
        observed_previous = os.umask(forensic.PREEXECUTION_DIAGNOSTIC_UMASK)
        assert observed_previous == 0o002
        root = tmp_path / "active-diagnostic"
        root.mkdir(mode=0o755)
        for fixture_id in forensic.PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES:
            evaluator._run_synthetic_preexecution_diagnostic_fixture(
                fixture_id=fixture_id,
                diagnostic_root=root / "synthetic" / fixture_id,
            )
        paths = evaluator._forensic_runtime_paths(root)
        for key in (
            "startup_stage_ledger",
            "child_stdout",
            "child_stderr",
            "child_traceback",
            "child_exception",
            "invocation",
            "environment",
            "command",
            "heartbeat",
            "read_guard_manifest",
            "read_guard_events",
            "forensic_freeze_custody",
            "last_stage_marker",
            "synthetic_results",
        ):
            evaluator._exclusive_bytes(paths[key], b"{}\n")
        inventory = forensic._validate_active_diagnostic_namespace(root)
        assert inventory["effective_umask"] == 0o022
        assert inventory["unexpected_paths"] == []
        assert inventory["pass"] is True
    finally:
        os.umask(inherited)


def test_real_diagnostic_entry_scopes_and_restores_umask_before_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forensic = SimpleNamespace(
        PREEXECUTION_DIAGNOSTIC_UMASK=0o022,
        PREEXECUTION_DIAGNOSTIC_EXPECTED_PREVIOUS_UMASK=0o002,
    )
    monkeypatch.setattr(
        evaluator, "_validate_forensic_constant_alignment", lambda: forensic
    )
    observed: dict[str, int] = {}

    def delegated(*, previous_umask: int) -> dict[str, bool]:
        sampled = os.umask(0o022)
        os.umask(sampled)
        observed.update(previous=previous_umask, effective=sampled)
        return {"pass": True}

    monkeypatch.setattr(
        evaluator,
        "_execute_preexecution_diagnostic_under_umask",
        delegated,
    )
    inherited = os.umask(0o002)
    try:
        assert evaluator.execute_preexecution_diagnostic() == {"pass": True}
        restored = os.umask(0o002)
        os.umask(restored)
        assert observed == {"previous": 0o002, "effective": 0o022}
        assert restored == 0o002
    finally:
        os.umask(inherited)


def test_terminal_bundle_is_reloaded_before_tracked_publication(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    custody = {"receipt": "custody"}
    synthetic = {"receipt": "synthetic"}
    runtime_result = {"schema": "runtime"}
    final_inventory = {"schema": "inventory"}
    publication = {
        "v2_spec_written": True,
        "automatic_execution_authorized": False,
        "pass": True,
    }

    class FakeForensic:
        def write_preexecution_terminal_bundle(self, **kwargs: object) -> dict[str, object]:
            events.append("runtime-terminal")
            assert kwargs == {
                "diagnostic_custody_receipt": custody,
                "synthetic_results_receipt": synthetic,
                "diagnostic_root": tmp_path,
            }
            return {
                "runtime_result": runtime_result,
                "final_namespace_inventory": final_inventory,
            }

        def load_and_validate_diagnostic_bundle(self, **kwargs: object) -> dict[str, object]:
            events.append("strict-reload")
            assert kwargs["diagnostic_root"] == tmp_path
            return {
                "diagnostic_custody": custody,
                "result": runtime_result,
                "final_namespace_inventory": final_inventory,
                "preflight_namespace_inventory": {"rows": []},
                "postflight_namespace_inventory": {"rows": []},
            }

        def write_forensic_result_artifacts(self, **kwargs: object) -> dict[str, object]:
            events.append("tracked-publication")
            assert kwargs["diagnostic_root"] == tmp_path
            return publication

    observed = evaluator._publish_preexecution_diagnostic_terminal(
        forensic=FakeForensic(),
        diagnostic_root=tmp_path,
        custody=custody,
        synthetic_receipt=synthetic,
    )
    assert observed is publication
    assert events == [
        "runtime-terminal",
        "strict-reload",
        "tracked-publication",
    ]


def test_terminal_bundle_drift_blocks_tracked_publication(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class FakeForensic:
        def write_preexecution_terminal_bundle(self, **_kwargs: object) -> dict[str, object]:
            events.append("runtime-terminal")
            return {
                "runtime_result": {"schema": "runtime"},
                "final_namespace_inventory": {"schema": "inventory"},
            }

        def load_and_validate_diagnostic_bundle(self, **_kwargs: object) -> dict[str, object]:
            events.append("strict-reload")
            return {
                "diagnostic_custody": {"drift": True},
                "result": {"schema": "runtime"},
                "final_namespace_inventory": {"schema": "inventory"},
                "preflight_namespace_inventory": {"rows": []},
                "postflight_namespace_inventory": {"rows": []},
            }

        def write_forensic_result_artifacts(self, **_kwargs: object) -> dict[str, object]:
            events.append("tracked-publication")
            return {"pass": True}

    with pytest.raises(
        evaluator.QualificationError,
        match="terminal diagnostic bundle custody drift",
    ):
        evaluator._publish_preexecution_diagnostic_terminal(
            forensic=FakeForensic(),
            diagnostic_root=tmp_path,
            custody={"receipt": "custody"},
            synthetic_receipt={"receipt": "synthetic"},
        )
    assert events == ["runtime-terminal", "strict-reload"]


def test_child_payload_is_rebuilt_against_parent_held_custody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zero = {
        "scientific_inputs_opened": 0,
        "failed_scientific_payloads_opened": 0,
        "outcome_rows_opened": 0,
        "tensor_reads": 0,
        "model_loads": 0,
        "training_steps": 0,
    }
    mismatch = {"mechanism": "WHOLE_RECORD_SCHEMA_MISMATCH"}

    def build(**kwargs: object) -> dict[str, object]:
        return {
            "launcher_process_identity": copy.deepcopy(
                kwargs["launcher_process_identity"]
            ),
            "child_process_identity": copy.deepcopy(
                kwargs["child_process_identity"]
            ),
            "repo_head": kwargs["repo_head"],
            "repo_clean": kwargs["repo_clean"],
            "scientific_contract_digest": kwargs[
                "scientific_contract_digest"
            ],
            "forensic_authority_source_closure": copy.deepcopy(
                kwargs["forensic_authority_source_closure"]
            ),
            "forensic_freeze_custody": {
                "content_digest": kwargs["forensic_freeze_custody"][
                    "content_digest"
                ]
            },
            "output_namespace_before": copy.deepcopy(
                kwargs["output_namespace_before"]
            ),
            "output_namespace_after": copy.deepcopy(
                kwargs["output_namespace_after"]
            ),
            "mismatch_evidence": copy.deepcopy(kwargs["mismatch_evidence"]),
            "committed_source_root_cause_proof": copy.deepcopy(
                kwargs["committed_source_root_cause_proof"]
            ),
            "current_runtime_context": copy.deepcopy(
                kwargs["current_runtime_context"]
            ),
            "scientific_counters": copy.deepcopy(kwargs["scientific_counters"]),
            "pass": True,
        }

    fake = SimpleNamespace(
        PREEXECUTION_DIAGNOSTIC_ROOT=(
            evaluator._forensic_contract().PREEXECUTION_DIAGNOSTIC_ROOT
        ),
        PREEXECUTION_CHILD_ZERO_SCIENTIFIC_COUNTERS=zero,
        validate_preexecution_child_result=lambda value: copy.deepcopy(value),
        build_preexecution_child_result=build,
    )
    monkeypatch.setattr(
        evaluator, "_validate_forensic_constant_alignment", lambda: fake
    )
    monkeypatch.setattr(
        evaluator,
        "_synthetic_archive_validation_projection_evidence",
        lambda: copy.deepcopy(mismatch),
    )
    launcher = {"pid": 11}
    child = {"pid": 12}
    closure = {"path": "closure.json", "content_digest": "a" * 64}
    freeze = {
        "repo_head": "b" * 40,
        "repo_clean": True,
        "scientific_contract_digest": "c" * 64,
        "forensic_authority_source_closure": closure,
        "committed_source_root_cause_proof": {"pass": True},
        "content_digest": "d" * 64,
    }
    namespace = [{"path": "/technical/only", "kind": "DIRECTORY"}]
    value = build(
        launcher_process_identity=launcher,
        child_process_identity=child,
        repo_head=freeze["repo_head"],
        repo_clean=True,
        scientific_contract_digest=freeze["scientific_contract_digest"],
        forensic_authority_source_closure=closure,
        forensic_freeze_custody=freeze,
        output_namespace_before=namespace,
        output_namespace_after=namespace,
        mismatch_evidence=mismatch,
        committed_source_root_cause_proof=freeze[
            "committed_source_root_cause_proof"
        ],
        current_runtime_context={"role": "CHILD_CURRENT_ONLY"},
        scientific_counters=zero,
    )
    assert evaluator._validate_preexecution_child_payload(
        value,
        launcher_process_identity=launcher,
        child_process_identity=child,
        forensic_freeze_custody=freeze,
        namespace_before=namespace,
        namespace_after=namespace,
    ) == value

    for field in (
        "repo_head",
        "forensic_authority_source_closure",
        "forensic_freeze_custody",
        "output_namespace_after",
        "mismatch_evidence",
        "committed_source_root_cause_proof",
    ):
        tampered = copy.deepcopy(value)
        tampered[field] = {"tampered": True}
        with pytest.raises(
            evaluator.QualificationError,
            match="does not match parent-held custody",
        ):
            evaluator._validate_preexecution_child_payload(
                tampered,
                launcher_process_identity=launcher,
                child_process_identity=child,
                forensic_freeze_custody=freeze,
                namespace_before=namespace,
                namespace_after=namespace,
            )


@pytest.mark.parametrize(
    ("fixture_id", "expected_returncode", "expected_kind"),
    (
        ("PASS", 0, "EXIT"),
        ("RAISE", 1, "EXIT"),
        ("EXIT_NONZERO", 23, "EXIT"),
        ("SIGTERM", -int(signal.SIGTERM), "SIGNAL"),
        ("MISSING_PATH", 1, "EXIT"),
        ("UNICODE_RAISE", 1, "EXIT"),
    ),
)
def test_synthetic_wrapper_fixtures_preserve_external_failure_custody(
    synthetic_observations: dict[str, dict[str, object]],
    fixture_id: str,
    expected_returncode: int,
    expected_kind: str,
) -> None:
    # The committed diagnostic launcher is the lexical TinyQuad venv
    # interpreter.  The repository's aggregate test command imports torch into
    # system Python through PYTHONPATH, which the deliberately isolated child
    # correctly ignores under -E.  Bind the real child interpreter explicitly.
    assert evaluator.FORENSIC_INTERPRETER.is_file()
    observed = synthetic_observations[fixture_id]
    root = Path(str(observed["diagnostic_root"]))
    execution = observed["execution"]
    assert execution["returncode"] == expected_returncode
    assert execution["termination"]["kind"] == expected_kind
    assert execution["cleanup"] == {
        "process_group_members_after_wait": [],
        "exact_nonlauncher_forensic_role_matches_after_wait": [],
        "scoped_dev_kfd_holders_after_wait": [],
        "current_launcher_excluded_from_role_scan": True,
        "cleanup_scope": (
            "TERMINATED_CHILD_PROCESS_GROUP_AND_NONLAUNCHER_FORENSIC_ROLES"
        ),
        "literal_zero_all_forensic_roles_claimed": False,
        "pass": True,
    }
    assert all(value == 0 for value in observed["scientific_counters"].values())
    assert execution["stdout"]["path"] != execution["stderr"]["path"]
    assert execution["stderr"]["path"] != execution["traceback"]["path"]
    assert Path(execution["stdout"]["path"]).is_file()
    assert Path(execution["stderr"]["path"]).is_file()
    assert Path(execution["traceback"]["path"]).is_file()
    assert Path(execution["structured_exception"]["path"]).is_file()
    assert Path(execution["heartbeat"]["path"]).stat().st_size > 0
    assert Path(execution["environment_receipt"]["path"]).is_file()
    assert Path(execution["command_receipt"]["path"]).is_file()
    environment_custody = execution["environment"][
        "result_environment_custody"
    ]
    assert environment_custody["key_count"] == len(
        environment_custody["key_names"]
    )
    assert len(environment_custody["canonical_map_sha256"]) == 64
    assert environment_custody["values_persisted"] is False
    command_receipt = json.loads(
        Path(execution["command_receipt"]["path"]).read_text()
    )
    assert command_receipt["popen_environment"] == environment_custody
    assert Path(execution["read_guard_manifest"]["path"]).is_file()
    assert execution["read_guard_events"]["bytes"] == 0
    lifecycle = execution["synthetic_technical_lifecycle_rows"]
    assert [(row["stage_id"], row["event"]) for row in lifecycle] == [
        (stage_id, event)
        for stage_id in evaluator._forensic_contract().PREEXECUTION_DIAGNOSTIC_SYNTHETIC_STAGE_IDS
        for event in ("STARTED", "COMPLETED")
    ]
    assert not (root / "technical_reservation").exists()
    last_stage = json.loads(
        Path(execution["last_stage_marker"]["path"]).read_text()
    )
    if fixture_id == "PASS":
        assert (last_stage["stage_id"], last_stage["event"]) == (
            "COMPLETE",
            "COMPLETED",
        )
    elif fixture_id == "MISSING_PATH":
        assert last_stage["stage_id"] == (
            "BEFORE_EVALUATOR_IMPORT_AND_ARGPARSE_HANDOFF"
        )
    else:
        assert (last_stage["stage_id"], last_stage["event"]) == (
            "VALIDATE_FORENSIC_AUTHORITY",
            "STARTED",
        )

    exception_path = Path(execution["structured_exception"]["path"])
    traceback_path = Path(execution["traceback"]["path"])
    if fixture_id == "PASS":
        assert exception_path.stat().st_size == 0
        assert traceback_path.stat().st_size == 0
        assert len(observed["startup_stage_rows"]) == 14
    elif fixture_id == "SIGTERM":
        assert exception_path.stat().st_size == 0
        assert execution["termination"]["signal_number_or_null"] == int(
            signal.SIGTERM
        )
    else:
        exception = json.loads(exception_path.read_text(encoding="utf-8"))
        assert exception["structured_capture_available"] is True
        assert exception["traceback_stream"] == "streams/child.traceback"
        assert traceback_path.stat().st_size > 0
        if fixture_id == "MISSING_PATH":
            assert exception["exception_type"] == "FileNotFoundError"
        if fixture_id == "UNICODE_RAISE":
            assert "H1–H4 / λ / 雪" in exception["exception_message"]


def test_forensic_popen_uses_the_same_environment_map_bound_by_receipt() -> None:
    tree = ast.parse(
        (
            evaluator.ROOT
            / "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
        ).read_text(encoding="utf-8")
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_forensic_child_with_external_stream_custody"
    )
    builder_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "build_environment_receipt"
    ]
    popen_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
    ]
    assert len(builder_calls) == len(popen_calls) == 1

    def keyword_value(call: ast.Call, name: str) -> ast.AST:
        return next(keyword.value for keyword in call.keywords if keyword.arg == name)

    assert ast.dump(
        keyword_value(builder_calls[0], "result_environment"),
        include_attributes=False,
    ) == ast.dump(
        ast.Name(id="environment", ctx=ast.Load()), include_attributes=False
    )
    assert ast.dump(
        keyword_value(popen_calls[0], "env"), include_attributes=False
    ) == ast.dump(ast.Name(id="environment", ctx=ast.Load()), include_attributes=False)


@pytest.fixture(scope="module")
def synthetic_observations(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, object]]:
    root = tmp_path_factory.mktemp("forensic-fixtures")
    forensic = evaluator._forensic_contract()
    return {
        fixture_id: evaluator._run_synthetic_preexecution_diagnostic_fixture(
            fixture_id=fixture_id,
            diagnostic_root=root / fixture_id.lower(),
        )
        for fixture_id in forensic.PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES
    }


def test_all_synthetic_writer_rows_round_trip_through_frozen_builder(
    synthetic_observations: dict[str, dict[str, object]],
) -> None:
    forensic = evaluator._forensic_contract()
    rows = [
        evaluator._synthetic_fixture_result_row(
            synthetic_observations[fixture_id]
        )
        for fixture_id in forensic.PREEXECUTION_DIAGNOSTIC_SYNTHETIC_FIXTURES
    ]
    receipt = forensic.build_synthetic_results_receipt(fixture_rows=rows)
    assert forensic.validate_synthetic_results_receipt(receipt) == receipt
    assert all(
        row["technical_resources_cleaned"] is True
        and row["technical_lifecycle"]["rows"] == 10
        and row["last_stage_marker"]["content_digest"]
        == row["last_stage_marker_value"]["content_digest"]
        for row in receipt["rows"]
    )
    expected_prefix_rows = {
        "PASS": 14,
        "RAISE": 5,
        "EXIT_NONZERO": 5,
        "SIGTERM": 5,
        "MISSING_PATH": 0,
        "UNICODE_RAISE": 5,
    }
    for row in receipt["rows"]:
        assert len(row["startup_stage_prefix_rows"]) == expected_prefix_rows[
            row["fixture_id"]
        ]
        if expected_prefix_rows[row["fixture_id"]] == 0:
            assert row["startup_stage_prefix"] is None
        else:
            assert row["startup_stage_prefix"]["rows"] == expected_prefix_rows[
                row["fixture_id"]
            ]


def test_fast_missing_path_fixture_repeatedly_binds_exact_child_identity(
    tmp_path: Path,
) -> None:
    for index in range(3):
        observed = evaluator._run_synthetic_preexecution_diagnostic_fixture(
            fixture_id="MISSING_PATH",
            diagnostic_root=tmp_path / f"missing-{index}",
        )
        execution = observed["execution"]
        assert execution["process_identity"]["role"] == (
            "PREEXECUTION_DIAGNOSTIC_SYNTHETIC_CHILD"
        )
        assert execution["returncode"] == 1
        assert execution["cleanup"]["pass"] is True
        exception = json.loads(
            Path(execution["structured_exception"]["path"]).read_text()
        )
        assert exception["exception_type"] == "FileNotFoundError"
