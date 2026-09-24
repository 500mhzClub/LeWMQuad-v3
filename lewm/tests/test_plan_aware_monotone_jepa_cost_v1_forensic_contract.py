from __future__ import annotations

import ast
import copy
import inspect
import os
from pathlib import Path
import stat

import pytest

from lewm.safety import (
    plan_aware_monotone_jepa_cost_v1_forensic_contract as forensic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _stat_row(path: Path, *, relative: str, kind: str) -> dict[str, object]:
    info = os.lstat(path)
    row: dict[str, object] = {
        "path": relative,
        "kind": kind,
        "mode": stat.S_IMODE(info.st_mode),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
    }
    if kind == "FILE":
        row["bytes"] = int(info.st_size)
    return row


def _assert_no_duplicate_literal_dict_keys(path: Path) -> None:
    parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        assert len(keys) == len(set(keys)), (
            f"duplicate literal mapping key in {path}: {keys}"
        )


def test_authority_builders_are_self_consistent_and_science_invariant() -> None:
    assert forensic.validate_forensic_contract(
        forensic.FORENSIC_CONTRACT
    ) == forensic.FORENSIC_CONTRACT
    assert forensic.validate_archive_inventory_authority(
        forensic.ARCHIVE_INVENTORY_AUTHORITY
    ) == forensic.ARCHIVE_INVENTORY_AUTHORITY
    assert forensic.validate_os_evidence_schema(
        forensic.OS_EVIDENCE_SCHEMA_AUTHORITY
    ) == forensic.OS_EVIDENCE_SCHEMA_AUTHORITY
    assert (
        forensic.build_conditional_v2_gate_authority()
        == forensic.CONDITIONAL_V2_GATE_AUTHORITY
    )
    assert (
        forensic.build_forensic_evaluator_fixture()
        == forensic.FORENSIC_EVALUATOR_FIXTURE
    )
    for value in (
        forensic.CONDITIONAL_V2_GATE_AUTHORITY,
        forensic.FORENSIC_EVALUATOR_FIXTURE,
        forensic.USER_CONDITIONAL_V2_SPEC_AUTHORITY,
    ):
        forensic.validate_self_digest(value)
    assert forensic.FORENSIC_PRIMARY_CLASSIFICATION == (
        "FINAL_CHILD_ROOT_CAUSE_IDENTIFIED"
    )
    assert forensic.FORENSIC_SECONDARY_MECHANISM == (
        "PREATTEMPT_CUSTODY_SCHEMA_MISMATCH"
    )
    assert forensic.SCIENTIFIC_DISPOSITION == (
        "PLAN_AWARE_MONOTONE_JEPA_COST_V1_TECHNICAL_NON_RESULT"
    )
    assert forensic.BASE.CONTRACT_SHA256 == (
        forensic.BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
    )


def test_forensic_python_sources_have_no_duplicate_literal_keys() -> None:
    for relative in (
        "lewm/safety/plan_aware_monotone_jepa_cost_v1_forensic_contract.py",
        "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py",
        "scripts/run_plan_aware_monotone_jepa_cost_v1_preexecution_diagnostic_child.py",
    ):
        _assert_no_duplicate_literal_dict_keys(REPO_ROOT / relative)


def test_exact_whole_record_schema_mismatch_fixture() -> None:
    value = forensic.build_archive_verification_record_equality_fixture()
    forensic.validate_self_digest(value)
    assert value["scientific_inputs_opened"] == 0
    assert value["defect_id"] == (
        "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
    )
    assert value["raw_records_equal"] is False
    assert value["identity_projections_equal"] is True
    assert value["defect_reproduced"] is True
    assert value["corrected_identity_comparison_passes"] is True
    assert value["full_inventory_verified_is_one_detail_not_sole_cause"] is True
    assert value["per_row_key_differences"][0]["minimal_only"] == [
        "partial_artifacts_reusable"
    ]
    assert set(value["per_row_key_differences"][1]["detailed_only"]) == {
        "full_inventory_verified",
        "pass",
        "persistence_receipt",
        "stage_c_executed",
    }


def test_committed_source_proof_opens_no_scientific_payload() -> None:
    value = forensic.build_committed_source_root_cause_proof(REPO_ROOT)
    forensic.validate_self_digest(value)
    assert value["defect_proved"] is True
    assert value["archive_payloads_opened"] == 0
    assert value["outcome_metric_or_tensor_values_opened"] == 0
    evidence = value["evidence"]
    assert evidence["whole_record_inequality"] == {
        "function": "_new_attempt",
        "lines": [2131],
        "comparison_count": 1,
        "field": "failed_archives",
    }
    assert evidence["runtime_minimal_custody_builder"][
        "minimal_archive_builder_function"
    ] == "execution_correction_2_failed_archive_custody"
    assert evidence["detailed_validator_return_key_sets"][
        "first_archive_function"
    ] == "validate_execution_correction_archive"
    assert evidence["detailed_validator_return_key_sets"][
        "second_archive_function"
    ] == "validate_execution_correction_2_archive"
    assert evidence["full_inventory_verified_is_one_detail_not_sole_cause"] is True


def test_overlay_source_closure_never_hashes_forbidden_science(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    original = forensic._read_no_follow_bound_file

    def recording(
        root: Path,
        relative: str | Path,
        *,
        expected_stat_rows: object,
    ) -> bytes:
        observed.append((Path(root) / relative).resolve(strict=False))
        return original(
            root, relative, expected_stat_rows=expected_stat_rows
        )

    monkeypatch.setattr(forensic, "_read_no_follow_bound_file", recording)
    value = forensic.build_forensic_source_closure(
        REPO_ROOT, require_complete=False
    )
    declared = {
        (REPO_ROOT / relative).resolve()
        for relative in forensic.FORENSIC_SOURCE_CLOSURE_DEFAULT_PATHS
    }
    forbidden = {
        Path(path).resolve(strict=False)
        for path in forensic.scientific_forbidden_bindings(
            REPO_ROOT
        ).values()
    }
    assert observed
    assert set(observed).issubset(declared)
    assert set(observed).isdisjoint(forbidden)
    assert value["scientific_value_handling"] == {
        "outcome_metric_or_tensor_values_inspected": False,
        "outcome_metric_or_tensor_values_interpreted": False,
        "outcome_informed_scientific_change": False,
    }


def test_base_authority_validation_skips_outcome_derived_route_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[dict[str, object], str]] = []

    def record(
        _repo_root: Path, binding: object, label: str
    ) -> None:
        assert isinstance(binding, dict)
        observed.append((copy.deepcopy(binding), label))

    monkeypatch.setattr(forensic, "_validate_bound_repo_file", record)
    receipt = forensic._validate_base_authorities_read_only(REPO_ROOT)
    assert receipt["outcome_derived_route_role_authority_files_opened"] == 0
    assert receipt[
        "route_role_authority_binding_reused_from_prior_source_closure_metadata"
    ] == forensic.BASE.ROUTE_ROLE_RECEIPT_BINDING
    labels = {label for _binding, label in observed}
    expected = {
        f"original_scientific:{name}"
        for name in forensic.BASE.BASE_SCIENTIFIC_AUTHORITY_BINDINGS
        if name != "route_role_authority"
    }
    expected.update(
        f"first_execution_correction:{name}"
        for name in forensic.BASE.EXECUTION_CORRECTION_2_BASE_AMENDMENT_AUTHORITY_BINDINGS
    )
    expected.update(
        f"second_execution_correction:{name}"
        for name in ("amendment", "output_schema", "fixture", "source_closure")
    )
    assert labels == expected
    route_path = str(forensic.BASE.ROUTE_ROLE_AUTHORITY_BINDING["path"])
    assert all("route_role" not in label for label in labels)
    assert all(str(binding.get("path")) != route_path for binding, _label in observed)


def test_bound_repo_authority_validator_uses_anchored_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = forensic.attach_self_digest(
        {"schema": "synthetic.technical.authority.v1", "pass": True}
    )
    payload = forensic.canonical_json_bytes(value) + b"\n"
    relative = Path("docs/synthetic_technical_authority.json")
    binding = forensic._artifact_binding(
        str(relative), payload, content_digest=value["content_digest"]
    )
    calls: list[tuple[Path, Path, object]] = []

    def anchored(
        root: Path,
        path: str | Path,
        *,
        expected_stat_rows: object,
    ) -> bytes:
        calls.append((Path(root), Path(path), expected_stat_rows))
        return payload

    monkeypatch.setattr(forensic, "_read_no_follow_bound_file", anchored)
    forensic._validate_bound_repo_file(tmp_path, binding, "synthetic-safe")
    assert calls == [(tmp_path, relative, None)]
    source = inspect.getsource(forensic._validate_bound_repo_file)
    parsed = ast.parse(source)
    calls_ast = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)]
    assert sum(
        isinstance(node.func, ast.Name)
        and node.func.id == "_read_no_follow_bound_file"
        for node in calls_ast
    ) == 1
    assert not any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"open", "read_bytes", "read_text"}
        for node in calls_ast
    )


def test_third_receipt_verifier_is_pinned_to_anchored_reader() -> None:
    source = inspect.getsource(forensic.verify_third_failure_technical_receipts)
    parsed = ast.parse(source)
    calls = [node for node in ast.walk(parsed) if isinstance(node, ast.Call)]
    assert sum(
        isinstance(node.func, ast.Name)
        and node.func.id == "_read_no_follow_bound_file"
        for node in calls
    ) == 1
    assert not any(
        (isinstance(node.func, ast.Name) and node.func.id == "open")
        or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"open", "read_bytes", "read_text"}
        )
        for node in calls
    )


def test_attempt_policy_and_conditional_v2_spec_are_exact() -> None:
    policy = forensic.ATTEMPT_ACCOUNTING_POLICY
    assert policy["third_incident_provisional_accounting"] == {
        "attempt_namespace_created": False,
        "attempt_reservation_created": False,
        "scientific_counters_zero": True,
        "attempt_consumed": False,
        "technical_startup_incident": True,
        "technical_startup_attempt_recorded": False,
        "scientific_attempt_consumed": False,
        "publication_attempt_recorded": False,
        "basis": (
            "technical failure receipts and namespace custody; this finding "
            "does not itself authorize another execution"
        ),
    }
    assert "independently validated" in policy["scientific_completion_boundary"]
    assert "no training" in policy["publication_retry_rule"]
    authority = forensic.USER_CONDITIONAL_V2_SPEC_AUTHORITY
    assert authority["source"] == "CURRENT_USER_INSTRUCTION"
    assert authority["specification_authorized_conditionally"] is True
    assert authority["execution_authorized"] is False
    assert authority["scientific_attempt_authorized"] is False

    decision = forensic.attach_self_digest({"v2_spec_authorized": True})
    specification = forensic.build_conditional_v2_spec_authority(decision)
    assert forensic.validate_conditional_v2_spec_authority(
        specification, decision=decision
    ) == specification
    assert len(specification["clauses"]) == 10
    assert specification["execution_authorized"] is False
    assert specification["scientific_attempt_authorized"] is False
    assert specification["conditional_stage_c_semantics"].startswith(
        "STAGE_C_ENTERED_ONLY_IF_FROZEN_GATE_AUTHORIZES"
    )
    markdown = forensic.build_conditional_v2_spec_markdown(specification)
    assert forensic.validate_conditional_v2_spec_markdown(
        markdown, specification=specification
    ) == markdown
    assert "No V2 execution" in markdown

    tampered = copy.deepcopy(specification)
    tampered["clauses"][0]["requirement"] = "tampered"
    tampered.pop("content_digest")
    tampered = forensic.attach_self_digest(tampered)
    with pytest.raises(forensic.ForensicContractError, match="drift"):
        forensic.validate_conditional_v2_spec_authority(
            tampered, decision=decision
        )


def test_runtime_result_and_tracked_publication_claims_are_separate() -> None:
    runtime_source = inspect.getsource(forensic.build_preexecution_runtime_result)
    result_source = inspect.getsource(forensic.build_forensic_result)
    writer_source = inspect.getsource(forensic.write_forensic_result_artifacts)
    assert '"v2_spec_written": False' in runtime_source
    assert '"publication_pending_until_writer_completes": True' in result_source
    assert '"v2_spec_written": True' not in result_source
    assert '"v2_spec_written": True' in writer_source
    assert "FAIL_CLOSED_ALL_OR_ABSENT_WITH_ROLLBACK" in writer_source


def test_secure_reader_binds_every_component_and_leaf(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "receipts"
    nested.mkdir(parents=True)
    root.chmod(0o755)
    nested.chmod(0o755)
    leaf = nested / "value.json"
    leaf.write_bytes(b"technical-only\n")
    leaf.chmod(0o644)
    rows = {
        ".": _stat_row(root, relative=".", kind="DIRECTORY"),
        "receipts": _stat_row(
            nested, relative="receipts", kind="DIRECTORY"
        ),
        "receipts/value.json": _stat_row(
            leaf, relative="receipts/value.json", kind="FILE"
        ),
    }
    assert forensic._read_no_follow_bound_file(
        root,
        "receipts/value.json",
        expected_stat_rows=rows,
    ) == b"technical-only\n"

    missing_leaf = {key: value for key, value in rows.items() if key != "receipts/value.json"}
    with pytest.raises(forensic.ForensicContractError, match="missing"):
        forensic._read_no_follow_bound_file(
            root,
            "receipts/value.json",
            expected_stat_rows=missing_leaf,
        )


@pytest.mark.parametrize("swap_kind", ("parent", "leaf", "hardlink"))
def test_secure_reader_rejects_link_swaps_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_kind: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    root.chmod(0o755)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "target.json"
    target.write_bytes(b"forbidden-target\n")
    reads: list[int] = []
    original_read = forensic.os.read

    def recording_read(fd: int, size: int) -> bytes:
        reads.append(fd)
        return original_read(fd, size)

    monkeypatch.setattr(forensic.os, "read", recording_read)
    if swap_kind == "parent":
        (root / "receipts").symlink_to(outside, target_is_directory=True)
        relative = "receipts/target.json"
    else:
        receipts = root / "receipts"
        receipts.mkdir()
        if swap_kind == "leaf":
            (receipts / "target.json").symlink_to(target)
        else:
            os.link(target, receipts / "target.json")
        relative = "receipts/target.json"
    with pytest.raises(forensic.ForensicContractError):
        forensic._read_no_follow_bound_file(
            root,
            relative,
            expected_stat_rows=None,
        )
    assert reads == []


def test_final_inventory_self_exclusion_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "diagnostic"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    root.chmod(0o755)
    receipts.chmod(0o755)
    data = receipts / "data.json"
    data.write_bytes(b"{}\n")
    data.chmod(0o644)
    runtime_paths = copy.deepcopy(forensic.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS)
    runtime_paths["final_namespace_inventory"] = (
        "receipts/final_namespace_inventory.json"
    )
    monkeypatch.setattr(forensic, "PREEXECUTION_DIAGNOSTIC_ROOT", root)
    monkeypatch.setattr(
        forensic, "PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS", runtime_paths
    )
    monkeypatch.setattr(
        forensic,
        "_expected_final_diagnostic_namespace",
        lambda: (
            {"receipts"},
            {
                "receipts/data.json",
                "receipts/final_namespace_inventory.json",
            },
        ),
    )
    before = forensic.build_final_diagnostic_namespace_inventory(root)
    assert all(
        "bytes" not in row
        for row in before["rows"]
        if row["kind"] == "DIRECTORY"
    )
    written = forensic.write_final_diagnostic_namespace_inventory(root)
    assert written == before
    assert forensic.validate_final_diagnostic_namespace_inventory(
        written, diagnostic_root=root
    ) == written


def test_tracked_writer_rolls_back_post_write_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = forensic._forensic_result_paths()
    payloads = {
        path: f"payload-{index}\n".encode("utf-8")
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        forensic,
        "load_and_validate_diagnostic_bundle",
        lambda **_kwargs: {"pass": True},
    )
    monkeypatch.setattr(
        forensic,
        "build_forensic_result_artifacts_from_bundle",
        lambda _bundle: {"tracked_payloads": payloads},
    )
    calls = 0

    def fail_second_read(
        root: Path,
        relative: str | Path,
        *,
        expected_stat_rows: object,
    ) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise forensic.ForensicContractError("injected verification failure")
        return (root / relative).read_bytes()

    monkeypatch.setattr(
        forensic, "_read_no_follow_bound_file", fail_second_read
    )
    with pytest.raises(
        forensic.ForensicContractError, match="injected verification failure"
    ):
        forensic.write_forensic_result_artifacts(repo_root=tmp_path)
    assert all(not (tmp_path / relative).exists() for relative in paths)


def test_exclusive_write_removes_leaf_after_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "receipt.json"
    original_fsync = forensic.os.fsync
    calls = 0

    def fail_parent_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected parent fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(forensic.os, "fsync", fail_parent_fsync)
    with pytest.raises(OSError, match="injected parent fsync failure"):
        forensic._exclusive_write(target, b"technical-only\n")
    assert calls >= 3
    assert not target.exists()
    assert not target.is_symlink()


def test_exclusive_write_surfaces_unlink_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "receipt.json"
    original_fsync = forensic.os.fsync
    original_unlink = forensic.os.unlink
    fsync_calls = 0

    def fail_parent_fsync(fd: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected parent fsync failure")
        original_fsync(fd)

    def fail_unlink(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected unlink failure")

    monkeypatch.setattr(forensic.os, "fsync", fail_parent_fsync)
    monkeypatch.setattr(forensic.os, "unlink", fail_unlink)
    with pytest.raises(
        forensic.ForensicContractError,
        match="exclusive-write cleanup failed",
    ):
        forensic._exclusive_write(target, b"technical-only\n")
    assert target.is_file()
    original_unlink(target)
    assert not target.exists()


def test_authority_postflight_rejects_preexisting_source_row_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = forensic.build_forensic_source_closure(
        REPO_ROOT, require_complete=False
    )
    complete = forensic.build_prospective_forensic_source_closure(REPO_ROOT)
    namespace = {"pass": True}
    source_proof = {"pass": True}
    scientific_invariants = {"pass": True}
    preparation = forensic.attach_self_digest(
        {
            "schema": forensic.FORENSIC_FREEZE_PREPARATION_SCHEMA,
            "experiment_id": forensic.FORENSIC_EXPERIMENT_ID,
            "freeze_mode": "PREEXECUTION_FORENSIC_AUTHORITY_PREPARATION",
            "base_head": forensic.SOURCE_COMMIT,
            "changed_paths": sorted(forensic.FORENSIC_CODE_AND_TEST_PATHS),
            "prospective_authorities": {
                str(path): forensic._artifact_binding(str(path), payload)
                for path, payload in forensic.forensic_authority_payloads().items()
            },
            "prospective_source_closure": partial,
            "prospective_complete_source_closure": complete,
            "namespace_and_process_custody": namespace,
            "committed_source_root_cause_proof": source_proof,
            "scientific_implementation_invariants": scientific_invariants,
            "scientific_inputs_opened": 0,
            "scientific_archive_payload_files_opened": 0,
            "files_reused": 0,
            "required_enclosing_commit_subject": (
                forensic.FORENSIC_FREEZE_COMMIT_SUBJECT
            ),
            "pass": True,
        }
    )
    changed = copy.deepcopy(complete)
    generated = {str(path) for path in forensic.forensic_authority_payloads()}
    row = next(item for item in changed["rows"] if item["path"] not in generated)
    row["sha256"] = "0" * 64
    changed = forensic.attach_self_digest(changed)
    monkeypatch.setattr(
        forensic,
        "build_prospective_forensic_source_closure",
        lambda _root: copy.deepcopy(changed),
    )
    monkeypatch.setattr(
        forensic,
        "_validate_forensic_prepublication_namespaces",
        lambda *_args, **_kwargs: copy.deepcopy(namespace),
    )
    monkeypatch.setattr(
        forensic,
        "build_committed_source_root_cause_proof",
        lambda _root: copy.deepcopy(source_proof),
    )
    monkeypatch.setattr(
        forensic,
        "_validate_scientific_argv_builder_invariants",
        lambda _root: copy.deepcopy(scientific_invariants),
    )
    with pytest.raises(
        forensic.ForensicContractError, match="preparation receipt drift"
    ):
        forensic.validate_forensic_authority_write(
            REPO_ROOT, preparation_receipt=preparation
        )


def test_public_authority_rollback_removes_and_proves_full_set_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in forensic.FORENSIC_GENERATED_AUTHORITY_PATHS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"temporary technical authority\n")
    monkeypatch.setattr(
        forensic,
        "_git_text",
        lambda _root, arguments: (
            forensic.SOURCE_COMMIT
            if tuple(arguments) == ("rev-parse", "HEAD")
            else ""
        ),
    )
    receipt = forensic.rollback_forensic_authorities(tmp_path)
    forensic.validate_self_digest(receipt)
    assert receipt["all_intended_paths_absent"] is True
    assert set(receipt["removed_paths"]) == set(
        forensic.FORENSIC_GENERATED_AUTHORITY_PATHS
    )
    assert all(
        not (tmp_path / relative).exists()
        and not (tmp_path / relative).is_symlink()
        for relative in forensic.FORENSIC_GENERATED_AUTHORITY_PATHS
    )


def test_result_commit_validator_pins_lineage_bundle_and_tracked_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_commit = "f" * 40
    freeze_commit = "e" * 40
    result_paths = forensic._forensic_result_paths()
    payloads = {
        path: f"bound-{index}\n".encode("utf-8")
        for index, path in enumerate(result_paths)
    }
    bundle = {
        "forensic_freeze_custody": {"repo_head": freeze_commit},
        "result": {"content_digest": "1" * 64},
        "final_namespace_inventory": {"content_digest": "2" * 64},
        "preflight_namespace_inventory": {"rows": []},
        "postflight_namespace_inventory": {"rows": []},
    }
    artifacts = {
        "result": {"source_freeze_commit": freeze_commit},
        "tracked_payloads": payloads,
        "conditional_v2_decision": {"content_digest": "3" * 64},
        "conditional_v2_spec_authority": {"content_digest": "4" * 64},
        "conditional_v2_spec_markdown": "# V2\n",
    }
    calls: list[str] = []

    def fake_git(_root: Path, arguments: list[str]) -> str:
        if arguments == ["rev-parse", "HEAD"]:
            return result_commit
        if arguments == ["status", "--porcelain=v1"]:
            return ""
        if arguments == ["rev-list", "--parents", "-n", "1", result_commit]:
            return f"{result_commit} {freeze_commit}"
        if arguments == ["show", "-s", "--format=%s", result_commit]:
            return forensic.FORENSIC_RESULT_COMMIT_SUBJECT
        if arguments == ["rev-list", "--parents", "-n", "1", freeze_commit]:
            return f"{freeze_commit} {forensic.SOURCE_COMMIT}"
        if arguments == ["show", "-s", "--format=%s", freeze_commit]:
            return forensic.FORENSIC_FREEZE_COMMIT_SUBJECT
        if arguments == [
            "diff",
            "--name-only",
            forensic.SOURCE_COMMIT,
            freeze_commit,
        ]:
            return "\n".join(forensic.FORENSIC_REQUIRED_CHANGED_PATHS)
        if arguments == ["diff", "--name-status", freeze_commit, result_commit]:
            return "\n".join(f"A\t{path}" for path in result_paths)
        raise AssertionError(f"unexpected git query: {arguments}")

    monkeypatch.setattr(forensic, "_git_text", fake_git)
    monkeypatch.setattr(
        forensic, "_git_is_ancestor", lambda *_arguments: True
    )
    monkeypatch.setattr(
        forensic,
        "_validate_forensic_authority_files",
        lambda _root: {"pass": True},
    )
    monkeypatch.setattr(
        forensic,
        "_validate_archive_custody_metadata_only",
        lambda: {"pass": True},
    )
    monkeypatch.setattr(
        forensic,
        "load_and_validate_diagnostic_bundle",
        lambda **_arguments: copy.deepcopy(bundle),
    )
    monkeypatch.setattr(
        forensic,
        "build_forensic_result_artifacts_from_bundle",
        lambda _bundle: copy.deepcopy(artifacts),
    )

    def validate_spec(value: object, *, decision: object) -> object:
        assert value == artifacts["conditional_v2_spec_authority"]
        assert decision == artifacts["conditional_v2_decision"]
        calls.append("spec-json")
        return value

    def validate_markdown(value: object, *, specification: object) -> object:
        assert value == artifacts["conditional_v2_spec_markdown"]
        assert specification == artifacts["conditional_v2_spec_authority"]
        calls.append("spec-markdown")
        return value

    monkeypatch.setattr(
        forensic, "validate_conditional_v2_spec_authority", validate_spec
    )
    monkeypatch.setattr(
        forensic, "validate_conditional_v2_spec_markdown", validate_markdown
    )
    observed = {path: payload for path, payload in payloads.items()}
    monkeypatch.setattr(
        forensic,
        "_read_no_follow_bound_file",
        lambda _root, relative, *, expected_stat_rows: observed[Path(relative)],
    )
    monkeypatch.setattr(
        forensic, "_active_forensic_or_scientific_processes", lambda _root: []
    )
    monkeypatch.setattr(forensic.BASE, "OUTPUT_ROOT", tmp_path / "scientific-output")

    receipt = forensic.validate_forensic_result_commit(
        tmp_path, diagnostic_root=tmp_path / "diagnostic"
    )
    forensic.validate_self_digest(receipt)
    assert receipt["result_commit"] == result_commit
    assert receipt["sole_parent_diagnostic_freeze_commit"] == freeze_commit
    assert receipt["result_changed_paths"] == sorted(str(path) for path in result_paths)
    assert receipt["tracked_bytes_equal_rebuilt_artifact_set"] is True
    assert receipt["scientific_output_root_absent"] is True
    assert receipt["scientific_attempt_namespaces"] == []
    assert receipt["exact_scientific_or_forensic_process_matches"] == []
    assert calls == ["spec-json", "spec-markdown"]

    observed[result_paths[0]] = b"tampered\n"
    with pytest.raises(forensic.ForensicContractError, match="byte drift"):
        forensic.validate_forensic_result_commit(
            tmp_path, diagnostic_root=tmp_path / "diagnostic"
        )
