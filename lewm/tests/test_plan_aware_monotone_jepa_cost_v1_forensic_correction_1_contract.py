from __future__ import annotations

import ast
from collections import UserDict
import copy
import hashlib
from pathlib import Path
import subprocess

import pytest

from lewm.safety import (
    plan_aware_monotone_jepa_cost_v1_forensic_contract as frozen,
)
from lewm.safety import (
    plan_aware_monotone_jepa_cost_v1_forensic_correction_1_contract as correction,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FIXTURES = (
    ("ORDINARY_FOUR_KEY_ACCEPT", "ACCEPT"),
    ("ROW_COUNTED_FIVE_KEY_ACCEPT", "ACCEPT"),
    ("FIVE_KEY_REJECTED_BY_ORDINARY", "REJECT"),
    ("FOUR_KEY_REJECTED_BY_ROW_COUNTED", "REJECT"),
    ("ROWS_MISSING", "REJECT"),
    ("EXTRA_SIXTH_FIELD", "REJECT"),
    ("ROWS_NEGATIVE", "REJECT"),
    ("ROWS_FLOAT", "REJECT"),
    ("ROWS_STRING", "REJECT"),
    ("ROWS_BOOLEAN", "REJECT"),
    ("SHA256_MALFORMED", "REJECT"),
    ("CONTENT_DIGEST_MALFORMED", "REJECT"),
    ("PATH_EMPTY", "REJECT"),
    ("TERMINAL_RECEIPT_REPEAT_IDENTICAL", "PASS/BYTE_IDENTICAL"),
    ("UTF8_SELF_DIGEST_REGENERATION", "PASS/CANONICAL_BYTE_IDENTICAL"),
    ("PRODUCER_TERMINAL_CHECKER_AGREEMENT", "PASS/EXACT"),
)
EXPECTED_ZERO_COUNTER_FIELDS = (
    "fit_states_opened",
    "fit_rows_opened",
    "calibration_states_opened",
    "calibration_rows_opened",
    "heldout_states_opened",
    "heldout_rows_opened",
    "route_labels_opened",
    "true_future_latents_opened",
    "predicted_latents_opened",
    "predictor_checkpoints_opened",
    "route_cost_checkpoints_opened",
    "model_initializations",
    "optimizer_updates",
    "training_epochs",
    "scientific_score_rows",
    "candidate_selections",
    "scientific_metrics",
    "scientific_payloads",
)


def _ordinary_binding() -> dict[str, object]:
    return {
        "path": "docs/H1–H4.json",
        "sha256": "1" * 64,
        "bytes": 17,
        "content_digest": "2" * 64,
    }


def _row_counted_binding() -> dict[str, object]:
    return {**_ordinary_binding(), "rows": 12}


def _production_closure_binding() -> dict[str, object]:
    return {
        **_ordinary_binding(),
        "path": str(correction.TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        "rows": len(correction.CORRECTION_SOURCE_CLOSURE_PATHS),
    }


def test_strict_four_and_five_key_validator_selection() -> None:
    ordinary = _ordinary_binding()
    row_counted = _row_counted_binding()

    assert frozen.validate_artifact_binding(
        ordinary, content_digest_required=True
    ) == ordinary
    assert correction.validate_row_counted_content_binding(
        row_counted
    ) == row_counted

    with pytest.raises(frozen.ForensicContractError):
        frozen.validate_artifact_binding(
            row_counted, content_digest_required=True
        )
    with pytest.raises(correction.ForensicCorrectionContractError):
        correction.validate_row_counted_content_binding(ordinary)


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("missing_rows", None),
        ("extra_sixth_field", True),
        ("rows_negative", -1),
        ("rows_float", 1.0),
        ("rows_string", "1"),
        ("rows_boolean", True),
        ("sha256_malformed", "A" * 64),
        ("content_digest_malformed", "B" * 64),
        ("path_empty", ""),
    ),
)
def test_row_counted_binding_rejects_required_invalid_cases(
    mutation: str, value: object
) -> None:
    candidate = _row_counted_binding()
    if mutation == "missing_rows":
        del candidate["rows"]
    elif mutation == "extra_sixth_field":
        candidate["extra"] = value
    elif mutation.startswith("rows_"):
        candidate["rows"] = value
    elif mutation == "sha256_malformed":
        candidate["sha256"] = value
    elif mutation == "content_digest_malformed":
        candidate["content_digest"] = value
    else:
        candidate["path"] = value

    with pytest.raises(correction.ForensicCorrectionContractError):
        correction.validate_row_counted_content_binding(candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("bytes", -1),
        ("bytes", 1.0),
        ("bytes", "1"),
        ("bytes", True),
        ("rows", None),
        ("path", None),
        ("sha256", None),
        ("content_digest", None),
        ("path", "docs/NUL\x00binding.json"),
        ("sha256", "a" * 63),
        ("sha256", "g" * 64),
        ("content_digest", "b" * 65),
        ("content_digest", "z" * 64),
    ),
)
def test_row_counted_binding_rejects_all_noncanonical_field_values(
    field: str, value: object
) -> None:
    candidate = _row_counted_binding()
    candidate[field] = value
    with pytest.raises(correction.ForensicCorrectionContractError):
        correction.validate_row_counted_content_binding(candidate)


def test_row_counted_binding_rejects_non_dict_mapping_without_coercion() -> None:
    with pytest.raises(correction.ForensicCorrectionContractError):
        correction.validate_row_counted_content_binding(
            UserDict(_row_counted_binding())
        )


def test_row_counted_binding_accepts_zero_counts_without_inference() -> None:
    candidate = _row_counted_binding()
    candidate["bytes"] = 0
    candidate["rows"] = 0
    assert correction.validate_row_counted_content_binding(candidate) == candidate


def test_exact_eighteen_scientific_counters_are_strictly_zero() -> None:
    assert correction.CORRECTION_SCIENTIFIC_COUNTER_FIELDS == (
        EXPECTED_ZERO_COUNTER_FIELDS
    )
    zeros = {key: 0 for key in EXPECTED_ZERO_COUNTER_FIELDS}
    assert correction.validate_correction_zero_scientific_counters(zeros) == zeros

    invalid_values = []
    missing = copy.deepcopy(zeros)
    missing.pop(EXPECTED_ZERO_COUNTER_FIELDS[-1])
    invalid_values.append(missing)
    invalid_values.append({**zeros, "unexpected_counter": 0})
    invalid_values.append({**zeros, "training_epochs": 1})
    invalid_values.append({**zeros, "training_epochs": True})
    for invalid in invalid_values:
        with pytest.raises(correction.ForensicCorrectionContractError):
            correction.validate_correction_zero_scientific_counters(invalid)


def test_closure_callsite_pins_path_and_row_count_explicitly() -> None:
    candidate = _row_counted_binding()
    assert correction.validate_row_counted_content_binding(
        candidate,
        expected_path="docs/H1–H4.json",
        expected_rows=12,
    ) == candidate
    with pytest.raises(correction.ForensicCorrectionContractError):
        correction.validate_row_counted_content_binding(
            candidate,
            expected_path="docs/H1–H4.json",
            expected_rows=13,
        )
    with pytest.raises(correction.ForensicCorrectionContractError):
        correction.validate_row_counted_content_binding(
            candidate,
            expected_path="docs/not-the-closure.json",
            expected_rows=12,
        )


def test_exact_sixteen_fixture_receipt_and_utf8_canonical_custody() -> None:
    receipt = correction.build_row_counted_binding_fixture_receipt()
    assert correction.validate_row_counted_binding_fixture_receipt(receipt) == receipt
    assert receipt["fixture_ids"] == list(
        fixture_id for fixture_id, _ in EXPECTED_FIXTURES
    )
    assert len(receipt["rows"]) == 16
    assert receipt["row_count"] == 16
    assert receipt["all_pass"] is True
    assert receipt["rows_zero_accepted_by_generic_row_counted_schema"] is True
    assert [(row["fixture_id"], row["expected"]) for row in receipt["rows"]] == list(
        EXPECTED_FIXTURES
    )
    assert all(row["pass"] is True for row in receipt["rows"])
    assert correction.UTF8_SELF_DIGEST_FIXTURE_LITERAL == "H1–H4"
    assert correction.UTF8_SELF_DIGEST_FIXTURE_LITERAL.encode("utf-8").hex(" ") == (
        "48 31 e2 80 93 48 34"
    )
    assert frozen.canonical_json_bytes(receipt) == frozen.canonical_json_bytes(
        correction.build_row_counted_binding_fixture_receipt()
    )


def test_terminal_receipt_is_byte_identical_and_all_consumers_agree() -> None:
    binding = _production_closure_binding()
    produced = correction.produce_row_counted_content_binding(binding)
    first = correction.build_row_counted_terminal_binding_receipt(produced)
    second = correction.build_row_counted_terminal_binding_receipt(
        copy.deepcopy(produced)
    )
    checked = correction.check_external_terminal_binding_receipt(first)

    assert first["schema"] == correction.ROW_COUNTED_TERMINAL_RECEIPT_SCHEMA
    assert frozen.canonical_json_bytes(first) == frozen.canonical_json_bytes(second)
    assert correction.validate_row_counted_terminal_binding_receipt(first) == first
    assert produced == first["source_closure"] == checked["source_closure"]


def test_terminal_receipt_rejects_tampered_five_key_binding() -> None:
    receipt = correction.build_row_counted_terminal_binding_receipt(
        _production_closure_binding()
    )
    tampered = copy.deepcopy(receipt)
    tampered["source_closure"]["rows"] = 13
    with pytest.raises(
        (correction.ForensicCorrectionContractError, frozen.ForensicContractError)
    ):
        correction.validate_row_counted_terminal_binding_receipt(tampered)


def test_frozen_four_key_validator_contract_is_byte_and_ast_bound() -> None:
    assert correction.FROZEN_FORENSIC_CONTRACT_SHA256 == (
        "a57c2d01d309cc28cbdd08c51cac55615c1e97370b5f4b38ab715ec123e578e6"
    )
    assert correction.FROZEN_FOUR_KEY_VALIDATOR_AST_SHA256 == (
        "1111ad326d17ea14f019b4bbbc1c62bb5622edbebbedbc751313366f2c2aa9ab"
    )
    ordinary = _ordinary_binding()
    assert frozen.validate_artifact_binding(
        ordinary, content_digest_required=True
    ) == ordinary

    source_path = (
        REPO_ROOT
        / "lewm/safety/plan_aware_monotone_jepa_cost_v1_forensic_contract.py"
    )
    source_bytes = source_path.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == (
        correction.FROZEN_FORENSIC_CONTRACT_SHA256
    )
    module = ast.parse(source_bytes.decode("utf-8"), filename=str(source_path))
    validators = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_artifact_binding"
    ]
    assert len(validators) == 1
    assert hashlib.sha256(
        ast.dump(validators[0], include_attributes=False).encode("utf-8")
    ).hexdigest() == correction.FROZEN_FOUR_KEY_VALIDATOR_AST_SHA256

    committed = subprocess.run(
        [
            "git",
            "show",
            (
                f"{correction.SOURCE_FORENSIC_FREEZE_COMMIT}:"
                f"{correction.FROZEN_FORENSIC_CONTRACT_PATH}"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    blob_oid = subprocess.run(
        [
            "git",
            "rev-parse",
            (
                f"{correction.SOURCE_FORENSIC_FREEZE_COMMIT}:"
                f"{correction.FROZEN_FORENSIC_CONTRACT_PATH}"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    assert committed == source_bytes
    assert blob_oid == correction.FROZEN_FORENSIC_CONTRACT_BLOB_OID


def test_lineage_roots_and_future_command_are_exact_and_nonexecuting() -> None:
    assert correction.PRESERVED_LINEAGE == {
        "scientific_source": "1d799eb24d8171cb6d90bc0d0e375d9e1b0cc4f0",
        "scientific_freeze": "9c1c3adcfb8382c33e8da8895dc345e006e92e43",
        "correction_1": "14625958c0fcc21af05b33fd30c6cc2fc8537745",
        "correction_2_base": "1c18af8c3c45f9b14992362f7e50a35b651c6997",
        "forensic_freeze": "fa80f01599e99a5fd5721481f00ab9d177e2f00f",
    }
    assert correction.FAILED_PREEXECUTION_DIAGNOSTIC_ROOT != (
        correction.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
    )
    assert correction.FUTURE_V2_SPEC_ONLY_COMMAND == {
        "argv": [
            "/home/andrewknowles/TinyQuadJEPA/bin/python",
            (
                "/home/andrewknowles/Workspace/LeWMQuad-v3/scripts/"
                "evaluate_plan_aware_monotone_jepa_cost_v2.py"
            ),
            "execute",
        ],
        "cwd": "/home/andrewknowles/Workspace/LeWMQuad-v3",
        "shell_rendering": (
            "/home/andrewknowles/TinyQuadJEPA/bin/python "
            "/home/andrewknowles/Workspace/LeWMQuad-v3/scripts/"
            "evaluate_plan_aware_monotone_jepa_cost_v2.py execute"
        ),
        "status": "REQUIRED_FUTURE_V2_IMPLEMENTATION_NOT_PRESENT_OR_RUNNABLE",
        "specification_only": True,
        "execution_authorized": False,
        "do_not_execute": True,
        "separate_explicit_scientific_authority_required": True,
    }
    v2_script = Path(correction.FUTURE_V2_SPEC_ONLY_COMMAND["argv"][1])
    assert not v2_script.exists()
    assert not v2_script.is_symlink()
    assert not hasattr(
        correction,
        "PREEXECUTION_DIAGNOSTIC_CORRECTION_SYNTHETIC_CHILD_SUBCOMMAND",
    )


def test_correction_v2_spec_preserves_both_exact_technical_corrections() -> None:
    decision = frozen.attach_self_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "conditional_v2_spec_decision.correction_1.v1"
            ),
            "v2_spec_authorized": True,
            "automatic_execution_authorized": False,
            "scientific_contract_change_authorized": False,
        }
    )
    specification = correction.build_correction_conditional_v2_spec_authority(
        decision
    )
    assert correction.validate_correction_conditional_v2_spec_authority(
        specification, decision=decision
    ) == specification
    frozen_base = frozen.build_conditional_v2_spec_authority(decision)
    assert specification["frozen_base_specification_authority"] == frozen_base
    assert specification["frozen_base_specification_markdown"] == (
        frozen.build_conditional_v2_spec_markdown(frozen_base)
    )
    assert specification["base_specification_bytes_unchanged"] is True
    assert specification["exact_technical_correction_count"] == 2
    prior, row_counted = specification["exact_technical_corrections"]
    assert prior["correction_id"] == frozen.FORENSIC_DEFECT_ID
    assert prior["required_contract"] == "STABLE_ARCHIVE_IDENTITY_PROJECTION"
    assert row_counted == {
        "correction_id": (
            "ROW_COUNTED_TERMINAL_CLOSURE_BINDING_SCHEMA_MISMATCH"
        ),
        "validator_id": correction.ROW_COUNTED_CONTENT_BINDING_VALIDATOR_ID,
        "validator_symbol": "STRICT_ROW_COUNTED_CONTENT_BINDING_VALIDATOR_V1",
        "exact_keys": [
            "path", "sha256", "bytes", "content_digest", "rows"
        ],
        "expected_path": str(correction.TRACKED_CORRECTION_SOURCE_CLOSURE_PATH),
        "expected_rows": 9,
        "production_selectors": {
            "corrected_child_producer": "produce_row_counted_content_binding",
            "terminal_validator": "validate_terminal_row_counted_content_binding",
            "independent_external_checker": (
                "check_external_row_counted_content_binding"
            ),
            "terminal_receipt_external_checker": (
                "check_external_terminal_binding_receipt"
            ),
        },
        "ordinary_four_key_validator_unchanged": True,
        "scientific_payload_or_logic_changed": False,
    }
    correction_clause = next(
        clause
        for clause in specification["clauses"]
        if clause["id"] == "EXACT_TECHNICAL_CORRECTION_ONLY"
    )
    assert "whole-record archive-custody equality" in correction_clause[
        "requirement"
    ]
    assert "row-counted five-key" in correction_clause["requirement"]
    markdown = correction.build_correction_conditional_v2_spec_markdown(
        specification
    )
    assert correction.validate_correction_conditional_v2_spec_markdown(
        markdown, specification=specification
    ) == markdown
    assert "STRICT_ROW_COUNTED_CONTENT_BINDING_VALIDATOR_V1" in markdown
    assert specification["execution_authorized"] is False
    assert specification["future_v2_script_present"] is False


def test_canonical_json_counter_reload_accepts_exact_key_set_order() -> None:
    reversed_counters = dict(
        reversed(list(correction.CORRECTION_ZERO_SCIENTIFIC_COUNTERS.items()))
    )
    reloaded = frozen._canonical_json_from_bound_bytes(
        frozen.canonical_json_bytes(reversed_counters) + b"\n",
        label="correction zero counters",
    )
    assert correction.validate_correction_zero_scientific_counters(
        reloaded
    ) == correction.CORRECTION_ZERO_SCIENTIFIC_COUNTERS


def test_correction_contract_scanner_helpers_are_fail_visible_for_aliases(
    tmp_path: Path,
) -> None:
    relative = str(
        correction.CORRECTION_CONTRACT_SCRIPT.relative_to(REPO_ROOT)
    )
    assert correction._correction_argv_has_direct_contract_reference(
        ["python", correction.CORRECTION_CONTRACT_MODULE, "malformed"]
    )
    assert correction._correction_argv_has_direct_contract_reference(
        ["python", relative, "malformed"]
    )
    alias = tmp_path / "correction-contract-alias.py"
    alias.symlink_to(correction.CORRECTION_CONTRACT_SCRIPT)
    assert correction._correction_argv_references_contract_script(
        ["python", alias.name, "malformed"], process_cwd=tmp_path
    )


def test_role_scan_custody_is_exact_and_does_not_omit_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        frozen,
        "_active_forensic_or_scientific_processes",
        lambda _root, *, exclude_pids: [],
    )
    monkeypatch.setattr(
        correction,
        "_active_correction_terminal_coordinators",
        lambda *, exclude_pids: [],
    )
    custody = correction._observe_correction_role_scan_custody(
        REPO_ROOT,
        gate="POSTCOMMIT_BEFORE_RESULT_READS",
        excluded_pids={123},
    )
    assert correction._validate_correction_role_scan_custody(
        custody,
        expected_gate="POSTCOMMIT_BEFORE_RESULT_READS",
        expected_excluded_pids={123},
        reverify_live=True,
    ) == custody
    assert custody["all_nonexcluded_roles_zero"] is True
    assert custody["nonexcluded_match_count"] == 0


def test_postcommit_replay_scans_roles_before_and_after_result_reads() -> None:
    source = (
        REPO_ROOT
        / "lewm/safety/plan_aware_monotone_jepa_cost_v1_"
        "forensic_correction_1_contract.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_correction_result_commit"
    ]
    assert len(functions) == 1
    body = ast.get_source_segment(source, functions[0])
    assert body is not None
    ordered = (
        'gate="POSTCOMMIT_BEFORE_RESULT_READS"',
        'head = BASE._git_text(root, ("rev-parse", "HEAD"))',
        "artifacts = correction_result_artifact_payloads(",
        'gate="POSTCOMMIT_AFTER_RESULT_READS"',
        "disposition = BASE.attach_self_digest(",
    )
    positions = [body.index(item) for item in ordered]
    assert positions == sorted(positions)
    assert '"pre_result_read_role_scan_custody"' in body
    assert '"post_result_read_role_scan_custody"' in body


def test_postcommit_failure_custody_is_strict_and_role_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        correction,
        "_correction_proc_argv",
        lambda _pid: correction.expected_correction_result_commit_validation_argv(),
    )
    monkeypatch.setattr(
        frozen,
        "_active_forensic_or_scientific_processes",
        lambda _root, *, exclude_pids: [],
    )
    monkeypatch.setattr(
        correction,
        "_active_correction_terminal_coordinators",
        lambda *, exclude_pids: [],
    )
    receipt = correction.build_correction_postcommit_replay_failure(
        RuntimeError("postcommit regression")
    )
    assert correction.validate_correction_postcommit_replay_failure(
        receipt
    ) == receipt
    assert receipt["failure_role_scan_custody"][
        "all_nonexcluded_roles_zero"
    ] is True
    tampered = copy.deepcopy(receipt)
    tampered["scientific_counters"].pop("scientific_payloads")
    tampered.pop("content_digest")
    tampered = frozen.attach_self_digest(tampered)
    with pytest.raises(correction.ForensicCorrectionContractError):
        correction.validate_correction_postcommit_replay_failure(tampered)


def test_failed_root_authority_and_fresh_root_boundary_are_exact() -> None:
    assert correction.FAILED_DIAGNOSTIC_ROOT_AUTHORITY == {
        "root": str(correction.FAILED_PREEXECUTION_DIAGNOSTIC_ROOT),
        "file_count": 87,
        "total_file_bytes": 269_462,
        "manifest_sha256": (
            "c2b9c4dc241eead083a8eb2a8eda074cb86e2b40a1a78d0a8f289482884c1385"
        ),
        "manifest_row_keys": ["path", "sha256", "bytes"],
        "technical_only": True,
        "immutable_and_nonreusable": True,
    }
    assert len(correction.FAILED_TECHNICAL_FILE_PATHS) == 87
    assert len(set(correction.FAILED_TECHNICAL_FILE_PATHS)) == 87
    assert correction.FAILED_PREEXECUTION_DIAGNOSTIC_ROOT.is_dir()
    assert not correction.FAILED_PREEXECUTION_DIAGNOSTIC_ROOT.is_symlink()
    assert not correction.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT.exists()
    assert not correction.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT.is_symlink()


def test_prelaunch_namespace_builder_and_validator_use_one_exact_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        frozen,
        "_active_forensic_or_scientific_processes",
        lambda _root, *, exclude_pids: [],
    )
    monkeypatch.setattr(
        correction,
        "_active_correction_terminal_coordinators",
        lambda *, exclude_pids: [],
    )
    monkeypatch.setattr(
        correction,
        "_anchored_leaf_absent",
        lambda _path: True,
    )
    observed = correction._validate_correction_prepublication_namespaces(
        REPO_ROOT,
        exclude_current_process=True,
    )
    assert observed["fresh_diagnostic_root_absent"] is True
    assert "fresh_root_absent" not in observed
    assert correction._validate_correction_prelaunch_namespace_custody(
        observed
    ) == observed


def test_correction_launcher_stages_and_reloads_before_returning() -> None:
    """Pin the correction-only staging order without executing the launcher."""

    evaluator_path = REPO_ROOT / "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
    source = evaluator_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(evaluator_path))
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_execute_preexecution_diagnostic_under_umask"
    ]
    assert len(functions) == 1
    body = ast.get_source_segment(source, functions[0])
    assert body is not None

    ordered_literals = (
        '_atomic_preexecution_json(paths["diagnostic_custody"], custody)',
        "correction.write_correction_terminal_bundle(",
        "correction.load_and_validate_correction_diagnostic_bundle(",
        'reloaded["diagnostic_custody"] != custody',
        'reloaded["synthetic_results"] != synthetic_receipt',
        'reloaded["result"] != staged["runtime_result"]',
        'reloaded["final_namespace_inventory"]',
        "return staged",
    )
    positions = [body.index(literal) for literal in ordered_literals]
    assert positions == sorted(positions)
    assert body.count("correction.write_correction_terminal_bundle(") == 1
    assert body.count(
        "correction.load_and_validate_correction_diagnostic_bundle("
    ) == 1

    base_publication = "return _publish_preexecution_diagnostic_terminal("
    assert body.count(base_publication) == 1
    assert body.index(base_publication) > body.index("return staged")


def test_production_corrected_child_survives_custody_canonical_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real five-key child and correction custody adapters."""

    source_commit = "a" * 40
    closure = correction.build_forensic_correction_source_closure(
        REPO_ROOT,
        prospective_payloads=correction.forensic_correction_authority_payloads(),
    )
    closure_binding = correction.correction_source_closure_binding(closure)
    root_cause = frozen.build_committed_source_root_cause_proof(REPO_ROOT)
    freeze = frozen.attach_self_digest(
        {
            "repo_head": source_commit,
            "correction_source_closure": closure_binding,
            "committed_source_root_cause_proof": root_cause,
        }
    )

    def process_identity(
        pid: int, argv: list[str], role: str
    ) -> dict[str, object]:
        return frozen.validate_process_identity(
            {
                "pid": pid,
                "process_group_id": pid,
                "start_time_ticks": pid * 10,
                "argv": argv,
                "argv_sha256": frozen.canonical_json_sha256(argv),
                "executable": "/usr/bin/python3",
                "role": role,
            }
        )

    launcher = process_identity(
        101,
        ["/usr/bin/python3", "correction-launcher"],
        "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_LAUNCHER",
    )
    outer_argv = ["/usr/bin/python3", "correction-child"]
    child_identity = process_identity(
        102,
        outer_argv,
        "PREEXECUTION_DIAGNOSTIC_CORRECTION_1_CHILD",
    )
    internal_argv = correction.expected_correction_child_inner_argv(
        launcher_pid=launcher["pid"],
        launcher_start_time_ticks=launcher["start_time_ticks"],
    )
    runtime_context = frozen.build_current_runtime_context(
        captured_at_ns=50,
        identity={
            "uid": 1000,
            "euid": 1000,
            "gid": 1000,
            "egid": 1000,
            "groups": [1000],
        },
        cwd=str(REPO_ROOT),
        umask={
            "value": frozen.PREEXECUTION_DIAGNOSTIC_UMASK,
            "sampled_and_restored": True,
        },
        rlimits={"RLIMIT_NOFILE": {"soft": 1024, "hard": 1024}},
        cpu={"affinity": [0], "cpu_count": 1},
        gpu={
            "visibility_environment": {
                "CUDA_VISIBLE_DEVICES": None,
                "NVIDIA_VISIBLE_DEVICES": None,
                "ROCR_VISIBLE_DEVICES": None,
                "HIP_VISIBLE_DEVICES": None,
            },
            "device_metadata": [],
            "device_files_opened": 0,
        },
        temp={
            "environment": {"TMPDIR": None, "TMP": None, "TEMP": None},
            "path_metadata": [],
        },
    )
    fixture = frozen.build_archive_verification_record_equality_fixture()
    mismatch = {
        "fixture": "MINIMAL_RUNTIME_VERSUS_DETAILED_VALIDATOR_ARCHIVE_CUSTODY",
        "raw_record_equality": False,
        "identity_projection_equality": True,
        "mismatch_mechanism": (
            "INCOMPATIBLE_ARCHIVE_CUSTODY_SCHEMA_WHOLE_RECORD_COMPARISON"
        ),
        "minimal_key_sets": [
            sorted(row) for row in fixture["runtime_minimal_records"]
        ],
        "detailed_key_sets": [
            sorted(row) for row in fixture["new_attempt_detailed_records"]
        ],
        "stable_identity_fields": fixture["identity_projection_fields"],
        "scientific_payloads_opened": 0,
        "pass": True,
    }
    monkeypatch.setattr(
        correction,
        "validate_correction_freeze_custody_receipt",
        lambda value, **_kwargs: value,
    )
    child = correction.build_corrected_preexecution_child_result(
        repo_root=REPO_ROOT,
        launcher_process_identity=launcher,
        child_process_identity=child_identity,
        repo_head=source_commit,
        repo_clean=True,
        scientific_contract_digest=(
            frozen.BASE.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        ),
        forensic_authority_source_closure=closure_binding,
        forensic_freeze_custody=freeze,
        output_namespace_before=[],
        output_namespace_after=[],
        mismatch_evidence=mismatch,
        committed_source_root_cause_proof=root_cause,
        current_runtime_context=runtime_context,
        scientific_counters=correction.CORRECTION_ZERO_SCIENTIFIC_COUNTERS,
    )
    assert child["terminal_binding_receipt"] == (
        correction.build_row_counted_terminal_binding_receipt(closure_binding)
    )

    environment_map = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "VIRTUAL_ENV": "/home/andrewknowles/TinyQuadJEPA",
        "PATH": "/home/andrewknowles/TinyQuadJEPA/bin",
    }
    environment = frozen.build_environment_receipt(
        inherited_python_keys_removed=[],
        inherited_environment_key_names=[],
        result_environment=environment_map,
        virtual_env=environment_map["VIRTUAL_ENV"],
        path_prepend=environment_map["PATH"],
    )
    diagnostic_root = correction.PREEXECUTION_DIAGNOSTIC_CORRECTION_ROOT
    fd_custody = {
        "traceback_fd": 3,
        "exception_fd": 4,
        "heartbeat_fd": 5,
        "read_guard_events_fd": 6,
        "paths": {
            label: str(
                (
                    diagnostic_root
                    / frozen.PREEXECUTION_DIAGNOSTIC_RUNTIME_PATHS[key]
                ).absolute()
            )
            for label, key in {
                "traceback": "child_traceback",
                "exception": "child_exception",
                "heartbeat": "heartbeat",
                "read_guard_events": "read_guard_events",
            }.items()
        },
        "pass": True,
    }
    command = frozen.build_command_receipt(
        diagnostic_root=diagnostic_root,
        outer_exact_argv=outer_argv,
        internal_exact_argv=internal_argv,
        fd_custody=fd_custody,
        environment_receipt=environment,
        cwd=REPO_ROOT,
    )
    invocation = frozen.build_invocation_receipt(
        source_commit=source_commit,
        diagnostic_root=diagnostic_root,
        launcher_process_identity=launcher,
        child_process_identity=child_identity,
        outer_exact_argv=outer_argv,
        internal_exact_argv=internal_argv,
        environment=environment,
        fd_custody=fd_custody,
        technical_runtime_context=runtime_context,
        started_monotonic_ns=100,
        namespace_before=[],
    )
    umask = frozen.build_diagnostic_umask_custody(
        previous_umask=frozen.PREEXECUTION_DIAGNOSTIC_EXPECTED_PREVIOUS_UMASK,
        restored_umask=frozen.PREEXECUTION_DIAGNOSTIC_EXPECTED_PREVIOUS_UMASK,
        set_before_root_creation=True,
        inherited_by_all_children=True,
        restoration_verified=True,
    )
    termination = {
        "kind": "EXIT",
        "exit_code_or_null": 0,
        "signal_number_or_null": None,
        "signal_name_or_null": None,
    }
    cleanup = {
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
    os_evidence = frozen.build_os_evidence_receipt(
        launcher_process_identity=launcher,
        child_process_identity=child_identity,
        started_monotonic_ns=100,
        ended_monotonic_ns=200,
        returncode=0,
        termination=termination,
        cleanup=cleanup,
        namespace_before=[],
        namespace_after=[],
        current_runtime_context=runtime_context,
    )
    preexecution = frozen.build_preexecution_only_receipt(
        source_commit=source_commit,
        namespace_before=[],
        namespace_after=[],
        scientific_counters=frozen.ZERO_SCIENTIFIC_COUNTERS,
    )
    synthetic = frozen.attach_self_digest({"pass": True})
    monkeypatch.setattr(
        frozen,
        "validate_synthetic_results_receipt",
        lambda value: value,
    )
    last_stage = frozen.build_last_stage_marker(
        producer_role="PREEXECUTION_DIAGNOSTIC_CHILD",
        stage_id="COMPLETE",
        event="COMPLETED",
        pid=child_identity["pid"],
        monotonic_ns=500,
    )
    startup_rows = []
    sequence = 0
    monotonic_ns = 100
    for stage_id in frozen.PREEXECUTION_DIAGNOSTIC_STAGE_IDS:
        for event in ("STARTED", "COMPLETED"):
            startup_rows.append(
                frozen.build_startup_stage_row(
                    sequence=sequence,
                    stage_id=stage_id,
                    event=event,
                    monotonic_ns=monotonic_ns,
                )
            )
            sequence += 1
            monotonic_ns += 1
    child_payload = correction._authority_bytes(child)
    streams = {
        label: frozen._artifact_binding(path, payload)
        for label, path, payload in (
            ("stdout", "streams/child.stdout", child_payload),
            ("stderr", "streams/child.stderr", b""),
            ("traceback", "streams/child.traceback", b""),
            ("exception", "receipts/child_exception.json", b""),
            ("heartbeat", "receipts/heartbeat.jsonl", b"x"),
            ("read_guard_events", "receipts/read_guard_events.jsonl", b""),
        )
    }
    custody = correction.build_correction_diagnostic_custody_receipt(
        repo_root=REPO_ROOT,
        source_commit=source_commit,
        correction_source_closure=closure,
        diagnostic_root=diagnostic_root,
        launcher_process_identity=launcher,
        child_process_identity=child_identity,
        forensic_freeze_custody_receipt=freeze,
        umask_custody_receipt=umask,
        invocation_receipt=invocation,
        environment_receipt=environment,
        command_receipt=command,
        read_guard_manifest=frozen.build_read_guard_manifest(REPO_ROOT),
        os_evidence_receipt=os_evidence,
        preexecution_only_receipt=preexecution,
        synthetic_results_receipt=synthetic,
        corrected_child_result=child,
        last_stage_marker_value=last_stage,
        startup_stage_rows=startup_rows,
        stream_bindings=streams,
        exception_observed=False,
    )
    reloaded = frozen._canonical_json_from_bound_bytes(
        correction._authority_bytes(custody),
        label="production correction custody bridge",
    )
    checked = correction._validate_correction_diagnostic_custody_envelope(
        reloaded
    )
    assert checked["preexecution_child_result"]["schema"] == (
        correction.CORRECTED_CHILD_RESULT_SCHEMA
    )
    assert checked["preexecution_child_result"]["terminal_binding_receipt"] == (
        child["terminal_binding_receipt"]
    )
