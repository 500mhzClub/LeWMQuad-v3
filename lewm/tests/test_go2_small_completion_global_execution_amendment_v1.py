from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat

import pytest

from lewm.oracle import go2_small_completion_global_execution_amendment_v1 as AUTH


def _source_bindings() -> list[dict[str, object]]:
    return [{
        "path": path,
        "role": role,
        "byte_count": index + 1,
        "sha256": f"{index + 1:064x}",
    } for index, (path, role) in enumerate(AUTH.SOURCE_SPECS)]


def _scientific_bindings() -> dict[str, str]:
    result = {
        key: f"{index + 20:064x}"
        for index, key in enumerate(AUTH.SCIENTIFIC_CONTRACT_BINDING_KEYS)
    }
    result["source_repository_commit"] = "a" * 40
    result["genesis_backend"] = "synthetic-test-backend"
    result["candidate_allocation_amendment_digest"] = \
        AUTH.CANDIDATE_ALLOCATION_AMENDMENT_DIGEST
    result["candidate_allocator_contract_digest"] = \
        AUTH.CANDIDATE_ALLOCATION_CONTRACT_DIGEST
    result["scorer_fit_allocation_design_digest"] = \
        AUTH.SCORER_FIT_ALLOCATION_DESIGN_DIGEST
    return result


def _input_bindings() -> dict[str, object]:
    return {
        "predecessor_scientific_input_bindings_digest": "b" * 64,
        "candidate_pool_scene_ids_digest": "c" * 64,
        "fixed_state_projection_digest": "d" * 64,
        "candidate_pool_count": 17,
        "fixed_state_count": 115,
        "selected_completion_scene_count": 5,
        "final_state_count": 120,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }


def _lineage_bindings() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for spec in AUTH._LINEAGE_SPECS:
        raw_sha256, byte_count = AUTH._EXPECTED_LINEAGE_RAW[str(spec["role"])]
        row = {
            **dict(spec),
            "raw_sha256": raw_sha256,
            "byte_count": byte_count,
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": False,
            "preserved_not_retried_or_reinterpreted": True,
        }
        result[str(spec["role"])] = row
    return result


def _report() -> dict[str, object]:
    return AUTH.build_coupling_report(
        source_repository_commit="e" * 40,
        source_bindings=_source_bindings(),
        scientific_contract_bindings=_scientific_bindings(),
        preoutcome_input_bindings=_input_bindings(),
    )


def _report_binding(report: dict[str, object]) -> dict[str, object]:
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    return AUTH.coupling_report_artifact_binding(report, raw)


def _source_correction_fixture(
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[
            dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Build exact synthetic V1/mixed witnesses and a three-file successor."""

    historical_sources = _source_bindings()
    report = AUTH.build_coupling_report(
        source_repository_commit=
            AUTH.ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        source_bindings=historical_sources,
        scientific_contract_bindings=_scientific_bindings(),
        preoutcome_input_bindings=_input_bindings(),
    )
    report_raw = AUTH._pretty_json_bytes(report)
    report_binding = AUTH.coupling_report_artifact_binding(report, report_raw)
    amendment = AUTH.build_execution_amendment(
        source_repository_commit=
            AUTH.ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        source_bindings=historical_sources,
        coupling_report_binding=report_binding,
        scientific_contract_bindings=_scientific_bindings(),
        preoutcome_input_bindings=_input_bindings(),
        predecessor_lineage=_lineage_bindings(),
    )
    amendment_raw = AUTH._pretty_json_bytes(amendment)
    exact_report_binding = {
        "path": str(AUTH.COUPLING_REPORT_RELATIVE_PATH),
        "schema": AUTH.REPORT_SCHEMA,
        "self_digest_key": AUTH.REPORT_SELF_KEY,
        "self_digest": report[AUTH.REPORT_SELF_KEY],
        "raw_sha256": hashlib.sha256(report_raw).hexdigest(),
        "byte_count": len(report_raw),
        "source_repository_commit":
            AUTH.ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
    }
    exact_amendment_binding = {
        "path": str(AUTH.EXECUTION_AMENDMENT_RELATIVE_PATH),
        "schema": AUTH.AMENDMENT_SCHEMA,
        "self_digest_key": AUTH.AMENDMENT_SELF_KEY,
        "self_digest": amendment[AUTH.AMENDMENT_SELF_KEY],
        "raw_sha256": hashlib.sha256(amendment_raw).hexdigest(),
        "byte_count": len(amendment_raw),
        "source_repository_commit":
            AUTH.ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
    }
    monkeypatch.setattr(
        AUTH, "ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING",
        exact_report_binding)
    monkeypatch.setattr(
        AUTH, "ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING",
        exact_amendment_binding)
    v1 = {
        "status": AUTH._V1_AUTHORITY_STATUS,
        "source_repository_commit":
            AUTH.ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        "coupling_report": report,
        "coupling_report_artifact_binding": exact_report_binding,
        "execution_amendment": amendment,
        "execution_amendment_artifact_binding": exact_amendment_binding,
    }

    mixed_unsigned = {
        "schema": (
            "go2_scorer_fit_preserved_state_mixed_precontract_"
            "disposition_reachability_v2"),
        "status": "PASS_PREOUTCOME_37_RETAINED_8_REPLACEMENT_DISPOSITION",
        "complete": True,
        "synthetic_test_only": True,
    }
    mixed_self = AUTH._legacy_default_json_digest(mixed_unsigned)
    mixed_payload = {
        **mixed_unsigned,
        "mixed_precontract_disposition_receipt_digest": mixed_self,
    }
    mixed_raw = AUTH._pretty_json_bytes(mixed_payload)
    mixed_binding = {
        "path": AUTH.HISTORICAL_MIXED_DISPOSITION_ARTIFACT_BINDING["path"],
        "self_digest_key":
            "mixed_precontract_disposition_receipt_digest",
        "self_digest": mixed_self,
        "raw_sha256": hashlib.sha256(mixed_raw).hexdigest(),
        "byte_count": len(mixed_raw),
    }
    monkeypatch.setattr(
        AUTH, "HISTORICAL_MIXED_DISPOSITION_ARTIFACT_BINDING",
        mixed_binding)
    mixed = {"payload": mixed_payload, "binding": mixed_binding}

    successor_sources = copy.deepcopy(historical_sources)
    allowed = set(AUTH.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(successor_sources):
        if row["path"] in allowed:
            row["byte_count"] = int(row["byte_count"]) + 100
            row["sha256"] = f"{index + 40:064x}"
    assert {
        row["path"] for old, row in zip(
            historical_sources, successor_sources, strict=True)
        if (old["byte_count"], old["sha256"])
        != (row["byte_count"], row["sha256"])
    } == allowed
    return v1, successor_sources, mixed


def _source_correction_amendment(
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    v1, sources, mixed = _source_correction_fixture(monkeypatch)
    amendment = AUTH.build_execution_amendment_v2(
        source_repository_commit="f" * 40,
        source_bindings=sources,
        v1_execution_authority=v1,
        historical_mixed_disposition_authority=mixed,
    )
    return amendment, v1, mixed


def _preplan_integration_correction(
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    v2, _v1, _mixed = _source_correction_amendment(monkeypatch)
    v2_raw = AUTH._pretty_json_bytes(v2)
    v2_binding = {
        "path": str(AUTH.EXECUTION_AMENDMENT_V2_RELATIVE_PATH),
        "schema": AUTH.AMENDMENT_V2_SCHEMA,
        "self_digest_key": AUTH.AMENDMENT_SELF_KEY,
        "self_digest": v2[AUTH.AMENDMENT_SELF_KEY],
        "raw_sha256": hashlib.sha256(v2_raw).hexdigest(),
        "byte_count": len(v2_raw),
        "source_repository_commit": "f" * 40,
    }
    monkeypatch.setattr(
        AUTH, "IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT", "f" * 40)
    monkeypatch.setattr(
        AUTH, "IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING",
        v2_binding)
    v2_authority = {"payload": v2, "binding": v2_binding}
    current_sources = copy.deepcopy(v2["source_bindings"])
    allowed = set(
        AUTH.PREPLAN_INTEGRATION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    for index, row in enumerate(current_sources):
        if row["path"] in allowed:
            row["byte_count"] = int(row["byte_count"]) + 200
            row["sha256"] = f"{index + 50:064x}"
    assert {
        row["path"] for old, row in zip(
            v2["source_bindings"], current_sources, strict=True)
        if (old["byte_count"], old["sha256"])
        != (row["byte_count"], row["sha256"])
    } == allowed
    correction = AUTH.build_preplan_integration_correction(
        source_repository_commit="9" * 40,
        source_bindings=current_sources,
        immutable_v2_execution_authority=v2_authority)
    return correction, v2_authority, {"sources": current_sources}


def test_coupling_report_is_closed_coupled_and_self_digested() -> None:
    report = _report()
    validated = AUTH.validate_coupling_report(
        report,
        expected_scientific_contract_bindings=_scientific_bindings(),
        expected_preoutcome_input_bindings=_input_bindings(),
        expected_source_repository_commit="e" * 40,
        validate_live_source=False,
    )

    assert validated == report
    assert report["classification"] == "COUPLED"
    assert report["selected_method"] == "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL"
    assert report["constraint_ids"] == list(AUTH.CONSTRAINT_IDS)
    assert len(report["constraint_inventory"]) == 22
    assert report["decisive_constraint_ids"] == [
        "GOAL_TYPE_CANDIDATE_FLOOR_CEILING",
        "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
        "ALL_40_COMPLETION_MASKS_PASS",
    ]
    assert report[AUTH.REPORT_SELF_KEY] == AUTH.canonical_digest({
        key: value for key, value in report.items()
        if key != AUTH.REPORT_SELF_KEY
    })
    assert report["scope"]["solver_invoked"] is False
    assert report["outcome_boundary"] == {
        "candidate_outcomes_consumed": False,
        "branch_labels_read": False,
        "frames_read_or_generated": False,
        "latents_read_or_generated": False,
        "scorer_metrics_read": False,
        "predictor_outputs_read": False,
    }


@pytest.mark.parametrize("mutation", [
    "drop_constraint", "add_constraint", "reclassify", "decouple",
    "change_scientific_binding", "change_input_count",
])
def test_coupling_report_rejects_every_surface_change(mutation: str) -> None:
    report = copy.deepcopy(_report())
    if mutation == "drop_constraint":
        report["constraint_inventory"].pop()
    elif mutation == "add_constraint":
        report["constraint_inventory"].append(copy.deepcopy(
            report["constraint_inventory"][0]))
    elif mutation == "reclassify":
        report["constraint_inventory"][11]["joint_reference"] = False
    elif mutation == "decouple":
        report["classification"] = "DECOUPLED"
    elif mutation == "change_scientific_binding":
        report["scientific_contract_bindings"]["selection_digest"] = "f" * 64
    else:
        report["preoutcome_input_bindings"]["candidate_pool_count"] = 18
    report[AUTH.REPORT_SELF_KEY] = AUTH.canonical_digest({
        key: value for key, value in report.items()
        if key != AUTH.REPORT_SELF_KEY
    })

    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_coupling_report(
            report,
            expected_scientific_contract_bindings=_scientific_bindings(),
            expected_preoutcome_input_bindings=_input_bindings(),
            validate_live_source=False,
        )


def test_scientific_bindings_require_frozen_allocation_digests() -> None:
    for key in (
            "candidate_allocation_amendment_digest",
            "candidate_allocator_contract_digest",
            "scorer_fit_allocation_design_digest"):
        bindings = _scientific_bindings()
        bindings[key] = "0" * 64
        with pytest.raises(AUTH.GlobalExecutionAmendmentError):
            AUTH.validate_scientific_contract_bindings(bindings)


def test_execution_amendment_supersedes_only_nonscientific_execution_choices(
        ) -> None:
    report = _report()
    amendment = AUTH.build_execution_amendment(
        source_repository_commit="e" * 40,
        source_bindings=_source_bindings(),
        coupling_report_binding=_report_binding(report),
        scientific_contract_bindings=_scientific_bindings(),
        preoutcome_input_bindings=_input_bindings(),
        predecessor_lineage=_lineage_bindings(),
    )
    validated = AUTH.validate_execution_amendment(
        amendment,
        expected_coupling_report_binding=_report_binding(report),
        expected_scientific_contract_bindings=_scientific_bindings(),
        expected_preoutcome_input_bindings=_input_bindings(),
        expected_source_repository_commit="e" * 40,
        validate_live_authorities=False,
    )

    assert validated == amendment
    supersession = amendment["supersession"]
    assert supersession["status"] == \
        AUTH.SUPERSEDED_EXTERNAL_ENUMERATION_STATUS
    assert supersession["superseded_execution_requirements"] == [
        {
            "constraint_id": "FIRST_PASSING_SCENE_COMBINATION",
            "status": AUTH.SUPERSEDED_EXTERNAL_ENUMERATION_STATUS,
            "requirement": (
                "external enumeration of 6,188 five-scene combinations and "
                "proof that the selected combination is globally "
                "lexicographically earliest"
            ),
        },
        {
            "constraint_id": "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
            "status": AUTH.SUPERSEDED_CANONICAL_TIE_BREAK_STATUS,
            "requirement": (
                "sequential identity-ordered lexicographic minimisation of the "
                "rotation vector before applying exact completion compatibility"
            ),
        },
    ]
    assert supersession["scientific_requirement_superseded"] is False
    assert supersession["selector_superseded"] is False
    assert supersession["candidate_allocation_constraints_superseded"] is False
    assert supersession["oracle_superseded"] is False
    assert supersession["scorer_protocol_superseded"] is False
    assert amendment["v1_disposition"] == AUTH.V1_FAILURE_STATUS
    assert amendment["v2_backend_disposition"] == AUTH.V2_BACKEND_DISPOSITION
    method = amendment["selected_execution_method"]
    assert method["single_global_model"] is True
    assert method["external_combination_enumeration"] is False
    assert method["solver_threads"] == 1
    assert method["old_allocate_then_post_gate_sequence_preserved"] is False
    assert method[
        "completion_eligibility_predicate_and_evidence_preserved"] is True
    assert method["stable_hash_pair_objective"] == \
        AUTH.STABLE_HASH_OBJECTIVE_CONTRACT
    assert method["stable_hash_pair_objective_digest"] == \
        AUTH.STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST
    assert method["superseded_execution_constraint_ids"] == [
        "FIRST_PASSING_SCENE_COMBINATION",
        "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
    ]
    assert "FIRST_PASSING_SCENE_COMBINATION" not in \
        method["preserved_mandatory_constraint_ids"]
    assert "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER" not in \
        method["preserved_mandatory_constraint_ids"]
    assert method["branch_or_downstream_metric_in_objective"] is False
    assert amendment["issuance_boundary"]["performance_benchmark_run"] is False
    assert amendment["continuation_authority"][
        "final_200_state_corpus_authorised"] is False


def test_dual_downstream_runtime_contract_and_stage_roles_are_exactly_bound(
        ) -> None:
    expected_runtime_contracts = {
        "genesis": {
            "role": "genesis_branch_generation",
            "interpreter_relative_path": (
                ".generated/venvs/genesis_render_vulkan/bin/python"
            ),
            "pyvenv_config_relative_path": (
                ".generated/venvs/genesis_render_vulkan/pyvenv.cfg"
            ),
            "pyvenv_config_sha256": (
                "41c6a8f52f3404bd3b7fcd805519c1976e2f4194ef9aa8eccf2f0919383386a9"
            ),
            "pyvenv_config_byte_count": 219,
            "python_version": "3.12.3",
            "genesis_version": "0.3.14",
            "torch_version": "2.12.0+cu130",
            "torch_cuda_runtime": "13.0",
            "torch_hip_runtime": None,
            "accelerator_available": False,
            "accelerator_device_count": 0,
            "accelerator_devices": [],
        },
        "rocm": {
                "role": "rocm_encoding_training_and_development",
                "interpreter_relative_path": (
                    ".generated/venvs/world_model_rocm_7_2_1_v1/bin/python"
                ),
                "pyvenv_config_relative_path": (
                    ".generated/venvs/world_model_rocm_7_2_1_v1/pyvenv.cfg"
                ),
                "pyvenv_config_sha256": (
                    "49222cc65a628e83d00d99da60f1dea8d59bc01a3ea9616227f330e2ecd50577"
                ),
            "pyvenv_config_byte_count": 223,
            "python_version": "3.12.3",
            "torch_version": "2.12.0+rocm7.2",
            "torch_cuda_runtime": None,
            "torch_hip_runtime": "7.2.53211",
            "accelerator_available": True,
            "accelerator_device_count": 2,
            "accelerator_devices": [
                {
                    "index": 0,
                    "name": "AMD Radeon AI PRO R9700",
                    "capability": [12, 0],
                    "gcn_arch_name": "gfx1201",
                    "multi_processor_count": 32,
                },
                {
                    "index": 1,
                    "name": "AMD Ryzen 9 9950X3D 16-Core Processor",
                    "capability": [10, 3],
                    "gcn_arch_name": "gfx1036",
                    "multi_processor_count": 1,
                },
            ],
        },
    }
    expected_stage_roles = {
        "six_branch_smoke": "genesis",
        "smoke_encoding": "rocm",
        "full_720_branch_corpus": "genesis",
        "full_latent_encoding": "rocm",
        "scorer_training_and_qualification": "rocm",
        "development_transfer": "rocm",
        "qualification_validation": "rocm",
        "development_validation": "rocm",
    }
    assert AUTH.DOWNSTREAM_RUNTIME_CONTRACTS == expected_runtime_contracts
    assert AUTH.DOWNSTREAM_STAGE_RUNTIME_ROLES == expected_stage_roles

    report = _report()
    amendment = AUTH.build_execution_amendment(
        source_repository_commit="e" * 40,
        source_bindings=_source_bindings(),
        coupling_report_binding=_report_binding(report),
        scientific_contract_bindings=_scientific_bindings(),
        preoutcome_input_bindings=_input_bindings(),
        predecessor_lineage=_lineage_bindings(),
    )
    continuation = amendment["continuation_authority"]
    assert continuation["downstream_runtime_contracts"] == \
        expected_runtime_contracts
    assert continuation["downstream_stage_runtime_roles"] == \
        expected_stage_roles
    assert continuation["downstream_uses_global_solver_interpreter"] == {
        "genesis": False,
        "rocm": False,
    }


@pytest.mark.parametrize("mutation", [
    "genesis_interpreter",
    "genesis_pyvenv_digest",
    "genesis_version",
    "rocm_interpreter",
    "rocm_torch_version",
    "rocm_accelerator_identity",
    "stage_runtime_role",
    "global_solver_interpreter",
])
def test_execution_amendment_rejects_downstream_runtime_mutation(
        mutation: str) -> None:
    report = _report()
    report_binding = _report_binding(report)
    amendment = AUTH.build_execution_amendment(
        source_repository_commit="e" * 40,
        source_bindings=_source_bindings(),
        coupling_report_binding=report_binding,
        scientific_contract_bindings=_scientific_bindings(),
        preoutcome_input_bindings=_input_bindings(),
        predecessor_lineage=_lineage_bindings(),
    )
    continuation = amendment["continuation_authority"]
    runtimes = continuation["downstream_runtime_contracts"]
    if mutation == "genesis_interpreter":
        runtimes["genesis"]["interpreter_relative_path"] = \
            ".generated/venvs/genesis_rocm/bin/python"
    elif mutation == "genesis_pyvenv_digest":
        runtimes["genesis"]["pyvenv_config_sha256"] = "0" * 64
    elif mutation == "genesis_version":
        runtimes["genesis"]["genesis_version"] = "0.3.15"
    elif mutation == "rocm_interpreter":
        runtimes["rocm"]["interpreter_relative_path"] = \
            ".generated/venvs/genesis_render_vulkan/bin/python"
    elif mutation == "rocm_torch_version":
        runtimes["rocm"]["torch_version"] = "2.12.0+cu130"
    elif mutation == "rocm_accelerator_identity":
        runtimes["rocm"]["accelerator_devices"][0]["gcn_arch_name"] = \
            "gfx1200"
    elif mutation == "stage_runtime_role":
        continuation["downstream_stage_runtime_roles"][
            "six_branch_smoke"] = "rocm"
    else:
        continuation["downstream_uses_global_solver_interpreter"][
            "rocm"] = True
    amendment[AUTH.AMENDMENT_SELF_KEY] = AUTH.canonical_digest({
        key: value for key, value in amendment.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })

    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_execution_amendment(
            amendment,
            expected_coupling_report_binding=report_binding,
            expected_scientific_contract_bindings=_scientific_bindings(),
            expected_preoutcome_input_bindings=_input_bindings(),
            expected_source_repository_commit="e" * 40,
            validate_live_authorities=False,
        )


def test_execution_amendment_rejects_extra_supersession() -> None:
    report = _report()
    amendment = AUTH.build_execution_amendment(
        source_repository_commit="e" * 40,
        source_bindings=_source_bindings(),
        coupling_report_binding=_report_binding(report),
        scientific_contract_bindings=_scientific_bindings(),
        preoutcome_input_bindings=_input_bindings(),
        predecessor_lineage=_lineage_bindings(),
    )
    amendment["supersession"]["selector_superseded"] = True
    amendment[AUTH.AMENDMENT_SELF_KEY] = AUTH.canonical_digest({
        key: value for key, value in amendment.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })

    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_execution_amendment(
            amendment,
            expected_coupling_report_binding=_report_binding(report),
            expected_scientific_contract_bindings=_scientific_bindings(),
            expected_preoutcome_input_bindings=_input_bindings(),
            validate_live_authorities=False,
        )


def test_lineage_bindings_are_exact_and_immutable() -> None:
    lineage = _lineage_bindings()
    assert AUTH.validate_predecessor_lineage_bindings(lineage) == lineage
    assert lineage["v1_immutable_failure"]["self_digest"] == \
        AUTH.V1_FAILURE_RECEIPT_DIGEST
    assert lineage["v2_contract"]["self_digest"] == AUTH.V2_CONTRACT_DIGEST
    assert lineage["v2_benchmark_failure"]["self_digest"] == \
        AUTH.V2_BENCHMARK_RECEIPT_DIGEST
    assert lineage["v2_terminal_failure"]["self_digest"] == \
        AUTH.V2_TERMINAL_FAILURE_RECEIPT_DIGEST

    for role, key, value in (
        ("v1_immutable_failure", "self_digest", "0" * 64),
        ("v1_immutable_failure", "raw_sha256", "0" * 64),
        ("v2_contract", "semantic_status", "INVALID"),
        ("v2_benchmark_failure", "candidate_outcomes_consumed", True),
        ("v2_terminal_failure", "preserved_not_retried_or_reinterpreted", False),
    ):
        changed = copy.deepcopy(lineage)
        changed[role][key] = value
        with pytest.raises(AUTH.GlobalExecutionAmendmentError):
            AUTH.validate_predecessor_lineage_bindings(changed)


def test_runtime_absence_registry_is_closed_and_detects_a_predating_output(
        tmp_path: Path) -> None:
    generated = tmp_path / AUTH.GENERATED_ROOT_RELATIVE_PATH
    scorer_fit = generated / "scorer_fit"
    scorer_fit.mkdir(parents=True)
    (tmp_path / AUTH.UTILITY_SCORER_ROOT_RELATIVE_PATH).mkdir(parents=True)
    rows = AUTH.audit_runtime_outputs_absent(root=tmp_path)
    assert rows == AUTH._expected_absence_rows()
    assert {label for label, _path, _kind in AUTH.NEW_RUNTIME_OUTPUT_PATHS} <= {
        row["label"] for row in rows
    }
    assert {
        label for label, _path, _kind in AUTH.NEW_RUNTIME_OUTPUT_PATHS
    } <= {row["label"] for row in rows}
    assert any(
        row["path"] == (
            ".generated/go2_utility_scorer_v1_2/"
            "scorer_contract_global_exact_successor_v1.json")
        for row in rows
    )

    forbidden = scorer_fit / "small_completion_global_exact_model_plan_v1.json"
    forbidden.write_text("{}\n")
    with pytest.raises(
            AUTH.GlobalExecutionAmendmentError,
            match="predates amendment: global_exact_model_plan"):
        AUTH.audit_runtime_outputs_absent(root=tmp_path)


def test_exclusive_json_is_read_only_fsynced_shape_and_never_overwrites(
        tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    payload = {"schema": "synthetic", "value": 7}
    AUTH._exclusive_json(path, payload, label="synthetic authority")

    raw = path.read_bytes()
    assert json.loads(raw) == payload
    assert raw.endswith(b"\n")
    assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    before = hashlib.sha256(raw).hexdigest()
    with pytest.raises(
            AUTH.GlobalExecutionAmendmentError, match="already exists"):
        AUTH._exclusive_json(
            path, {"schema": "synthetic", "value": 8},
            label="synthetic authority")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_report_binding_must_precede_and_share_amendment_commit() -> None:
    report = _report()
    binding = _report_binding(report)
    changed = dict(binding)
    changed["source_repository_commit"] = "f" * 40
    with pytest.raises(
            AUTH.GlobalExecutionAmendmentError,
            match="source commits differ"):
        AUTH.build_execution_amendment(
            source_repository_commit="e" * 40,
            source_bindings=_source_bindings(),
            coupling_report_binding=changed,
            scientific_contract_bindings=_scientific_bindings(),
            preoutcome_input_bindings=_input_bindings(),
            predecessor_lineage=_lineage_bindings(),
        )


def test_exact_intended_engine_and_runner_are_source_bound() -> None:
    assert AUTH.ENGINE_SOURCE_PATH == \
        "lewm/oracle/go2_small_completion_global_exact_model_v1.py"
    assert AUTH.RUNNER_SOURCE_PATH == \
        "scripts/run_go2_small_completion_global_exact_v1.py"
    assert AUTH.ENGINE_SOURCE_PATH in AUTH.EXPECTED_SOURCE_PATHS
    assert AUTH.RUNNER_SOURCE_PATH in AUTH.EXPECTED_SOURCE_PATHS
    assert {
        "scripts/encode_go2_branch_corpus_v1_2.py",
        "scripts/train_go2_utility_scorer_v1_2.py",
        "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
    } < set(AUTH.EXPECTED_SOURCE_PATHS)


def test_exact_hash_objective_and_boundary_fixture_are_frozen() -> None:
    objective = AUTH.STABLE_HASH_OBJECTIVE_CONTRACT
    assert objective["domain_separation_utf8"] == \
        "LEWM_GO2_SMALL_COMPLETION_GLOBAL_EXACT_PAIR_OBJECTIVE_V1"
    assert objective["pair_digest"] == "SHA-256(preimage_construction)"
    assert "canonical pair identity JSON bytes" in objective["variable_order"]
    assert "first_10_hex_digits" in objective["coefficient_construction"]
    assert objective["coefficient_range"] == [1, 1099511627776]
    assert "below 2^47" in objective["exact_binary64_sum_boundary"]
    assert objective["outcome_or_downstream_fields_consumed"] == []
    assert AUTH.STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST == \
        AUTH.canonical_digest(objective)

    fixtures = AUTH.FIXTURE_VALIDATION_CONTRACT
    fixture_id = "MULTIPLE_FEASIBLE_OLD_CANONICAL_MASK_FAIL_LATER_JOINT_VALID"
    assert fixture_id in fixtures["required_fixture_ids"]
    boundary = fixtures["mandatory_boundary_fixture"]
    assert boundary["fixture_id"] == fixture_id
    assert boundary["old_identity_ordered_canonical_vector_mask_passes"] is False
    assert boundary["later_hard_feasible_vector_mask_passes"] is True
    assert boundary["new_global_model_returns_mask_valid_solution"] is True
    assert boundary["every_scientific_constraint_still_validates"] is True


def test_source_correction_literal_historical_bindings_and_digest_convention(
        ) -> None:
    assert AUTH.ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT == \
        "1ebc1378e81b7704768c30d3b2b4b165180a93b9"
    assert AUTH.ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING == {
        "path": str(AUTH.COUPLING_REPORT_RELATIVE_PATH),
        "schema": AUTH.REPORT_SCHEMA,
        "self_digest_key": AUTH.REPORT_SELF_KEY,
        "self_digest": (
            "4433cc9e44a1caa44ec3dea73096b414b8db09a64a525a091ac48cf4eb290e76"
        ),
        "raw_sha256": (
            "0fe164fd20183f030d7cd5c410802d7e244a91c82b4919335e151158c3c30a83"
        ),
        "byte_count": 24_256,
        "source_repository_commit":
            "1ebc1378e81b7704768c30d3b2b4b165180a93b9",
    }
    assert AUTH.ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING["self_digest"] \
        == "52e00a327b944a72bcb954b48d7bf0503dfc2a71f3bc7c62c20298d495993b37"
    assert AUTH.ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING["raw_sha256"] \
        == "a4e97420f86515b5b4d1171bac903173fc542c680c50b61b8a8234d2b7fc97c4"
    assert AUTH.ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING["byte_count"] \
        == 32_128
    assert AUTH.HISTORICAL_MIXED_DISPOSITION_ARTIFACT_BINDING == {
        "path": (
            ".generated/go2_branch_corpus_v1_2/scorer_fit/"
            "preserved_state_mixed_precontract_disposition_reachability_v2.json"
        ),
        "self_digest_key":
            "mixed_precontract_disposition_receipt_digest",
        "self_digest": (
            "fef1a98980bc41d63434367f518ff2876dbcf93afbea52ff8f555300d3220604"
        ),
        "raw_sha256": (
            "faa71a30cc720b6ed19cf44e4b1c5d5d9f03fc15c7069f1a3ff0ff44ac953958"
        ),
        "byte_count": 29_403,
    }
    literal = {"b": 2, "a": 1}
    assert AUTH._legacy_default_json_digest(literal) == \
        "d8497d9d82770a70729261095aa98f7ef5154d7af499f8037b6ca250296785a6"
    assert AUTH._legacy_default_json_digest(literal) != \
        AUTH.canonical_digest(literal)


def test_source_correction_v2_is_closed_preserves_v1_and_binds_true_boundary(
        monkeypatch: pytest.MonkeyPatch) -> None:
    amendment, v1, mixed = _source_correction_amendment(monkeypatch)
    frozen_v1 = copy.deepcopy(v1)
    frozen_mixed = copy.deepcopy(mixed)

    assert AUTH.validate_execution_amendment_v2(
        amendment, validate_live_authorities=False) == amendment
    assert v1 == frozen_v1
    assert mixed == frozen_mixed
    assert amendment["schema"] == AUTH.AMENDMENT_V2_SCHEMA
    assert amendment["status"] == AUTH.AMENDMENT_V2_STATUS
    assert amendment["amendment_version"] == 2
    assert amendment["source_repository_commit"] == "f" * 40
    assert amendment["v1_execution_authority"] == v1
    assert amendment["historical_mixed_disposition_authority"] == mixed
    assert amendment["failed_attempt_disposition"] == \
        AUTH.FAILED_SOURCE_TRANSITION_DISPOSITION
    failure = amendment["failed_attempt_disposition"]
    assert failure["mandatory_synthetic_fixture_suite_completed"] is True
    assert failure["synthetic_fixture_solver_invoked"] is True
    assert failure["optional_completion_rotation_vectors_parsed"] == 17
    assert failure["scientific_masks_accessed"] is True
    assert failure["frozen_45_check_mask_evidence_read_and_validated"] is True
    assert failure["preserved_phase1_vector_mapping_returned"] is False
    assert failure["production_instance_built"] is False
    assert failure["runner_plan_written"] is False
    assert failure["scientific_production_solver_invoked"] is False
    assert failure["candidate_outcomes_consumed"] is False
    assert failure["downstream_started"] is False
    assert amendment["source_correction"][
        "observed_changed_source_paths"] == sorted(
            AUTH.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    assert amendment["source_correction"]["scientific_contract_changed"] \
        is False
    assert amendment[AUTH.AMENDMENT_SELF_KEY] == AUTH.canonical_digest({
        key: value for key, value in amendment.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })
    assert str(AUTH.EXECUTION_AMENDMENT_V2_RELATIVE_PATH) not in {
        row["path"] for row in AUTH._expected_absence_rows()
    }
    raw = AUTH._pretty_json_bytes(amendment)
    assert AUTH.execution_amendment_v2_artifact_binding(
        amendment, raw)["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(
            AUTH.GlobalExecutionAmendmentError,
            match="raw bytes changed"):
        AUTH.execution_amendment_v2_artifact_binding(
            amendment, json.dumps(amendment, sort_keys=True).encode())


@pytest.mark.parametrize("field", [
    "mandatory_synthetic_fixture_suite_completed",
    "synthetic_fixture_solver_invoked",
    "optional_completion_masks_accessed",
    "frozen_45_check_mask_evidence_read_and_validated",
    "scientific_masks_accessed",
    "preserved_phase1_vector_mapping_assembled",
    "preserved_phase1_vector_mapping_returned",
    "candidate_outcomes_consumed",
    "candidate_branch_outcomes_inspected",
    "branch_labels_read",
    "frames_or_latents_created",
    "scorer_or_predictor_accessed",
    "production_instance_built",
    "production_model_built",
    "model_execution_plan_built",
    "runner_plan_written",
    "scientific_production_solver_invoked",
    "terminal_receipt_written",
    "joint_receipt_written",
    "candidate_allocation_manifest_written",
    "phase2_revalidation_receipt_written",
    "state_manifest_written",
    "successor_scorer_contract_written",
    "downstream_started",
    "performance_benchmark_run",
    "v1_or_v2_benchmark_retried",
])
def test_source_correction_rejects_resigned_failed_boundary_mutation(
        field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    amendment, _v1, _mixed = _source_correction_amendment(monkeypatch)
    changed = copy.deepcopy(amendment)
    value = changed["failed_attempt_disposition"][field]
    changed["failed_attempt_disposition"][field] = (
        value + 1 if type(value) is int else not value)
    changed["failed_attempt_disposition_digest"] = AUTH.canonical_digest(
        changed["failed_attempt_disposition"])
    changed[AUTH.AMENDMENT_SELF_KEY] = AUTH.canonical_digest({
        key: item for key, item in changed.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })

    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_execution_amendment_v2(
            changed, validate_live_authorities=False)


@pytest.mark.parametrize("field", [
    "legacy_active_mixed_disposition_loader_changed",
    "scene_or_state_pool_changed",
    "candidate_bank_or_frequency_changed",
    "selector_or_allocation_constraint_changed",
    "model_objective_or_solver_setting_changed",
    "oracle_render_preprocess_or_target_encoder_changed",
    "scorer_architecture_or_qualification_changed",
    "scientific_contract_changed",
    "candidate_outcome_or_downstream_metric_used",
    "would_be_1ebc_operational_scorer_digest_preserved",
    "current_operational_scorer_digest_bound_only_after_valid_manifest",
])
def test_source_correction_rejects_resigned_scientific_disposition_mutation(
        field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    amendment, _v1, _mixed = _source_correction_amendment(monkeypatch)
    changed = copy.deepcopy(amendment)
    changed["source_correction"][field] = not changed["source_correction"][field]
    changed["source_correction_digest"] = AUTH.canonical_digest(
        changed["source_correction"])
    changed[AUTH.AMENDMENT_SELF_KEY] = AUTH.canonical_digest({
        key: item for key, item in changed.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })

    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_execution_amendment_v2(
            changed, validate_live_authorities=False)


def test_source_correction_rejects_artifact_or_source_scope_change(
        monkeypatch: pytest.MonkeyPatch) -> None:
    amendment, v1, mixed = _source_correction_amendment(monkeypatch)

    changed = copy.deepcopy(amendment)
    changed["v1_execution_authority"][
        "coupling_report_artifact_binding"]["raw_sha256"] = "0" * 64
    changed[AUTH.AMENDMENT_SELF_KEY] = AUTH.canonical_digest({
        key: item for key, item in changed.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })
    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_execution_amendment_v2(
            changed, validate_live_authorities=False)

    changed = copy.deepcopy(amendment)
    changed["historical_mixed_disposition_authority"]["payload"][
        "synthetic_test_only"] = False
    changed[AUTH.AMENDMENT_SELF_KEY] = AUTH.canonical_digest({
        key: item for key, item in changed.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })
    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_execution_amendment_v2(
            changed, validate_live_authorities=False)

    successor_sources = copy.deepcopy(amendment["source_bindings"])
    outside = next(
        row for row in successor_sources
        if row["path"] not in AUTH.SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    outside["sha256"] = "0" * 64
    with pytest.raises(
            AUTH.GlobalExecutionAmendmentError,
            match="unauthorised source path"):
        AUTH.build_execution_amendment_v2(
            source_repository_commit="f" * 40,
            source_bindings=successor_sources,
            v1_execution_authority=v1,
            historical_mixed_disposition_authority=mixed)


def test_source_correction_issue_is_exclusive_and_reopens_exactly(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    amendment, v1, mixed = _source_correction_amendment(monkeypatch)
    scorer_fit = tmp_path / AUTH.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    (tmp_path / AUTH.UTILITY_SCORER_ROOT_RELATIVE_PATH).mkdir(parents=True)
    monkeypatch.setattr(
        AUTH, "_clean_source_commit", lambda *, root: "f" * 40)
    monkeypatch.setattr(
        AUTH, "_read_source_bindings",
        lambda *, root: copy.deepcopy(amendment["source_bindings"]))
    monkeypatch.setattr(
        AUTH, "load_historical_v1_execution_authority",
        lambda *, root: copy.deepcopy(v1))
    monkeypatch.setattr(
        AUTH, "load_predecessor_lineage",
        lambda *, root: copy.deepcopy(
            v1["execution_amendment"]["immutable_predecessor_lineage"]))
    monkeypatch.setattr(
        AUTH, "audit_runtime_outputs_absent",
        lambda *, root: copy.deepcopy(
            amendment["runtime_outputs_absent_at_issue"]))
    path = tmp_path / AUTH.EXECUTION_AMENDMENT_V2_RELATIVE_PATH

    issued = AUTH.issue_execution_amendment_v2(
        path, historical_mixed_disposition_authority=mixed,
        source_repository_commit="f" * 40, root=tmp_path)
    raw = path.read_bytes()
    assert issued == amendment
    assert json.loads(raw) == amendment
    assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    assert AUTH.execution_amendment_v2_artifact_binding(issued, raw)[
        "execution_amendment_digest"] == amendment[AUTH.AMENDMENT_SELF_KEY]

    reopened = AUTH.issue_execution_amendment_v2(
        path, historical_mixed_disposition_authority=mixed,
        source_repository_commit="f" * 40, root=tmp_path)
    assert reopened == issued
    assert path.read_bytes() == raw


def test_preplan_correction_literal_v2_binding_is_frozen() -> None:
    assert AUTH.IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING == {
        "path": str(AUTH.EXECUTION_AMENDMENT_V2_RELATIVE_PATH),
        "schema": AUTH.AMENDMENT_V2_SCHEMA,
        "self_digest_key": AUTH.AMENDMENT_SELF_KEY,
        "self_digest": (
            "36454a1626345da92468038e50e130db103a4196d924f24dca9e2a9e8d38dcd3"
        ),
        "raw_sha256": (
            "da176fa54456e3827a444c7e583487d54e549e2afb488ad891393a0cbe56658e"
        ),
        "byte_count": 131_997,
        "source_repository_commit":
            "5e92a43814d6eb81fc5cfe9adb6d9c380b1c3e72",
    }
    assert AUTH.PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH == (
        AUTH.SCORER_FIT_RELATIVE_PATH /
        "small_completion_global_exact_preplan_integration_correction_v1.json")


def test_preplan_correction_is_closed_preserves_v2_and_exact_failures(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction, v2_authority, _current = _preplan_integration_correction(
        monkeypatch)
    v2 = v2_authority["payload"]
    frozen_v2 = copy.deepcopy(v2_authority)
    assert AUTH.validate_preplan_integration_correction(
        correction, validate_live_authorities=False) == correction
    assert v2_authority == frozen_v2
    assert correction["schema"] == \
        AUTH.PREPLAN_INTEGRATION_CORRECTION_SCHEMA
    assert correction["status"] == \
        AUTH.PREPLAN_INTEGRATION_CORRECTION_STATUS
    assert correction["amendment_version"] == 2
    assert correction["preplan_integration_correction_version"] == 1
    assert correction["immutable_v2_execution_authority"] == v2_authority
    for key in AUTH._PREPLAN_PRESERVED_V2_FIELDS:
        assert correction[key] == v2[key]
    assert correction["v2_post_install_reopen_failure"] == \
        AUTH.V2_POST_INSTALL_REOPEN_FAILURE
    assert correction["v2_post_install_reopen_failure"][
        "v2_artifact_remains_valid"] is True
    assert correction["v2_post_install_reopen_failure"][
        "runtime_outputs_absent_during_subsequent_validation"] is True
    assert correction["post_v2_preplan_failed_attempt_disposition"] == \
        AUTH.POST_V2_PREPLAN_FAILED_ATTEMPT_DISPOSITION
    failure = correction["post_v2_preplan_failed_attempt_disposition"]
    assert failure["pre_mask_v2_context_validated"] is True
    assert failure["mask_context_completed_and_returned"] is True
    assert failure["optional_completion_rotation_vectors_parsed"] == 17
    assert failure["optional_completion_masks_accessed"] is True
    assert failure["preserved_phase1_vector_mapping_returned"] is True
    assert failure["scientific_masks_accessed"] is True
    assert failure["builder_fixed_and_optional_rows_assembled"] is True
    assert failure["production_instance_construction_entered"] is True
    assert failure["production_instance_built"] is False
    assert failure["production_instance_returned"] is False
    boundary = correction["issuance_boundary"]
    assert boundary["historical_scientific_masks_accessed"] is True
    assert boundary["scientific_masks_accessed_during_this_issuance"] is False
    assert boundary["new_attempt_mask_context_started"] is False
    assert boundary["candidate_outcomes_consumed"] is False
    source_correction = correction["preplan_integration_correction"]
    assert source_correction["observed_changed_source_paths"] == sorted(
        AUTH.PREPLAN_INTEGRATION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    assert source_correction["builder_optional_candidate_projection_changed"] \
        is False
    assert source_correction["scientific_contract_changed"] is False
    assert correction[AUTH.AMENDMENT_SELF_KEY] == AUTH.canonical_digest({
        key: value for key, value in correction.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })
    absence_paths = {
        row["path"] for row in AUTH._expected_absence_rows()
    }
    assert str(AUTH.PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH) not in (
        absence_paths)
    for _label, path, _kind in AUTH.NEW_RUNTIME_OUTPUT_PATHS:
        assert str(path) in absence_paths


@pytest.mark.parametrize("field", [
    "v2_post_install_reopen_failure",
    "post_v2_preplan_failed_attempt_disposition",
    "preplan_integration_correction",
    "immutable_v2_execution_authority",
])
def test_preplan_correction_rejects_resigned_nested_mutation(
        field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    correction, _v2, _current = _preplan_integration_correction(monkeypatch)
    changed = copy.deepcopy(correction)
    if field == "immutable_v2_execution_authority":
        changed[field]["binding"]["raw_sha256"] = "0" * 64
    else:
        changed[field]["synthetic_tamper"] = True
        digest_key = {
            "v2_post_install_reopen_failure":
                "v2_post_install_reopen_failure_digest",
            "post_v2_preplan_failed_attempt_disposition":
                "post_v2_preplan_failed_attempt_disposition_digest",
            "preplan_integration_correction":
                "preplan_integration_correction_digest",
        }[field]
        changed[digest_key] = AUTH.canonical_digest(changed[field])
    changed[AUTH.AMENDMENT_SELF_KEY] = AUTH.canonical_digest({
        key: value for key, value in changed.items()
        if key != AUTH.AMENDMENT_SELF_KEY
    })
    with pytest.raises(AUTH.GlobalExecutionAmendmentError):
        AUTH.validate_preplan_integration_correction(
            changed, validate_live_authorities=False)


def test_preplan_correction_rejects_source_change_outside_exact_four_paths(
        monkeypatch: pytest.MonkeyPatch) -> None:
    correction, v2_authority, current = _preplan_integration_correction(
        monkeypatch)
    sources = copy.deepcopy(current["sources"])
    outside = next(
        row for row in sources
        if row["path"] not in
        AUTH.PREPLAN_INTEGRATION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS)
    outside["sha256"] = "0" * 64
    with pytest.raises(
            AUTH.GlobalExecutionAmendmentError,
            match="unauthorised source path"):
        AUTH.build_preplan_integration_correction(
            source_repository_commit=correction["source_repository_commit"],
            source_bindings=sources,
            immutable_v2_execution_authority=v2_authority)


def test_preplan_correction_issue_is_exclusive_and_active_loader_reopens(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    correction, v2_authority, current = _preplan_integration_correction(
        monkeypatch)
    scorer_fit = tmp_path / AUTH.SCORER_FIT_RELATIVE_PATH
    scorer_fit.mkdir(parents=True)
    (tmp_path / AUTH.UTILITY_SCORER_ROOT_RELATIVE_PATH).mkdir(parents=True)
    monkeypatch.setattr(
        AUTH, "_clean_source_commit",
        lambda *, root: correction["source_repository_commit"])
    monkeypatch.setattr(
        AUTH, "_read_source_bindings",
        lambda *, root: copy.deepcopy(current["sources"]))
    monkeypatch.setattr(
        AUTH, "load_immutable_v2_execution_authority",
        lambda *, root: copy.deepcopy(v2_authority))
    monkeypatch.setattr(
        AUTH, "load_predecessor_lineage",
        lambda *, root: copy.deepcopy(
            correction["immutable_predecessor_lineage"]))
    monkeypatch.setattr(
        AUTH, "audit_runtime_outputs_absent",
        lambda *, root: copy.deepcopy(
            correction["runtime_outputs_absent_at_issue"]))
    path = tmp_path / AUTH.PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH

    issued = AUTH.issue_preplan_integration_correction(
        path, source_repository_commit=correction[
            "source_repository_commit"], root=tmp_path)
    raw = path.read_bytes()
    assert issued == correction
    assert json.loads(raw) == correction
    assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    active = AUTH.load_active_execution_authority(root=tmp_path)
    assert active["execution_amendment"] == correction
    assert active["immutable_v2_execution_authority"] == v2_authority
    assert active["source_transition_digest"] == correction[
        AUTH.AMENDMENT_SELF_KEY]

    reopened = AUTH.issue_preplan_integration_correction(
        path, source_repository_commit=correction[
            "source_repository_commit"], root=tmp_path)
    assert reopened == issued
    assert path.read_bytes() == raw
