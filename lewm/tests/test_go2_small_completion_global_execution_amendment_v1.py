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
