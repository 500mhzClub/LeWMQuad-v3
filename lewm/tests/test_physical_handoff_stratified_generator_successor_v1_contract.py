from __future__ import annotations

import copy
import hashlib

import pytest

from lewm.safety import (
    physical_handoff_stratified_generator_successor_v1_contract as C,
)


def test_exact_successor_identity_seed_block_and_allocation() -> None:
    assert C.EXPERIMENT_ID == "PHYSICAL_HANDOFF_STRATIFIED_GENERATOR_SUCCESSOR_V1"
    assert C.IDENTITY_NAMESPACE == "phsgs-v1"
    assert C.PROCEDURAL_SEED_BASE == 8221747320681816064
    assert C.MAX_CANDIDATE_COUNT == 4096
    assert C.derive_procedural_seed(C.FAMILY_IDS[0], 0, 0) == C.PROCEDURAL_SEED_BASE
    assert C.derive_procedural_seed(C.FAMILY_IDS[-1], 15, 63) == 8221747320681820159
    seeds = sorted(
        C.derive_procedural_seed(family, stratum, attempt)
        for family in C.FAMILY_IDS
        for stratum in range(C.STRATA_PER_FAMILY)
        for attempt in range(C.MAX_ATTEMPTS_PER_STREAM)
    )
    assert len(seeds) == len(set(seeds)) == 4096
    assert hashlib.sha256(C.canonical_json_bytes(seeds)[:-1]).hexdigest() == (
        "d6e9bfad1d89f2e4cc1baf91dc7a6636e818137751631c1ba20e770af304a9a1"
    )
    assert C.candidate_index(C.FAMILY_IDS[0], 0, 1) == C.STREAM_COUNT
    identity_fields = {
        "stream_id",
        "candidate_spec_id",
        "scene_id",
        "state_id",
        "episode_id",
        "graph_id",
    }
    assert all(
        row[field].startswith("phsgs-v1-")
        for row in C.build_candidate_identity_manifest()
        for field in identity_fields
    )


def test_exact_output_branches_questions_claims_and_receipt_absence() -> None:
    contract = C.validate_contract(C.build_contract())
    assert contract["scientific_questions"] == list(C.SCIENTIFIC_QUESTIONS)
    assert contract["claims_boundary"] == C.CLAIMS_BOUNDARY
    assert contract["output"]["success_inventory"] == list(C.OUTPUT_LEAVES)
    assert contract["output"]["generator_terminal_inventory"] == list(
        C.OUTPUT_LEAVES[:5] + C.OUTPUT_LEAVES[-4:]
    )
    assert len(C.SUCCESS_OUTPUT_LEAVES) == 19
    assert len(C.GENERATOR_TERMINAL_OUTPUT_LEAVES) == 9
    assert contract["output"]["external_regeneration_receipt"] is None
    prohibited = contract["output"]["prohibited_external_publication_authority"]
    assert prohibited["paths"] == list(C.PROHIBITED_EXTERNAL_PUBLICATION_PATHS)
    assert len(prohibited["paths"]) == 3
    assert prohibited["required_absent_before_publication"] is True
    assert prohibited["required_absent_after_publication"] is True


def test_v4_context_records_exact_offset_index_correction_and_interpretation() -> None:
    context = C.validate_v4_context(C.build_v4_context())
    serialized = C.canonical_json_bytes(context)
    for fragment in (
        b'"offset_opening_shortfall_strata":[0,1,2,4,5,6,10]',
        b'"requested_offset_shortfall_stratum_index":8',
        b'"corrected_offset_shortfall_stratum_index":5',
        b'"requested_offset_stratum_index_8_canonical_evidence":{"adequate":true',
        b'"corrected_offset_stratum_index_5_canonical_evidence":{"adequate":false',
    ):
        assert fragment in serialized
    assert context["v4_interpretation"] == list(C.V4_INTERPRETATION_SENTENCES)
    assert context["v4_terminal_identity_count"] == 256
    assert context["v4_teacher_execution_count"] == 248
    assert context["v4_hard_technical_stop_count"] == 0
    assert context["v4_models_trained"] == 0
    resolution = C.V4_SHORTFALL_RESOLUTION_AUTHORITY
    assert resolution["canonical_stream_count"] == 12
    assert resolution["canonical_stream_order"] == [
        {"family": family, "stratum_index": stratum}
        for family, stratum in C.V4_CANONICAL_SHORTFALL_STREAMS
    ]
    offset = {
        row["stratum_index"]
        for row in resolution["canonical_stream_order"]
        if row["family"] == "OFFSET_OPENING"
    }
    assert offset == {0, 1, 2, 4, 5, 6, 10}
    assert 8 not in offset


def test_source_observation_is_freeze_only_and_result_commit_is_not_identity() -> None:
    freeze = "a" * 40
    value = C.build_source_freeze_observation(
        source_freeze_commit=freeze,
        source_freeze_tree_oid="b" * 40,
        source_closure_content_digest="c" * 64,
        source_closure_file_sha256="d" * 64,
        observed_head_commit_at_scientific_reduction=freeze,
    )
    assert C.validate_source_freeze_observation(
        value, source_freeze_commit=freeze
    ) == value
    with pytest.raises(C.PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError):
        C.build_source_freeze_observation(
            source_freeze_commit=freeze,
            source_freeze_tree_oid="b" * 40,
            source_closure_content_digest="c" * 64,
            source_closure_file_sha256="d" * 64,
            observed_head_commit_at_scientific_reduction="e" * 40,
        )
    assert C.EXPECTED_COMMIT_AUTHORITY[
        "result_commit_participates_in_result_json_content_identity"
    ] is False
    assert C.EXPECTED_COMMIT_AUTHORITY[
        "result_markdown_participates_in_result_json_content_identity"
    ] is False


def test_material_contract_rejects_schema_or_extra_field() -> None:
    authority = C.PREDECESSOR_IDENTITY_PROJECTION_AUTHORITY
    nonoverlap = {
        "seed_nonoverlap": {
            "generated_seed_count": C.MAX_CANDIDATE_COUNT,
            "generated_seed_unique_count": C.MAX_CANDIDATE_COUNT,
            "registries": [
                {
                    "registry": name,
                    "count": value["count"],
                    "canonical_sorted_unique_no_lf_sha256": value[
                        "canonical_sorted_unique_no_lf_sha256"
                    ],
                    "successor_overlap_count": 0,
                }
                for name, value in sorted(
                    C.REQUIRED_PREDECESSOR_SEED_REGISTRY_AUTHORITY.items()
                )
            ],
            "all_overlap_counts_zero": True,
        },
        "broad_predecessor_identity_projection": copy.deepcopy(
            authority["broad_predecessor_identity_projection"]
        ),
        "v4_registered_and_fixture_identity_count": authority[
            "v4_registered_and_fixture_identity_count"
        ],
        "v4_registered_and_fixture_identity_projection_sha256": authority[
            "v4_registered_and_fixture_identity_projection_sha256"
        ],
        "successor_identity_count": authority["successor_identity_count"],
        "successor_identity_projection_sha256": authority[
            "successor_identity_projection_sha256"
        ],
        "successor_namespace": C.IDENTITY_NAMESPACE,
        "successor_namespace_required_prefix": f"{C.IDENTITY_NAMESPACE}-",
        "broad_predecessor_identity_overlap_count": 0,
        "broad_predecessor_scene_hash_overlap_count": 0,
        "v4_registered_and_fixture_identity_overlap_count": 0,
        "v4_fixture_seed_overlap_count": 0,
        "structured_semantic_overlap_is_not_an_identity_gate": True,
        "all_identity_and_seed_overlap_counts_zero": True,
    }
    value = C.build_material_contract("a" * 40, "b" * 64, nonoverlap)
    assert C.validate_material_contract(value) == value
    tampered = copy.deepcopy(value)
    tampered.pop("content_digest")
    tampered["unexpected"] = 0
    tampered = C.attach_content_digest(tampered)
    with pytest.raises(
        C.PhysicalHandoffStratifiedGeneratorSuccessorV1ContractError,
        match="field drift",
    ):
        C.validate_material_contract(tampered)


def test_role_salt_and_evidence_only_packaging_ramifications_are_explicit() -> None:
    allocation = C.GENERATOR_ALLOCATION_AUTHORITY
    assert allocation["role_assignment_hash_experiment_salt"] == (
        C.V4.V3.V2.V1.EXPERIMENT_ID
    )
    assert allocation["within_stream_execution_order"].startswith("strict increasing")
    assert allocation["inter_stream_scheduling"] == (
        "unconstrained and scientifically irrelevant"
    )
    persistence = C.IMPLEMENTATION_PERSISTENCE_RAMIFICATIONS_AUTHORITY
    assert persistence["formula_or_threshold_change"] is False
    assert persistence["scientific_interpretation_change"] is False
    assert persistence["encoding_runtime_bound_ancillary_evidence"][
        "checkpoint_inference_is_runtime_bound_not_reexecuted_by_reducer"
    ] is True


def test_physical_teacher_is_only_the_user_facing_fourth_comparator_alias() -> None:
    authority = C.validate_content_digest(C.HELDOUT_COMPARATOR_ALIAS_AUTHORITY)
    assert authority["internal_condition_order"] == list(C.HELDOUT_CONDITION_IDS)
    assert authority["internal_condition_order"][3] == "TEACHER_TRACE"
    assert authority["user_facing_comparator_order"][3] == "PHYSICAL_TEACHER"
    assert authority["user_facing_to_internal_condition_id"][
        "PHYSICAL_TEACHER"
    ] == "TEACHER_TRACE"
    assert authority["physical_teacher_comparator_ordinal"] == 4
    assert authority["internal_condition_ids_renamed"] is False
    assert authority["new_teacher_execution"] is False
    assert authority["model_policy_formula_threshold_or_gate_change"] is False
    assert C.FROZEN_DOWNSTREAM_AUTHORITY[
        "heldout_comparator_alias_authority"
    ] == authority


def test_qualification_candidate_lifecycle_is_explicit_and_science_invariant() -> None:
    authority = C.validate_content_digest(
        C.QUALIFICATION_CANDIDATE_LIFECYCLE_AUTHORITY
    )
    assert authority["content_digest"] == (
        "dd01d16f90709fd25034ab6bba0f2815498b2642e37a2c4dae7de6ef9e8e24a7"
    )
    assert authority["candidate_boundary_sequence"] == [
        "_qualify constructs a complete raw packet or raises",
        (
            "destroy every owned qualification Scene exactly once in reverse "
            "creation order"
        ),
        "clear all owned qualification session references",
        "call lewm_genesis.scene_builder.shutdown_genesis() exactly once",
        "run Python garbage collection",
        "return the raw packet or re-raise the primary exception",
    ]
    assert authority["owned_scene_release"]["destroy_exactly_once"] is True
    assert authority["owned_scene_release"]["destroy_order"] == (
        "reverse creation order"
    )
    reset = authority["process_global_reset"]
    assert reset["public_helper"] == "lewm_genesis.scene_builder.shutdown_genesis"
    assert reset["helper_invocations_per_qualification_candidate_call"] == 1
    assert reset["genesis_destroy_call"] == "gs.destroy()"
    assert reset["applies_after_final_or_initial_tipped_candidate"] is True
    assert reset["applies_on_python_exception_path"] is True
    assert reset["precedes_outer_candidate_validation_and_persistence"] is True
    assert reset["conditional_on_a_next_attempt"] is False
    reinitialization = authority["next_candidate_reinitialization"]
    assert reinitialization["entrypoint"] == (
        "lewm_genesis.scene_builder.initialize_genesis"
    )
    assert reinitialization["genesis_seed_mapping"] == (
        "int(procedural_seed) & 0x7FFFFFFF"
    )
    failure = authority["failure_policy"]
    assert failure[
        "cleanup_only_failure_hard_stops_before_return_or_persistence"
    ] is True
    assert failure[
        "process_global_shutdown_failure_hard_stops_before_return_or_persistence"
    ] is True
    assert failure["primary_base_exception_is_preserved"] is True
    prohibited = authority["prohibited_mechanisms"]
    assert all(value is False for value in prohibited.values())
    scientific_effect = authority["scientific_effect"]
    assert scientific_effect["lifecycle_only"] is True
    assert all(
        value is False
        for key, value in scientific_effect.items()
        if key != "lifecycle_only"
    )

    contract = C.validate_contract(C.build_contract())
    assert contract["execution_and_fault_authority"][
        "qualification_candidate_lifecycle"
    ] == authority
    assert contract["runtime_policy"][
        "qualification_candidate_lifecycle"
    ] == authority
    helper_path = "lewm_genesis/lewm_genesis/scene_builder.py"
    assert len(C.TRACKED_SOURCE_PATHS) == 16
    assert C.TRACKED_SOURCE_PATHS.count(helper_path) == 1
    assert len(C.SOURCE_CLOSURE_PATHS) == 97
    assert C.SOURCE_CLOSURE_PATHS.count(helper_path) == 1


def test_public_contract_surface_is_complete() -> None:
    assert "QUALIFICATION_CANDIDATE_LIFECYCLE_AUTHORITY" in C.__all__
    for name in (
        "build_material_contract",
        "validate_material_contract",
        "build_source_freeze_observation",
        "validate_source_freeze_observation",
        "expected_material_inventory_counts",
    ):
        assert name in C.__all__
        assert callable(getattr(C, name))
