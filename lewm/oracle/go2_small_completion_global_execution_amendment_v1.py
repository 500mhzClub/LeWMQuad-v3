"""Prospective authority for the outcome-free small-completion global model.

This module records the bounded source inspection which established that
completion-scene selection and candidate rotation assignment are coupled.  It
then issues the sole prospective execution amendment authorising one global
exact feasibility model.  The external 6,188-combination enumeration and the
inner identity-ordered lexicographic rotation tie-break are superseded as
non-scientific execution machinery; every scientific selector, allocation
margin, allowed subset, oracle, corpus and scorer contract remains frozen.

Importing this module performs no file access, solver work, mask access, or
outcome access.  Report construction is pure.  The issue functions are the
only writers and install their dedicated JSON artifacts with ``O_EXCL``, file
and directory ``fsync``, and read-only mode.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from fnmatch import fnmatchcase
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

REPORT_SCHEMA = "go2_small_completion_scene_rotation_coupling_report_v1"
REPORT_STATUS = "PASS_SOURCE_ONLY_COUPLING_CLASSIFICATION"
REPORT_SELF_KEY = "coupling_report_digest"
AMENDMENT_SCHEMA = "go2_small_completion_global_exact_execution_amendment_v1"
AMENDMENT_STATUS = "ISSUED_PROSPECTIVE_ONE_GLOBAL_EXACT_MODEL_AUTHORITY"
AMENDMENT_SELF_KEY = "execution_amendment_digest"
AMENDMENT_V2_SCHEMA = (
    "go2_small_completion_global_exact_execution_amendment_v2")
AMENDMENT_V2_STATUS = (
    "ISSUED_PROSPECTIVE_SOURCE_CORRECTED_ONE_GLOBAL_EXACT_MODEL_AUTHORITY")
PREPLAN_INTEGRATION_CORRECTION_SCHEMA = (
    "go2_small_completion_global_exact_preplan_integration_correction_v1")
PREPLAN_INTEGRATION_CORRECTION_STATUS = (
    "ISSUED_PROSPECTIVE_CANONICAL_BOUNDARY_VALIDATION_CORRECTION")

GENERATED_ROOT_RELATIVE_PATH = Path(".generated/go2_branch_corpus_v1_2")
UTILITY_SCORER_ROOT_RELATIVE_PATH = Path(".generated/go2_utility_scorer_v1_2")
MANAGED_GENERATED_ROOTS = (
    GENERATED_ROOT_RELATIVE_PATH,
    UTILITY_SCORER_ROOT_RELATIVE_PATH,
)
SCORER_FIT_RELATIVE_PATH = GENERATED_ROOT_RELATIVE_PATH / "scorer_fit"
COUPLING_REPORT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "small_completion_scene_rotation_coupling_report_v1.json"
)
EXECUTION_AMENDMENT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "small_completion_global_exact_execution_amendment_v1.json"
)
EXECUTION_AMENDMENT_V2_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "small_completion_global_exact_execution_amendment_v2.json"
)
PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "small_completion_global_exact_preplan_integration_correction_v1.json"
)

ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT = (
    "1ebc1378e81b7704768c30d3b2b4b165180a93b9")
ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING = {
    "path": str(COUPLING_REPORT_RELATIVE_PATH),
    "schema": REPORT_SCHEMA,
    "self_digest_key": REPORT_SELF_KEY,
    "self_digest": (
        "4433cc9e44a1caa44ec3dea73096b414b8db09a64a525a091ac48cf4eb290e76"
    ),
    "raw_sha256": (
        "0fe164fd20183f030d7cd5c410802d7e244a91c82b4919335e151158c3c30a83"
    ),
    "byte_count": 24_256,
    "source_repository_commit": ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
}
ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING = {
    "path": str(EXECUTION_AMENDMENT_RELATIVE_PATH),
    "schema": AMENDMENT_SCHEMA,
    "self_digest_key": AMENDMENT_SELF_KEY,
    "self_digest": (
        "52e00a327b944a72bcb954b48d7bf0503dfc2a71f3bc7c62c20298d495993b37"
    ),
    "raw_sha256": (
        "a4e97420f86515b5b4d1171bac903173fc542c680c50b61b8a8234d2b7fc97c4"
    ),
    "byte_count": 32_128,
    "source_repository_commit": ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
}
HISTORICAL_MIXED_DISPOSITION_ARTIFACT_BINDING = {
    "path": (
        ".generated/go2_branch_corpus_v1_2/scorer_fit/"
        "preserved_state_mixed_precontract_disposition_reachability_v2.json"
    ),
    "self_digest_key": "mixed_precontract_disposition_receipt_digest",
    "self_digest": (
        "fef1a98980bc41d63434367f518ff2876dbcf93afbea52ff8f555300d3220604"
    ),
    "raw_sha256": (
        "faa71a30cc720b6ed19cf44e4b1c5d5d9f03fc15c7069f1a3ff0ff44ac953958"
    ),
    "byte_count": 29_403,
}
SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_small_completion_global_execution_amendment_v1.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_small_completion_global_exact_v1.py",
)
IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT = (
    "5e92a43814d6eb81fc5cfe9adb6d9c380b1c3e72")
IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING = {
    "path": str(EXECUTION_AMENDMENT_V2_RELATIVE_PATH),
    "schema": AMENDMENT_V2_SCHEMA,
    "self_digest_key": AMENDMENT_SELF_KEY,
    "self_digest": (
        "36454a1626345da92468038e50e130db103a4196d924f24dca9e2a9e8d38dcd3"
    ),
    "raw_sha256": (
        "da176fa54456e3827a444c7e583487d54e549e2afb488ad891393a0cbe56658e"
    ),
    "byte_count": 131_997,
    "source_repository_commit": IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT,
}
PREPLAN_INTEGRATION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_small_completion_global_execution_amendment_v1.py",
    "lewm/oracle/go2_small_completion_global_exact_model_v1.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_small_completion_global_exact_v1.py",
)
V2_POST_INSTALL_REOPEN_FAILURE = {
    "status": (
        "IMMUTABLE_VALID_V2_ARTIFACT_COMMAND_RETURN_FAILURE_"
        "POST_INSTALL_REOPEN_PATH_DEFECT"),
    "disposition": (
        "V2_DURABLY_INSTALLED_AND_REMAINS_VALID_COMMAND_FAILED_ONLY_DURING_"
        "POST_INSTALL_CANONICAL_REOPEN"),
    "argv": [
        "python3", "scripts/run_go2_small_completion_global_exact_v1.py",
        "--stage", "issue-source-correction",
    ],
    "exit_code": 1,
    "exception_type": (
        "lewm.oracle.go2_small_completion_global_execution_amendment_v1."
        "GlobalExecutionAmendmentError"),
    "exception_message": (
        "source-corrected execution amendment logical path changed"),
    "v2_artifact_durably_installed": True,
    "v2_artifact_remains_valid": True,
    "canonical_loader_subsequently_exact_validated": True,
    "runtime_outputs_absent_during_subsequent_validation": True,
    "post_install_validation_defect_only": True,
    "v2_scientific_or_execution_authority_invalidated": False,
    "scientific_masks_accessed_during_command": False,
    "candidate_outcomes_consumed": False,
    "production_instance_or_model_built": False,
    "runner_or_model_plan_written": False,
    "scientific_production_solver_invoked": False,
}
POST_V2_PREPLAN_FAILED_ATTEMPT_DISPOSITION = {
    "status": "IMMUTABLE_FAILED_POST_V2_PREPLAN_CANONICAL_BOUNDARY_VALIDATION",
    "disposition": (
        "AFTER_17_OPTIONAL_VECTORS_AND_7_PRESERVED_VECTORS_BEFORE_FIRST_"
        "PRODUCTION_INSTANCE_RETURN_BEFORE_MODEL_PLAN_OR_SCIENTIFIC_SOLVE"),
    "argv": [
        "python3", "scripts/run_go2_small_completion_global_exact_v1.py",
        "--stage", "solve-and-continue",
    ],
    "exit_code": 1,
    "exception_type": (
        "lewm.oracle.go2_small_completion_global_exact_model_v1."
        "GlobalExactModelError"),
    "exception_message": "raw optional candidate identity changed",
    "mandatory_synthetic_fixture_suite_completed": True,
    "synthetic_fixture_solver_invoked": True,
    "pre_mask_v2_context_validated": True,
    "mask_context_completed_and_returned": True,
    "optional_completion_rotation_vectors_parsed": 17,
    "optional_completion_masks_accessed": True,
    "frozen_45_check_mask_evidence_read_and_validated": True,
    "preserved_phase1_vector_mapping_assembled": True,
    "preserved_phase1_vector_mapping_returned": True,
    "scientific_masks_accessed": True,
    "builder_fixed_and_optional_rows_assembled": True,
    "production_instance_construction_entered": True,
    "failed_at_first_candidate_structural_scene_projection": True,
    "candidate_outcomes_consumed": False,
    "candidate_branch_outcomes_inspected": False,
    "branch_labels_read": False,
    "frames_or_latents_created": False,
    "scorer_or_predictor_accessed": False,
    "production_instance_built": False,
    "production_instance_returned": False,
    "production_model_built": False,
    "model_execution_plan_built": False,
    "runner_plan_written": False,
    "scientific_production_solver_invoked": False,
    "terminal_receipt_written": False,
    "joint_receipt_written": False,
    "candidate_allocation_manifest_written": False,
    "phase2_revalidation_receipt_written": False,
    "state_manifest_written": False,
    "successor_scorer_contract_written": False,
    "downstream_started": False,
    "performance_benchmark_run": False,
    "v1_or_v2_benchmark_retried": False,
}
FAILED_SOURCE_TRANSITION_DISPOSITION = {
    "status": "IMMUTABLE_FAILED_PRE_PLAN_SOURCE_VALIDATION",
    "disposition": (
        "AFTER_17_OPTIONAL_MASKS_AND_FROZEN_45_CHECK_EVIDENCE_BEFORE_7_VECTOR_"
        "MAPPING_RETURN_BEFORE_PRODUCTION_PLAN_OR_SOLVE"
    ),
    "argv": [
        "python3", "scripts/run_go2_small_completion_global_exact_v1.py",
        "--stage", "solve-and-continue",
    ],
    "exit_code": 1,
    "exception_type": (
        "lewm.oracle.go2_scorer_state_selector_amendment_v2."
        "StateSelectorAmendmentError"
    ),
    "exception_message": "mixed precontract disposition source binding mismatch",
    "mandatory_synthetic_fixture_suite_completed": True,
    "synthetic_fixture_solver_invoked": True,
    "optional_completion_rotation_vectors_parsed": 17,
    "optional_completion_masks_accessed": True,
    "frozen_45_check_mask_evidence_read_and_validated": True,
    "scientific_masks_accessed": True,
    "preserved_phase1_vector_mapping_assembled": False,
    "preserved_phase1_vector_mapping_returned": False,
    "candidate_outcomes_consumed": False,
    "candidate_branch_outcomes_inspected": False,
    "branch_labels_read": False,
    "frames_or_latents_created": False,
    "scorer_or_predictor_accessed": False,
    "production_instance_built": False,
    "production_model_built": False,
    "model_execution_plan_built": False,
    "runner_plan_written": False,
    "scientific_production_solver_invoked": False,
    "terminal_receipt_written": False,
    "joint_receipt_written": False,
    "candidate_allocation_manifest_written": False,
    "phase2_revalidation_receipt_written": False,
    "state_manifest_written": False,
    "successor_scorer_contract_written": False,
    "downstream_started": False,
    "performance_benchmark_run": False,
    "v1_or_v2_benchmark_retried": False,
}

ENGINE_SOURCE_PATH = (
    "lewm/oracle/go2_small_completion_global_exact_model_v1.py"
)
RUNNER_SOURCE_PATH = "scripts/run_go2_small_completion_global_exact_v1.py"
GENESIS_DOWNSTREAM_INTERPRETER_RELATIVE_PATH = (
    ".generated/venvs/genesis_render_vulkan/bin/python"
)
GENESIS_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH = (
    ".generated/venvs/genesis_render_vulkan/pyvenv.cfg"
)
GENESIS_DOWNSTREAM_PYVENV_CONFIG_SHA256 = (
    "41c6a8f52f3404bd3b7fcd805519c1976e2f4194ef9aa8eccf2f0919383386a9"
)
GENESIS_DOWNSTREAM_PYVENV_CONFIG_BYTE_COUNT = 219
ROCM_DOWNSTREAM_INTERPRETER_RELATIVE_PATH = (
    ".generated/venvs/world_model_rocm_7_2_1_v1/bin/python"
)
ROCM_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH = (
    ".generated/venvs/world_model_rocm_7_2_1_v1/pyvenv.cfg"
)
ROCM_DOWNSTREAM_PYVENV_CONFIG_SHA256 = (
    "49222cc65a628e83d00d99da60f1dea8d59bc01a3ea9616227f330e2ecd50577"
)
ROCM_DOWNSTREAM_PYVENV_CONFIG_BYTE_COUNT = 223
DOWNSTREAM_PYTHON_VERSION = "3.12.3"
GENESIS_DOWNSTREAM_RUNTIME_CONTRACT = {
    "role": "genesis_branch_generation",
    "interpreter_relative_path":
        GENESIS_DOWNSTREAM_INTERPRETER_RELATIVE_PATH,
    "pyvenv_config_relative_path":
        GENESIS_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH,
    "pyvenv_config_sha256": GENESIS_DOWNSTREAM_PYVENV_CONFIG_SHA256,
    "pyvenv_config_byte_count":
        GENESIS_DOWNSTREAM_PYVENV_CONFIG_BYTE_COUNT,
    "python_version": DOWNSTREAM_PYTHON_VERSION,
    "genesis_version": "0.3.14",
    "torch_version": "2.12.0+cu130",
    "torch_cuda_runtime": "13.0",
    "torch_hip_runtime": None,
    "accelerator_available": False,
    "accelerator_device_count": 0,
    "accelerator_devices": [],
}
ROCM_DOWNSTREAM_RUNTIME_CONTRACT = {
    "role": "rocm_encoding_training_and_development",
    "interpreter_relative_path": ROCM_DOWNSTREAM_INTERPRETER_RELATIVE_PATH,
    "pyvenv_config_relative_path": ROCM_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH,
    "pyvenv_config_sha256": ROCM_DOWNSTREAM_PYVENV_CONFIG_SHA256,
    "pyvenv_config_byte_count": ROCM_DOWNSTREAM_PYVENV_CONFIG_BYTE_COUNT,
    "python_version": DOWNSTREAM_PYTHON_VERSION,
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
}
DOWNSTREAM_RUNTIME_CONTRACTS = {
    "genesis": GENESIS_DOWNSTREAM_RUNTIME_CONTRACT,
    "rocm": ROCM_DOWNSTREAM_RUNTIME_CONTRACT,
}
DOWNSTREAM_STAGE_RUNTIME_ROLES = {
    "six_branch_smoke": "genesis",
    "smoke_encoding": "rocm",
    "full_720_branch_corpus": "genesis",
    "full_latent_encoding": "rocm",
    "scorer_training_and_qualification": "rocm",
    "development_transfer": "rocm",
    "qualification_validation": "rocm",
    "development_validation": "rocm",
}
SOURCE_SPECS = (
    ("lewm/oracle/go2_small_completion_global_execution_amendment_v1.py",
     "coupling_report_and_execution_authority"),
    (ENGINE_SOURCE_PATH, "one_global_exact_model"),
    ("lewm/oracle/go2_candidate_allocation_v1_2.py",
     "frozen_candidate_allocation_constraints"),
    ("lewm/oracle/go2_scorer_state_selector_amendment_v2.py",
     "frozen_completion_rotation_eligibility"),
    ("lewm/oracle/go2_parallel_small_completion_search_v1.py",
     "superseded_external_enumeration_lineage_and_manifest_materialisation"),
    ("scripts/build_go2_branch_corpus_v1_2.py",
     "frozen_identity_and_downstream_corpus_integration"),
    ("scripts/encode_go2_branch_corpus_v1_2.py",
     "frozen_encoding_science_with_successor_lineage_admission"),
    ("scripts/train_go2_utility_scorer_v1_2.py",
     "frozen_training_science_with_successor_lineage_admission"),
    ("scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
     "frozen_development_transfer_science_with_successor_lineage_admission"),
    (RUNNER_SOURCE_PATH, "prospective_execution_runner"),
)
EXPECTED_SOURCE_PATHS = tuple(path for path, _role in SOURCE_SPECS)

V1_FAILURE_STATUS = "IMMUTABLE_FAIL_COLD_START_INCLUDED_IN_FIRST_TIMED_WAVE"
V1_FAILURE_RECEIPT_DIGEST = (
    "afb4c190cf7d2e93b678a546fc233340102c6f5260110b1471752bc54a0e88d6"
)
V1_FAILURE_RAW_SHA256 = (
    "cc3b07b3ed470058dc395d0eb34d5d6cd83e8edc0140e4c18f249d4d4747fe5b"
)
V1_FAILURE_BYTE_COUNT = 2_688

V2_SOURCE_REPOSITORY_COMMIT = "112579b680a83df35b72100e5ecc528b5b34e18f"
V2_CONTRACT_DIGEST = (
    "05e91f432b82ad6d30ab6d8a1d4431b4e9c4ccd2470439b17535486d394667f2"
)
V2_BENCHMARK_RECEIPT_DIGEST = (
    "77f9e8d44aab6ca3d7b60c2c46a94cb4afd1be4dc5bc47a67521d70220cf69b9"
)
V2_TERMINAL_FAILURE_RECEIPT_DIGEST = (
    "ae8f23d1127e24a52208d0c48f1635fc8f28b157ed0790ac3f2695a15672d67e"
)
V2_CONTRACT_RAW_SHA256 = (
    "891bda434e7913f488506d5084a5c3634c156c71f12d2a42cdb0c0a9d23d6271"
)
V2_CONTRACT_BYTE_COUNT = 16_131
V2_BENCHMARK_RAW_SHA256 = (
    "09a03fa6d07af8c5436949be5ea5d533de570260b62e559ab19259ccb4ddbe68"
)
V2_BENCHMARK_BYTE_COUNT = 10_020
V2_TERMINAL_FAILURE_RAW_SHA256 = (
    "31718fc436a1f5bdea23f9fc756fc285a2a22a97a497fe36699cc50798bc9582"
)
V2_TERMINAL_FAILURE_BYTE_COUNT = 72_037
V2_BACKEND_DISPOSITION = (
    "NOT_AUTHORISED_FOR_SCIENTIFIC_SEARCH_AFTER_V2_PERFORMANCE_FAILURE"
)
SUPERSEDED_EXTERNAL_ENUMERATION_STATUS = (
    "SUPERSEDED_PRE_OUTCOME_UNNECESSARY_LEXICOGRAPHIC_EXHAUSTION"
)
SUPERSEDED_CANONICAL_TIE_BREAK_STATUS = (
    "SUPERSEDED_PRE_OUTCOME_NONSCIENTIFIC_TIE_BREAK"
)

CANDIDATE_ALLOCATION_AMENDMENT_DIGEST = (
    "4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc"
)
CANDIDATE_ALLOCATION_CONTRACT_DIGEST = (
    "bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e"
)
SCORER_FIT_ALLOCATION_DESIGN_DIGEST = (
    "a587b1de264dfb54176aa231e5183ae4b7b4229bbf65c02d62438f86af5e7116"
)

SCIENTIFIC_CONTRACT_BINDING_KEYS = (
    "selection_digest",
    "scorer_fit_allocation_design_digest",
    "candidate_allocator_contract_digest",
    "candidate_allocation_amendment_digest",
    "pre_identity_allocation_validation_digest",
    "invalid_scorer_identity_exclusion_digest",
    "state_selector_amendment_digest",
    "state_selector_feasibility_receipt_digest",
    "candidate_bank_digest",
    "clean_source_launch_receipt_digest",
    "source_repository_commit",
    "clean_source_binding_digest",
    "bound_implementations_digest",
    "scorer_contract_artifact_digest",
    "mixed_precontract_disposition_receipt_digest",
    "progress_contract_digest",
    "safety_contract_digest",
    "oracle_v1_2_digest",
    "scorer_contract_v1_2_digest",
    "boundary_digest",
    "render_contract_digest",
    "preprocess_contract_digest",
    "textured_v03_renderer_contract_digest",
    "preprocessing_digest",
    "target_encoder_digest",
    "target_encoder_checkpoint_sha256",
    "genesis_backend",
)

PREOUTCOME_INPUT_BINDING_KEYS = frozenset({
    "predecessor_scientific_input_bindings_digest",
    "candidate_pool_scene_ids_digest",
    "fixed_state_projection_digest",
    "candidate_pool_count",
    "fixed_state_count",
    "selected_completion_scene_count",
    "final_state_count",
    "candidate_outcomes_consumed",
    "scientific_masks_accessed",
})

NEW_RUNTIME_OUTPUT_PATHS = (
    ("global_exact_model_plan", SCORER_FIT_RELATIVE_PATH /
     "small_completion_global_exact_model_plan_v1.json", "file"),
    ("global_exact_terminal_result", SCORER_FIT_RELATIVE_PATH /
     "small_completion_global_exact_terminal_result_v1.json", "file"),
    ("global_exact_terminal_infeasibility", SCORER_FIT_RELATIVE_PATH /
     "small_completion_global_exact_terminal_infeasibility_v1.json", "file"),
    ("global_exact_joint_receipt", SCORER_FIT_RELATIVE_PATH /
     "small_completion_global_exact_joint_receipt_v1.json", "file"),
    ("candidate_allocation_manifest", SCORER_FIT_RELATIVE_PATH /
     "candidate_allocation_manifest.json", "file"),
    ("small_terminal_state_shard", SCORER_FIT_RELATIVE_PATH /
     "state_shard_small_enclosed_maze.json", "file"),
    ("preserved_state_revalidation", SCORER_FIT_RELATIVE_PATH /
     "preserved_state_revalidation_reachability_v2.json", "file"),
    ("complete_state_manifest", SCORER_FIT_RELATIVE_PATH /
     "state_manifest.json", "file"),
    ("branch_rows", SCORER_FIT_RELATIVE_PATH / "branch_rows.jsonl", "file"),
    ("corpus_receipt", SCORER_FIT_RELATIVE_PATH / "corpus_receipt.json", "file"),
    ("latents_index", SCORER_FIT_RELATIVE_PATH / "latents_index.json", "file"),
    ("row_records", SCORER_FIT_RELATIVE_PATH / "row_records", "directory"),
    ("frames", SCORER_FIT_RELATIVE_PATH / "frames", "directory"),
    ("successor_scorer_contract", UTILITY_SCORER_ROOT_RELATIVE_PATH /
     "scorer_contract_global_exact_successor_v1.json", "file"),
)

# Exact frozen outcome/scorer/predictor surface from the active selector.  It
# is rechecked independently at this successor's issuance rather than merely
# trusting the earlier phase-1 attestation.
PHASE1_FORBIDDEN_EXACT_FILE_PATHS = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/state_manifest.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "candidate_allocation_manifest.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/branch_rows.jsonl",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/corpus_receipt.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/smoke_branch_receipt.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/smoke_encoding_receipt.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/latents_index.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "encoding_invocation_summary.json",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/context.f16",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/horizon.f16",
    ".generated/go2_utility_scorer_v1_2/qualification.json",
    ".generated/go2_utility_scorer_v1_2/scorer_package.pt",
    ".generated/go2_utility_scorer_v1_2/scorer_package_receipt.json",
    ".generated/go2_utility_scorer_v1_2/"
    "counterfactual_development_transfer_v1_2/development_transfer_spec.json",
    ".generated/go2_utility_scorer_v1_2/"
    "counterfactual_development_transfer_v1_2/result.json",
)
PHASE1_FORBIDDEN_DIRECTORY_ROOTS = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/row_records",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/frames",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/latents/context",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/latents/horizon",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "invalid_attempts/redrive_records",
    ".generated/go2_branch_corpus_v1_2/scorer_fit/invalid_attempts/latents",
    ".generated/go2_utility_scorer_v1_2/registered_initialisations",
    ".generated/go2_utility_scorer_v1_2/training",
    ".generated/go2_utility_scorer_v1_2/invalid_attempts",
    ".generated/go2_utility_scorer_v1_2/"
    "counterfactual_development_transfer_v1_2",
)
PHASE1_FORBIDDEN_GLOB_PATTERNS = (
    ".generated/go2_branch_corpus_v1_2/scorer_fit/branch_summary_*.json",
    ".generated/go2_utility_scorer_v1_2/no_latent_baseline_*.pt",
    ".generated/go2_utility_scorer_v1_2/"
    "no_latent_baseline_*.receipt.json",
    ".generated/go2_utility_scorer_v1_2/failed_scorer_*.pt",
)

_LINEAGE_SPECS = (
    {
        "role": "v1_immutable_failure",
        "path": str(SCORER_FIT_RELATIVE_PATH /
                    "small_completion_parallel_prefix_benchmark_v1.json"),
        "schema": "go2_parallel_small_completion_search_v1_benchmark_receipt",
        "self_digest_key": "benchmark_receipt_digest",
        "self_digest": V1_FAILURE_RECEIPT_DIGEST,
        "semantic_status": V1_FAILURE_STATUS,
    },
    {
        "role": "v2_contract",
        "path": str(SCORER_FIT_RELATIVE_PATH /
                    "small_completion_parallel_prefix_benchmark_v2_contract.json"),
        "schema": "go2_parallel_small_completion_benchmark_v2_contract",
        "self_digest_key": "benchmark_v2_contract_digest",
        "self_digest": V2_CONTRACT_DIGEST,
        "semantic_status": "IMMUTABLE_ISSUED_V2_CONTRACT",
    },
    {
        "role": "v2_benchmark_failure",
        "path": str(SCORER_FIT_RELATIVE_PATH /
                    "small_completion_parallel_prefix_benchmark_v2.json"),
        "schema": "go2_parallel_small_completion_search_v2_benchmark_receipt",
        "self_digest_key": "benchmark_receipt_digest",
        "self_digest": V2_BENCHMARK_RECEIPT_DIGEST,
        "semantic_status": "IMMUTABLE_FAIL_MEDIAN_AND_MAXIMUM_GATE",
    },
    {
        "role": "v2_terminal_failure",
        "path": str(SCORER_FIT_RELATIVE_PATH /
                    "small_completion_parallel_terminal_failure_v2.json"),
        "schema": "go2_parallel_small_completion_benchmark_v2_terminal_failure",
        "self_digest_key": "terminal_failure_receipt_digest",
        "self_digest": V2_TERMINAL_FAILURE_RECEIPT_DIGEST,
        "semantic_status": "IMMUTABLE_ONE_SHOT_V2_FAILURE",
    },
)
_EXPECTED_LINEAGE_RAW = {
    "v1_immutable_failure": (V1_FAILURE_RAW_SHA256, V1_FAILURE_BYTE_COUNT),
    "v2_contract": (V2_CONTRACT_RAW_SHA256, V2_CONTRACT_BYTE_COUNT),
    "v2_benchmark_failure": (
        V2_BENCHMARK_RAW_SHA256, V2_BENCHMARK_BYTE_COUNT),
    "v2_terminal_failure": (
        V2_TERMINAL_FAILURE_RAW_SHA256, V2_TERMINAL_FAILURE_BYTE_COUNT),
}


def _constraint(
        constraint_id: str, origin: str, symbols: Sequence[str], *,
        scene: bool, rotation: bool, joint: bool, sensitive: bool,
        exact_rule: str, derived_from: Sequence[str] = (),
        scientific_status: str = "PRESERVED_MANDATORY",
        ) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id,
        "origin": origin,
        "source_symbols": list(symbols),
        "scene_identity_reference": scene,
        "rotation_subset_identity_reference": rotation,
        "joint_reference": joint,
        "selection_sensitive": sensitive,
        "exact_rule": exact_rule,
        "derived_from": list(derived_from),
        "scientific_status": scientific_status,
    }


# Closed inventory: validators require exact byte-equivalent JSON structure and
# reject missing, added, reordered or reclassified constraints.
CONSTRAINT_INVENTORY = (
    _constraint(
        "SCENE_POOL_CURSOR_EXCLUSION_UNIQUENESS", "DOMAIN",
        ("_cursor_restricted_completion_rows",
         "_small_completion_candidates_from_feasibility",
         "_parallel_small_search_inputs"), scene=True, rotation=False,
        joint=False, sensitive=True,
        exact_rule=(
            "eligible completion scenes are strictly after the frozen cursor, "
            "exclude every fixed scene, and have unique sorted scene identities"
        )),
    _constraint(
        "SELECT_FIVE_DISTINCT_SCENES", "STRUCTURAL",
        ("build_search_plan", "unrank_combination"), scene=True,
        rotation=False, joint=False, sensitive=True,
        exact_rule="exactly five distinct scenes are selected without replacement"),
    _constraint(
        "SELECTED_ORDINAL_DEFINES_IDENTITY_AND_SPLIT_ROLE", "STRUCTURAL",
        ("_parallel_selected_completion_states",), scene=True,
        rotation=False, joint=False, sensitive=True,
        exact_rule=(
            "selected scenes are structurally ordered; ordinal zero is calibration "
            "and ordinals one through four are fit, with bound state identities"
        )),
    _constraint(
        "FINAL_IDENTITY_SHAPE_120_8X3_4FIT_1CAL", "STRUCTURAL",
        ("_normalise_identity_states", "_parallel_rank_identity_material"),
        scene=True, rotation=False, joint=False, sensitive=True,
        exact_rule=(
            "the manifest has 120 unique identities in eight families and three "
            "strata, with four fit and one calibration identity per cell"
        )),
    _constraint(
        "ROTATION_BLOCK_CATALOGUE_12_OF_6", "DOMAIN",
        ("candidate_block", "ROTATION_BLOCKS"), scene=False, rotation=True,
        joint=False, sensitive=False,
        exact_rule=(
            "the only subsets are twelve six-candidate rotations of offsets "
            "[0,1,3,5,8,10] modulo twelve"
        )),
    _constraint(
        "ROTATION_BLOCK_FORWARD_TURNING_REVERSE_SEMANTICS", "DOMAIN",
        ("_subset_catalogue", "validate_allocation_manifest"), scene=False,
        rotation=True, joint=False, sensitive=False,
        exact_rule=(
            "every assigned subset has forward and turning behaviour; reverse is "
            "not mandatory per subset and remains globally exact-60"
        )),
    _constraint(
        "EXACTLY_ONE_ROTATION_PER_STATE", "MILP_HARD_MARGIN",
        ("_constraint_system",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule="sum_r y[state,r] equals one for every active identity"),
    _constraint(
        "FIT_FAMILY_STRATUM_CANDIDATE_EXACT_2", "MILP_HARD_MARGIN",
        ("_constraint_system",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "for every family, stratum and candidate, incidence across the four "
            "fit identities equals two"
        )),
    _constraint(
        "CALIBRATION_STRATUM_CANDIDATE_EXACT_4", "MILP_HARD_MARGIN",
        ("_constraint_system",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "for every stratum and candidate, incidence across eight calibration "
            "identities equals four"
        )),
    _constraint(
        "CALIBRATION_FAMILY_CANDIDATE_1_TO_2", "MILP_HARD_MARGIN",
        ("_constraint_system",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "for every family and candidate, incidence across three calibration "
            "identities is between one and two inclusive"
        )),
    _constraint(
        "GOAL_TYPE_CANDIDATE_FLOOR_CEILING", "MILP_HARD_MARGIN",
        ("_constraint_system", "_small_completion_candidates_from_feasibility"),
        scene=True, rotation=True, joint=True, sensitive=True,
        exact_rule=(
            "within every snapshot-bound landmark material, each candidate count "
            "is floor(n_goal/2) or ceil(n_goal/2)"
        )),
    _constraint(
        "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY", "POST_ALLOCATION_GATE",
        ("completion_enriched_eligibility",
         "completion_rotation_eligibility_vector",
         "_all_completion_masks_pass"), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "the identity-owned evidence row indexed by the assigned rotation has "
            "the exact assigned subset and eligible=true"
        )),
    _constraint(
        "ALL_40_COMPLETION_MASKS_PASS", "POST_ALLOCATION_GATE",
        ("_all_completion_masks_pass",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule="all forty completion identities pass their exact allocated mask",
        derived_from=("COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",)),
    _constraint(
        "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER", "CANONICALIZATION",
        ("_normalise_identity_states", "_lexicographic_rotations",
         "validate_allocation_manifest"), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "the allocator orders identities by digest then state ID and freezes a "
            "deterministic rotation vector"
        ), scientific_status=SUPERSEDED_CANONICAL_TIE_BREAK_STATUS),
    _constraint(
        "FIRST_PASSING_SCENE_COMBINATION", "EXECUTION_POLICY",
        ("OrderedFrontier", "run_scientific_parallel_search"), scene=True,
        rotation=True, joint=True, sensitive=True,
        exact_rule=(
            "the predecessor implementation enumerates all five-scene combinations "
            "lexicographically and commits the first passing rank"
        ), scientific_status=SUPERSEDED_EXTERNAL_ENUMERATION_STATUS),
    _constraint(
        "GLOBAL_CANDIDATE_EXACT_60", "DERIVED_VALIDATION",
        ("_validate_counts",), scene=False, rotation=True, joint=False,
        sensitive=False,
        exact_rule="each candidate appears in exactly sixty state subsets",
        derived_from=("FIT_FAMILY_STRATUM_CANDIDATE_EXACT_2",
                      "CALIBRATION_STRATUM_CANDIDATE_EXACT_4")),
    _constraint(
        "FIT_CALIBRATION_CANDIDATE_EXACT_48_12", "DERIVED_VALIDATION",
        ("_validate_counts",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "each candidate appears forty-eight times in fit and twelve times in "
            "calibration"
        )),
    _constraint(
        "STRATUM_CANDIDATE_EXACT_20", "DERIVED_VALIDATION",
        ("_validate_counts",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule="each candidate appears exactly twenty times per stratum"),
    _constraint(
        "FAMILY_CANDIDATE_BALANCE_7_8", "DERIVED_VALIDATION",
        ("_validate_counts",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "within every family, six candidates appear seven times and six "
            "appear eight times"
        )),
    _constraint(
        "FAMILY_STRATUM_CANDIDATE_BALANCE_2_3", "DERIVED_VALIDATION",
        ("_validate_counts",), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "within every family-stratum cell, six candidates appear twice and "
            "six appear three times"
        )),
    _constraint(
        "REVERSE_CANDIDATE_IN_60_DISTINCT_STATES", "DERIVED_VALIDATION",
        ("_post_identity_pre_outcome_validation",
         "validate_allocation_manifest"), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule="candidate ten occurs in exactly sixty distinct state subsets"),
    _constraint(
        "ASSIGNMENT_BLOCK_CONTINGENCY_AND_PAIRWISE_INTEGRITY",
        "DERIVED_VALIDATION",
        ("_contingency_tables", "_pairwise_cooccurrence",
         "validate_allocation_manifest"), scene=True, rotation=True, joint=True,
        sensitive=True,
        exact_rule=(
            "assignment blocks, contingency tables, subset usage and pairwise "
            "co-occurrence are exact deterministic projections of the manifest"
        )),
)

CONSTRAINT_IDS = tuple(row["constraint_id"] for row in CONSTRAINT_INVENTORY)
DECISIVE_COUPLING_CONSTRAINT_IDS = (
    "GOAL_TYPE_CANDIDATE_FLOOR_CEILING",
    "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
    "ALL_40_COMPLETION_MASKS_PASS",
)

PRESERVED_MANDATORY_CONSTRAINT_IDS = tuple(
    row["constraint_id"] for row in CONSTRAINT_INVENTORY
    if row["scientific_status"] == "PRESERVED_MANDATORY"
)

STABLE_HASH_OBJECTIVE_CONTRACT = {
    "schema": "go2_small_completion_global_exact_pair_objective_v1",
    "purpose": "deterministic_tie_break_only_after_all_mandatory_constraints",
    "direction": "minimize",
    "domain_separation_utf8": (
        "LEWM_GO2_SMALL_COMPLETION_GLOBAL_EXACT_PAIR_OBJECTIVE_V1"
    ),
    "preimage_construction": (
        "domain_separation_utf8 || 0x00 || canonical_pair_identity_json"
    ),
    "canonical_json": {
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": True,
        "allow_nan": False,
    },
    "selectable_pair_identity_fields": [
        "kind=selectable_completion", "structural_scene_identity_digest",
        "assigned_split_role", "candidate_rotation_index",
        "candidate_indices",
    ],
    "fixed_pair_identity_fields": [
        "kind=fixed_state", "state_identity_digest", "split_role",
        "candidate_rotation_index", "candidate_indices",
    ],
    "pair_digest": "SHA-256(preimage_construction)",
    "variable_order": (
        "ascending full 32-byte pair_digest, then ascending canonical pair "
        "identity JSON bytes as the explicit SHA-256 collision tie-break"
    ),
    "coefficient_construction": (
        "1 + unsigned_big_endian_integer(first_10_hex_digits_of_pair_digest)"
    ),
    "coefficient_range": [1, 1099511627776],
    "exact_binary64_sum_boundary": (
        "exactly 120 selected pair variables imply objective sum below 2^47, "
        "so every integer coefficient and complete sum is exact in binary64"
    ),
    "linear_objective": "sum(pair_coefficient * binary_pair_variable)",
    "equal_objective_solution_tie_break": (
        "frozen variable order, constraint order, one solver thread, solver seed "
        "and branching policy; repeated fixture solves must return identical bytes"
    ),
    "outcome_or_downstream_fields_consumed": [],
}
STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST = hashlib.sha256(json.dumps(
    STABLE_HASH_OBJECTIVE_CONTRACT, sort_keys=True, separators=(",", ":"),
    ensure_ascii=True, allow_nan=False,
).encode("utf-8")).hexdigest()

FIXTURE_VALIDATION_CONTRACT = {
    "status": "MANDATORY_BEFORE_SCIENTIFIC_SOLVE",
    "control_method": "TRACTABLE_EXHAUSTIVE_TRUSTED_CONTROL_ENUMERATION",
    "required_fixture_ids": [
        "KNOWN_FEASIBLE",
        "KNOWN_INFEASIBLE",
        "MULTIPLE_FEASIBLE_OLD_CANONICAL_MASK_FAIL_LATER_JOINT_VALID",
        "FIT_CALIBRATION_CONSTRAINTS",
        "RESIDUAL_CANDIDATE_FREQUENCY_CONSTRAINTS",
    ],
    "requirements": {
        "new_and_exhaustive_control_agree_on_feasibility": True,
        "returned_solutions_satisfy_every_frozen_constraint": True,
        "infeasible_fixtures_proved_infeasible": True,
        "repeated_runs_same_solution_and_digest": True,
        "same_lexicographically_earliest_solution_required": False,
        "candidate_outcomes_consumed": False,
    },
    "mandatory_boundary_fixture": {
        "fixture_id": (
            "MULTIPLE_FEASIBLE_OLD_CANONICAL_MASK_FAIL_LATER_JOINT_VALID"
        ),
        "at_least_two_hard_margin_feasible_rotation_vectors": True,
        "old_identity_ordered_canonical_vector_mask_passes": False,
        "later_hard_feasible_vector_mask_passes": True,
        "old_and_new_methods_agree_underlying_hard_margin_feasibility": True,
        "new_global_model_returns_mask_valid_solution": True,
        "new_solution_may_differ_from_old_canonical_vector": True,
        "every_scientific_constraint_still_validates": True,
    },
}

_HEX = frozenset("0123456789abcdef")


class GlobalExecutionAmendmentError(RuntimeError):
    """The coupling report or prospective execution authority is invalid."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GlobalExecutionAmendmentError(
            "value is not canonical JSON") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _legacy_default_json_digest(value: Any) -> str:
    """Reproduce the frozen selector receipt's historical JSON convention."""

    try:
        encoded = json.dumps(
            value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GlobalExecutionAmendmentError(
            "legacy value is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str) and len(value) == length
        and all(character in _HEX for character in value)
    )


def _without(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _pin_relative(root: Path, relative: str | Path, *, label: str) -> Path:
    repository = Path(root).resolve()
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise GlobalExecutionAmendmentError(
            f"{label} path is not repository-relative")
    if any(part == "sealed" or part == "sealed_test.json"
           or part.startswith("sealed_") for part in candidate_relative.parts):
        raise GlobalExecutionAmendmentError(f"{label} path enters sealed custody")
    candidate = repository / candidate_relative
    cursor = repository
    for part in candidate_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GlobalExecutionAmendmentError(f"{label} path is symlinked")
    return candidate


def _pin_generated(
        root: Path, relative: str | Path, *, label: str,
        allow_managed_root: bool = False) -> Path:
    """Resolve one exact managed generated-root alias, never sealed paths."""

    repository = Path(root).resolve()
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise GlobalExecutionAmendmentError(
            f"{label} path is not repository-relative")
    matches: list[tuple[Path, Path]] = []
    for logical in MANAGED_GENERATED_ROOTS:
        try:
            matches.append((logical, candidate_relative.relative_to(logical)))
        except ValueError:
            continue
    if len(matches) != 1:
        raise GlobalExecutionAmendmentError(
            f"{label} escaped the managed generated roots")
    managed_root, suffix = matches[0]
    if ((not suffix.parts and not allow_managed_root) or any(
            part == "sealed" or part == "sealed_test.json"
            or part.startswith("sealed_") for part in suffix.parts)):
        raise GlobalExecutionAmendmentError(f"{label} path is inaccessible")
    logical_root = repository / managed_root
    if logical_root.is_symlink():
        target = logical_root.readlink()
        target = target if target.is_absolute() else logical_root.parent / target
        if (target.name != logical_root.name
                or any(part == ".." or part == "sealed"
                       or part == "sealed_test.json" or part.startswith("sealed_")
                       for part in target.parts)):
            raise GlobalExecutionAmendmentError(
                "managed generated-root alias identity changed")
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise GlobalExecutionAmendmentError(
                "managed generated-root alias target is unavailable") from exc
    elif logical_root.exists():
        if not logical_root.is_dir():
            raise GlobalExecutionAmendmentError(
                "managed generated root is unavailable")
        canonical_root = logical_root.resolve(strict=True)
    else:
        # A wholly absent managed root is valid outcome-absence evidence.  Pin
        # its lexical location only after proving that no existing ancestor is
        # an alias.
        cursor = repository
        for part in managed_root.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise GlobalExecutionAmendmentError(
                    "absent managed generated-root ancestor is symlinked")
        canonical_root = logical_root
    if ((canonical_root.exists() and not canonical_root.is_dir())
            or canonical_root.name != logical_root.name):
        raise GlobalExecutionAmendmentError(
            "managed generated root identity changed")
    cursor = canonical_root
    for part in suffix.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise GlobalExecutionAmendmentError(f"{label} path is symlinked")
    return canonical_root.joinpath(*suffix.parts)


def _require_logical_path(
        path: Path, *, expected_relative: Path, root: Path, label: str) -> Path:
    repository = Path(root).resolve()
    supplied = Path(path)
    supplied_absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    if supplied_absolute.absolute() != (repository / expected_relative).absolute():
        raise GlobalExecutionAmendmentError(f"{label} logical path changed")
    return _pin_generated(root, expected_relative, label=label)


def _validate_source_bindings(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(SOURCE_SPECS):
        raise GlobalExecutionAmendmentError("source-binding coverage changed")
    validated: list[dict[str, Any]] = []
    for (expected_path, expected_role), raw in zip(
            SOURCE_SPECS, rows, strict=True):
        if not isinstance(raw, Mapping):
            raise GlobalExecutionAmendmentError("source binding is malformed")
        row = dict(raw)
        if (set(row) != {"path", "role", "byte_count", "sha256"}
                or row.get("path") != expected_path
                or row.get("role") != expected_role
                or isinstance(row.get("byte_count"), bool)
                or not isinstance(row.get("byte_count"), int)
                or row["byte_count"] <= 0
                or not _is_hex(row.get("sha256"), 64)):
            raise GlobalExecutionAmendmentError("source binding changed")
        validated.append(row)
    return validated


def _read_source_bindings(*, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative, role in SOURCE_SPECS:
        path = _pin_relative(root, relative, label="global-model source")
        if not path.is_file() or path.is_symlink():
            raise GlobalExecutionAmendmentError(
                f"expected source is unavailable: {relative}")
        raw = path.read_bytes()
        rows.append({
            "path": relative,
            "role": role,
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return rows


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GlobalExecutionAmendmentError(
            f"git source-custody check failed: {' '.join(args)}") from exc


def _clean_source_commit(*, root: Path) -> str:
    commit = _git("rev-parse", "HEAD", root=root).decode().strip()
    if not _is_hex(commit, 40):
        raise GlobalExecutionAmendmentError("source commit is malformed")
    if _git("status", "--porcelain=v1", "--untracked-files=all", root=root):
        raise GlobalExecutionAmendmentError(
            "report/amendment issuance requires a clean source commit")
    tracked = _git(
        "ls-files", "--error-unmatch", "--", *EXPECTED_SOURCE_PATHS,
        root=root,
    ).decode().splitlines()
    if len(tracked) != len(EXPECTED_SOURCE_PATHS) or set(tracked) != set(
            EXPECTED_SOURCE_PATHS):
        raise GlobalExecutionAmendmentError(
            "global-model source closure is not fully tracked")
    for relative in EXPECTED_SOURCE_PATHS:
        live = _pin_relative(root, relative, label="global-model source").read_bytes()
        if _git("show", f"HEAD:{relative}", root=root) != live:
            raise GlobalExecutionAmendmentError(
                f"live source differs from clean commit: {relative}")
    return commit


def validate_scientific_contract_bindings(
        payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GlobalExecutionAmendmentError(
            "scientific contract bindings must be a mapping")
    bindings = dict(payload)
    if set(bindings) != set(SCIENTIFIC_CONTRACT_BINDING_KEYS):
        raise GlobalExecutionAmendmentError(
            "scientific contract binding surface changed")
    for key in SCIENTIFIC_CONTRACT_BINDING_KEYS:
        value = bindings[key]
        if key == "genesis_backend":
            if not isinstance(value, str) or not value:
                raise GlobalExecutionAmendmentError(
                    "genesis backend binding is malformed")
        elif key == "source_repository_commit":
            if not _is_hex(value, 40):
                raise GlobalExecutionAmendmentError(
                    "predecessor source commit binding is malformed")
        elif not _is_hex(value, 64):
            raise GlobalExecutionAmendmentError(
                f"scientific digest is malformed: {key}")
    if (bindings["candidate_allocation_amendment_digest"]
            != CANDIDATE_ALLOCATION_AMENDMENT_DIGEST
            or bindings["candidate_allocator_contract_digest"]
            != CANDIDATE_ALLOCATION_CONTRACT_DIGEST
            or bindings["scorer_fit_allocation_design_digest"]
            != SCORER_FIT_ALLOCATION_DESIGN_DIGEST):
        raise GlobalExecutionAmendmentError(
            "frozen candidate-allocation binding changed")
    return bindings


def validate_preoutcome_input_bindings(
        payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GlobalExecutionAmendmentError(
            "pre-outcome input bindings must be a mapping")
    bindings = dict(payload)
    if set(bindings) != PREOUTCOME_INPUT_BINDING_KEYS:
        raise GlobalExecutionAmendmentError(
            "pre-outcome input binding surface changed")
    for key in (
            "predecessor_scientific_input_bindings_digest",
            "candidate_pool_scene_ids_digest", "fixed_state_projection_digest"):
        if not _is_hex(bindings.get(key), 64):
            raise GlobalExecutionAmendmentError(
                f"pre-outcome input digest is malformed: {key}")
    if (bindings.get("candidate_pool_count") != 17
            or bindings.get("fixed_state_count") != 115
            or bindings.get("selected_completion_scene_count") != 5
            or bindings.get("final_state_count") != 120
            or bindings.get("candidate_outcomes_consumed") is not False
            or bindings.get("scientific_masks_accessed") is not False):
        raise GlobalExecutionAmendmentError(
            "pre-outcome scientific input shape changed")
    return bindings


def _expected_absence_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(label: str, path: str | Path, kind: str) -> None:
        relative = str(path)
        if relative in seen:
            return
        seen.add(relative)
        rows.append({
            "label": label,
            "path": relative,
            "expected_kind": kind,
            "exists": False,
            "symlink": False,
            "artifact_absent": True,
        })

    for label, path, kind in NEW_RUNTIME_OUTPUT_PATHS:
        add(label, path, kind)
    for index, path in enumerate(PHASE1_FORBIDDEN_EXACT_FILE_PATHS):
        add(f"phase1_forbidden_exact_file_{index:02d}", path, "file")
    for index, path in enumerate(PHASE1_FORBIDDEN_DIRECTORY_ROOTS):
        add(f"phase1_forbidden_directory_{index:02d}", path, "directory")
    for index, path in enumerate(PHASE1_FORBIDDEN_GLOB_PATTERNS):
        add(f"phase1_forbidden_glob_{index:02d}", path, "glob")
    return rows


def audit_runtime_outputs_absent(*, root: Path = ROOT) -> list[dict[str, Any]]:
    rows = _expected_absence_rows()
    for row in rows:
        if row["expected_kind"] == "glob":
            pattern = Path(str(row["path"]))
            parent = _pin_generated(
                root, pattern.parent, label=row["label"],
                allow_managed_root=True)
            if parent.exists():
                if parent.is_symlink() or not parent.is_dir():
                    raise GlobalExecutionAmendmentError(
                        f"runtime glob parent changed: {row['label']}")
                matches = [entry for entry in parent.iterdir()
                           if fnmatchcase(entry.name, pattern.name)]
                if any(entry.is_symlink() for entry in matches):
                    raise GlobalExecutionAmendmentError(
                        f"runtime glob match is symlinked: {row['label']}")
                if matches:
                    raise GlobalExecutionAmendmentError(
                        f"runtime/outcome output predates amendment: {row['label']}")
            continue
        path = _pin_generated(root, row["path"], label=str(row["label"]))
        if path.exists() or path.is_symlink():
            raise GlobalExecutionAmendmentError(
                f"runtime/outcome output predates amendment: {row['label']}")
    return rows


def build_coupling_report(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        scientific_contract_bindings: Mapping[str, Any],
        preoutcome_input_bindings: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the exact source-only coupling report without file access."""

    if not _is_hex(source_repository_commit, 40):
        raise GlobalExecutionAmendmentError("source commit is malformed")
    sources = _validate_source_bindings(list(source_bindings))
    scientific = validate_scientific_contract_bindings(
        scientific_contract_bindings)
    inputs = validate_preoutcome_input_bindings(preoutcome_input_bindings)
    inventory = copy.deepcopy(list(CONSTRAINT_INVENTORY))
    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": REPORT_STATUS,
        "complete": True,
        "source_repository_commit": source_repository_commit,
        "expected_source_paths": list(EXPECTED_SOURCE_PATHS),
        "source_bindings": sources,
        "source_binding_set_digest": canonical_digest(sources),
        "scientific_contract_bindings": scientific,
        "scientific_contract_bindings_digest": canonical_digest(scientific),
        "preoutcome_input_bindings": inputs,
        "preoutcome_input_bindings_digest": canonical_digest(inputs),
        "scope": {
            "fixed_state_count": 115,
            "candidate_scene_count": 17,
            "selected_candidate_scene_count": 5,
            "final_state_count": 120,
            "rotation_count": 12,
            "candidates_per_rotation": 6,
            "preoutcome_authority_artifacts_read_for_binding": True,
            "generated_candidate_state_or_mask_artifact_contents_read_for_classification":
                False,
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": False,
            "solver_invoked": False,
            "exhaustive_scientific_search_started": False,
        },
        "allocator_scene_id_boundary": {
            "raw_scene_id_consumed_by_allocator": False,
            "allocator_identity_fields": [
                "state_id", "state_identity_digest", "family", "stratum",
                "split_role", "goal_type",
            ],
            "scene_derived_goal_type_is_hard_margin_input": True,
            "scene_pool_membership_eligibility_is_selection_only": True,
            "scene_pool_membership_eligibility_is_not_the_coupling_witness":
                True,
            "identity_owned_completion_rotation_vector_is_post_gate_input": True,
            "identity_owned_rotation_compatibility_is_joint_not_scene_pool_eligibility":
                True,
            "no_other_direct_scene_id_rotation_rule_found": True,
        },
        "constraint_inventory": inventory,
        "constraint_inventory_digest": canonical_digest(inventory),
        "constraint_ids": list(CONSTRAINT_IDS),
        "classification": "COUPLED",
        "classification_rule": (
            "COUPLED iff at least one selection-sensitive mandatory hard or "
            "post-allocation constraint jointly references scene identity and "
            "rotation/subset identity; membership in the seventeen-scene "
            "eligible pool is selection-only and is distinct from the exact "
            "twelve-entry identity-owned rotation-compatibility vector"
        ),
        "decisive_constraint_ids": list(DECISIVE_COUPLING_CONSTRAINT_IDS),
        "selected_method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL",
        "decoupled_deterministic_selection_applicable": False,
        "global_exact_model_applicable": True,
        "fixture_validation_contract": copy.deepcopy(
            FIXTURE_VALIDATION_CONTRACT),
        "outcome_boundary": {
            "candidate_outcomes_consumed": False,
            "branch_labels_read": False,
            "frames_read_or_generated": False,
            "latents_read_or_generated": False,
            "scorer_metrics_read": False,
            "predictor_outputs_read": False,
        },
    }
    payload[REPORT_SELF_KEY] = canonical_digest(payload)
    return payload


_REPORT_KEYS = frozenset({
    "schema", "status", "complete", "source_repository_commit",
    "expected_source_paths", "source_bindings", "source_binding_set_digest",
    "scientific_contract_bindings", "scientific_contract_bindings_digest",
    "preoutcome_input_bindings", "preoutcome_input_bindings_digest", "scope",
    "allocator_scene_id_boundary", "constraint_inventory",
    "constraint_inventory_digest", "constraint_ids", "classification",
    "classification_rule", "decisive_constraint_ids", "selected_method",
    "decoupled_deterministic_selection_applicable",
    "global_exact_model_applicable", "fixture_validation_contract",
    "outcome_boundary", REPORT_SELF_KEY,
})


def validate_coupling_report(
        payload: Mapping[str, Any], *,
        expected_scientific_contract_bindings: Mapping[str, Any],
        expected_preoutcome_input_bindings: Mapping[str, Any],
        expected_source_repository_commit: str | None = None,
        root: Path = ROOT, validate_live_source: bool = True,
        ) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GlobalExecutionAmendmentError("coupling report is not a mapping")
    report = dict(payload)
    sources = _validate_source_bindings(report.get("source_bindings"))
    scientific = validate_scientific_contract_bindings(
        expected_scientific_contract_bindings)
    inputs = validate_preoutcome_input_bindings(
        expected_preoutcome_input_bindings)
    commit = report.get("source_repository_commit")
    if (set(report) != _REPORT_KEYS
            or report.get("schema") != REPORT_SCHEMA
            or report.get("status") != REPORT_STATUS
            or report.get("complete") is not True
            or not _is_hex(commit, 40)
            or (expected_source_repository_commit is not None
                and commit != expected_source_repository_commit)
            or report.get("expected_source_paths") != list(EXPECTED_SOURCE_PATHS)
            or report.get("source_binding_set_digest")
            != canonical_digest(sources)
            or report.get("scientific_contract_bindings") != scientific
            or report.get("scientific_contract_bindings_digest")
            != canonical_digest(scientific)
            or report.get("preoutcome_input_bindings") != inputs
            or report.get("preoutcome_input_bindings_digest")
            != canonical_digest(inputs)
            or report.get("constraint_inventory")
            != list(CONSTRAINT_INVENTORY)
            or report.get("constraint_inventory_digest")
            != canonical_digest(list(CONSTRAINT_INVENTORY))
            or report.get("constraint_ids") != list(CONSTRAINT_IDS)
            or report.get("classification") != "COUPLED"
            or report.get("decisive_constraint_ids")
            != list(DECISIVE_COUPLING_CONSTRAINT_IDS)
            or report.get("selected_method")
            != "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL"
            or report.get("decoupled_deterministic_selection_applicable")
            is not False
            or report.get("global_exact_model_applicable") is not True
            or report.get("fixture_validation_contract")
            != FIXTURE_VALIDATION_CONTRACT
            or report.get(REPORT_SELF_KEY)
            != canonical_digest(_without(report, REPORT_SELF_KEY))):
        raise GlobalExecutionAmendmentError("coupling report binding changed")
    rebuilt = build_coupling_report(
        source_repository_commit=str(commit), source_bindings=sources,
        scientific_contract_bindings=scientific,
        preoutcome_input_bindings=inputs)
    if rebuilt != report:
        raise GlobalExecutionAmendmentError("coupling report is not exact")
    if validate_live_source:
        live_commit = _clean_source_commit(root=root)
        if live_commit != commit or _read_source_bindings(root=root) != sources:
            raise GlobalExecutionAmendmentError(
                "live source differs from coupling report")
    return report


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise GlobalExecutionAmendmentError(f"{label} is unavailable")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlobalExecutionAmendmentError(f"{label} JSON is corrupt") from exc
    if not isinstance(payload, dict):
        raise GlobalExecutionAmendmentError(f"{label} root is not a mapping")
    return payload, raw


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=True,
            allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GlobalExecutionAmendmentError(
            "artifact is not canonical JSON") from exc


def _exclusive_json(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    encoded = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise GlobalExecutionAmendmentError(f"{label} already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o444, follow_symlinks=False)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # A partial immutable artifact is never silently removed or replaced.
        raise
    reopened, raw = _load_json(path, label=label)
    if (reopened != dict(payload) or raw != encoded
            or stat.S_IMODE(path.stat().st_mode) & 0o222):
        raise GlobalExecutionAmendmentError(
            f"{label} exact read-only reopen changed")


def load_coupling_report(
        path: Path, *,
        expected_scientific_contract_bindings: Mapping[str, Any],
        expected_preoutcome_input_bindings: Mapping[str, Any],
        expected_source_repository_commit: str | None = None,
        root: Path = ROOT, validate_live_source: bool = True,
        ) -> dict[str, Any]:
    pinned = _require_logical_path(
        path, expected_relative=COUPLING_REPORT_RELATIVE_PATH,
        root=root, label="coupling report")
    payload, _raw = _load_json(pinned, label="coupling report")
    return validate_coupling_report(
        payload,
        expected_scientific_contract_bindings=
            expected_scientific_contract_bindings,
        expected_preoutcome_input_bindings=expected_preoutcome_input_bindings,
        expected_source_repository_commit=expected_source_repository_commit,
        root=root, validate_live_source=validate_live_source)


def issue_coupling_report(
        path: Path, *,
        scientific_contract_bindings: Mapping[str, Any],
        preoutcome_input_bindings: Mapping[str, Any],
        source_repository_commit: str | None = None,
        root: Path = ROOT,
        ) -> dict[str, Any]:
    """Install the one coupling report before any global-model runtime output."""

    pinned = _require_logical_path(
        path, expected_relative=COUPLING_REPORT_RELATIVE_PATH,
        root=root, label="coupling report")
    if not pinned.parent.is_dir() or pinned.parent.is_symlink():
        raise GlobalExecutionAmendmentError(
            "coupling-report parent is unavailable")
    if pinned.exists() or pinned.is_symlink():
        return load_coupling_report(
            path,
            expected_scientific_contract_bindings=
                scientific_contract_bindings,
            expected_preoutcome_input_bindings=preoutcome_input_bindings,
            expected_source_repository_commit=source_repository_commit,
            root=root)
    audit_runtime_outputs_absent(root=root)
    commit = _clean_source_commit(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise GlobalExecutionAmendmentError("requested source commit is not live")
    report = build_coupling_report(
        source_repository_commit=commit,
        source_bindings=_read_source_bindings(root=root),
        scientific_contract_bindings=scientific_contract_bindings,
        preoutcome_input_bindings=preoutcome_input_bindings)
    audit_runtime_outputs_absent(root=root)
    _exclusive_json(pinned, report, label="coupling report")
    return load_coupling_report(
        path,
        expected_scientific_contract_bindings=scientific_contract_bindings,
        expected_preoutcome_input_bindings=preoutcome_input_bindings,
        expected_source_repository_commit=commit, root=root)


def _validate_lineage_payload(role: str, payload: Mapping[str, Any]) -> None:
    if role == "v1_immutable_failure":
        if (payload.get("passes") is not False
                or payload.get("candidate_outcomes_consumed") is not False):
            raise GlobalExecutionAmendmentError(
                "immutable V1 failure semantics changed")
    elif role == "v2_contract":
        if (payload.get("source_repository_commit")
                != V2_SOURCE_REPOSITORY_COMMIT
                or payload.get("candidate_outcomes_consumed_at_issue") is not False
                or payload.get("scientific_masks_accessed_at_issue") is not False):
            raise GlobalExecutionAmendmentError(
                "immutable V2 contract semantics changed")
    elif role == "v2_benchmark_failure":
        if (payload.get("benchmark_v2_contract_digest") != V2_CONTRACT_DIGEST
                or payload.get("passes") is not False
                or payload.get("median_gate_passes") is not False
                or payload.get("maximum_gate_passes") is not False
                or payload.get("candidate_outcomes_consumed") is not False
                or payload.get("scientific_masks_accessed") is not False):
            raise GlobalExecutionAmendmentError(
                "immutable V2 benchmark failure semantics changed")
    elif role == "v2_terminal_failure":
        if (payload.get("status") != "IMMUTABLE_ONE_SHOT_V2_FAILURE"
                or payload.get("failure_kind") != "BENCHMARK_GATE_FAIL"
                or payload.get("benchmark_v2_contract_digest")
                != V2_CONTRACT_DIGEST
                or payload.get("benchmark_receipt_digest")
                != V2_BENCHMARK_RECEIPT_DIGEST
                or payload.get("overall_v2_passes") is not False
                or payload.get("scientific_search_plan_issued") is not False
                or payload.get("scientific_search_started") is not False
                or payload.get("candidate_outcomes_consumed") is not False
                or payload.get("scientific_masks_accessed") is not False
                or payload.get("v2_retry_permitted") is not False):
            raise GlobalExecutionAmendmentError(
                "immutable V2 terminal failure semantics changed")
    else:  # pragma: no cover - all callers iterate the closed registry.
        raise GlobalExecutionAmendmentError("unknown immutable lineage role")


def load_predecessor_lineage(*, root: Path = ROOT) -> dict[str, Any]:
    """Read only the four explicitly bound pre-outcome V1/V2 records."""

    result: dict[str, Any] = {}
    for spec in _LINEAGE_SPECS:
        path = _pin_generated(root, spec["path"], label=spec["role"])
        payload, raw = _load_json(path, label=spec["role"])
        self_key = spec["self_digest_key"]
        if (payload.get("schema") != spec["schema"]
                or payload.get(self_key) != spec["self_digest"]
                or canonical_digest(_without(payload, self_key))
                != spec["self_digest"]):
            raise GlobalExecutionAmendmentError(
                f"immutable lineage changed: {spec['role']}")
        _validate_lineage_payload(str(spec["role"]), payload)
        result[str(spec["role"])] = {
            **dict(spec),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": False,
            "preserved_not_retried_or_reinterpreted": True,
        }
    for role, (raw_sha256, byte_count) in _EXPECTED_LINEAGE_RAW.items():
        if (result[role]["raw_sha256"] != raw_sha256
                or result[role]["byte_count"] != byte_count):
            raise GlobalExecutionAmendmentError(
                f"immutable raw lineage binding changed: {role}")
    return result


def validate_predecessor_lineage_bindings(
        payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GlobalExecutionAmendmentError("lineage bindings are not a mapping")
    lineage = {str(key): dict(value) if isinstance(value, Mapping) else value
               for key, value in payload.items()}
    if set(lineage) != {str(spec["role"]) for spec in _LINEAGE_SPECS}:
        raise GlobalExecutionAmendmentError("lineage role coverage changed")
    for spec in _LINEAGE_SPECS:
        role = str(spec["role"])
        row = lineage[role]
        expected_keys = set(spec) | {
            "raw_sha256", "byte_count", "candidate_outcomes_consumed",
            "scientific_masks_accessed",
            "preserved_not_retried_or_reinterpreted",
        }
        if (not isinstance(row, dict) or set(row) != expected_keys
                or any(row.get(key) != value for key, value in spec.items())
                or not _is_hex(row.get("raw_sha256"), 64)
                or isinstance(row.get("byte_count"), bool)
                or not isinstance(row.get("byte_count"), int)
                or row["byte_count"] <= 0
                or row.get("candidate_outcomes_consumed") is not False
                or row.get("scientific_masks_accessed") is not False
                or row.get("preserved_not_retried_or_reinterpreted") is not True):
            raise GlobalExecutionAmendmentError("lineage binding changed")
    for role, (raw_sha256, byte_count) in _EXPECTED_LINEAGE_RAW.items():
        if (lineage[role]["raw_sha256"] != raw_sha256
                or lineage[role]["byte_count"] != byte_count):
            raise GlobalExecutionAmendmentError(
                f"raw lineage binding changed: {role}")
    return lineage


def coupling_report_artifact_binding(
        report: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return {
        "path": str(COUPLING_REPORT_RELATIVE_PATH),
        "schema": REPORT_SCHEMA,
        "coupling_report_digest": report[REPORT_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": report["source_repository_commit"],
        "classification": "COUPLED",
    }


def _validate_report_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GlobalExecutionAmendmentError("coupling-report binding is malformed")
    binding = dict(payload)
    if (set(binding) != {
            "path", "schema", "coupling_report_digest", "raw_sha256",
            "byte_count", "source_repository_commit", "classification"}
            or binding.get("path") != str(COUPLING_REPORT_RELATIVE_PATH)
            or binding.get("schema") != REPORT_SCHEMA
            or binding.get("classification") != "COUPLED"
            or not _is_hex(binding.get("coupling_report_digest"), 64)
            or not _is_hex(binding.get("raw_sha256"), 64)
            or not _is_hex(binding.get("source_repository_commit"), 40)
            or isinstance(binding.get("byte_count"), bool)
            or not isinstance(binding.get("byte_count"), int)
            or binding["byte_count"] <= 0):
        raise GlobalExecutionAmendmentError("coupling-report binding changed")
    return binding


def build_execution_amendment(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        coupling_report_binding: Mapping[str, Any],
        scientific_contract_bindings: Mapping[str, Any],
        preoutcome_input_bindings: Mapping[str, Any],
        predecessor_lineage: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the exact prospective amendment without reading or writing files."""

    if not _is_hex(source_repository_commit, 40):
        raise GlobalExecutionAmendmentError("source commit is malformed")
    sources = _validate_source_bindings(list(source_bindings))
    report_binding = _validate_report_binding(coupling_report_binding)
    scientific = validate_scientific_contract_bindings(
        scientific_contract_bindings)
    inputs = validate_preoutcome_input_bindings(preoutcome_input_bindings)
    lineage = validate_predecessor_lineage_bindings(predecessor_lineage)
    if report_binding["source_repository_commit"] != source_repository_commit:
        raise GlobalExecutionAmendmentError(
            "coupling report and amendment source commits differ")
    absence = _expected_absence_rows()
    payload: dict[str, Any] = {
        "schema": AMENDMENT_SCHEMA,
        "status": AMENDMENT_STATUS,
        "complete": True,
        "source_repository_commit": source_repository_commit,
        "expected_source_paths": list(EXPECTED_SOURCE_PATHS),
        "source_bindings": sources,
        "source_binding_set_digest": canonical_digest(sources),
        "coupling_report": report_binding,
        "scientific_contract_bindings": scientific,
        "scientific_contract_bindings_digest": canonical_digest(scientific),
        "preoutcome_input_bindings": inputs,
        "preoutcome_input_bindings_digest": canonical_digest(inputs),
        "immutable_predecessor_lineage": lineage,
        "immutable_predecessor_lineage_digest": canonical_digest(lineage),
        "v1_disposition": V1_FAILURE_STATUS,
        "v2_backend_disposition": V2_BACKEND_DISPOSITION,
        "supersession": {
            "status": SUPERSEDED_EXTERNAL_ENUMERATION_STATUS,
            "superseded_execution_requirements": [
                {
                    "constraint_id": "FIRST_PASSING_SCENE_COMBINATION",
                    "status": SUPERSEDED_EXTERNAL_ENUMERATION_STATUS,
                    "requirement": (
                        "external enumeration of 6,188 five-scene combinations "
                        "and proof that the selected combination is globally "
                        "lexicographically earliest"
                    ),
                },
                {
                    "constraint_id":
                        "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
                    "status": SUPERSEDED_CANONICAL_TIE_BREAK_STATUS,
                    "requirement": (
                        "sequential identity-ordered lexicographic minimisation "
                        "of the rotation vector before applying exact completion "
                        "compatibility"
                    ),
                },
            ],
            "scientific_requirement_superseded": False,
            "selector_superseded": False,
            "candidate_allocation_constraints_superseded": False,
            "oracle_superseded": False,
            "scorer_protocol_superseded": False,
        },
        "preserved_scientific_contract": {
            "scene_and_state_pool": "unchanged",
            "candidate_bank": "unchanged",
            "candidate_frequencies": "unchanged",
            "six_candidate_subset_size": "unchanged",
            "allowed_rotation_blocks": "unchanged",
            "family_and_stratum_quotas": "unchanged",
            "fit_calibration_split_96_24": "unchanged",
            "completion_enrichment_rule": "unchanged",
            "candidate_allocation_amendment_digest":
                CANDIDATE_ALLOCATION_AMENDMENT_DIGEST,
            "candidate_allocator_constraint_digest":
                CANDIDATE_ALLOCATION_CONTRACT_DIGEST,
            "oracle_v1_2": "unchanged",
            "render_preprocess_target_encoder_contracts": "unchanged",
            "scorer_architecture_and_qualification_criteria": "unchanged",
        },
        "selected_execution_method": {
            "classification": "COUPLED",
            "method": "ONE_GLOBAL_EXACT_FEASIBILITY_MODEL",
            "decision_variables": "binary selected-scene/state by allowed rotation",
            "mandatory_constraints_before_tie_break": True,
            "exactly_five_selected_completion_scenes": True,
            "one_rotation_per_selected_scene": True,
            "one_rotation_per_fixed_state_or_exact_residual_equivalent": True,
            "constraint_inventory_ids": list(CONSTRAINT_IDS),
            "preserved_mandatory_constraint_ids":
                list(PRESERVED_MANDATORY_CONSTRAINT_IDS),
            "superseded_execution_constraint_ids": [
                "FIRST_PASSING_SCENE_COMBINATION",
                "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
            ],
            "completion_eligibility_position": (
                "integrated hard scene-rotation compatibility in the single "
                "global model"
            ),
            "old_allocate_then_post_gate_sequence_preserved": False,
            "completion_eligibility_predicate_and_evidence_preserved": True,
            "stable_hash_pair_objective": copy.deepcopy(
                STABLE_HASH_OBJECTIVE_CONTRACT),
            "stable_hash_pair_objective_digest":
                STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST,
            "branch_or_downstream_metric_in_objective": False,
            "global_lexicographically_earliest_scene_set_required": False,
            "single_global_model": True,
            "external_combination_enumeration": False,
            "solver_threads": 1,
            "freeze_before_solve": [
                "variable_order", "constraint_order", "solver_version",
                "solver_seed", "thread_count", "presolve_settings",
                "search_branching_policy", "all_input_digests",
            ],
            "infeasible_effect": (
                "issue immutable exact-infeasibility receipt and stop without "
                "relaxing another scientific condition"
            ),
        },
        "fixture_validation_contract": copy.deepcopy(
            FIXTURE_VALIDATION_CONTRACT),
        "runtime_outputs_absent_at_issue": absence,
        "runtime_outputs_absent_at_issue_digest": canonical_digest(absence),
        "issuance_boundary": {
            "coupling_report_issued_first": True,
            "source_tree_clean_and_committed": True,
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": False,
            "branch_labels_read": False,
            "frames_or_latents_created": False,
            "scorer_or_predictor_accessed": False,
            "solver_invoked": False,
            "performance_benchmark_run": False,
            "v1_or_v2_retried": False,
        },
        "continuation_authority": {
            "on_valid_manifest": (
                "freeze complete 120-state manifest then continue directly through "
                "the already-authorised scorer smoke, 720-branch fit corpus, shared "
                "heads, no-latent baseline, one qualification and one development "
                "transfer analysis"
            ),
            "final_200_state_corpus_authorised": False,
            "another_approval_stop_required": False,
            "another_infrastructure_benchmark_authorised": False,
            "downstream_runtime_contracts": copy.deepcopy(
                DOWNSTREAM_RUNTIME_CONTRACTS),
            "downstream_stage_runtime_roles": copy.deepcopy(
                DOWNSTREAM_STAGE_RUNTIME_ROLES),
            "downstream_uses_global_solver_interpreter": {
                "genesis": False,
                "rocm": False,
            },
        },
    }
    payload[AMENDMENT_SELF_KEY] = canonical_digest(payload)
    return payload


_AMENDMENT_KEYS = frozenset({
    "schema", "status", "complete", "source_repository_commit",
    "expected_source_paths", "source_bindings", "source_binding_set_digest",
    "coupling_report", "scientific_contract_bindings",
    "scientific_contract_bindings_digest", "preoutcome_input_bindings",
    "preoutcome_input_bindings_digest", "immutable_predecessor_lineage",
    "immutable_predecessor_lineage_digest", "v1_disposition",
    "v2_backend_disposition", "supersession", "preserved_scientific_contract",
    "selected_execution_method", "fixture_validation_contract",
    "runtime_outputs_absent_at_issue", "runtime_outputs_absent_at_issue_digest",
    "issuance_boundary", "continuation_authority", AMENDMENT_SELF_KEY,
})


def validate_execution_amendment(
        payload: Mapping[str, Any], *,
        expected_coupling_report_binding: Mapping[str, Any],
        expected_scientific_contract_bindings: Mapping[str, Any],
        expected_preoutcome_input_bindings: Mapping[str, Any],
        expected_source_repository_commit: str | None = None,
        root: Path = ROOT, validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GlobalExecutionAmendmentError("execution amendment is not a mapping")
    amendment = dict(payload)
    sources = _validate_source_bindings(amendment.get("source_bindings"))
    report_binding = _validate_report_binding(
        expected_coupling_report_binding)
    scientific = validate_scientific_contract_bindings(
        expected_scientific_contract_bindings)
    inputs = validate_preoutcome_input_bindings(
        expected_preoutcome_input_bindings)
    lineage = validate_predecessor_lineage_bindings(
        amendment.get("immutable_predecessor_lineage", {}))
    commit = amendment.get("source_repository_commit")
    expected_absence = _expected_absence_rows()
    if (set(amendment) != _AMENDMENT_KEYS
            or amendment.get("schema") != AMENDMENT_SCHEMA
            or amendment.get("status") != AMENDMENT_STATUS
            or amendment.get("complete") is not True
            or not _is_hex(commit, 40)
            or (expected_source_repository_commit is not None
                and commit != expected_source_repository_commit)
            or amendment.get("expected_source_paths")
            != list(EXPECTED_SOURCE_PATHS)
            or amendment.get("source_binding_set_digest")
            != canonical_digest(sources)
            or amendment.get("coupling_report") != report_binding
            or amendment.get("scientific_contract_bindings") != scientific
            or amendment.get("scientific_contract_bindings_digest")
            != canonical_digest(scientific)
            or amendment.get("preoutcome_input_bindings") != inputs
            or amendment.get("preoutcome_input_bindings_digest")
            != canonical_digest(inputs)
            or amendment.get("immutable_predecessor_lineage_digest")
            != canonical_digest(lineage)
            or amendment.get("v1_disposition") != V1_FAILURE_STATUS
            or amendment.get("v2_backend_disposition")
            != V2_BACKEND_DISPOSITION
            or amendment.get("fixture_validation_contract")
            != FIXTURE_VALIDATION_CONTRACT
            or amendment.get("runtime_outputs_absent_at_issue")
            != expected_absence
            or amendment.get("runtime_outputs_absent_at_issue_digest")
            != canonical_digest(expected_absence)
            or amendment.get(AMENDMENT_SELF_KEY)
            != canonical_digest(_without(amendment, AMENDMENT_SELF_KEY))):
        raise GlobalExecutionAmendmentError(
            "global exact execution amendment binding changed")
    rebuilt = build_execution_amendment(
        source_repository_commit=str(commit), source_bindings=sources,
        coupling_report_binding=report_binding,
        scientific_contract_bindings=scientific,
        preoutcome_input_bindings=inputs,
        predecessor_lineage=lineage)
    if rebuilt != amendment:
        raise GlobalExecutionAmendmentError("execution amendment is not exact")
    if validate_live_authorities:
        if (_clean_source_commit(root=root) != commit
                or _read_source_bindings(root=root) != sources
                or load_predecessor_lineage(root=root) != lineage):
            raise GlobalExecutionAmendmentError(
                "live source or immutable predecessor lineage changed")
        report_path = _pin_generated(
            root, COUPLING_REPORT_RELATIVE_PATH, label="coupling report")
        report, raw = _load_json(report_path, label="coupling report")
        validated_report = validate_coupling_report(
            report, expected_scientific_contract_bindings=scientific,
            expected_preoutcome_input_bindings=inputs,
            expected_source_repository_commit=str(commit), root=root,
            validate_live_source=True)
        if coupling_report_artifact_binding(validated_report, raw) != report_binding:
            raise GlobalExecutionAmendmentError(
                "live coupling-report artifact binding changed")
        if require_runtime_outputs_absent:
            audit_runtime_outputs_absent(root=root)
    return amendment


def load_execution_amendment(
        path: Path, *,
        expected_coupling_report_binding: Mapping[str, Any],
        expected_scientific_contract_bindings: Mapping[str, Any],
        expected_preoutcome_input_bindings: Mapping[str, Any],
        expected_source_repository_commit: str | None = None,
        root: Path = ROOT, validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    pinned = _require_logical_path(
        path, expected_relative=EXECUTION_AMENDMENT_RELATIVE_PATH,
        root=root, label="execution amendment")
    payload, _raw = _load_json(pinned, label="execution amendment")
    return validate_execution_amendment(
        payload,
        expected_coupling_report_binding=expected_coupling_report_binding,
        expected_scientific_contract_bindings=
            expected_scientific_contract_bindings,
        expected_preoutcome_input_bindings=expected_preoutcome_input_bindings,
        expected_source_repository_commit=expected_source_repository_commit,
        root=root, validate_live_authorities=validate_live_authorities,
        require_runtime_outputs_absent=require_runtime_outputs_absent)


def issue_execution_amendment(
        path: Path, *, coupling_report_path: Path,
        scientific_contract_bindings: Mapping[str, Any],
        preoutcome_input_bindings: Mapping[str, Any],
        source_repository_commit: str | None = None,
        root: Path = ROOT,
        ) -> dict[str, Any]:
    """Install the prospective amendment after the report and before solving."""

    amendment_path = _require_logical_path(
        path, expected_relative=EXECUTION_AMENDMENT_RELATIVE_PATH,
        root=root, label="execution amendment")
    report_path = _require_logical_path(
        coupling_report_path, expected_relative=COUPLING_REPORT_RELATIVE_PATH,
        root=root, label="coupling report")
    if (not amendment_path.parent.is_dir() or amendment_path.parent.is_symlink()
            or not report_path.is_file() or report_path.is_symlink()):
        raise GlobalExecutionAmendmentError(
            "coupling report must exist before amendment issuance")
    commit = _clean_source_commit(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise GlobalExecutionAmendmentError("requested source commit is not live")
    report, report_raw = _load_json(report_path, label="coupling report")
    validated_report = validate_coupling_report(
        report,
        expected_scientific_contract_bindings=scientific_contract_bindings,
        expected_preoutcome_input_bindings=preoutcome_input_bindings,
        expected_source_repository_commit=commit, root=root,
        validate_live_source=True)
    report_binding = coupling_report_artifact_binding(
        validated_report, report_raw)
    if amendment_path.exists() or amendment_path.is_symlink():
        return load_execution_amendment(
            path, expected_coupling_report_binding=report_binding,
            expected_scientific_contract_bindings=
                scientific_contract_bindings,
            expected_preoutcome_input_bindings=preoutcome_input_bindings,
            expected_source_repository_commit=commit, root=root)
    first_absence = audit_runtime_outputs_absent(root=root)
    lineage = load_predecessor_lineage(root=root)
    amendment = build_execution_amendment(
        source_repository_commit=commit,
        source_bindings=_read_source_bindings(root=root),
        coupling_report_binding=report_binding,
        scientific_contract_bindings=scientific_contract_bindings,
        preoutcome_input_bindings=preoutcome_input_bindings,
        predecessor_lineage=lineage)
    second_absence = audit_runtime_outputs_absent(root=root)
    if (first_absence != second_absence
            or second_absence != amendment["runtime_outputs_absent_at_issue"]):
        raise GlobalExecutionAmendmentError(
            "runtime-output absence changed before amendment install")
    _exclusive_json(amendment_path, amendment, label="execution amendment")
    return load_execution_amendment(
        path, expected_coupling_report_binding=report_binding,
        expected_scientific_contract_bindings=scientific_contract_bindings,
        expected_preoutcome_input_bindings=preoutcome_input_bindings,
        expected_source_repository_commit=commit, root=root,
        require_runtime_outputs_absent=True)


_V1_AUTHORITY_KEYS = frozenset({
    "status", "source_repository_commit", "coupling_report",
    "coupling_report_artifact_binding", "execution_amendment",
    "execution_amendment_artifact_binding",
})
_V1_AUTHORITY_STATUS = (
    "IMMUTABLE_VALID_V1_AUTHORITY_SUPERSEDED_ONLY_FOR_SOURCE_CLOSURE")
_V2_REPLACED_V1_FIELDS = frozenset({
    "schema", "status", "source_repository_commit", "source_bindings",
    "source_binding_set_digest", "runtime_outputs_absent_at_issue",
    "runtime_outputs_absent_at_issue_digest", "issuance_boundary",
    AMENDMENT_SELF_KEY,
})
_V2_PRESERVED_V1_FIELDS = tuple(sorted(
    _AMENDMENT_KEYS - _V2_REPLACED_V1_FIELDS))
_AMENDMENT_V2_KEYS = frozenset({
    *_AMENDMENT_KEYS,
    "amendment_version", "v1_execution_authority",
    "historical_mixed_disposition_authority",
    "failed_attempt_disposition", "failed_attempt_disposition_digest",
    "source_correction", "source_correction_digest",
})


def _exact_artifact_binding(
        value: Any, expected: Mapping[str, Any], *, label: str,
        ) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise GlobalExecutionAmendmentError(f"{label} binding changed")
    return dict(expected)


def validate_historical_v1_execution_authority(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the issued V1 report/amendment without current-source replay."""

    if not isinstance(value, Mapping) or set(value) != _V1_AUTHORITY_KEYS:
        raise GlobalExecutionAmendmentError(
            "historical V1 execution authority is not closed")
    authority = copy.deepcopy(dict(value))
    if (authority.get("status") != _V1_AUTHORITY_STATUS
            or authority.get("source_repository_commit")
            != ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT):
        raise GlobalExecutionAmendmentError(
            "historical V1 execution authority disposition changed")
    report_binding = _exact_artifact_binding(
        authority.get("coupling_report_artifact_binding"),
        ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING,
        label="historical coupling report")
    amendment_binding = _exact_artifact_binding(
        authority.get("execution_amendment_artifact_binding"),
        ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING,
        label="historical execution amendment")
    report = authority.get("coupling_report")
    amendment = authority.get("execution_amendment")
    if not isinstance(report, Mapping) or not isinstance(amendment, Mapping):
        raise GlobalExecutionAmendmentError(
            "historical V1 execution authority payload is missing")
    report = dict(report)
    amendment = dict(amendment)
    report_raw = _pretty_json_bytes(report)
    amendment_raw = _pretty_json_bytes(amendment)
    if (report.get(REPORT_SELF_KEY) != report_binding["self_digest"]
            or hashlib.sha256(report_raw).hexdigest()
            != report_binding["raw_sha256"]
            or len(report_raw) != report_binding["byte_count"]
            or amendment.get(AMENDMENT_SELF_KEY)
            != amendment_binding["self_digest"]
            or hashlib.sha256(amendment_raw).hexdigest()
            != amendment_binding["raw_sha256"]
            or len(amendment_raw) != amendment_binding["byte_count"]):
        raise GlobalExecutionAmendmentError(
            "historical V1 authority bytes changed")
    scientific = validate_scientific_contract_bindings(
        report.get("scientific_contract_bindings", {}))
    preoutcome = validate_preoutcome_input_bindings(
        report.get("preoutcome_input_bindings", {}))
    validated_report = validate_coupling_report(
        report, expected_scientific_contract_bindings=scientific,
        expected_preoutcome_input_bindings=preoutcome,
        expected_source_repository_commit=
            ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        validate_live_source=False)
    legacy_report_binding = coupling_report_artifact_binding(
        validated_report, report_raw)
    if amendment.get("coupling_report") != legacy_report_binding:
        raise GlobalExecutionAmendmentError(
            "historical amendment report binding changed")
    validate_execution_amendment(
        amendment, expected_coupling_report_binding=legacy_report_binding,
        expected_scientific_contract_bindings=scientific,
        expected_preoutcome_input_bindings=preoutcome,
        expected_source_repository_commit=
            ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        validate_live_authorities=False)
    return authority


def load_historical_v1_execution_authority(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen only the two immutable issued V1 authority artifacts."""

    report_path = _pin_generated(
        root, COUPLING_REPORT_RELATIVE_PATH,
        label="historical coupling report")
    amendment_path = _pin_generated(
        root, EXECUTION_AMENDMENT_RELATIVE_PATH,
        label="historical execution amendment")
    report, report_raw = _load_json(
        report_path, label="historical coupling report")
    amendment, amendment_raw = _load_json(
        amendment_path, label="historical execution amendment")
    value = {
        "status": _V1_AUTHORITY_STATUS,
        "source_repository_commit":
            ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        "coupling_report": report,
        "coupling_report_artifact_binding": {
            **ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING,
            "raw_sha256": hashlib.sha256(report_raw).hexdigest(),
            "byte_count": len(report_raw),
        },
        "execution_amendment": amendment,
        "execution_amendment_artifact_binding": {
            **ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING,
            "raw_sha256": hashlib.sha256(amendment_raw).hexdigest(),
            "byte_count": len(amendment_raw),
        },
    }
    return validate_historical_v1_execution_authority(value)


def validate_historical_mixed_disposition_authority(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact d9d disposition supplied by its archived loader."""

    if (not isinstance(value, Mapping)
            or set(value) != {"payload", "binding"}
            or not isinstance(value.get("payload"), Mapping)):
        raise GlobalExecutionAmendmentError(
            "historical mixed disposition authority is not closed")
    result = copy.deepcopy(dict(value))
    binding = _exact_artifact_binding(
        result.get("binding"), HISTORICAL_MIXED_DISPOSITION_ARTIFACT_BINDING,
        label="historical mixed disposition")
    payload = dict(result["payload"])
    self_key = str(binding["self_digest_key"])
    raw = _pretty_json_bytes(payload)
    if (payload.get("schema")
            != "go2_scorer_fit_preserved_state_mixed_precontract_"
               "disposition_reachability_v2"
            or payload.get("status")
            != "PASS_PREOUTCOME_37_RETAINED_8_REPLACEMENT_DISPOSITION"
            or payload.get(self_key) != binding["self_digest"]
            or _legacy_default_json_digest(_without(payload, self_key))
            != binding["self_digest"]
            or hashlib.sha256(raw).hexdigest() != binding["raw_sha256"]
            or len(raw) != binding["byte_count"]):
        raise GlobalExecutionAmendmentError(
            "historical mixed disposition bytes changed")
    return result


def _changed_source_paths(
        historical: Sequence[Mapping[str, Any]],
        successor: Sequence[Mapping[str, Any]],
        ) -> list[str]:
    before = {str(row["path"]): dict(row) for row in historical}
    after = {str(row["path"]): dict(row) for row in successor}
    if set(before) != set(after):
        raise GlobalExecutionAmendmentError(
            "source correction closure path set changed")
    return sorted(
        path for path in before
        if (before[path]["sha256"], before[path]["byte_count"])
        != (after[path]["sha256"], after[path]["byte_count"]))


def build_execution_amendment_v2(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        v1_execution_authority: Mapping[str, Any],
        historical_mixed_disposition_authority: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the source-only V2 successor without file or mask access."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT):
        raise GlobalExecutionAmendmentError(
            "source-correction commit is malformed or not a successor")
    v1 = validate_historical_v1_execution_authority(
        v1_execution_authority)
    mixed = validate_historical_mixed_disposition_authority(
        historical_mixed_disposition_authority)
    current_sources = _validate_source_bindings(list(source_bindings))
    v1_amendment = dict(v1["execution_amendment"])
    historical_sources = _validate_source_bindings(
        v1_amendment["source_bindings"])
    changed_paths = _changed_source_paths(
        historical_sources, current_sources)
    if changed_paths != sorted(SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS):
        raise GlobalExecutionAmendmentError(
            "source correction changed an unauthorised source path")
    absence = _expected_absence_rows()
    failure = copy.deepcopy(FAILED_SOURCE_TRANSITION_DISPOSITION)
    source_correction = {
        "status": "SOURCE_ONLY_HISTORICAL_AUTHORITY_VALIDATION_CORRECTION",
        "defect": (
            "HISTORICAL_MIXED_PRECONTRACT_DISPOSITION_WAS_VALIDATED_AS_"
            "CURRENT_SOURCE"
        ),
        "correction": (
            "GLOBAL_EXACT_PATH_USES_EXACT_D9D_HISTORICAL_DISPOSITION_"
            "AUTHORITY_WITHOUT_CHANGING_THE_LEGACY_CURRENT_SOURCE_LOADER"
        ),
        "historical_source_repository_commit":
            ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        "successor_source_repository_commit": source_repository_commit,
        "allowed_changed_source_paths":
            list(SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed_paths,
        "historical_source_binding_set_digest": v1_amendment[
            "source_binding_set_digest"],
        "successor_source_binding_set_digest": canonical_digest(
            current_sources),
        "legacy_active_mixed_disposition_loader_changed": False,
        "global_exact_historical_loader_added": True,
        "scene_or_state_pool_changed": False,
        "candidate_bank_or_frequency_changed": False,
        "selector_or_allocation_constraint_changed": False,
        "model_objective_or_solver_setting_changed": False,
        "oracle_render_preprocess_or_target_encoder_changed": False,
        "scorer_architecture_or_qualification_changed": False,
        "scientific_contract_changed": False,
        "candidate_outcome_or_downstream_metric_used": False,
        "would_be_1ebc_operational_scorer_digest_preserved": False,
        "current_operational_scorer_digest_bound_only_after_valid_manifest":
            True,
    }
    payload: dict[str, Any] = {
        "schema": AMENDMENT_V2_SCHEMA,
        "status": AMENDMENT_V2_STATUS,
        "complete": True,
        "amendment_version": 2,
        "source_repository_commit": source_repository_commit,
        "expected_source_paths": list(EXPECTED_SOURCE_PATHS),
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        **{key: copy.deepcopy(v1_amendment[key])
           for key in _V2_PRESERVED_V1_FIELDS},
        "runtime_outputs_absent_at_issue": absence,
        "runtime_outputs_absent_at_issue_digest": canonical_digest(absence),
        "issuance_boundary": {
            "v1_report_and_amendment_preserved_immutable": True,
            "source_tree_clean_and_committed": True,
            "failed_attempt_preserved": True,
            "mandatory_synthetic_fixture_suite_previously_completed": True,
            "candidate_masks_previously_accessed": True,
            "frozen_45_check_mask_evidence_previously_validated": True,
            "seven_vector_mapping_previously_returned": False,
            "candidate_outcomes_consumed": False,
            "production_instance_or_model_built": False,
            "runner_or_model_plan_written": False,
            "scientific_production_solver_invoked": False,
            "performance_benchmark_run": False,
            "v1_or_v2_benchmark_retried": False,
        },
        "v1_execution_authority": copy.deepcopy(v1),
        "historical_mixed_disposition_authority": copy.deepcopy(mixed),
        "failed_attempt_disposition": failure,
        "failed_attempt_disposition_digest": canonical_digest(failure),
        "source_correction": source_correction,
        "source_correction_digest": canonical_digest(source_correction),
    }
    if set(payload) != _AMENDMENT_V2_KEYS - {AMENDMENT_SELF_KEY}:
        raise GlobalExecutionAmendmentError(
            "source-corrected amendment construction surface changed")
    payload[AMENDMENT_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_execution_amendment_v2(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _AMENDMENT_V2_KEYS:
        raise GlobalExecutionAmendmentError(
            "source-corrected amendment is not closed")
    amendment = copy.deepcopy(dict(payload))
    expected = build_execution_amendment_v2(
        source_repository_commit=str(amendment.get(
            "source_repository_commit", "")),
        source_bindings=amendment.get("source_bindings", []),
        v1_execution_authority=amendment.get("v1_execution_authority", {}),
        historical_mixed_disposition_authority=amendment.get(
            "historical_mixed_disposition_authority", {}),
    )
    if (amendment != expected
            or amendment.get(AMENDMENT_SELF_KEY)
            != canonical_digest(_without(amendment, AMENDMENT_SELF_KEY))):
        raise GlobalExecutionAmendmentError(
            "source-corrected amendment binding changed")
    if validate_live_authorities:
        if (_clean_source_commit(root=root)
                != amendment["source_repository_commit"]
                or _read_source_bindings(root=root)
                != amendment["source_bindings"]
                or load_historical_v1_execution_authority(root=root)
                != amendment["v1_execution_authority"]
                or load_predecessor_lineage(root=root)
                != amendment["immutable_predecessor_lineage"]):
            raise GlobalExecutionAmendmentError(
                "live source or historical authority differs from V2")
        if require_runtime_outputs_absent:
            audit_runtime_outputs_absent(root=root)
    return amendment


def execution_amendment_v2_artifact_binding(
        amendment: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    validated = validate_execution_amendment_v2(
        amendment, validate_live_authorities=False)
    if raw != _pretty_json_bytes(validated):
        raise GlobalExecutionAmendmentError(
            "source-corrected amendment raw bytes changed")
    return {
        "path": str(EXECUTION_AMENDMENT_V2_RELATIVE_PATH),
        "schema": AMENDMENT_V2_SCHEMA,
        "execution_amendment_digest": validated[AMENDMENT_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "historical_source_repository_commit":
            ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT,
        "source_repository_commit": validated["source_repository_commit"],
    }


def load_execution_amendment_v2(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    supplied = (root / EXECUTION_AMENDMENT_V2_RELATIVE_PATH
                if path is None else path)
    pinned = _require_logical_path(
        supplied, expected_relative=EXECUTION_AMENDMENT_V2_RELATIVE_PATH,
        root=root, label="source-corrected execution amendment")
    payload, _raw = _load_json(
        pinned, label="source-corrected execution amendment")
    validated = validate_execution_amendment_v2(
        payload, root=root,
        validate_live_authorities=validate_live_authorities,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    execution_amendment_v2_artifact_binding(validated, _raw)
    return validated


def issue_execution_amendment_v2(
        path: Path, *,
        historical_mixed_disposition_authority: Mapping[str, Any],
        source_repository_commit: str | None = None,
        root: Path = ROOT,
        ) -> dict[str, Any]:
    """Install the sole source-corrected successor after its clean commit."""

    amendment_path = _require_logical_path(
        path, expected_relative=EXECUTION_AMENDMENT_V2_RELATIVE_PATH,
        root=root, label="source-corrected execution amendment")
    if not amendment_path.parent.is_dir() or amendment_path.parent.is_symlink():
        raise GlobalExecutionAmendmentError(
            "source-corrected amendment parent is unavailable")
    commit = _clean_source_commit(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise GlobalExecutionAmendmentError(
            "requested source-correction commit is not live")
    v1 = load_historical_v1_execution_authority(root=root)
    mixed = validate_historical_mixed_disposition_authority(
        historical_mixed_disposition_authority)
    if amendment_path.exists() or amendment_path.is_symlink():
        return load_execution_amendment_v2(
            amendment_path, root=root, require_runtime_outputs_absent=True)
    first_absence = audit_runtime_outputs_absent(root=root)
    amendment = build_execution_amendment_v2(
        source_repository_commit=commit,
        source_bindings=_read_source_bindings(root=root),
        v1_execution_authority=v1,
        historical_mixed_disposition_authority=mixed)
    second_absence = audit_runtime_outputs_absent(root=root)
    if (first_absence != second_absence
            or second_absence != amendment["runtime_outputs_absent_at_issue"]):
        raise GlobalExecutionAmendmentError(
            "runtime-output absence changed before V2 amendment install")
    _exclusive_json(
        amendment_path, amendment,
        label="source-corrected execution amendment")
    return load_execution_amendment_v2(
        amendment_path, root=root, require_runtime_outputs_absent=True)


def load_source_corrected_execution_authority(
        *, root: Path = ROOT,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    """Return the closed active V2 authority projection for builder context."""

    path = _pin_generated(
        root, EXECUTION_AMENDMENT_V2_RELATIVE_PATH,
        label="source-corrected execution amendment")
    amendment, raw = _load_json(
        path, label="source-corrected execution amendment")
    amendment = validate_execution_amendment_v2(
        amendment, root=root, validate_live_authorities=True,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    v1 = amendment["v1_execution_authority"]
    report = dict(v1["coupling_report"])
    return {
        "coupling_report": report,
        "coupling_report_binding": coupling_report_artifact_binding(
            report, _pretty_json_bytes(report)),
        "execution_amendment": amendment,
        "execution_amendment_binding":
            execution_amendment_v2_artifact_binding(amendment, raw),
        "v1_execution_amendment": dict(v1["execution_amendment"]),
        "v1_execution_amendment_binding": dict(
            v1["execution_amendment_artifact_binding"]),
        "historical_mixed_disposition_authority": copy.deepcopy(
            amendment["historical_mixed_disposition_authority"]),
        "scientific_contract_bindings": dict(
            amendment["scientific_contract_bindings"]),
        "preoutcome_input_bindings": dict(
            amendment["preoutcome_input_bindings"]),
        "source_transition_digest": amendment[AMENDMENT_SELF_KEY],
        "candidate_outcomes_consumed": False,
    }


_IMMUTABLE_V2_AUTHORITY_KEYS = frozenset({"payload", "binding"})
_PREPLAN_REPLACED_V2_FIELDS = frozenset({
    "schema", "status", "source_repository_commit", "source_bindings",
    "source_binding_set_digest", "runtime_outputs_absent_at_issue",
    "runtime_outputs_absent_at_issue_digest", "issuance_boundary",
    AMENDMENT_SELF_KEY,
})
_PREPLAN_PRESERVED_V2_FIELDS = tuple(sorted(
    _AMENDMENT_V2_KEYS - _PREPLAN_REPLACED_V2_FIELDS))
_PREPLAN_CORRECTION_KEYS = frozenset({
    *_AMENDMENT_V2_KEYS,
    "preplan_integration_correction_version",
    "immutable_v2_execution_authority",
    "v2_post_install_reopen_failure",
    "v2_post_install_reopen_failure_digest",
    "post_v2_preplan_failed_attempt_disposition",
    "post_v2_preplan_failed_attempt_disposition_digest",
    "preplan_integration_correction",
    "preplan_integration_correction_digest",
})


def validate_immutable_v2_execution_authority(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete immutable V2 payload and literal raw binding."""

    if (not isinstance(value, Mapping)
            or set(value) != _IMMUTABLE_V2_AUTHORITY_KEYS
            or not isinstance(value.get("payload"), Mapping)):
        raise GlobalExecutionAmendmentError(
            "immutable V2 execution authority is not closed")
    authority = copy.deepcopy(dict(value))
    binding = _exact_artifact_binding(
        authority.get("binding"),
        IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING,
        label="immutable V2 execution amendment")
    payload = validate_execution_amendment_v2(
        authority["payload"], validate_live_authorities=False)
    raw = _pretty_json_bytes(payload)
    if (payload.get("schema") != AMENDMENT_V2_SCHEMA
            or payload.get("status") != AMENDMENT_V2_STATUS
            or payload.get("source_repository_commit")
            != IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT
            or payload.get(AMENDMENT_SELF_KEY) != binding["self_digest"]
            or binding.get("self_digest_key") != AMENDMENT_SELF_KEY
            or hashlib.sha256(raw).hexdigest() != binding["raw_sha256"]
            or len(raw) != binding["byte_count"]):
        raise GlobalExecutionAmendmentError(
            "immutable V2 execution amendment bytes changed")
    return authority


def load_immutable_v2_execution_authority(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen V2 as exact historical authority without current-source replay."""

    path = _pin_generated(
        root, EXECUTION_AMENDMENT_V2_RELATIVE_PATH,
        label="immutable V2 execution amendment")
    payload, raw = _load_json(path, label="immutable V2 execution amendment")
    binding = {
        **IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING,
        "self_digest": payload.get(AMENDMENT_SELF_KEY),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": payload.get("source_repository_commit"),
    }
    return validate_immutable_v2_execution_authority({
        "payload": payload,
        "binding": binding,
    })


def build_preplan_integration_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_v2_execution_authority: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the orthogonal source-only correction around immutable V2."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT):
        raise GlobalExecutionAmendmentError(
            "preplan-correction commit is malformed or not a successor")
    v2_authority = validate_immutable_v2_execution_authority(
        immutable_v2_execution_authority)
    v2 = dict(v2_authority["payload"])
    current_sources = _validate_source_bindings(list(source_bindings))
    v2_sources = _validate_source_bindings(v2["source_bindings"])
    changed_paths = _changed_source_paths(v2_sources, current_sources)
    if changed_paths != sorted(
            PREPLAN_INTEGRATION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS):
        raise GlobalExecutionAmendmentError(
            "preplan integration correction changed an unauthorised source path")

    absence = _expected_absence_rows()
    post_install_failure = copy.deepcopy(V2_POST_INSTALL_REOPEN_FAILURE)
    preplan_failure = copy.deepcopy(
        POST_V2_PREPLAN_FAILED_ATTEMPT_DISPOSITION)
    correction = {
        "status": "SOURCE_ONLY_CANONICAL_BOUNDARY_VALIDATION_CORRECTION",
        "defect": (
            "GLOBAL_EXACT_MODEL_ACCEPTED_ONLY_A_TWO_FIELD_BOUNDARY_WHILE_THE_"
            "FROZEN_OPTIONAL_CANDIDATES_CARRY_THE_COMPLETE_CANONICAL_BOUNDARY"
        ),
        "correction": (
            "VALIDATE_AND_HASH_THE_COMPLETE_TEN_FIELD_CANONICAL_BOUNDARY_"
            "WITHOUT_PROJECTING_AWAY_FROZEN_STRUCTURAL_EVIDENCE"),
        "historical_source_repository_commit":
            IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT,
        "successor_source_repository_commit": source_repository_commit,
        "allowed_changed_source_paths": list(
            PREPLAN_INTEGRATION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed_paths,
        "historical_source_binding_set_digest": v2[
            "source_binding_set_digest"],
        "successor_source_binding_set_digest": canonical_digest(
            current_sources),
        "canonical_boundary_keys": [
            "boundary_digest", "command_block_tick", "decimation_phase",
            "episode_step", "observation_emission_phase_ns", "reset",
            "sim_time_ns", "source_step", "terminated", "truncated",
        ],
        "canonical_boundary_digest": v2[
            "scientific_contract_bindings"]["boundary_digest"],
        "builder_optional_candidate_projection_changed": False,
        "model_canonical_boundary_validation_corrected": True,
        "scene_or_state_pool_changed": False,
        "candidate_bank_or_frequency_changed": False,
        "selector_or_allocation_constraint_changed": False,
        "model_scientific_constraint_changed": False,
        "model_objective_or_solver_setting_changed": False,
        "oracle_render_preprocess_or_target_encoder_changed": False,
        "scorer_architecture_or_qualification_changed": False,
        "scientific_contract_changed": False,
        "candidate_outcome_or_downstream_metric_used": False,
        "historical_scorer_identity_lineage_preserved": True,
        "current_operational_scorer_digest_bound_only_after_valid_manifest":
            True,
    }
    payload: dict[str, Any] = {
        **{key: copy.deepcopy(v2[key])
           for key in _PREPLAN_PRESERVED_V2_FIELDS},
        "schema": PREPLAN_INTEGRATION_CORRECTION_SCHEMA,
        "status": PREPLAN_INTEGRATION_CORRECTION_STATUS,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "runtime_outputs_absent_at_issue": absence,
        "runtime_outputs_absent_at_issue_digest": canonical_digest(absence),
        "issuance_boundary": {
            "immutable_v1_and_v2_authorities_preserved": True,
            "source_tree_clean_and_committed": True,
            "v2_post_install_reopen_failure_preserved": True,
            "post_v2_preplan_failure_preserved": True,
            "mandatory_synthetic_fixture_suite_previously_completed": True,
            "historical_optional_completion_vectors_parsed": 17,
            "historical_frozen_45_check_mask_evidence_validated": True,
            "historical_seven_vector_mapping_assembled_and_returned": True,
            "historical_scientific_masks_accessed": True,
            "scientific_masks_accessed_during_this_issuance": False,
            "new_attempt_mask_context_started": False,
            "candidate_outcomes_consumed": False,
            "production_instance_or_model_built": False,
            "runner_or_model_plan_written": False,
            "scientific_production_solver_invoked": False,
            "performance_benchmark_run": False,
            "v1_or_v2_benchmark_retried": False,
        },
        "preplan_integration_correction_version": 1,
        "immutable_v2_execution_authority": copy.deepcopy(v2_authority),
        "v2_post_install_reopen_failure": post_install_failure,
        "v2_post_install_reopen_failure_digest": canonical_digest(
            post_install_failure),
        "post_v2_preplan_failed_attempt_disposition": preplan_failure,
        "post_v2_preplan_failed_attempt_disposition_digest": canonical_digest(
            preplan_failure),
        "preplan_integration_correction": correction,
        "preplan_integration_correction_digest": canonical_digest(correction),
    }
    if set(payload) != _PREPLAN_CORRECTION_KEYS - {AMENDMENT_SELF_KEY}:
        raise GlobalExecutionAmendmentError(
            "preplan integration correction construction surface changed")
    payload[AMENDMENT_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_preplan_integration_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    """Validate the active correction and every immutable predecessor."""

    if (not isinstance(payload, Mapping)
            or set(payload) != _PREPLAN_CORRECTION_KEYS):
        raise GlobalExecutionAmendmentError(
            "preplan integration correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != PREPLAN_INTEGRATION_CORRECTION_SCHEMA
            or correction.get("status")
            != PREPLAN_INTEGRATION_CORRECTION_STATUS
            or correction.get("amendment_version") != 2
            or correction.get("preplan_integration_correction_version") != 1):
        raise GlobalExecutionAmendmentError(
            "preplan integration correction version changed")
    expected = build_preplan_integration_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_v2_execution_authority=correction.get(
            "immutable_v2_execution_authority", {}),
    )
    if (correction != expected
            or correction.get(AMENDMENT_SELF_KEY)
            != canonical_digest(_without(correction, AMENDMENT_SELF_KEY))):
        raise GlobalExecutionAmendmentError(
            "preplan integration correction binding changed")
    if validate_live_authorities:
        if (_clean_source_commit(root=root)
                != correction["source_repository_commit"]
                or _read_source_bindings(root=root)
                != correction["source_bindings"]
                or load_immutable_v2_execution_authority(root=root)
                != correction["immutable_v2_execution_authority"]
                or load_predecessor_lineage(root=root)
                != correction["immutable_predecessor_lineage"]):
            raise GlobalExecutionAmendmentError(
                "live source or immutable authority differs from correction")
        if require_runtime_outputs_absent:
            audit_runtime_outputs_absent(root=root)
    return correction


def preplan_integration_correction_artifact_binding(
        correction: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    validated = validate_preplan_integration_correction(
        correction, validate_live_authorities=False)
    if raw != _pretty_json_bytes(validated):
        raise GlobalExecutionAmendmentError(
            "preplan integration correction raw bytes changed")
    return {
        "path": str(PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH),
        "schema": PREPLAN_INTEGRATION_CORRECTION_SCHEMA,
        "execution_amendment_digest": validated[AMENDMENT_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "v2_source_repository_commit": IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT,
        "source_repository_commit": validated["source_repository_commit"],
    }


def load_preplan_integration_correction(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    supplied = (root / PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH
                if path is None else path)
    pinned = _require_logical_path(
        supplied, expected_relative=PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH,
        root=root, label="preplan integration correction")
    payload, raw = _load_json(pinned, label="preplan integration correction")
    validated = validate_preplan_integration_correction(
        payload, root=root,
        validate_live_authorities=validate_live_authorities,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    preplan_integration_correction_artifact_binding(validated, raw)
    return validated


def issue_preplan_integration_correction(
        path: Path, *, source_repository_commit: str | None = None,
        root: Path = ROOT,
        ) -> dict[str, Any]:
    """Install the sole prospective integration correction after clean commit."""

    correction_path = _require_logical_path(
        path, expected_relative=PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH,
        root=root, label="preplan integration correction")
    if (not correction_path.parent.is_dir()
            or correction_path.parent.is_symlink()):
        raise GlobalExecutionAmendmentError(
            "preplan integration correction parent is unavailable")
    commit = _clean_source_commit(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise GlobalExecutionAmendmentError(
            "requested preplan-correction commit is not live")
    if correction_path.exists() or correction_path.is_symlink():
        return load_preplan_integration_correction(
            path, root=root, require_runtime_outputs_absent=True)
    v2_authority = load_immutable_v2_execution_authority(root=root)
    first_absence = audit_runtime_outputs_absent(root=root)
    correction = build_preplan_integration_correction(
        source_repository_commit=commit,
        source_bindings=_read_source_bindings(root=root),
        immutable_v2_execution_authority=v2_authority)
    second_absence = audit_runtime_outputs_absent(root=root)
    if (first_absence != second_absence
            or second_absence != correction["runtime_outputs_absent_at_issue"]):
        raise GlobalExecutionAmendmentError(
            "runtime-output absence changed before correction install")
    _exclusive_json(
        correction_path, correction, label="preplan integration correction")
    return load_preplan_integration_correction(
        path, root=root, require_runtime_outputs_absent=True)


def load_active_execution_authority(
        *, root: Path = ROOT,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    """Return the closed active correction projection for runtime consumers."""

    path = _pin_generated(
        root, PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH,
        label="preplan integration correction")
    correction, raw = _load_json(path, label="preplan integration correction")
    correction = validate_preplan_integration_correction(
        correction, root=root, validate_live_authorities=True,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    v1 = correction["v1_execution_authority"]
    report = dict(v1["coupling_report"])
    return {
        "coupling_report": report,
        "coupling_report_binding": coupling_report_artifact_binding(
            report, _pretty_json_bytes(report)),
        "execution_amendment": correction,
        "execution_amendment_binding":
            preplan_integration_correction_artifact_binding(correction, raw),
        "immutable_v2_execution_authority": copy.deepcopy(
            correction["immutable_v2_execution_authority"]),
        "v1_execution_amendment": dict(v1["execution_amendment"]),
        "v1_execution_amendment_binding": dict(
            v1["execution_amendment_artifact_binding"]),
        "historical_mixed_disposition_authority": copy.deepcopy(
            correction["historical_mixed_disposition_authority"]),
        "scientific_contract_bindings": dict(
            correction["scientific_contract_bindings"]),
        "preoutcome_input_bindings": dict(
            correction["preoutcome_input_bindings"]),
        "source_transition_digest": correction[AMENDMENT_SELF_KEY],
        "candidate_outcomes_consumed": False,
    }


__all__ = [
    "AMENDMENT_SCHEMA", "AMENDMENT_SELF_KEY", "AMENDMENT_STATUS",
    "AMENDMENT_V2_SCHEMA", "AMENDMENT_V2_STATUS",
    "CANDIDATE_ALLOCATION_AMENDMENT_DIGEST",
    "CANDIDATE_ALLOCATION_CONTRACT_DIGEST", "CONSTRAINT_IDS",
    "CONSTRAINT_INVENTORY", "COUPLING_REPORT_RELATIVE_PATH",
    "DECISIVE_COUPLING_CONSTRAINT_IDS", "ENGINE_SOURCE_PATH",
    "DOWNSTREAM_PYTHON_VERSION", "DOWNSTREAM_RUNTIME_CONTRACTS",
    "DOWNSTREAM_STAGE_RUNTIME_ROLES",
    "GENESIS_DOWNSTREAM_INTERPRETER_RELATIVE_PATH",
    "GENESIS_DOWNSTREAM_PYVENV_CONFIG_BYTE_COUNT",
    "GENESIS_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH",
    "GENESIS_DOWNSTREAM_PYVENV_CONFIG_SHA256",
    "GENESIS_DOWNSTREAM_RUNTIME_CONTRACT",
    "ROCM_DOWNSTREAM_INTERPRETER_RELATIVE_PATH",
    "ROCM_DOWNSTREAM_PYVENV_CONFIG_BYTE_COUNT",
    "ROCM_DOWNSTREAM_PYVENV_CONFIG_RELATIVE_PATH",
    "ROCM_DOWNSTREAM_PYVENV_CONFIG_SHA256",
    "ROCM_DOWNSTREAM_RUNTIME_CONTRACT",
    "EXECUTION_AMENDMENT_RELATIVE_PATH",
    "EXECUTION_AMENDMENT_V2_RELATIVE_PATH", "EXPECTED_SOURCE_PATHS",
    "FAILED_SOURCE_TRANSITION_DISPOSITION",
    "FIXTURE_VALIDATION_CONTRACT", "GlobalExecutionAmendmentError",
    "HISTORICAL_MIXED_DISPOSITION_ARTIFACT_BINDING",
    "IMMUTABLE_V2_EXECUTION_AMENDMENT_ARTIFACT_BINDING",
    "IMMUTABLE_V2_SOURCE_REPOSITORY_COMMIT",
    "NEW_RUNTIME_OUTPUT_PATHS",
    "ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING",
    "ORIGINAL_EXECUTION_AMENDMENT_ARTIFACT_BINDING",
    "ORIGINAL_GLOBAL_EXACT_SOURCE_REPOSITORY_COMMIT",
    "REPORT_SCHEMA", "REPORT_SELF_KEY",
    "REPORT_STATUS", "RUNNER_SOURCE_PATH", "SCIENTIFIC_CONTRACT_BINDING_KEYS",
    "STABLE_HASH_OBJECTIVE_CONTRACT",
    "STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST",
    "SUPERSEDED_CANONICAL_TIE_BREAK_STATUS",
    "SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "POST_V2_PREPLAN_FAILED_ATTEMPT_DISPOSITION",
    "PREPLAN_INTEGRATION_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "PREPLAN_INTEGRATION_CORRECTION_RELATIVE_PATH",
    "PREPLAN_INTEGRATION_CORRECTION_SCHEMA",
    "PREPLAN_INTEGRATION_CORRECTION_STATUS",
    "SUPERSEDED_EXTERNAL_ENUMERATION_STATUS", "V1_FAILURE_RECEIPT_DIGEST",
    "V1_FAILURE_STATUS", "V2_BACKEND_DISPOSITION",
    "V2_BENCHMARK_RECEIPT_DIGEST", "V2_CONTRACT_DIGEST",
    "V2_SOURCE_REPOSITORY_COMMIT", "V2_TERMINAL_FAILURE_RECEIPT_DIGEST",
    "audit_runtime_outputs_absent", "build_coupling_report",
    "build_execution_amendment", "build_execution_amendment_v2",
    "build_preplan_integration_correction",
    "canonical_digest",
    "coupling_report_artifact_binding", "issue_coupling_report",
    "execution_amendment_v2_artifact_binding",
    "issue_execution_amendment", "issue_execution_amendment_v2",
    "issue_preplan_integration_correction",
    "load_active_execution_authority",
    "load_coupling_report", "load_execution_amendment",
    "load_execution_amendment_v2",
    "load_immutable_v2_execution_authority",
    "load_preplan_integration_correction",
    "load_historical_v1_execution_authority",
    "load_predecessor_lineage", "load_source_corrected_execution_authority",
    "validate_coupling_report", "validate_execution_amendment",
    "validate_execution_amendment_v2",
    "validate_immutable_v2_execution_authority",
    "validate_preplan_integration_correction",
    "validate_historical_mixed_disposition_authority",
    "validate_historical_v1_execution_authority",
    "validate_predecessor_lineage_bindings",
    "validate_preoutcome_input_bindings",
    "validate_scientific_contract_bindings",
    "V2_POST_INSTALL_REOPEN_FAILURE",
    "preplan_integration_correction_artifact_binding",
]
