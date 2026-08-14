"""Prospective authority for the outcome-free small-completion global model.

This module records the bounded source inspection which established that
completion-scene selection and candidate rotation assignment are coupled.  It
then issues the sole prospective execution amendment authorising one global
exact feasibility model.  The external 6,188-combination enumeration and the
inner identity-ordered lexicographic rotation tie-break are superseded as
non-scientific execution machinery; every scientific selector, allocation
margin, allowed subset, oracle, corpus and scorer contract remains frozen.

Importing this module performs no file access, solver work, mask access, or
outcome access.  Report construction is pure.  The two issue functions are
the only writers and install their dedicated JSON artifacts with ``O_EXCL``,
file and directory ``fsync``, and read-only mode.
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


__all__ = [
    "AMENDMENT_SCHEMA", "AMENDMENT_SELF_KEY", "AMENDMENT_STATUS",
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
    "EXECUTION_AMENDMENT_RELATIVE_PATH", "EXPECTED_SOURCE_PATHS",
    "FIXTURE_VALIDATION_CONTRACT", "GlobalExecutionAmendmentError",
    "NEW_RUNTIME_OUTPUT_PATHS", "REPORT_SCHEMA", "REPORT_SELF_KEY",
    "REPORT_STATUS", "RUNNER_SOURCE_PATH", "SCIENTIFIC_CONTRACT_BINDING_KEYS",
    "STABLE_HASH_OBJECTIVE_CONTRACT",
    "STABLE_HASH_OBJECTIVE_CONTRACT_DIGEST",
    "SUPERSEDED_CANONICAL_TIE_BREAK_STATUS",
    "SUPERSEDED_EXTERNAL_ENUMERATION_STATUS", "V1_FAILURE_RECEIPT_DIGEST",
    "V1_FAILURE_STATUS", "V2_BACKEND_DISPOSITION",
    "V2_BENCHMARK_RECEIPT_DIGEST", "V2_CONTRACT_DIGEST",
    "V2_SOURCE_REPOSITORY_COMMIT", "V2_TERMINAL_FAILURE_RECEIPT_DIGEST",
    "audit_runtime_outputs_absent", "build_coupling_report",
    "build_execution_amendment", "canonical_digest",
    "coupling_report_artifact_binding", "issue_coupling_report",
    "issue_execution_amendment", "load_coupling_report",
    "load_execution_amendment", "load_predecessor_lineage",
    "validate_coupling_report", "validate_execution_amendment",
    "validate_predecessor_lineage_bindings",
    "validate_preoutcome_input_bindings",
    "validate_scientific_contract_bindings",
]
