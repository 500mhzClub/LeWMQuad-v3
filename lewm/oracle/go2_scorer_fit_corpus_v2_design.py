"""Prospective full-bank scorer-fit corpus V2 design authority.

The six-of-twelve allocation is an immutable exact infeasibility.  This
module supersedes only that partial-subset layer and freezes the prospective
replacement in which every selected state receives candidates 0--11 once.
It contains no simulator, solver, branch, frame, latent, scorer, predictor,
checkpoint, or outcome reader.

Historical scientific authorities remain immutable.  The later preselection
source correction may reopen only the exact issued classification/design and
their frozen lineage for custody validation; it cannot replace either
payload.  Every authority is installed with ``O_EXCL``, file and directory
``fsync``, and read-only permissions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from fnmatch import fnmatchcase
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

from lewm.oracle import (
    go2_small_completion_global_execution_amendment_v1 as GLOBAL_AUTHORITY,
)


ROOT = Path(__file__).resolve().parents[2]

MASK_CLASSIFICATION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_rotation_mask_classification")
MASK_CLASSIFICATION_STATUS = (
    "PASS_ALL_OLD_ROTATION_CONDITIONS_PARTIAL_SUBSET_ALLOCATION_ONLY")
MASK_CLASSIFICATION_SELF_KEY = "rotation_mask_classification_digest"
DESIGN_SCHEMA = "go2_scorer_fit_corpus_v2_design_amendment"
DESIGN_STATUS = "ISSUED_PROSPECTIVE_FULL_TWELVE_CANDIDATE_BANK"
DESIGN_SELF_KEY = "scorer_fit_corpus_v2_design_digest"
SOURCE_CORRECTION_SCHEMA = (
    "go2_scorer_fit_corpus_v2_preselection_source_correction_v1")
SOURCE_CORRECTION_STATUS = (
    "ISSUED_PRESELECTION_SOURCE_CORRECTION_AFTER_REGISTERED_"
    "DEVELOPMENT_MANIFEST_ALIAS_FAILURE")
SOURCE_CORRECTION_SELF_KEY = (
    "scorer_fit_corpus_v2_source_correction_digest")

BRANCH_GENERATED_ROOT_RELATIVE_PATH = Path(
    ".generated/go2_branch_corpus_v1_2")
UTILITY_V2_ROOT_RELATIVE_PATH = Path(
    ".generated/go2_utility_scorer_fit_corpus_v2")
MANAGED_GENERATED_ROOTS = (
    BRANCH_GENERATED_ROOT_RELATIVE_PATH,
    Path(".generated/go2_utility_scorer_v1_2"),
    UTILITY_V2_ROOT_RELATIVE_PATH,
)
SCORER_FIT_RELATIVE_PATH = BRANCH_GENERATED_ROOT_RELATIVE_PATH / "scorer_fit"
MASK_CLASSIFICATION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_rotation_mask_classification.json"
)
DESIGN_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_design_amendment.json"
)
SOURCE_CORRECTION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE_PATH /
    "scorer_fit_corpus_v2_preselection_source_correction_v1.json"
)

ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT = (
    "76bc465cb33ef94d535b433c83660d94335bee00")
SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/encode_go2_branch_corpus_v1_2.py",
    "scripts/train_go2_utility_scorer_v1_2.py",
    "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)

TERMINAL_SOURCE_REPOSITORY_COMMIT = (
    "e1bdbe7adc15d0aa85f69ffb9e97fa198eb152c5")
ACTIVE_GLOBAL_AMENDMENT_DIGEST = (
    "f4cbc2e5e7baa1c4cebda3dfde0e5f9744aa50a7d1d1c5c53704338a1bb6f822")
GLOBAL_EXACT_MODEL_DIGEST = (
    "57770fe998cb7d7cb952b122077fe7dd8daeadce5f70fa3150c6f8641e5162b4")
EXACT_INFEASIBILITY_DIGEST = (
    "eb9d347fee6d4b498cf02a2c0af51483304d9489aa98050a8212e99c737135dc")
TERMINAL_RECEIPT_DIGEST = (
    "d1a17c289e3993f6cad30cc3ff1725246075881c292e930e776d8bcd7fb215d4")
GLOBAL_MODEL_PLAN_DIGEST = (
    "c638e18ce376ed1cfa263c042ed9e89428c0ac5283b7f0c05429ed63baa1d082")
MODEL_EXECUTION_PLAN_DIGEST = (
    "c50375e73e3396561808619b3f882367f8befcdb88b99708364014cfb49c0d98")

ACTIVE_GLOBAL_AMENDMENT_BINDING = {
    "path": str(
        SCORER_FIT_RELATIVE_PATH /
        "small_completion_global_exact_preplan_integration_correction_v1.json"),
    "schema": GLOBAL_AUTHORITY.PREPLAN_INTEGRATION_CORRECTION_SCHEMA,
    "self_digest_key": GLOBAL_AUTHORITY.AMENDMENT_SELF_KEY,
    "self_digest": ACTIVE_GLOBAL_AMENDMENT_DIGEST,
    "raw_sha256": (
        "1c70ff5db95388dcc5dc5af004e7b0912091380bf7d81a98efa459cc6433f576"),
    "byte_count": 282_516,
    "source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
}
GLOBAL_MODEL_PLAN_BINDING = {
    "path": str(
        SCORER_FIT_RELATIVE_PATH /
        "small_completion_global_exact_model_plan_v1.json"),
    "schema": "go2_small_completion_global_exact_runner_plan_v1",
    "self_digest_key": "global_exact_model_plan_digest",
    "self_digest": GLOBAL_MODEL_PLAN_DIGEST,
    "raw_sha256": (
        "9d5fed3d851cfa7944dbccc53eaec4265b784042e590f471be4d1726c46ac9dd"),
    "byte_count": 15_420,
    "source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
}
GLOBAL_TERMINAL_INFEASIBILITY_BINDING = {
    "path": str(
        SCORER_FIT_RELATIVE_PATH /
        "small_completion_global_exact_terminal_infeasibility_v1.json"),
    "schema": "go2_small_completion_global_exact_runner_terminal_v1",
    "self_digest_key": "global_exact_terminal_result_digest",
    "self_digest": TERMINAL_RECEIPT_DIGEST,
    "raw_sha256": (
        "83724fa4ae370efd610d50be21a7424d8f7ab445c7679efbf5ff6af698684efb"),
    "byte_count": 5_108,
    "source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
}

CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9")
ACTIVE_SELECTOR_DIGEST = (
    "8c1d9f5ff1430fda6d9d80512afdba3070c78301befa57604aafcad9cb5c880b")
ORACLE_V1_2_DIGEST = (
    "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4")
SIX_OF_TWELVE_SUPERSESSION = (
    "SUPERSEDED_PRE_OUTCOME_SIX_OF_TWELVE_ALLOCATION_EXACTLY_INFEASIBLE")

FAMILIES = (
    "large_enclosed_maze", "local_composite_motifs", "loop_alias_stress",
    "medium_enclosed_maze", "open_obstacle_field", "rough_local_dynamics",
    "small_enclosed_maze", "visual_sensor_stress",
)
STRATA = ("general", "safety_enriched", "completion_enriched")
SPLIT_ROLES = ("fit", "calibration")
CANDIDATE_INDICES = tuple(range(12))
STATE_COUNT = 120
ASSIGNMENT_COUNT = 1_440

FULL_BANK_COUNT_CONTRACT: dict[str, Any] = {
    "schema": "go2_scorer_fit_corpus_v2_full_bank_count_contract",
    "state_count": STATE_COUNT,
    "family_count": 8,
    "states_per_family": 15,
    "strata": list(STRATA),
    "states_per_family_stratum": 5,
    "fit_state_count": 96,
    "calibration_state_count": 24,
    "fit_per_family_stratum": 4,
    "calibration_per_family_stratum": 1,
    "candidate_indices": list(CANDIDATE_INDICES),
    "candidate_count": 12,
    "candidates_per_state": 12,
    "assignments_total": ASSIGNMENT_COUNT,
    "per_candidate": {
        "overall": 120,
        "fit": 96,
        "calibration": 24,
        "per_stratum": 40,
        "per_family": 15,
        "fit_per_family": 12,
        "calibration_per_family": 3,
        "per_family_stratum": 5,
    },
    "unordered_candidate_pair_cooccurrence": 120,
    "candidate_by_family_distributions_identical": True,
    "candidate_by_stratum_distributions_identical": True,
    "candidate_by_split_distributions_identical": True,
    "candidate_by_goal_type_distributions_identical": True,
}

COMPLETION_ORDER_DOMAIN = (
    "LEWM_GO2_SCORER_FIT_CORPUS_V2_FULL_BANK_SMALL_COMPLETION_ORDER_V1")
COMPLETION_ORDERING_CONTRACT: dict[str, Any] = {
    "schema": "go2_scorer_fit_corpus_v2_small_completion_order_contract",
    "domain_separation_utf8": COMPLETION_ORDER_DOMAIN,
    "active_selector_contract_digest": ACTIVE_SELECTOR_DIGEST,
    "preimage": (
        "domain_separation_utf8 || 0x00 || active_selector_contract_digest || "
        "0x00 || compact_canonical_complete_structural_state_identity_json || "
        "0x00 || compact_canonical_designated_goal_identity_json"
    ),
    "hash": "SHA-256",
    "order": (
        "ascending SHA-256, then ascending complete structural identity JSON "
        "bytes as the final structural tie-break"
    ),
    "selection": {
        "calibration": (
            "first calibration-eligible full-bank-valid identity in order"),
        "fit": (
            "first four distinct fit-eligible full-bank-valid identities in "
            "the same order, excluding the calibration scene"),
        "failure": "fewer than one calibration and four fit scenes",
    },
    "candidate_outcome_or_downstream_metric_fields": [],
    "optimisation_or_solver": False,
}

PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS: dict[str, Any] = {
    "selection_digest": (
        "c20b4feceb865b25fb24e5534be5f84d14a5795d069ca2b0c14cd3f23d8ca9dd"),
    "scorer_fit_allocation_design_digest": (
        "a587b1de264dfb54176aa231e5183ae4b7b4229bbf65c02d62438f86af5e7116"),
    "candidate_allocator_contract_digest": (
        "bb2d9956947be64985f15970dc30f9f0e37cda8012f7c7f5da8808c5d601de5e"),
    "candidate_allocation_amendment_digest": (
        "4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc"),
    "pre_identity_allocation_validation_digest": (
        "46efa42e3bdcad6df6cdcd4e404c2e8a796a9a331109a433cfbfffcfa18bf60d"),
    "invalid_scorer_identity_exclusion_digest": (
        "6d644c34b822fb5fb8e30906875047d1677aa730c2db584470cabdbe8bf6abc3"),
    "state_selector_amendment_digest": ACTIVE_SELECTOR_DIGEST,
    "state_selector_feasibility_receipt_digest": (
        "0e2013f40f506da6485bb5e2fe5a3108595243aeb9141a6437f8cac023642482"),
    "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
    "clean_source_launch_receipt_digest": (
        "1656fbd691a63a63338bcd2d4707275ac67fbcfdc90c8a2afe5504876a543e3e"),
    "source_repository_commit": (
        "d9d129e2bbea8519f7ed3186f3cfb3c661baba04"),
    "clean_source_binding_digest": (
        "563dfbe7f8a08b4cc059ca2f3f495f201fcf9693529ed783d79bb3dcb37663ee"),
    "bound_implementations_digest": (
        "4b5c70bcd35af13d284c55cb045af7c0b6d592be6e8e1f78a3a9b4293c0ac5f7"),
    "scorer_contract_artifact_digest": (
        "ed038cc3b773705c5ae1f4a1bc21d420ed43f0c6d7a408a5c911e250960ef78f"),
    "mixed_precontract_disposition_receipt_digest": (
        "fef1a98980bc41d63434367f518ff2876dbcf93afbea52ff8f555300d3220604"),
    "progress_contract_digest": (
        "840328d918f446bad1a5855e72f13f8937fc9a42eafd87818bf8cd94305e2c3d"),
    "safety_contract_digest": (
        "5cf4572be2490c1b6f748abc704fff3a3c15fb1ea8dc060e49314e2bbaf01e0f"),
    "oracle_v1_2_digest": ORACLE_V1_2_DIGEST,
    "scorer_contract_v1_2_digest": (
        "f268763ed9365205cd0b0001c4527afbf5e5d948846dbb891225a48acb74113a"),
    "boundary_digest": (
        "1faae05f843e6f02f0f354c63ab3bcad9404111140146b1355d025da3d0c7a92"),
    "render_contract_digest": (
        "2faa22e3b10a2c4199bdabdbc0ed0e1ff9c7c4ac48bb489daeb0fd70d5b65c17"),
    "preprocess_contract_digest": (
        "2688ca405ed7e8bb86e82f1d111b7b865466f4d497b973a04a52af846b5da6a9"),
    "textured_v03_renderer_contract_digest": (
        "df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b"),
    "preprocessing_digest": (
        "8e6aa177b094ea91d27b3c91bcd8f01835b8be5fc51796d145314982ea930fe5"),
    "target_encoder_digest": (
        "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"),
    "target_encoder_checkpoint_sha256": (
        "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"),
    "genesis_backend": "cpu",
}

FROZEN_PREDICTOR_QUALIFICATION: dict[str, Any] = {
    "commit": "ee47b47e7964c16360f265c4cfbe7f8181d16402",
    "stage_a_identity_digest": (
        "ce2cbbe8dab9a89ad6f85d16c56a9d712d791c8bbfd8925a8f01efc0c039705a"),
    "branch_corpus_digest": (
        "f84eb3271f1a3b7052bbf2e84240453e84772b0a530e60ec47f723a44e2e10e9"),
    "target_latent_index_digest": (
        "861285ec9c8fc6c92c6f3a31cade0f031172bf6818d76d1899634a60c7e5c291"),
    "direct_fidelity_and_retrieval_result_digest": (
        "3b5c500b4b1326056ce18c6276d7842f4230faec36f8f29cc65945f54527bbcb"),
    "occupancy_result_digest": (
        "09dc413d9ce30c2cb19c99e93eeaad410983a7f53575387bc6694f3844a070d6"),
    "occupancy_gate_digest": (
        "4bf9a92144fa728d953c9dffebb235c9b476ded59d7462a107fe2e6ade0894e4"),
    "result_report": {
        "path": (
            "docs/lewm_go2_counterfactual_predictor_qualification_v1_2_"
            "result_2026-08-11.md"),
        "raw_sha256": (
            "14a4276b1caee817a7097eb78f187c9e38b6d4c7eb70a16f88ec32ccd223894b"),
        "byte_count": 28_437,
    },
    "modified_or_rerun": False,
}

PRIOR_PREOUTCOME_FAILURE_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "role": "candidate_allocation_infeasibility",
        "path": (
            "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "allocation_failure_2026-08-11.json"),
        "schema": "go2_shared_utility_scorer_v1_2_preoutcome_failure_receipt",
        "semantic_status": "BLOCKED_PRE_OUTCOME_FROZEN_CONTRACT_INCONSISTENCY",
        "self_digest_canonicalization": "COMPACT_CANONICAL_JSON",
        "self_digest_key": "failure_receipt_digest",
        "self_digest": (
            "550c52f9a3ff04f8a564f6f28e75e9d36fc8bc0f73da4795b95dedc3ad2e3cab"),
        "raw_sha256": (
            "3e224158d43a4e75fc7a60436feaeb00cd538a5fabfae5a92983f7ede612df99"),
        "byte_count": 8_034,
    },
    {
        "role": "state_selection_contract_infeasibility",
        "path": (
            "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "state_selection_failure_2026-08-11.json"),
        "schema": (
            "lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "state_selection_failure_v1"),
        "semantic_status": "BLOCKED_PRE_OUTCOME_SCIENTIFIC_CONTRACT_INFEASIBILITY",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": "failure_report_digest",
        "self_digest": (
            "47c2bcc7cfaf79b328cd5a1cf2823554f2553fc419e020559fde1351df2ca75f"),
        "raw_sha256": (
            "26e372229af4bcde8242062b0b9f9d9c5ba85bdb073537812e9395a608c2455d"),
        "byte_count": 8_586,
    },
    {
        "role": "selector_feasibility_amendment_v2_failure_report",
        "path": (
            "docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "selector_feasibility_failure_2026-08-12.json"),
        "schema": (
            "lewm_go2_shared_utility_scorer_v1_2_preoutcome_"
            "selector_feasibility_failure_v1"),
        "semantic_status": "TERMINAL_PRE_OUTCOME_SELECTOR_FEASIBILITY_FAILURE",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": "failure_report_digest",
        "self_digest": (
            "81637cdf3889dc0856ea97aee9a644f182855ef49c4e466eee3f8aed4134a0b8"),
        "raw_sha256": (
            "db2e025fe71164943bb214a602dfdeb249d629de2420aa758bb97717c8974b49"),
        "byte_count": 28_618,
    },
    {
        "role": "selector_feasibility_v1_failed_census",
        "path": str(
            SCORER_FIT_RELATIVE_PATH / "state_selector_feasibility_receipt.json"),
        "schema": "go2_scorer_fit_state_selector_feasibility_receipt_v1",
        "semantic_status": "FAIL_OUTCOME_FREE_SELECTOR_FEASIBILITY",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": "state_selector_feasibility_receipt_digest",
        "self_digest": (
            "2310c3d1b138b605fda483b39cbd4775479cbcc502a4e3707e7a8670457f54d7"),
        "raw_sha256": (
            "28e852792b5de24b2d008c5bb3f95521da668927e555deb9eb3c508bb6b0e59f"),
        "byte_count": 1_194_515,
    },
    {
        "role": "preserved_state_precontract_revalidation_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "preserved_state_precontract_revalidation_reachability_v2.json"),
        "schema": (
            "go2_scorer_fit_preserved_state_precontract_revalidation_"
            "reachability_v2"),
        "semantic_status": "FAIL_PRECONTRACT_IDENTITY_REVALIDATION",
        "self_digest_canonicalization": "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS",
        "self_digest_key": (
            "preserved_state_precontract_revalidation_receipt_digest"),
        "self_digest": (
            "0316e7f9b8462670eabe76da5fefc003274b4d08355373d14e1100cd6165e8e3"),
        "raw_sha256": (
            "2e49ed6e47caa98ed30ce45bddccd07cd3fad4c1950e5c6ff3fe051d0307de25"),
        "byte_count": 334_260,
    },
)

IMMUTABLE_V1_V2_FAILURE_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "role": "v1_immutable_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_prefix_benchmark_v1.json"),
        "schema": "go2_parallel_small_completion_search_v1_benchmark_receipt",
        "self_digest_key": "benchmark_receipt_digest",
        "self_digest": GLOBAL_AUTHORITY.V1_FAILURE_RECEIPT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V1_FAILURE_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V1_FAILURE_BYTE_COUNT,
    },
    {
        "role": "v2_contract",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_prefix_benchmark_v2_contract.json"),
        "schema": "go2_parallel_small_completion_benchmark_v2_contract",
        "self_digest_key": "benchmark_v2_contract_digest",
        "self_digest": GLOBAL_AUTHORITY.V2_CONTRACT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V2_CONTRACT_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V2_CONTRACT_BYTE_COUNT,
    },
    {
        "role": "v2_benchmark_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_prefix_benchmark_v2.json"),
        "schema": "go2_parallel_small_completion_search_v2_benchmark_receipt",
        "self_digest_key": "benchmark_receipt_digest",
        "self_digest": GLOBAL_AUTHORITY.V2_BENCHMARK_RECEIPT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V2_BENCHMARK_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V2_BENCHMARK_BYTE_COUNT,
    },
    {
        "role": "v2_terminal_failure",
        "path": str(
            SCORER_FIT_RELATIVE_PATH /
            "small_completion_parallel_terminal_failure_v2.json"),
        "schema": "go2_parallel_small_completion_benchmark_v2_terminal_failure",
        "self_digest_key": "terminal_failure_receipt_digest",
        "self_digest": GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_RECEIPT_DIGEST,
        "raw_sha256": GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_RAW_SHA256,
        "byte_count": GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_BYTE_COUNT,
    },
)

SOURCE_SPECS = (
    ("lewm/oracle/go2_scorer_fit_corpus_v2_design.py", "v2_design_authority"),
    ("lewm/oracle/go2_small_completion_global_execution_amendment_v1.py",
     "immutable_exact_infeasibility_lineage_authority"),
    ("lewm/oracle/go2_candidate_allocation_v1_2.py",
     "superseded_partial_subset_allocator_lineage"),
    ("lewm/oracle/go2_scorer_state_selector_amendment_v2.py",
     "active_outcome_free_state_selector"),
    ("lewm/oracle/go2_branch_oracle_v1_2.py", "frozen_oracle_v1_2"),
    ("scripts/dev_action_slew_reconstruction_v1.py",
     "frozen_full_bank_completion_reachability_slew"),
    ("lewm/oracle/go2_scorer_contract_v1_2.py", "frozen_scorer_science"),
    ("lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
     "full_bank_v2_successor_scorer_contract"),
    ("scripts/build_go2_branch_corpus_v1_2.py", "v2_identity_and_branch_builder"),
    ("scripts/encode_go2_branch_corpus_v1_2.py", "frozen_target_encoder_route"),
    ("scripts/train_go2_utility_scorer_v1_2.py", "frozen_training_route"),
    ("scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
     "frozen_development_transfer_route"),
    ("scripts/run_go2_scorer_fit_full_bank_v2.py", "v2_execution_runner"),
)
EXPECTED_SOURCE_PATHS = tuple(path for path, _role in SOURCE_SPECS)

V2_PREOUTCOME_ARTIFACT_PATHS = (
    SCORER_FIT_RELATIVE_PATH / "state_manifest_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_assignment_manifest_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_preoutcome_state_revalidation_v2.json",
    SCORER_FIT_RELATIVE_PATH / "state_shard_small_enclosed_maze_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_small_completion_selection_v2.json",
    SCORER_FIT_RELATIVE_PATH / "full_bank_preoutcome_feasibility_failure_v2.json",
)
V2_SUCCESSOR_CONTRACT_PATH = (
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_fit_corpus_v2_scorer_contract.json")
V2_ALWAYS_ABSENT_PATHS = (
    BRANCH_GENERATED_ROOT_RELATIVE_PATH / "final_eval/state_manifest.json",
)
V2_RUNTIME_OUTPUT_PATHS = (
    SCORER_FIT_RELATIVE_PATH / "branch_rows_v2.jsonl",
    SCORER_FIT_RELATIVE_PATH / "corpus_receipt_v2.json",
    SCORER_FIT_RELATIVE_PATH / "smoke_branch_receipt_v2.json",
    SCORER_FIT_RELATIVE_PATH / "latents_index_v2.json",
    SCORER_FIT_RELATIVE_PATH / "smoke_encoding_receipt_v2.json",
    SCORER_FIT_RELATIVE_PATH / "encoding_invocation_summary_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "label_distributions_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "completion_degeneracy_failure_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "no_latent_baseline_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "no_latent_baseline_v2.receipt.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_package_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "scorer_package_receipt_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH / "failed_scorer_v2.pt",
    UTILITY_V2_ROOT_RELATIVE_PATH / "qualification_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/development_transfer_spec_v2.json",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/result_v2.json",
)
# Compatibility aggregate for callers that need the complete pre-design
# absence surface.  Phase-aware code should use
# :func:`audit_v2_runtime_outputs_absent`.
V2_FUTURE_OUTPUT_PATHS = (
    *V2_PREOUTCOME_ARTIFACT_PATHS,
    *V2_ALWAYS_ABSENT_PATHS,
    V2_SUCCESSOR_CONTRACT_PATH,
    *V2_RUNTIME_OUTPUT_PATHS,
)
V2_FUTURE_OUTPUT_DIRECTORIES = (
    SCORER_FIT_RELATIVE_PATH / "row_records_v2",
    SCORER_FIT_RELATIVE_PATH / "frames_v2",
    SCORER_FIT_RELATIVE_PATH / "latents_v2/context",
    SCORER_FIT_RELATIVE_PATH / "latents_v2/horizon",
    SCORER_FIT_RELATIVE_PATH / "invalid_attempts_v2",
    SCORER_FIT_RELATIVE_PATH / "superseded_receipts_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH / "initialisations_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH / "training_v2",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/score_shards",
    UTILITY_V2_ROOT_RELATIVE_PATH /
    "counterfactual_development_transfer_v2/invalid_attempts",
)
V2_FUTURE_OUTPUT_GLOBS = (
    SCORER_FIT_RELATIVE_PATH / "branch_summary_v2_*.json",
)

_ROTATION_IDS = tuple(
    row["constraint_id"] for row in GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY
    if row["rotation_subset_identity_reference"]
)
EXPECTED_ROTATION_CONSTRAINT_IDS = (
    "ROTATION_BLOCK_CATALOGUE_12_OF_6",
    "ROTATION_BLOCK_FORWARD_TURNING_REVERSE_SEMANTICS",
    "EXACTLY_ONE_ROTATION_PER_STATE",
    "FIT_FAMILY_STRATUM_CANDIDATE_EXACT_2",
    "CALIBRATION_STRATUM_CANDIDATE_EXACT_4",
    "CALIBRATION_FAMILY_CANDIDATE_1_TO_2",
    "GOAL_TYPE_CANDIDATE_FLOOR_CEILING",
    "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
    "ALL_40_COMPLETION_MASKS_PASS",
    "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
    "FIRST_PASSING_SCENE_COMBINATION",
    "GLOBAL_CANDIDATE_EXACT_60",
    "FIT_CALIBRATION_CANDIDATE_EXACT_48_12",
    "STRATUM_CANDIDATE_EXACT_20",
    "FAMILY_CANDIDATE_BALANCE_7_8",
    "FAMILY_STRATUM_CANDIDATE_BALANCE_2_3",
    "REVERSE_CANDIDATE_IN_60_DISTINCT_STATES",
    "ASSIGNMENT_BLOCK_CONTINGENCY_AND_PAIRWISE_INTEGRITY",
)
COUPLING_INVENTORY_DIGEST = (
    "6a195d65d826a883057aa6ab5c535c48aefdc6457dc7ee427b41b12a4e1f7dce")
ROTATION_CONSTRAINT_INVENTORY_DIGEST = (
    "81cc9ff06fdca5359c9dab060625599298257026246c75c312761b16f2dddad3")

_ALLOCATION_BALANCE_IDS = frozenset({
    "ROTATION_BLOCK_CATALOGUE_12_OF_6",
    "ROTATION_BLOCK_FORWARD_TURNING_REVERSE_SEMANTICS",
    "EXACTLY_ONE_ROTATION_PER_STATE",
    "FIT_FAMILY_STRATUM_CANDIDATE_EXACT_2",
    "CALIBRATION_STRATUM_CANDIDATE_EXACT_4",
    "CALIBRATION_FAMILY_CANDIDATE_1_TO_2",
    "GOAL_TYPE_CANDIDATE_FLOOR_CEILING",
    "GLOBAL_CANDIDATE_EXACT_60",
    "FIT_CALIBRATION_CANDIDATE_EXACT_48_12",
    "STRATUM_CANDIDATE_EXACT_20",
    "FAMILY_CANDIDATE_BALANCE_7_8",
    "FAMILY_STRATUM_CANDIDATE_BALANCE_2_3",
    "REVERSE_CANDIDATE_IN_60_DISTINCT_STATES",
    "ASSIGNMENT_BLOCK_CONTINGENCY_AND_PAIRWISE_INTEGRITY",
})
_SUBSET_REACHABILITY_IDS = frozenset({
    "COMPLETION_ASSIGNED_ROTATION_ELIGIBILITY",
    "ALL_40_COMPLETION_MASKS_PASS",
})
_EXECUTION_POLICY_IDS = frozenset({
    "CANONICAL_ROTATION_VECTOR_BY_IDENTITY_ORDER",
    "FIRST_PASSING_SCENE_COMBINATION",
})

_HEX = frozenset("0123456789abcdef")

PREDECESSOR_VALIDATION_PROJECTION: dict[str, Any] = {
    "schema": "go2_scorer_fit_corpus_v2_predecessor_validation",
    "active_global_amendment_exact_raw_self_schema_source_validated": True,
    "global_exact_model_plan_exact_raw_self_schema_source_validated": True,
    "terminal_infeasibility_exact_raw_self_schema_source_validated": True,
    "terminal_nested_model_and_infeasibility_digests_validated": True,
    "immutable_v1_v2_failure_count_validated": 4,
    "prior_preoutcome_failure_count_validated": len(
        PRIOR_PREOUTCOME_FAILURE_BINDINGS),
    "frozen_predictor_qualification_report_raw_binding_validated": True,
    "historical_generated_artifacts_reopened_for_lineage_custody": True,
    "historical_generated_artifacts_used_to_classify_rotation_conditions": False,
    "historical_generated_artifacts_used_to_select_states": False,
    "solve_or_solver_invocation": False,
    "candidate_outcome_or_branch_label_read": False,
    "frame_latent_scorer_metric_or_predictor_score_read": False,
}


class ScorerFitCorpusV2DesignError(RuntimeError):
    """The prospective design, classification, or custody binding changed."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScorerFitCorpusV2DesignError(
            "value is not compact canonical JSON") from exc


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=True,
            allow_nan=False,
        ) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScorerFitCorpusV2DesignError(
            "value is not canonical JSON") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _without(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _is_hex(value: Any, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(character in _HEX for character in value))


def _validate_source_bindings(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(SOURCE_SPECS):
        raise ScorerFitCorpusV2DesignError("source binding coverage changed")
    result: list[dict[str, Any]] = []
    for (path, role), raw in zip(SOURCE_SPECS, rows, strict=True):
        if not isinstance(raw, Mapping):
            raise ScorerFitCorpusV2DesignError("source binding is malformed")
        row = dict(raw)
        if (set(row) != {"path", "role", "byte_count", "sha256"}
                or row.get("path") != path or row.get("role") != role
                or isinstance(row.get("byte_count"), bool)
                or not isinstance(row.get("byte_count"), int)
                or row["byte_count"] <= 0
                or not _is_hex(row.get("sha256"), 64)):
            raise ScorerFitCorpusV2DesignError("source binding changed")
        result.append(row)
    return result


def _changed_source_paths(
        historical: Sequence[Mapping[str, Any]],
        corrected: Sequence[Mapping[str, Any]],
        ) -> list[str]:
    before = {str(row["path"]): dict(row) for row in historical}
    after = {str(row["path"]): dict(row) for row in corrected}
    if set(before) != set(after):
        raise ScorerFitCorpusV2DesignError(
            "source-correction closure path set changed")
    return sorted(
        path for path in before
        if (before[path]["sha256"], before[path]["byte_count"])
        != (after[path]["sha256"], after[path]["byte_count"]))


def _expected_absence_rows(*, phase: str) -> list[dict[str, Any]]:
    if phase not in {"design", "successor_contract", "post_contract_pre_branch"}:
        raise ScorerFitCorpusV2DesignError("unknown V2 absence-audit phase")
    exact_paths = list(V2_ALWAYS_ABSENT_PATHS)
    exact_paths.extend(V2_RUNTIME_OUTPUT_PATHS)
    if phase == "design":
        exact_paths.extend(V2_PREOUTCOME_ARTIFACT_PATHS)
    if phase in {"design", "successor_contract"}:
        exact_paths.append(V2_SUCCESSOR_CONTRACT_PATH)
    exact_paths.extend(Path(path) for path in
                       GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_EXACT_FILE_PATHS)
    directory_paths = list(V2_FUTURE_OUTPUT_DIRECTORIES)
    directory_paths.extend(Path(path) for path in
                           GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_DIRECTORY_ROOTS)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, paths in (("file", exact_paths), ("directory", directory_paths)):
        for relative in paths:
            key = str(relative)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"path": key, "expected_kind": kind, "exists": False})
    patterns = list(V2_FUTURE_OUTPUT_GLOBS)
    patterns.extend(Path(path) for path in
                    GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_GLOB_PATTERNS)
    for pattern in patterns:
        rows.append({
            "path": str(pattern), "expected_kind": "glob", "exists": False,
        })
    return rows


def _validate_absence_projection(value: Any, *, phase: str) -> list[dict[str, Any]]:
    expected = _expected_absence_rows(phase=phase)
    if not isinstance(value, list) or value != expected:
        raise ScorerFitCorpusV2DesignError(
            "source-correction output-absence projection changed")
    return copy.deepcopy(expected)


def _predecessor_lineage() -> dict[str, Any]:
    return {
        "schema": "go2_scorer_fit_corpus_v2_exact_infeasibility_lineage",
        "terminal_source_repository_commit": TERMINAL_SOURCE_REPOSITORY_COMMIT,
        "active_global_amendment": copy.deepcopy(ACTIVE_GLOBAL_AMENDMENT_BINDING),
        "global_model_plan": copy.deepcopy(GLOBAL_MODEL_PLAN_BINDING),
        "terminal_infeasibility": copy.deepcopy(
            GLOBAL_TERMINAL_INFEASIBILITY_BINDING),
        "active_global_amendment_digest": ACTIVE_GLOBAL_AMENDMENT_DIGEST,
        "global_exact_model_digest": GLOBAL_EXACT_MODEL_DIGEST,
        "exact_infeasibility_digest": EXACT_INFEASIBILITY_DIGEST,
        "terminal_receipt_digest": TERMINAL_RECEIPT_DIGEST,
        "model_execution_plan_digest": MODEL_EXECUTION_PLAN_DIGEST,
        "terminal_status": "INFEASIBLE",
        "six_of_twelve_design_closed_exactly_infeasible": True,
        "six_of_twelve_model_retried_or_reinterpreted": False,
        "candidate_outcomes_consumed_at_proof": False,
        "branch_labels_read_at_proof": False,
        "frames_or_latents_created_at_proof": False,
        "scorer_or_predictor_accessed_at_proof": False,
        "scientific_conditions_relaxed": False,
        "immutable_v1_failure": {
            "status": GLOBAL_AUTHORITY.V1_FAILURE_STATUS,
            "receipt_digest": GLOBAL_AUTHORITY.V1_FAILURE_RECEIPT_DIGEST,
        },
        "immutable_v2_failure": {
            "source_repository_commit": GLOBAL_AUTHORITY.V2_SOURCE_REPOSITORY_COMMIT,
            "contract_digest": GLOBAL_AUTHORITY.V2_CONTRACT_DIGEST,
            "benchmark_receipt_digest": (
                GLOBAL_AUTHORITY.V2_BENCHMARK_RECEIPT_DIGEST),
            "terminal_failure_receipt_digest": (
                GLOBAL_AUTHORITY.V2_TERMINAL_FAILURE_RECEIPT_DIGEST),
            "backend_disposition": GLOBAL_AUTHORITY.V2_BACKEND_DISPOSITION,
        },
        "immutable_v1_v2_failure_bindings": copy.deepcopy(
            list(IMMUTABLE_V1_V2_FAILURE_BINDINGS)),
        "prior_preoutcome_failure_bindings": copy.deepcopy(
            list(PRIOR_PREOUTCOME_FAILURE_BINDINGS)),
        "all_invalid_attempt_and_preoutcome_lineage_transitively_bound_by": (
            ACTIVE_GLOBAL_AMENDMENT_DIGEST),
        "frozen_predictor_qualification": copy.deepcopy(
            FROZEN_PREDICTOR_QUALIFICATION),
    }


def _validate_predecessor_lineage(value: Any) -> dict[str, Any]:
    expected = _predecessor_lineage()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ScorerFitCorpusV2DesignError(
            "immutable exact-infeasibility lineage changed")
    return expected


def _validate_predecessor_validation_projection(value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping)
            or dict(value) != PREDECESSOR_VALIDATION_PROJECTION):
        raise ScorerFitCorpusV2DesignError(
            "historical predecessor validation projection changed")
    return copy.deepcopy(PREDECESSOR_VALIDATION_PROJECTION)


def _classification_reason(constraint_id: str) -> tuple[str, str]:
    if constraint_id in _ALLOCATION_BALANCE_IDS:
        return (
            "RETIRED_PARTIAL_SUBSET_DECISION_OR_BALANCE",
            "The condition exists only because six candidates were selected "
            "from twelve; assigning all twelve makes it either inapplicable "
            "or an algebraic identity under the V2 count contract.",
        )
    if constraint_id in _SUBSET_REACHABILITY_IDS:
        return (
            "REPLACED_BY_FULL_BANK_L_MAX_STATE_REVALIDATION",
            "The old mask compares completion enrichment with L_max of one "
            "allocated six-candidate subset.  It does not establish branch "
            "executability.  V2 recomputes the unchanged selector predicate "
            "with L_max over candidate indices 0--11.",
        )
    if constraint_id in _EXECUTION_POLICY_IDS:
        return (
            "RETIRED_NONSCIENTIFIC_ALLOCATION_EXECUTION_POLICY",
            "The condition canonicalizes or searches scene/rotation choices; "
            "V2 has no rotation choice or scene/rotation optimization.",
        )
    raise ScorerFitCorpusV2DesignError(
        f"unclassified old rotation condition: {constraint_id}")


def build_rotation_mask_classification(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        predecessor_validation: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the closed source-only classification; perform no file access."""

    if not _is_hex(source_repository_commit, 40):
        raise ScorerFitCorpusV2DesignError("source commit is malformed")
    sources = _validate_source_bindings(list(source_bindings))
    predecessor_projection = _validate_predecessor_validation_projection(
        predecessor_validation)
    if (_ROTATION_IDS != EXPECTED_ROTATION_CONSTRAINT_IDS
            or canonical_digest(list(GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY))
            != COUPLING_INVENTORY_DIGEST
            or canonical_digest([
                row for row in GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY
                if row["rotation_subset_identity_reference"]
            ]) != ROTATION_CONSTRAINT_INVENTORY_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "immutable coupling constraint inventory changed")
    rows: list[dict[str, Any]] = []
    by_id = {row["constraint_id"]: row
             for row in GLOBAL_AUTHORITY.CONSTRAINT_INVENTORY}
    for constraint_id in EXPECTED_ROTATION_CONSTRAINT_IDS:
        disposition, rationale = _classification_reason(constraint_id)
        old = by_id[constraint_id]
        rows.append({
            "constraint_id": constraint_id,
            "classification": "PARTIAL_SUBSET_ALLOCATION_ONLY",
            "v2_disposition": disposition,
            "origin": old["origin"],
            "source_symbols": list(old["source_symbols"]),
            "old_exact_rule": old["exact_rule"],
            "old_scientific_status": old["scientific_status"],
            "rationale": rationale,
        })
    payload: dict[str, Any] = {
        "schema": MASK_CLASSIFICATION_SCHEMA,
        "status": MASK_CLASSIFICATION_STATUS,
        "source_repository_commit": source_repository_commit,
        "source_bindings": sources,
        "source_binding_set_digest": canonical_digest(sources),
        "coupling_report_binding": copy.deepcopy(
            GLOBAL_AUTHORITY.ORIGINAL_COUPLING_REPORT_ARTIFACT_BINDING),
        "coupling_inventory_digest": COUPLING_INVENTORY_DIGEST,
        "rotation_constraint_inventory_digest": (
            ROTATION_CONSTRAINT_INVENTORY_DIGEST),
        "predecessor_validation": predecessor_projection,
        "classification_values": [
            "PARTIAL_SUBSET_ALLOCATION_ONLY",
            "TRUE_BRANCH_EXECUTION_REQUIREMENT",
        ],
        "conditions": rows,
        "counts": {
            "old_rotation_related_condition_count": len(rows),
            "partial_subset_allocation_only_count": len(rows),
            "true_branch_execution_requirement_count": 0,
        },
        "true_branch_execution_test": {
            "qualifying_reasons": [
                "candidate branch cannot execute from the state",
                "required planning-time input is unavailable",
                "canonical snapshot contract is violated",
                "required oracle or latent record cannot be produced",
            ],
            "matching_old_rotation_condition_ids": [],
            "all_twelve_candidates_use_the_same_executor": True,
            "candidate_indices_accepted_by_executor": list(CANDIDATE_INDICES),
        },
        "retained_non_rotation_completion_requirements": {
            "finite_graph_reachability": True,
            "absolute_body_bearing_deg_max": 75.0,
            "snapshot_goal_claimed": False,
            "snapshot_task_completed": False,
            "snapshot_terminated": False,
            "snapshot_truncated": False,
            "canonical_snapshot_boundary": True,
            "all_oracle_and_scorer_inputs_available": True,
            "completion_radius_m": 0.75,
            "horizon_blocks": 4,
            "horizon_ticks": 20,
            "full_bank_l_max_candidate_indices": list(CANDIDATE_INDICES),
            "branch_execution_used_for_revalidation": False,
        },
        "outcome_access": {
            "candidate_outcome_read": False,
            "branch_label_read": False,
            "frame_or_latent_read": False,
            "scorer_metric_or_predictor_score_read": False,
            "historical_receipts_reopened_for_exact_lineage_custody": True,
            "historical_receipts_used_for_classification": False,
        },
    }
    payload[MASK_CLASSIFICATION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_rotation_mask_classification(
        payload: Mapping[str, Any], *, validate_live_source: bool = False,
        root: Path = ROOT) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ScorerFitCorpusV2DesignError("mask classification is not a mapping")
    value = copy.deepcopy(dict(payload))
    if (value.get("schema") != MASK_CLASSIFICATION_SCHEMA
            or value.get("status") != MASK_CLASSIFICATION_STATUS
            or value.get(MASK_CLASSIFICATION_SELF_KEY)
            != canonical_digest(_without(value, MASK_CLASSIFICATION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError("mask classification binding changed")
    expected = build_rotation_mask_classification(
        source_repository_commit=str(value.get("source_repository_commit", "")),
        source_bindings=value.get("source_bindings", []),
        predecessor_validation=value.get("predecessor_validation", {}))
    if value != expected:
        raise ScorerFitCorpusV2DesignError("mask classification is not exact")
    if validate_live_source:
        commit, sources = clean_source_authority(root=root)
        if commit != value["source_repository_commit"] or sources != value["source_bindings"]:
            raise ScorerFitCorpusV2DesignError(
                "live source differs from mask classification")
    return value


def rotation_mask_classification_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    value = validate_rotation_mask_classification(payload)
    if raw != _pretty_json_bytes(value):
        raise ScorerFitCorpusV2DesignError(
            "mask classification raw bytes changed")
    return {
        "path": str(MASK_CLASSIFICATION_RELATIVE_PATH),
        "schema": MASK_CLASSIFICATION_SCHEMA,
        "self_digest_key": MASK_CLASSIFICATION_SELF_KEY,
        "self_digest": value[MASK_CLASSIFICATION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": value["source_repository_commit"],
    }


def _validate_classification_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScorerFitCorpusV2DesignError(
            "rotation-mask classification binding is malformed")
    row = dict(value)
    required = {
        "path", "schema", "self_digest_key", "self_digest", "raw_sha256",
        "byte_count", "source_repository_commit",
    }
    if (set(row) != required
            or row.get("path") != str(MASK_CLASSIFICATION_RELATIVE_PATH)
            or row.get("schema") != MASK_CLASSIFICATION_SCHEMA
            or row.get("self_digest_key") != MASK_CLASSIFICATION_SELF_KEY
            or not _is_hex(row.get("self_digest"), 64)
            or not _is_hex(row.get("raw_sha256"), 64)
            or isinstance(row.get("byte_count"), bool)
            or not isinstance(row.get("byte_count"), int)
            or row["byte_count"] <= 0
            or not _is_hex(row.get("source_repository_commit"), 40)):
        raise ScorerFitCorpusV2DesignError(
            "rotation-mask classification binding changed")
    return row


def build_design_amendment(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        rotation_mask_classification_binding: Mapping[str, Any],
        predecessor_validation: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the prospective V2 corpus design without reading any artifact."""

    if not _is_hex(source_repository_commit, 40):
        raise ScorerFitCorpusV2DesignError("source commit is malformed")
    sources = _validate_source_bindings(list(source_bindings))
    classification = _validate_classification_binding(
        rotation_mask_classification_binding)
    predecessor_projection = _validate_predecessor_validation_projection(
        predecessor_validation)
    if classification["source_repository_commit"] != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "classification and design source commits differ")
    lineage = _predecessor_lineage()
    payload: dict[str, Any] = {
        "schema": DESIGN_SCHEMA,
        "status": DESIGN_STATUS,
        "corpus_design_version": "scorer_fit_corpus_v2_full_bank_v1",
        "source_repository_commit": source_repository_commit,
        "source_bindings": sources,
        "source_binding_set_digest": canonical_digest(sources),
        "rotation_mask_classification": classification,
        "predecessor_validation": predecessor_projection,
        "preoutcome_lineage": lineage,
        "preoutcome_lineage_digest": canonical_digest(lineage),
        "preserved_scientific_contract_bindings": copy.deepcopy(
            PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS),
        "preserved_scientific_contract_bindings_digest": canonical_digest(
            PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS),
        "scientific_binding_dispositions": {
            "lineage_only_superseded_partial_allocation_bindings": [
                "scorer_fit_allocation_design_digest",
                "candidate_allocator_contract_digest",
                "candidate_allocation_amendment_digest",
                "pre_identity_allocation_validation_digest",
            ],
            "all_other_bound_selector_exclusion_oracle_render_preprocess_"
            "encoder_scorer_fields": "PRESERVED_ACTIVE_UNCHANGED",
            "lineage_binding_deleted_or_reinterpreted": False,
        },
        "supersession": {
            "status": SIX_OF_TWELVE_SUPERSESSION,
            "superseded_only": [
                "six candidates per state", "candidate rotations",
                "candidate exact-60 appearances", "scene-to-rotation assignment",
                "rotation eligibility used for subset allocation",
                "residual candidate-frequency MILPs",
                "partial-subset candidate co-occurrence balancing",
                "external or global scene/rotation optimization",
            ],
            "selector_superseded": False,
            "identity_exclusions_superseded": False,
            "oracle_superseded": False,
            "render_preprocess_encoder_superseded": False,
            "scorer_architecture_or_qualification_superseded": False,
        },
        "active_full_bank_rule": {
            "candidate_indices": list(CANDIDATE_INDICES),
            "every_selected_state_receives_every_candidate_exactly_once": True,
            "per_state_subset_decision": False,
            "per_state_rotation_decision": False,
            "candidate_assignment_optimization": False,
            "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        },
        "count_contract": copy.deepcopy(FULL_BANK_COUNT_CONTRACT),
        "count_contract_digest": canonical_digest(FULL_BANK_COUNT_CONTRACT),
        "state_design": {
            "families": list(FAMILIES),
            "strata": list(STRATA),
            "states_total": 120,
            "states_per_family": 15,
            "states_per_family_stratum": 5,
            "fit_states": 96,
            "calibration_states": 24,
            "fit_per_family_stratum": 4,
            "calibration_per_family_stratum": 1,
            "scene_disjoint_fit_calibration": True,
            "fixed_outcome_free_state_count": 115,
            "unresolved_small_completion_state_count": 5,
            "eligible_small_completion_scene_count": 17,
        },
        "small_completion_selection": copy.deepcopy(
            COMPLETION_ORDERING_CONTRACT),
        "small_completion_selection_digest": canonical_digest(
            COMPLETION_ORDERING_CONTRACT),
        "completion_full_bank_revalidation": {
            "formula": "max(d0 - 0.75 m, 0) <= L_max",
            "l_max_scope": "maximum over candidate indices 0--11",
            "actual_previous_applied_command": True,
            "frozen_slew_limiter": True,
            "exact_twenty_tick_plans": True,
            "branch_execution": False,
            "realised_outcome": False,
        },
        "preoutcome_feasibility_terminal": {
            "path": str(
                SCORER_FIT_RELATIVE_PATH /
                "full_bank_preoutcome_feasibility_failure_v2.json"),
            "condition": (
                "fewer than one calibration-eligible or four distinct fit-"
                "eligible full-bank-valid small completion scenes"),
            "mutually_exclusive_with_success_manifests": True,
            "candidate_outcomes_consumed": False,
            "automatic_selector_or_candidate_bank_revision": False,
        },
        "preserved_nonallocation_science": {
            "oracle_v1_2_digest": ORACLE_V1_2_DIGEST,
            "horizon_seconds": 2.0,
            "horizon_blocks": 4,
            "snapshot_time_goal_binding": True,
            "textured_v03_rendering": True,
            "crop_preprocessing_target_encoder": True,
            "scorer_architecture_training_and_qualification": True,
            "no_latent_baseline": True,
            "final_200_state_corpus_authorized_in_this_pass": False,
            "future_final_evaluation_rule": (
                "when separately authorised, final_eval selection excludes all "
                "120 scenes in the active scorer-fit V2 manifest"),
            "preexisting_reserved_final_evaluation_scene_set": False,
            "final_evaluation_manifest_absent_at_issue": True,
        },
        "selection_field_policy": {
            "candidate_outcome": False,
            "branch_label": False,
            "frame": False,
            "latent": False,
            "scorer_metric": False,
            "predictor_output": False,
            "historical_receipts_reopened_for_exact_lineage_custody": True,
            "historical_receipts_used_for_selection": False,
        },
        "issuance_boundary": {
            "source_tree_clean_and_committed": True,
            "classification_issued_first": True,
            "v1_and_v2_failures_immutable": True,
            "six_of_twelve_exact_infeasibility_immutable": True,
            "v2_state_selection_started": False,
            "v2_branch_execution_started": False,
            "milp_or_cp_sat_run": False,
            "performance_benchmark_run": False,
            "candidate_outcome_or_downstream_metric_used": False,
        },
    }
    payload[DESIGN_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_design_amendment(
        payload: Mapping[str, Any], *, validate_live_source: bool = False,
        root: Path = ROOT) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ScorerFitCorpusV2DesignError("design amendment is not a mapping")
    value = copy.deepcopy(dict(payload))
    if (value.get("schema") != DESIGN_SCHEMA
            or value.get("status") != DESIGN_STATUS
            or value.get(DESIGN_SELF_KEY)
            != canonical_digest(_without(value, DESIGN_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError("design amendment binding changed")
    expected = build_design_amendment(
        source_repository_commit=str(value.get("source_repository_commit", "")),
        source_bindings=value.get("source_bindings", []),
        rotation_mask_classification_binding=value.get(
            "rotation_mask_classification", {}),
        predecessor_validation=value.get("predecessor_validation", {}))
    if value != expected:
        raise ScorerFitCorpusV2DesignError("design amendment is not exact")
    _validate_predecessor_lineage(value["preoutcome_lineage"])
    if validate_live_source:
        commit, sources = clean_source_authority(root=root)
        if commit != value["source_repository_commit"] or sources != value["source_bindings"]:
            raise ScorerFitCorpusV2DesignError(
                "live source differs from design amendment")
    return value


def design_amendment_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    value = validate_design_amendment(payload)
    if raw != _pretty_json_bytes(value):
        raise ScorerFitCorpusV2DesignError("design amendment raw bytes changed")
    return {
        "path": str(DESIGN_RELATIVE_PATH),
        "schema": DESIGN_SCHEMA,
        "self_digest_key": DESIGN_SELF_KEY,
        "self_digest": value[DESIGN_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": value["source_repository_commit"],
    }


_ISSUED_DESIGN_AUTHORITY_KEYS = frozenset({
    "rotation_mask_classification_payload",
    "rotation_mask_classification_binding",
    "design_amendment_payload",
    "design_amendment_binding",
})

PRESELECTION_ALIAS_FAILURE_BOUNDARY = {
    "status": (
        "IMMUTABLE_PRESELECTION_FAILURE_REGISTERED_DEVELOPMENT_STAGE_A_"
        "MANIFEST_ALIAS_GUARD"),
    "issued_source_repository_commit":
        ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
    "classification_issued": True,
    "design_amendment_issued": True,
    "freeze_manifests_stage_entered": True,
    "preoutcome_input_loader_entered": True,
    "active_design_authority_validated": True,
    "predecessor_fixed_state_count_validated": 115,
    "eligible_small_completion_scene_count_validated": 17,
    "historical_preserved_rotation_evidence_loaded": True,
    "historical_rotation_evidence_used_as_active_mask": False,
    "factorial_exclusion_setup_loaded": True,
    "invalid_identity_index_setup_loaded": True,
    "failure_site": (
        "INITIAL_REGISTERED_EXCLUSION_MANIFEST_PATH_GUARD"),
    "failed_registered_path": (
        ".generated/go2_counterfactual_fidelity_v1_2/"
        "stage_a_identity_manifest.json"),
    "oracle_v1_1_identity_manifest_json_read": False,
    "oracle_v1_2_identity_manifest_json_read": False,
    "development_stage_a_identity_manifest_json_read": False,
    "exclusion_authority_returned": False,
    "candidate_revalidation_started": False,
    "completion_ordering_started": False,
    "small_completion_selection_started": False,
    "preoutcome_manifest_or_selection_artifact_issued": False,
    "solver_or_optimisation_invoked": False,
    "branch_execution_started": False,
    "candidate_outcome_or_branch_label_read": False,
    "frame_or_latent_created_or_read": False,
    "scorer_metric_or_predictor_output_read": False,
    "successor_scorer_contract_issued": False,
    "final_200_state_corpus_generated": False,
    "nothing_running": True,
}

_SOURCE_CORRECTION_KEYS = frozenset({
    "schema", "status", "complete", "source_correction_version",
    "source_repository_commit", "source_bindings",
    "source_binding_set_digest", "historical_source_repository_commit",
    "immutable_issued_design_authority",
    "immutable_issued_design_authority_digest",
    "preserved_scientific_design_digest",
    "preserved_rotation_mask_classification_digest",
    "runtime_outputs_absent_at_issue",
    "runtime_outputs_absent_at_issue_digest",
    "preselection_alias_failure_boundary",
    "preselection_alias_failure_boundary_digest",
    "source_correction", "source_correction_material_digest",
    "issuance_boundary", SOURCE_CORRECTION_SELF_KEY,
})


def validate_immutable_issued_design_authority(
        value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate immutable issued payloads and their exact canonical bytes."""

    if not isinstance(value, Mapping) or set(value) != _ISSUED_DESIGN_AUTHORITY_KEYS:
        raise ScorerFitCorpusV2DesignError(
            "immutable issued V2 design authority is not closed")
    authority = copy.deepcopy(dict(value))
    classification = validate_rotation_mask_classification(
        authority["rotation_mask_classification_payload"],
        validate_live_source=False)
    design = validate_design_amendment(
        authority["design_amendment_payload"], validate_live_source=False)
    classification_binding = rotation_mask_classification_artifact_binding(
        classification, _pretty_json_bytes(classification))
    design_binding = design_amendment_artifact_binding(
        design, _pretty_json_bytes(design))
    if (authority["rotation_mask_classification_binding"]
            != classification_binding
            or authority["design_amendment_binding"] != design_binding
            or classification["source_repository_commit"]
            != ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT
            or design["source_repository_commit"]
            != ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT
            or design["rotation_mask_classification"]
            != classification_binding):
        raise ScorerFitCorpusV2DesignError(
            "immutable issued V2 design payload or artifact binding changed")
    return authority


def build_preselection_source_correction(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        immutable_issued_design_authority: Mapping[str, Any],
        runtime_outputs_absent_at_issue: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Build the orthogonal source-only correction without generated access."""

    if (not _is_hex(source_repository_commit, 40)
            or source_repository_commit
            == ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction commit is malformed or not new")
    current_sources = _validate_source_bindings(list(source_bindings))
    issued = validate_immutable_issued_design_authority(
        immutable_issued_design_authority)
    classification = issued["rotation_mask_classification_payload"]
    design = issued["design_amendment_payload"]
    if classification["source_bindings"] != design["source_bindings"]:
        raise ScorerFitCorpusV2DesignError(
            "issued classification/design source closures differ")
    changed = _changed_source_paths(
        classification["source_bindings"], current_sources)
    if changed != sorted(SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "preselection correction changed an unauthorised source path")
    absence = _validate_absence_projection(
        list(runtime_outputs_absent_at_issue), phase="design")
    failure = copy.deepcopy(PRESELECTION_ALIAS_FAILURE_BOUNDARY)
    correction = {
        "status": "SOURCE_ONLY_REGISTERED_DEVELOPMENT_MANIFEST_ALIAS_CORRECTION",
        "defect": (
            "REGISTERED_DEVELOPMENT_STAGE_A_MANIFEST_ALIAS_FAILED_THE_"
            "INITIAL_EXCLUSION_PATH_GUARD"),
        "correction": (
            "RESOLVE_AND_VALIDATE_THE_REGISTERED_DEVELOPMENT_MANIFEST_"
            "THROUGH_ITS_EXACT_IMMUTABLE_ALIAS_TARGET"),
        "historical_source_repository_commit":
            ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        "successor_source_repository_commit": source_repository_commit,
        "allowed_changed_source_paths": list(
            SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS),
        "observed_changed_source_paths": changed,
        "historical_source_binding_set_digest": classification[
            "source_binding_set_digest"],
        "successor_source_binding_set_digest": canonical_digest(
            current_sources),
        "old_classification_or_design_overwritten": False,
        "old_classification_or_design_reissued": False,
        "old_scientific_design_digest_preserved": True,
        "scene_state_or_candidate_pool_changed": False,
        "candidate_bank_or_frequency_changed": False,
        "selector_exclusion_rule_or_quota_changed": False,
        "oracle_render_preprocess_or_target_encoder_changed": False,
        "scorer_architecture_training_or_qualification_changed": False,
        "scientific_contract_changed": False,
        "candidate_outcome_or_downstream_metric_used": False,
    }
    payload: dict[str, Any] = {
        "schema": SOURCE_CORRECTION_SCHEMA,
        "status": SOURCE_CORRECTION_STATUS,
        "complete": True,
        "source_correction_version": 1,
        "source_repository_commit": source_repository_commit,
        "source_bindings": current_sources,
        "source_binding_set_digest": canonical_digest(current_sources),
        "historical_source_repository_commit":
            ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT,
        "immutable_issued_design_authority": copy.deepcopy(issued),
        "immutable_issued_design_authority_digest": canonical_digest(issued),
        "preserved_scientific_design_digest": design[DESIGN_SELF_KEY],
        "preserved_rotation_mask_classification_digest": classification[
            MASK_CLASSIFICATION_SELF_KEY],
        "runtime_outputs_absent_at_issue": absence,
        "runtime_outputs_absent_at_issue_digest": canonical_digest(absence),
        "preselection_alias_failure_boundary": failure,
        "preselection_alias_failure_boundary_digest": canonical_digest(failure),
        "source_correction": correction,
        "source_correction_material_digest": canonical_digest(correction),
        "issuance_boundary": {
            "source_tree_clean_and_committed": True,
            "classification_and_design_preserved_immutable": True,
            "failure_preserved_truthfully": True,
            "double_runtime_output_absence_audit_required": True,
            "preselection_only": True,
            "selection_or_manifest_issued": False,
            "branch_execution_started": False,
            "candidate_outcomes_consumed": False,
            "solver_or_optimisation_invoked": False,
        },
    }
    if set(payload) != _SOURCE_CORRECTION_KEYS - {SOURCE_CORRECTION_SELF_KEY}:
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction construction surface changed")
    payload[SOURCE_CORRECTION_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_preselection_source_correction(
        payload: Mapping[str, Any], *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    """Validate the correction and immutable old scientific authorities."""

    if not isinstance(payload, Mapping) or set(payload) != _SOURCE_CORRECTION_KEYS:
        raise ScorerFitCorpusV2DesignError(
            "preselection source correction is not closed")
    correction = copy.deepcopy(dict(payload))
    if (correction.get("schema") != SOURCE_CORRECTION_SCHEMA
            or correction.get("status") != SOURCE_CORRECTION_STATUS
            or correction.get("complete") is not True
            or correction.get("source_correction_version") != 1):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction version changed")
    expected = build_preselection_source_correction(
        source_repository_commit=str(correction.get(
            "source_repository_commit", "")),
        source_bindings=correction.get("source_bindings", []),
        immutable_issued_design_authority=correction.get(
            "immutable_issued_design_authority", {}),
        runtime_outputs_absent_at_issue=correction.get(
            "runtime_outputs_absent_at_issue", []),
    )
    if (correction != expected
            or correction.get(SOURCE_CORRECTION_SELF_KEY)
            != canonical_digest(_without(
                correction, SOURCE_CORRECTION_SELF_KEY))):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction binding changed")
    if validate_live_authorities:
        commit, sources = clean_source_authority(root=root)
        if (commit != correction["source_repository_commit"]
                or sources != correction["source_bindings"]
                or _load_issued_design_authority_for_source_correction(
                    root=root)
                != correction["immutable_issued_design_authority"]):
            raise ScorerFitCorpusV2DesignError(
                "live source or immutable issued authority differs from correction")
        if require_runtime_outputs_absent:
            observed = audit_v2_runtime_outputs_absent(
                root=root, phase="design")
            if observed != correction["runtime_outputs_absent_at_issue"]:
                raise ScorerFitCorpusV2DesignError(
                    "runtime-output absence changed after correction issuance")
    return correction


def preselection_source_correction_artifact_binding(
        payload: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    correction = validate_preselection_source_correction(
        payload, validate_live_authorities=False)
    if raw != _pretty_json_bytes(correction):
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction raw bytes changed")
    return {
        "path": str(SOURCE_CORRECTION_RELATIVE_PATH),
        "schema": SOURCE_CORRECTION_SCHEMA,
        "self_digest_key": SOURCE_CORRECTION_SELF_KEY,
        "self_digest": correction[SOURCE_CORRECTION_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": correction["source_repository_commit"],
    }


def completion_order_material(
        complete_structural_state_identity: Mapping[str, Any],
        designated_goal_identity: Mapping[str, Any], *,
        active_selector_digest: str = ACTIVE_SELECTOR_DIGEST,
        ) -> dict[str, Any]:
    """Return the exact outcome-free material used to order one identity."""

    if (not isinstance(complete_structural_state_identity, Mapping)
            or not isinstance(designated_goal_identity, Mapping)
            or not complete_structural_state_identity
            or not designated_goal_identity
            or active_selector_digest != ACTIVE_SELECTOR_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "completion ordering identity or selector binding changed")
    # Canonicalization also rejects NaN, infinity and non-JSON objects.
    structural = copy.deepcopy(dict(complete_structural_state_identity))
    goal = copy.deepcopy(dict(designated_goal_identity))
    _json_bytes(structural)
    _json_bytes(goal)
    return {
        "domain_separation_utf8": COMPLETION_ORDER_DOMAIN,
        "active_selector_contract_digest": active_selector_digest,
        "complete_structural_state_identity": structural,
        "designated_goal_identity": goal,
    }


def completion_order_key(
        complete_structural_state_identity: Mapping[str, Any],
        designated_goal_identity: Mapping[str, Any], *,
        active_selector_digest: str = ACTIVE_SELECTOR_DIGEST,
        ) -> tuple[str, bytes]:
    """Return SHA-256 primary order and structural bytes collision tie-break."""

    material = completion_order_material(
        complete_structural_state_identity, designated_goal_identity,
        active_selector_digest=active_selector_digest)
    structural = _json_bytes({
        "complete_structural_state_identity": material[
            "complete_structural_state_identity"],
        "designated_goal_identity": material["designated_goal_identity"],
    })
    preimage = (
        COMPLETION_ORDER_DOMAIN.encode("utf-8") + b"\x00"
        + active_selector_digest.encode("ascii") + b"\x00"
        + _json_bytes(material["complete_structural_state_identity"]) + b"\x00"
        + _json_bytes(material["designated_goal_identity"])
    )
    return hashlib.sha256(preimage).hexdigest(), structural


def _pin_relative(root: Path, relative: str | Path, *, label: str) -> Path:
    repository = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or any(
            part == "sealed" or part == "sealed_test.json"
            or part.startswith("sealed_") for part in rel.parts):
        raise ScorerFitCorpusV2DesignError(f"{label} path is inaccessible")
    cursor = repository
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink() and rel != SCORER_FIT_RELATIVE_PATH:
            # Exact generated-root aliases are handled by _pin_generated.
            raise ScorerFitCorpusV2DesignError(f"{label} path is symlinked")
    return repository / rel


def _pin_generated(root: Path, relative: str | Path, *, label: str) -> Path:
    repository = Path(root).resolve()
    rel = Path(relative)
    matches: list[tuple[Path, Path]] = []
    for managed_root in MANAGED_GENERATED_ROOTS:
        try:
            matches.append((managed_root, rel.relative_to(managed_root)))
        except ValueError:
            continue
    if len(matches) != 1:
        raise ScorerFitCorpusV2DesignError(
            f"{label} escaped the managed generated roots")
    managed_root, suffix = matches[0]
    if not suffix.parts or any(
            part == "sealed" or part == "sealed_test.json"
            or part.startswith("sealed_") or part == ".."
            for part in suffix.parts):
        raise ScorerFitCorpusV2DesignError(f"{label} path is inaccessible")
    logical = repository / managed_root
    if logical.is_symlink():
        target = logical.readlink()
        target = target if target.is_absolute() else logical.parent / target
        if target.name != logical.name:
            raise ScorerFitCorpusV2DesignError(
                "managed generated-root alias changed")
        base = target.resolve(strict=True)
    else:
        base = logical.resolve(strict=True) if logical.exists() else logical
    cursor = base
    for part in suffix.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ScorerFitCorpusV2DesignError(f"{label} path is symlinked")
    return base.joinpath(*suffix.parts)


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScorerFitCorpusV2DesignError(
            f"git source-custody check failed: {' '.join(args)}") from exc


def clean_source_authority(*, root: Path = ROOT) -> tuple[str, list[dict[str, Any]]]:
    """Bind the exact clean implementation closure without tree-wide export."""

    repository = Path(root).resolve()
    commit = _git("rev-parse", "HEAD", root=repository).decode().strip()
    if not _is_hex(commit, 40):
        raise ScorerFitCorpusV2DesignError("source commit is malformed")
    if _git("status", "--porcelain=v1", "--untracked-files=all", root=repository):
        raise ScorerFitCorpusV2DesignError(
            "V2 authority issuance requires a clean source commit")
    tracked = _git(
        "ls-files", "--error-unmatch", "--", *EXPECTED_SOURCE_PATHS,
        root=repository,
    ).decode().splitlines()
    if len(tracked) != len(EXPECTED_SOURCE_PATHS) or set(tracked) != set(
            EXPECTED_SOURCE_PATHS):
        raise ScorerFitCorpusV2DesignError(
            "V2 source closure is not fully tracked")
    rows: list[dict[str, Any]] = []
    for relative, role in SOURCE_SPECS:
        path = _pin_relative(repository, relative, label="V2 source")
        if not path.is_file() or path.is_symlink():
            raise ScorerFitCorpusV2DesignError(
                f"V2 source unavailable: {relative}")
        raw = path.read_bytes()
        if _git("show", f"HEAD:{relative}", root=repository) != raw:
            raise ScorerFitCorpusV2DesignError(
                f"V2 source differs from clean commit: {relative}")
        rows.append({
            "path": relative, "role": role, "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return commit, rows


def audit_v2_runtime_outputs_absent(
        *, root: Path = ROOT, phase: str = "design",
        ) -> list[dict[str, Any]]:
    """Check the exact phase-appropriate V2 output-absence surface.

    ``design`` additionally requires all preoutcome V2 artifacts and the
    successor contract to be absent. ``successor_contract`` permits the five
    issued preoutcome artifacts but still requires the contract itself to be
    absent. ``post_contract_pre_branch`` permits both while retaining every
    branch/frame/latent/training/qualification/development absence gate.
    """

    if phase not in {"design", "successor_contract", "post_contract_pre_branch"}:
        raise ScorerFitCorpusV2DesignError("unknown V2 absence-audit phase")
    exact_paths = list(V2_ALWAYS_ABSENT_PATHS)
    exact_paths.extend(V2_RUNTIME_OUTPUT_PATHS)
    if phase == "design":
        exact_paths.extend(V2_PREOUTCOME_ARTIFACT_PATHS)
    if phase in {"design", "successor_contract"}:
        exact_paths.append(V2_SUCCESSOR_CONTRACT_PATH)
    exact_paths.extend(Path(path) for path in
                       GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_EXACT_FILE_PATHS)
    directory_paths = list(V2_FUTURE_OUTPUT_DIRECTORIES)
    directory_paths.extend(Path(path) for path in
                           GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_DIRECTORY_ROOTS)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, paths in (("file", exact_paths), ("directory", directory_paths)):
        for relative in paths:
            key = str(relative)
            if key in seen:
                continue
            seen.add(key)
            path = _pin_generated(root, relative, label="V2 outcome absence")
            if path.exists() or path.is_symlink():
                raise ScorerFitCorpusV2DesignError(
                    f"V2 outcome-bearing path predates design: {relative}")
            rows.append({"path": key, "expected_kind": kind, "exists": False})
    patterns = list(V2_FUTURE_OUTPUT_GLOBS)
    patterns.extend(Path(path) for path in
                    GLOBAL_AUTHORITY.PHASE1_FORBIDDEN_GLOB_PATTERNS)
    for pattern_raw in patterns:
        pattern = Path(pattern_raw)
        parent = _pin_generated(
            root, pattern.parent / "__v2_absence_probe__",
            label="V2 outcome glob absence").parent
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise ScorerFitCorpusV2DesignError(
                    "V2 outcome glob parent changed")
            if any(fnmatchcase(entry.name, pattern.name) for entry in parent.iterdir()):
                raise ScorerFitCorpusV2DesignError(
                    f"V2 outcome-bearing glob predates design: {pattern}")
        rows.append({
            "path": str(pattern), "expected_kind": "glob", "exists": False,
        })
    return rows


def audit_v2_outcome_outputs_absent(*, root: Path = ROOT) -> list[dict[str, Any]]:
    """Compatibility name for the complete pre-design absence audit."""

    return audit_v2_runtime_outputs_absent(root=root, phase="design")


def _exclusive_json(path: Path, value: Mapping[str, Any], *, label: str) -> bytes:
    raw = _pretty_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o444)
    except OSError as exc:
        raise ScorerFitCorpusV2DesignError(
            f"cannot exclusively create {label}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Never unlink a possibly durable authority after exclusive creation.
        raise
    return raw


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ScorerFitCorpusV2DesignError(f"{label} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScorerFitCorpusV2DesignError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ScorerFitCorpusV2DesignError(f"{label} is not an object")
    return value, raw


def _artifact_path_from_binding(
        binding: Mapping[str, Any], *, root: Path, label: str) -> Path:
    relative = Path(str(binding.get("path", "")))
    if any(root_path == relative or root_path in relative.parents
           for root_path in MANAGED_GENERATED_ROOTS):
        return _pin_generated(root, relative, label=label)
    return _pin_relative(root, relative, label=label)


def _validate_exact_json_binding(
        binding: Mapping[str, Any], *, root: Path, label: str,
        ) -> dict[str, Any]:
    path = _artifact_path_from_binding(binding, root=root, label=label)
    payload, raw = _load_json(path, label=label)
    self_key = str(binding.get("self_digest_key", ""))
    canonicalization = binding.get(
        "self_digest_canonicalization", "COMPACT_CANONICAL_JSON")
    digest_payload = _without(payload, self_key)
    if canonicalization == "COMPACT_CANONICAL_JSON":
        recomputed_self_digest = canonical_digest(digest_payload)
    elif canonicalization == "JSON_DUMPS_SORT_KEYS_DEFAULT_SEPARATORS":
        recomputed_self_digest = hashlib.sha256(json.dumps(
            digest_payload, sort_keys=True, ensure_ascii=True,
            allow_nan=False).encode("utf-8")).hexdigest()
    else:
        raise ScorerFitCorpusV2DesignError(
            f"unknown historical self-digest convention: {label}")
    if (len(raw) != binding.get("byte_count")
            or hashlib.sha256(raw).hexdigest() != binding.get("raw_sha256")
            or payload.get("schema") != binding.get("schema")
            or ("semantic_status" in binding
                and payload.get("status") != binding.get("semantic_status"))
            or payload.get(self_key) != binding.get("self_digest")
            or recomputed_self_digest != binding.get("self_digest")):
        raise ScorerFitCorpusV2DesignError(
            f"immutable historical artifact changed: {label}")
    return payload


def validate_historical_predecessor_artifacts(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen only the exact frozen lineage paths and validate them solve-free.

    This is custody validation for issuance.  None of the reopened JSON is
    supplied to the mask classifier or state selector.
    """

    active = _validate_exact_json_binding(
        ACTIVE_GLOBAL_AMENDMENT_BINDING, root=root,
        label="active global exact amendment")
    try:
        GLOBAL_AUTHORITY.validate_preplan_integration_correction(
            active, root=root, validate_live_authorities=False,
            require_runtime_outputs_absent=False)
    except Exception as exc:
        raise ScorerFitCorpusV2DesignError(
            "active global exact amendment historical validation failed") from exc
    if (active.get("source_repository_commit")
            != TERMINAL_SOURCE_REPOSITORY_COMMIT
            or active.get("scientific_contract_bindings")
            != PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS
            or active.get("preoutcome_input_bindings", {}).get(
                "candidate_outcomes_consumed") is not False):
        raise ScorerFitCorpusV2DesignError(
            "active global exact amendment scientific binding changed")

    plan = _validate_exact_json_binding(
        GLOBAL_MODEL_PLAN_BINDING, root=root, label="global exact model plan")
    nested_plan = plan.get("model_execution_plan")
    if (plan.get("status") != "FROZEN_PRE_SOLVE"
            or plan.get("source_repository_commit")
            != TERMINAL_SOURCE_REPOSITORY_COMMIT
            or plan.get("execution_amendment_digest")
            != ACTIVE_GLOBAL_AMENDMENT_DIGEST
            or plan.get("model_execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or not isinstance(nested_plan, Mapping)
            or nested_plan.get("model_digest") != GLOBAL_EXACT_MODEL_DIGEST
            or nested_plan.get("execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or nested_plan.get("candidate_outcomes_consumed") is not False
            or nested_plan.get("external_combination_enumeration") is not False
            or nested_plan.get("performance_gate") is not None
            or canonical_digest(_without(
                nested_plan, "execution_plan_digest"))
            != MODEL_EXECUTION_PLAN_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "global exact model plan semantic binding changed")

    terminal = _validate_exact_json_binding(
        GLOBAL_TERMINAL_INFEASIBILITY_BINDING, root=root,
        label="global exact terminal infeasibility")
    result = terminal.get("model_execution_result")
    exact = result.get("exact_model_result") if isinstance(result, Mapping) else None
    if (terminal.get("status") != "INFEASIBLE"
            or terminal.get("execution_amendment_digest")
            != ACTIVE_GLOBAL_AMENDMENT_DIGEST
            or terminal.get("global_exact_model_plan_digest")
            != GLOBAL_MODEL_PLAN_DIGEST
            or terminal.get("model_execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or terminal.get("exact_infeasibility_proved") is not True
            or terminal.get("scientific_conditions_relaxed") is not False
            or terminal.get("candidate_outcomes_consumed") is not False
            or terminal.get("branch_labels_read") is not False
            or terminal.get("scorer_or_predictor_accessed") is not False
            or not isinstance(result, Mapping)
            or result.get("model_digest") != GLOBAL_EXACT_MODEL_DIGEST
            or result.get("execution_plan_digest")
            != MODEL_EXECUTION_PLAN_DIGEST
            or result.get("status") != "EXACT_GLOBAL_ALLOCATION_INFEASIBLE"
            or result.get("candidate_outcomes_consumed") is not False
            or result.get("materialized_allocation") is not None
            or result.get("execution_result_digest")
            != terminal.get("model_execution_result_digest")
            or canonical_digest(_without(result, "execution_result_digest"))
            != result.get("execution_result_digest")
            or not isinstance(exact, Mapping)
            or exact.get("model_digest") != GLOBAL_EXACT_MODEL_DIGEST
            or exact.get("infeasibility_digest") != EXACT_INFEASIBILITY_DIGEST
            or exact.get("candidate_outcomes_consumed") is not False
            or canonical_digest(_without(exact, "infeasibility_digest"))
            != EXACT_INFEASIBILITY_DIGEST):
        raise ScorerFitCorpusV2DesignError(
            "global exact terminal infeasibility semantic binding changed")
    try:
        from lewm.oracle import go2_small_completion_global_exact_model_v1 as model
        model.validate_solver_runtime_identity_record(
            nested_plan.get("solver_runtime_identity", {}))
        model.validate_solver_runtime_identity_record(
            exact.get("solver_runtime_identity", {}))
    except Exception as exc:
        raise ScorerFitCorpusV2DesignError(
            "frozen solve-free runtime identity changed") from exc

    try:
        predecessor_lineage = GLOBAL_AUTHORITY.load_predecessor_lineage(root=root)
    except Exception as exc:
        raise ScorerFitCorpusV2DesignError(
            "immutable V1/V2 failure validation failed") from exc
    if (len(predecessor_lineage) != len(IMMUTABLE_V1_V2_FAILURE_BINDINGS)
            or {row["role"] for row in IMMUTABLE_V1_V2_FAILURE_BINDINGS}
            != set(predecessor_lineage)):
        raise ScorerFitCorpusV2DesignError(
            "immutable V1/V2 lineage coverage changed")

    prior_payloads: dict[str, dict[str, Any]] = {}
    for binding in PRIOR_PREOUTCOME_FAILURE_BINDINGS:
        prior_payloads[str(binding["role"])] = _validate_exact_json_binding(
            binding, root=root, label=str(binding["role"]))
    frozen_predictor = prior_payloads[
        "candidate_allocation_infeasibility"].get(
            "frozen_predictor_qualification")
    expected_predictor_projection = {
        key: value for key, value in FROZEN_PREDICTOR_QUALIFICATION.items()
        if key != "result_report"
    }
    if frozen_predictor != expected_predictor_projection:
        raise ScorerFitCorpusV2DesignError(
            "frozen predictor qualification lineage changed")

    report_binding = FROZEN_PREDICTOR_QUALIFICATION["result_report"]
    report_path = _artifact_path_from_binding(
        report_binding, root=root, label="predictor qualification report")
    if not report_path.is_file() or report_path.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "predictor qualification report is unavailable")
    report_raw = report_path.read_bytes()
    if (len(report_raw) != report_binding["byte_count"]
            or hashlib.sha256(report_raw).hexdigest()
            != report_binding["raw_sha256"]):
        raise ScorerFitCorpusV2DesignError(
            "predictor qualification report binding changed")
    return copy.deepcopy(PREDECESSOR_VALIDATION_PROJECTION)


def load_rotation_mask_classification(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_source: bool = True) -> dict[str, Any]:
    expected = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "mask classification logical path changed")
    payload, raw = _load_json(expected, label="mask classification")
    value = validate_rotation_mask_classification(
        payload, validate_live_source=validate_live_source, root=root)
    rotation_mask_classification_artifact_binding(value, raw)
    return value


def issue_rotation_mask_classification(
        path: Path | None = None, *, root: Path = ROOT) -> dict[str, Any]:
    expected = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "mask classification logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "mask classification parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_rotation_mask_classification(root=root)
    commit, sources = clean_source_authority(root=root)
    predecessor = validate_historical_predecessor_artifacts(root=root)
    before = audit_v2_outcome_outputs_absent(root=root)
    value = build_rotation_mask_classification(
        source_repository_commit=commit, source_bindings=sources,
        predecessor_validation=predecessor)
    if (before != audit_v2_outcome_outputs_absent(root=root)
            or predecessor
            != validate_historical_predecessor_artifacts(root=root)):
        raise ScorerFitCorpusV2DesignError(
            "V2 predecessor or outcome absence changed before classification install")
    _exclusive_json(expected, value, label="mask classification")
    return load_rotation_mask_classification(root=root)


def load_design_amendment(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_source: bool = True) -> dict[str, Any]:
    expected = _pin_generated(root, DESIGN_RELATIVE_PATH, label="design amendment")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError("design amendment logical path changed")
    payload, raw = _load_json(expected, label="design amendment")
    value = validate_design_amendment(
        payload, validate_live_source=validate_live_source, root=root)
    design_amendment_artifact_binding(value, raw)
    classification = load_rotation_mask_classification(
        root=root, validate_live_source=validate_live_source)
    classification_path = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    _classification_payload, classification_raw = _load_json(
        classification_path, label="mask classification")
    if value["rotation_mask_classification"] != (
            rotation_mask_classification_artifact_binding(
                classification, classification_raw)):
        raise ScorerFitCorpusV2DesignError(
            "design does not bind the installed classification")
    return value


def issue_design_amendment(
        path: Path | None = None, *, root: Path = ROOT) -> dict[str, Any]:
    expected = _pin_generated(root, DESIGN_RELATIVE_PATH, label="design amendment")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError("design amendment logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError("design amendment parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_design_amendment(root=root)
    classification = load_rotation_mask_classification(root=root)
    classification_path = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH, label="mask classification")
    _classification_payload, classification_raw = _load_json(
        classification_path, label="mask classification")
    classification_binding = rotation_mask_classification_artifact_binding(
        classification, classification_raw)
    commit, sources = clean_source_authority(root=root)
    predecessor = validate_historical_predecessor_artifacts(root=root)
    before = audit_v2_outcome_outputs_absent(root=root)
    value = build_design_amendment(
        source_repository_commit=commit, source_bindings=sources,
        rotation_mask_classification_binding=classification_binding,
        predecessor_validation=predecessor)
    if (before != audit_v2_outcome_outputs_absent(root=root)
            or predecessor
            != validate_historical_predecessor_artifacts(root=root)):
        raise ScorerFitCorpusV2DesignError(
            "V2 predecessor or outcome absence changed before design install")
    _exclusive_json(expected, value, label="design amendment")
    return load_design_amendment(root=root)


def _load_issued_design_authority_for_source_correction(
        *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen only the two issued artifacts with historical source replay off."""

    classification = load_rotation_mask_classification(
        root=root, validate_live_source=False)
    classification_path = _pin_generated(
        root, MASK_CLASSIFICATION_RELATIVE_PATH,
        label="immutable issued mask classification")
    _payload, classification_raw = _load_json(
        classification_path, label="immutable issued mask classification")
    design = load_design_amendment(root=root, validate_live_source=False)
    design_path = _pin_generated(
        root, DESIGN_RELATIVE_PATH, label="immutable issued design amendment")
    _payload, design_raw = _load_json(
        design_path, label="immutable issued design amendment")
    return validate_immutable_issued_design_authority({
        "rotation_mask_classification_payload": classification,
        "rotation_mask_classification_binding":
            rotation_mask_classification_artifact_binding(
                classification, classification_raw),
        "design_amendment_payload": design,
        "design_amendment_binding": design_amendment_artifact_binding(
            design, design_raw),
    })


def load_preselection_source_correction(
        path: Path | None = None, *, root: Path = ROOT,
        validate_live_authorities: bool = True,
        require_runtime_outputs_absent: bool = False,
        ) -> dict[str, Any]:
    expected = _pin_generated(
        root, SOURCE_CORRECTION_RELATIVE_PATH,
        label="preselection source correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction logical path changed")
    payload, raw = _load_json(expected, label="preselection source correction")
    correction = validate_preselection_source_correction(
        payload, root=root,
        validate_live_authorities=validate_live_authorities,
        require_runtime_outputs_absent=require_runtime_outputs_absent)
    preselection_source_correction_artifact_binding(correction, raw)
    return correction


def issue_preselection_source_correction(
        path: Path | None = None, *, root: Path = ROOT,
        source_repository_commit: str | None = None,
        ) -> dict[str, Any]:
    """Install the sole correction after a clean commit and two absence audits."""

    expected = _pin_generated(
        root, SOURCE_CORRECTION_RELATIVE_PATH,
        label="preselection source correction")
    supplied = expected if path is None else Path(path).absolute()
    if supplied.absolute() != expected.absolute():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction logical path changed")
    if not expected.parent.is_dir() or expected.parent.is_symlink():
        raise ScorerFitCorpusV2DesignError(
            "preselection source-correction parent is unavailable")
    if expected.exists() or expected.is_symlink():
        return load_preselection_source_correction(root=root)

    commit, sources = clean_source_authority(root=root)
    if source_repository_commit is not None and commit != source_repository_commit:
        raise ScorerFitCorpusV2DesignError(
            "requested preselection source-correction commit is not live")
    immutable = _load_issued_design_authority_for_source_correction(root=root)
    first_absence = audit_v2_runtime_outputs_absent(root=root, phase="design")
    correction = build_preselection_source_correction(
        source_repository_commit=commit,
        source_bindings=sources,
        immutable_issued_design_authority=immutable,
        runtime_outputs_absent_at_issue=first_absence)
    second_absence = audit_v2_runtime_outputs_absent(root=root, phase="design")
    second_commit, second_sources = clean_source_authority(root=root)
    if (first_absence != second_absence
            or second_absence != correction["runtime_outputs_absent_at_issue"]
            or (commit, sources) != (second_commit, second_sources)
            or immutable
            != _load_issued_design_authority_for_source_correction(root=root)):
        raise ScorerFitCorpusV2DesignError(
            "source, immutable authority, or output absence changed before install")
    _exclusive_json(expected, correction, label="preselection source correction")
    return load_preselection_source_correction(root=root)


def load_active_design_authority(*, root: Path = ROOT) -> dict[str, Any]:
    """Return immutable science plus the mandatory corrected source authority."""

    correction = load_preselection_source_correction(root=root)
    immutable = correction["immutable_issued_design_authority"]
    classification = immutable["rotation_mask_classification_payload"]
    classification_binding = immutable[
        "rotation_mask_classification_binding"]
    design = immutable["design_amendment_payload"]
    design_binding = immutable["design_amendment_binding"]
    correction_path = _pin_generated(
        root, SOURCE_CORRECTION_RELATIVE_PATH,
        label="preselection source correction")
    _payload, correction_raw = _load_json(
        correction_path, label="preselection source correction")
    return {
        "rotation_mask_classification": classification,
        "rotation_mask_classification_binding": classification_binding,
        "design_amendment": design,
        "design_amendment_binding": design_binding,
        "source_correction": correction,
        "source_correction_binding":
            preselection_source_correction_artifact_binding(
                correction, correction_raw),
        "source_correction_digest": correction[SOURCE_CORRECTION_SELF_KEY],
        "active_source_repository_commit": correction[
            "source_repository_commit"],
        "active_selector_digest": ACTIVE_SELECTOR_DIGEST,
        "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        "candidate_indices": list(CANDIDATE_INDICES),
        "count_contract": copy.deepcopy(FULL_BANK_COUNT_CONTRACT),
        "completion_ordering_contract": copy.deepcopy(
            COMPLETION_ORDERING_CONTRACT),
        "preserved_scientific_contract_bindings": copy.deepcopy(
            PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS),
        "candidate_outcomes_consumed": False,
    }


__all__ = [
    "ACTIVE_GLOBAL_AMENDMENT_BINDING", "ACTIVE_GLOBAL_AMENDMENT_DIGEST",
    "ACTIVE_SELECTOR_DIGEST", "ASSIGNMENT_COUNT", "CANDIDATE_BANK_DIGEST",
    "CANDIDATE_INDICES", "COMPLETION_ORDER_DOMAIN",
    "COMPLETION_ORDERING_CONTRACT", "DESIGN_RELATIVE_PATH", "DESIGN_SCHEMA",
    "DESIGN_SELF_KEY", "DESIGN_STATUS", "EXACT_INFEASIBILITY_DIGEST",
    "EXPECTED_ROTATION_CONSTRAINT_IDS", "EXPECTED_SOURCE_PATHS", "FAMILIES",
    "FROZEN_PREDICTOR_QUALIFICATION", "FULL_BANK_COUNT_CONTRACT",
    "GLOBAL_EXACT_MODEL_DIGEST", "GLOBAL_MODEL_PLAN_BINDING",
    "GLOBAL_TERMINAL_INFEASIBILITY_BINDING", "MASK_CLASSIFICATION_RELATIVE_PATH",
    "ISSUED_FULL_BANK_V2_SOURCE_REPOSITORY_COMMIT",
    "MASK_CLASSIFICATION_SCHEMA", "MASK_CLASSIFICATION_SELF_KEY",
    "MASK_CLASSIFICATION_STATUS", "ORACLE_V1_2_DIGEST",
    "PRESERVED_SCIENTIFIC_CONTRACT_BINDINGS", "PRIOR_PREOUTCOME_FAILURE_BINDINGS",
    "PREDECESSOR_VALIDATION_PROJECTION", "PRESELECTION_ALIAS_FAILURE_BOUNDARY",
    "SIX_OF_TWELVE_SUPERSESSION", "SOURCE_SPECS", "SPLIT_ROLES", "STATE_COUNT",
    "SOURCE_CORRECTION_ALLOWED_CHANGED_SOURCE_PATHS",
    "SOURCE_CORRECTION_RELATIVE_PATH", "SOURCE_CORRECTION_SCHEMA",
    "SOURCE_CORRECTION_SELF_KEY", "SOURCE_CORRECTION_STATUS",
    "STRATA", "ScorerFitCorpusV2DesignError", "TERMINAL_RECEIPT_DIGEST",
    "TERMINAL_SOURCE_REPOSITORY_COMMIT", "V2_FUTURE_OUTPUT_PATHS",
    "V2_PREOUTCOME_ARTIFACT_PATHS", "V2_RUNTIME_OUTPUT_PATHS",
    "V2_SUCCESSOR_CONTRACT_PATH", "audit_v2_outcome_outputs_absent",
    "audit_v2_runtime_outputs_absent", "build_design_amendment",
    "build_preselection_source_correction",
    "build_rotation_mask_classification", "canonical_digest",
    "clean_source_authority", "completion_order_key", "completion_order_material",
    "design_amendment_artifact_binding", "issue_design_amendment",
    "issue_preselection_source_correction",
    "issue_rotation_mask_classification", "load_active_design_authority",
    "load_design_amendment", "load_preselection_source_correction",
    "load_rotation_mask_classification",
    "preselection_source_correction_artifact_binding",
    "rotation_mask_classification_artifact_binding", "validate_design_amendment",
    "validate_immutable_issued_design_authority",
    "validate_historical_predecessor_artifacts",
    "validate_preselection_source_correction",
    "validate_rotation_mask_classification",
]
