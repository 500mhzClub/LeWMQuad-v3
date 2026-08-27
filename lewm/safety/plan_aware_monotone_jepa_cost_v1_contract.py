"""Prospective contract for ``PLAN_AWARE_MONOTONE_JEPA_COST_V1``.

This module is deliberately payload-free.  Importing it does not open the
Route-Intent panel, a latent tensor, a predictor checkpoint, a predecessor
result, a simulator, or a GPU.  It freezes the development-only experiment,
its custody barriers, schemas, deterministic decision rules, and small pure
helpers used to construct receipts.

The experiment trains a *route-only* ranker.  It is not a safety model and it
does not alter the unresolved protected-contact requirements.  Contact,
viability and stuck fields are never score inputs or route targets.  Oracle
viability is used only as the prospectively authorised training-set
conditioning mask and evaluation population.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_ID = "PLAN_AWARE_MONOTONE_JEPA_COST_V1"
DATE = "2026-08-27"
CONTRACT_SCHEMA_VERSION = "plan_aware_monotone_jepa_cost_v1.contract.v1"
OUTPUT_SCHEMA_VERSION = "plan_aware_monotone_jepa_cost_v1.output.v1"
FIXTURE_SCHEMA_VERSION = "plan_aware_monotone_jepa_cost_v1.evaluator_fixture.v1"
SOURCE_CLOSURE_SCHEMA_VERSION = "plan_aware_monotone_jepa_cost_v1.source_closure.v1"
ROUTE_ROLE_AUTHORITY_SCHEMA_VERSION = (
    "plan_aware_monotone_jepa_cost_v1.route_role_authority.v1"
)
EXECUTION_CORRECTION_AMENDMENT_SCHEMA_VERSION = (
    "plan_aware_monotone_jepa_cost_v1.execution_correction_amendment.v1"
)
EXECUTION_CORRECTION_REPLAY_SCHEMA_VERSION = (
    "plan_aware_monotone_jepa_cost_v1.execution_correction_replay.v1"
)
EXECUTION_CORRECTION_OUTPUT_SCHEMA_VERSION = (
    "plan_aware_monotone_jepa_cost_v1.execution_correction_output.v1"
)
EXECUTION_CORRECTION_FIXTURE_SCHEMA_VERSION = (
    "plan_aware_monotone_jepa_cost_v1.execution_correction_fixture.v1"
)
EXECUTION_CORRECTION_SOURCE_CLOSURE_SCHEMA_VERSION = (
    "plan_aware_monotone_jepa_cost_v1.execution_correction_source_closure.v1"
)

SOURCE_COMMIT = "1d799eb24d8171cb6d90bc0d0e375d9e1b0cc4f0"
REQUIRED_REQUIREMENTS_ANCESTOR = "b29eae1929725a4cc26a35d95662b545daee4553"
PREDECESSOR_EXECUTION_FREEZE_COMMIT = "b06905c2724a1ecb825db25b54ec1b3a43336bbf"
INITIAL_EXECUTION_FREEZE_COMMIT = "9c1c3adcfb8382c33e8da8895dc345e006e92e43"
SCIENTIFIC_AUTHORITY_CONTRACT_SHA256 = (
    "1667f325be2c835a6222dc90bb684f373a06b365d59b70e9746fd7adb052c382"
)
PREDICTOR_SEED = 2026080901
RANKER_SEED = 2026082701
PAIRWISE_UTILITY_TOLERANCE = 1e-12
RESULT_COMMIT_BINDING_POLICY = "ENCLOSING_GIT_COMMIT_AFTER_BYTE_FINALIZATION"
CONTRACT_FREEZE_COMMIT_SUBJECT = "Freeze plan-aware monotone JEPA route cost"
EXECUTION_CORRECTION_FREEZE_COMMIT_SUBJECT = (
    "Freeze plan-aware JEPA execution correction amendment"
)
CONTRACT_FREEZE_ANCESTRY_POLICY = (
    "normally execution HEAD is the clean direct single-parent contract-freeze "
    "child of SOURCE_COMMIT; only validated TRAINING_SMOKE failure custody permits "
    "a clean no-merge linear descendant correction freeze with every prior freeze "
    "in ancestry, exact freeze subject, closure-covered diff, and no artifact reuse; "
    "source_freeze_commit equals contract_freeze_commit"
)

CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS = (
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preregistration_2026-08-27.md",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_contract.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_output_schema.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_evaluator_fixture.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_route_role_authority.json",
)

PRIOR_SMOKE_FAILURE_CUSTODY_RECORD_FIELDS = (
    "archive_path",
    "failure_receipt",
    "source_freeze_commit",
    "source_closure",
    "inventory",
    "files_reused",
)
EXECUTION_RETRY_POLICY = {
    "default": "NO_RETRY",
    "eligible_failure_phase": "TRAINING_SMOKE",
    "eligible_failure_receipt_exact": {
        "phase": "TRAINING_SMOKE",
        "full_training_epochs_completed": 0,
        "calibration_rows_opened": 0,
        "heldout_rows_opened": 0,
        "final_checkpoint_published": False,
        "partial_artifacts_reusable": False,
        "nothing_running": True,
    },
    "eligible_retry_action": "WHOLLY_FRESH_CORRECTED_ATTEMPT",
    "automatic_retry": False,
    "same_source_freeze_retry": False,
    "prior_artifact_reuse": False,
    "active_experiment_processes_required": 0,
    "corrected_freeze": {
        "clean_worktree": True,
        "no_merge_linear_descendant_of_source": True,
        "every_prior_failure_source_freeze_must_be_strict_ancestor": True,
        "head_commit_subject": CONTRACT_FREEZE_COMMIT_SUBJECT,
        "changed_paths_must_be_in_source_closure_or_generated_authority": True,
        "source_closure_and_route_role_authority_must_validate_exactly": True,
        "immutable_scientific_authority_paths": list(
            CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS
        ),
        "immutable_authority_reference": (
            "byte hashes in every failed attempt's archived pre-smoke "
            "receipts/source_closure.json"
        ),
        "permitted_changes_after_fit_open": (
            "Python implementation/tests covered by the source closure and the "
            "refreshed tracked source-closure receipt only"
        ),
        "mutable_python_correction_required_against_every_archive": True,
        "empty_or_closure_only_descendant_retry": False,
        "scientific_preregistration_loss_gate_schema_role_or_policy_change": False,
    },
    "prior_smoke_failure_custody": {
        "field": "prior_smoke_failure_custody",
        "normal_value": [],
        "include_every_discovered_prior_smoke_failure_archive": True,
        "record_fields": list(PRIOR_SMOKE_FAILURE_CUSTODY_RECORD_FIELDS),
        "failure_receipt_fields": ["path", "sha256", "bytes", "content_digest"],
        "source_closure": (
            "mandatory exact {path,sha256,bytes,content_digest} binding to the "
            "failed attempt's pre-smoke receipts/source_closure.json snapshot"
        ),
        "inventory_fields": ["files", "bytes", "manifest_sha256"],
        "files_reused_required_value": 0,
        "bind_at_runtime_before_fresh_attempt": True,
        "persist_in": ["preexecution", "persistence", "result"],
    },
    "later_failure": {
        "definition": "any failure after TRAINING_SMOKE",
        "retry_allowed": False,
    },
}

EXECUTION_CORRECTION_FAILED_ARCHIVE = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    ".plan_aware_monotone_jepa_cost_v1.failed-1787837464936614634-641931"
)
BASE_SCIENTIFIC_AUTHORITY_BINDINGS = {
    "preregistration": {
        "path": "docs/lewm_plan_aware_monotone_jepa_cost_v1_preregistration_2026-08-27.md",
        "sha256": "9c06e0788fc30a02201678ebc42857bea644bdad75f4aaa20ec03c5d7c6718e6",
        "bytes": 9_419,
    },
    "contract": {
        "path": "docs/lewm_plan_aware_monotone_jepa_cost_v1_contract.json",
        "sha256": "f79146ae2183d18d289c691cc41a9326e9ea0c35ff8ae6fcfbf44d0f8edc9604",
        "content_digest": SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
        "bytes": 38_779,
    },
    "output_schema": {
        "path": "docs/lewm_plan_aware_monotone_jepa_cost_v1_output_schema.json",
        "sha256": "ed4d4bc21e0edc841ac3ed339d8e6918aabb32ad389084f9b013a0828d4479d4",
        "content_digest": (
            "e66798b015a2068f5be252dd3c0e4bc0b84aef4cf262d9b27b38089d65098338"
        ),
        "bytes": 19_364,
    },
    "evaluator_fixture": {
        "path": "docs/lewm_plan_aware_monotone_jepa_cost_v1_evaluator_fixture.json",
        "sha256": "6a53ca01a35b3edd2494126c1093b8a629789eac4941b79b5bf36f3ef91ce2c1",
        "content_digest": (
            "440e21d4bb557a16e8d60a01eb7a0708438ff56b9f4c0947b371ed30a11478e3"
        ),
        "bytes": 1_420,
    },
    "route_role_authority": {
        "path": "docs/lewm_plan_aware_monotone_jepa_cost_v1_route_role_authority.json",
        "sha256": "fa06b4bcfe10608625ae6dc586a9946755ac11d9f8b69a5895f049a3a5f59cc4",
        "content_digest": (
            "0f338807d55475a6059d742393c7c1a05e47fe4ee86b9588f364dd78888d154d"
        ),
        "bytes": 10_110,
    },
    "source_closure": {
        "path": "docs/lewm_plan_aware_monotone_jepa_cost_v1_source_closure.json",
        "sha256": "56e135c5acb015ff01a833ae7708a4e0f0f7d6b65ddda1693a9bd03221d054e7",
        "content_digest": (
            "19d450c44709419225bbe1588f697119fc73bd95c0a323c4e6113c28276bffd8"
        ),
        "bytes": 19_097,
    },
}
EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS = (
    {
        "path": "aggregates/stage_a_gate_evidence.json",
        "sha256": "18c4bf82cff32718820373da674a3f8838262ba18b40bfa1cf78b049cc854dd0",
        "bytes": 5_110,
    },
    {
        "path": "checkpoints/latent_true_future_final_epoch_060.pt",
        "sha256": "e2d34a764e86bc35e881d960a0a1e9f691a89c621a9f56134f0cfe21de7b8415",
        "bytes": 948_821,
    },
    {
        "path": "checkpoints/no_latent_final_epoch_060.pt",
        "sha256": "fc2715c65156dddceebbf091728ffca32e3ad40ff75e65696ee7b4f1da89027b",
        "bytes": 117_533,
    },
    {
        "path": "ledgers/route_only_targets.jsonl",
        "sha256": "9e14f309a92ba95fdc37f4ee7d8d0ae5d8b984634facce874c3ff2c0d1f42f32",
        "bytes": 394_179,
    },
    {
        "path": "ledgers/stage_a_raw_cost_rereduced.jsonl",
        "sha256": "ca2951df537d6781408395fd67adb75be32385bb8a9050ea66988c4e0e3e1cf8",
        "bytes": 666_894,
    },
    {
        "path": "ledgers/stage_a_true_future.jsonl",
        "sha256": "da34fed8975fbe5a61e3ea3da563466e612a08fa7eab1c6bc4d993f1af72af79",
        "bytes": 2_690_554,
    },
    {
        "path": "ledgers/training_epochs.jsonl",
        "sha256": "80e409850701ecb34505815a42986e3fc939d62c70bc4f2d28e7c001ff5f8c34",
        "bytes": 47_966,
    },
    {
        "path": "logs/stage_b_proprio_predictor_materialisation.log",
        "sha256": "1984906d2d7d19aadad34702b0b38dc1ab65e68167ea2dc3465fb2188cd24d4f",
        "bytes": 9_305,
    },
    {
        "path": "receipts/evaluation_contract.json",
        "sha256": "7a87c76d626f65723b00fd5f6689dc0ca5fda89987e10130a3b8d95382924106",
        "bytes": 21_519,
    },
    {
        "path": "receipts/failure.json",
        "sha256": "0a0f63d5beee712134b80217c87d815cd8d7ed5800d49cd7246eb687dc2e6448",
        "bytes": 9_014,
    },
    {
        "path": "receipts/preexecution.json",
        "sha256": "138067161d369cc45290a6fb1154d8b49c09304815ceb228273c748bee66f210",
        "bytes": 9_033,
    },
    {
        "path": "receipts/source_closure.json",
        "sha256": "56e135c5acb015ff01a833ae7708a4e0f0f7d6b65ddda1693a9bd03221d054e7",
        "bytes": 19_097,
    },
    {
        "path": "receipts/stage_b_gate.json",
        "sha256": "49d5fe2788b5326c718d39ed3b58f618bf25bca1b8d3aa3886d400d9227cafd9",
        "bytes": 1_052,
    },
    {
        "path": "receipts/training.json",
        "sha256": "6e8d4c9fc968459a1014863abeb2f0cff2668b61e44ea4a15149cf93f58fd00a",
        "bytes": 42_718,
    },
    {
        "path": "receipts/training_smoke.json",
        "sha256": "834f2275e09aaaa094c7a1ab7c20a360644036ed38bceb3cc3d71baced339505",
        "bytes": 1_977,
    },
)
EXECUTION_CORRECTION_ARCHIVE_INVENTORY = {
    "files": 15,
    "bytes": 4_984_772,
    "manifest_sha256": (
        "e8a70f94d56fede1d86c6fc63f2fc73c25c28080b18f020d917fc6567b55f20d"
    ),
    "rows": [copy.deepcopy(row) for row in EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS],
}
EXECUTION_CORRECTION_RECEIPT_CONTENT_DIGESTS = {
    "receipts/preexecution.json": (
        "c4e40a2e8c12cb7cab575079ac27405eeff3098e4753155945c4bd940d0c08ec"
    ),
    "receipts/source_closure.json": (
        "19d450c44709419225bbe1588f697119fc73bd95c0a323c4e6113c28276bffd8"
    ),
    "receipts/training_smoke.json": (
        "57b2e8bf5f7776584f9d9459a543be47e1108b679a4cc80beba4444e93d4bf5f"
    ),
    "receipts/training.json": (
        "120b088a35f578eb43966b289a70de226fe846ec111268d37520e9a9293a59f8"
    ),
    "receipts/evaluation_contract.json": (
        "5a5e8421023066f41c7e91fb74c4b5239b68379f125919ec7d9ad45f9961516b"
    ),
    "receipts/stage_b_gate.json": (
        "b0bb015a3deccb7a8d9c37fdf94f86724001a2df81f289f46828482d5376295f"
    ),
    "receipts/failure.json": (
        "05cdaf9fd9ece7de818c68092265d2075e88a44a5739ca499cfeb378f1b70212"
    ),
}

EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS = (
    "checkpoints/latent_true_future_final_epoch_060.pt",
    "checkpoints/no_latent_final_epoch_060.pt",
    "ledgers/route_only_targets.jsonl",
    "ledgers/stage_a_raw_cost_rereduced.jsonl",
    "ledgers/stage_a_true_future.jsonl",
    "ledgers/training_epochs.jsonl",
    "receipts/training.json",
    "receipts/training_smoke.json",
)
EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS = {
    "receipts/evaluation_contract.json": (
        "source_freeze_commit",
        "content_digest",
    ),
    "aggregates/stage_a_gate_evidence.json": (
        "source_freeze_commit",
        "evaluation_contract_content_digest",
        "content_digest",
    ),
    "receipts/stage_b_gate.json": (
        "contract_freeze_commit",
        "evaluation_contract.sha256",
        "stage_a_gate_evidence.sha256",
        "content_digest",
    ),
}
EXECUTION_CORRECTION_NORMALIZED_REPLAY_DIGESTS = {
    "receipts/evaluation_contract.json": (
        "116f469cafc1b2a173e46315ff193bd2913b600609a0765f598cee89e959a2bd"
    ),
    "aggregates/stage_a_gate_evidence.json": (
        "682ebca8c0a0ee3c41aa5dc75b8e73ef0ed2b473819afab9fe86201dba2241d7"
    ),
    "receipts/stage_b_gate.json": (
        "9326fa447f286673a1b109b3115adc4af2895694076999198b6e13d7052f3920"
    ),
}
EXECUTION_CORRECTION_FORBIDDEN_BEFORE_REPLAY_PREFIXES = (
    "stage_b/",
    "stage_c/",
)
EXECUTION_CORRECTION_FORBIDDEN_BEFORE_REPLAY_GLOBS = (
    "ledgers/stage_b_*",
    "ledgers/stage_c_*",
    "aggregates/stage_b_*",
    "aggregates/stage_c_*",
    "receipts/stage_b_proprio_*",
    "receipts/stage_c_*",
    "logs/stage_b_*",
    "logs/stage_c_*",
)
EXECUTION_CORRECTION_ENVIRONMENT_PROBE = {
    "failure_cause": "CONDITIONAL_CHILD_ENVIRONMENT_CONTAMINATION",
    "failed_child_environment": {
        "PYTHONPATH": "/usr/lib/python3/dist-packages",
        "selected_typing_extensions": {
            "path": "/usr/lib/python3/dist-packages/typing_extensions.py",
            "sha256": "4da413a94b4b5196b8cb390a1358630a10c858f987d72daa1b4157617f248a09",
            "bytes": 117_599,
            "Sentinel_present": False,
        },
    },
    "only_authorised_environment_change": {
        "remove": [
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONSTARTUP",
        ],
        "set_shared": {"PYTHONNOUSERSITE": "1"},
        "per_interpreter": {
            "cpu_child": {
                "interpreter": (
                    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan/bin/python"
                ),
                "VIRTUAL_ENV": (
                    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan"
                ),
                "PATH_prepend": (
                    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan/bin"
                ),
            },
            "gpu_child": {
                "interpreter": "/home/andrewknowles/TinyQuadJEPA/bin/python",
                "VIRTUAL_ENV": "/home/andrewknowles/TinyQuadJEPA",
                "PATH_prepend": "/home/andrewknowles/TinyQuadJEPA/bin",
            },
        },
        "interpreter_flags": ["-E", "-s"],
        "scope": (
            "the complete conditional helper chain: helper parent, CPU context-state "
            "children, GPU predictor children and Stage-C children"
        ),
    },
    "required_preflight": {
        "flags": ["-E", "-s"],
        "cpu_child": {
            "interpreter": (
                "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                "genesis_render_vulkan/bin/python"
            ),
            "typing_extensions": {
                "path": (
                    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan/lib/python3.12/site-packages/"
                    "typing_extensions.py"
                ),
                "sha256": "433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8",
                "bytes": 160_429,
                "Sentinel_present": True,
            },
            "pydantic_core": {
                "path": (
                    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan/lib/python3.12/site-packages/"
                    "pydantic_core/__init__.py"
                ),
                "sha256": "9cad6292b75254af606a970aae9bff6e5ae9f0b08089cd63acafa391b60794d2",
                "bytes": 5_115,
                "version": "2.46.4",
            },
            "genesis": {
                "path": (
                    "/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/venvs/"
                    "genesis_render_vulkan/lib/python3.12/site-packages/"
                    "genesis/__init__.py"
                ),
                "sha256": "bb47cd957ba9cfae37489adc46631f8b97cffb8d14ee4cc90c51336a44292abb",
                "bytes": 17_972,
                "version": "0.3.14",
                "import_required": True,
            },
        },
        "gpu_child": {
            "interpreter": "/home/andrewknowles/TinyQuadJEPA/bin/python",
            "typing_extensions": {
                "path": (
                    "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/"
                    "site-packages/typing_extensions.py"
                ),
                "sha256": "433d11d170d3a24d2eb065ebc1bfe848cea7e3d7ce68567ab52bea2d4c2f7ed8",
                "bytes": 160_429,
                "Sentinel_present": True,
            },
            "torch": {
                "path": (
                    "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/"
                    "site-packages/torch/__init__.py"
                ),
                "sha256": "a75f512e441b3c35a63561a06a958c54ce917e3913913564a429adcbdbac6c3d",
                "bytes": 103_060,
                "version": "2.10.0.dev20250926+rocm6.3",
                "import_required": True,
            },
        },
    },
}
EXECUTION_CORRECTION_POLICY = {
    "authorisation": "ONE_BOUND_EXECUTION_ONLY_CORRECTION_ATTEMPT",
    "failed_source_freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
    "scientific_authority_contract_sha256": SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
    "correction_freeze_commit_subject": EXECUTION_CORRECTION_FREEZE_COMMIT_SUBJECT,
    "fresh_attempt_required": True,
    "failed_archive_files_reused": 0,
    "direct_checkpoint_or_ledger_reuse": False,
    "automatic_retry": False,
    "maximum_fresh_attempts": 1,
    "permitted_change": (
        "sole execution-semantic change: conditional child environment "
        "construction and import probe"
    ),
    "non_scientific_plumbing_changes": [
        "separate amendment authority and source closure",
        "runtime custody validation and persistence",
        "pre-Stage-B scientific replay validation",
        "tests for the amendment and lifecycle barriers",
    ],
    "forbidden_changes": [
        "ranker architecture, weights, seed, optimizer, epochs or loss",
        "panel, split, route roles, targets or admissibility conditioning",
        "metric, threshold, gate, primary classification or next-decision rule",
        "predictor checkpoint, source mapping or Stage-C derangement semantics",
        "deployment-safety scope or claim",
    ],
    "pre_stage_b_replay_gate": {
        "byte_exact_paths": list(EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS),
        "normalized_exact_paths": {
            path: {
                "excluded_paths": list(
                    EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS[path]
                ),
                "scientific_content_digest": (
                    EXECUTION_CORRECTION_NORMALIZED_REPLAY_DIGESTS[path]
                ),
            }
            for path in EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS
        },
        "excluded_field_policy": (
            "only wrapper source/closure/contract/amendment bindings and their "
            "dependent content digests are excluded; scientific experiment digest, "
            "training history, model parameter digests, score/outcome rows, "
            "derangement maps, gate criteria and gate booleans remain included"
        ),
        "must_pass_before_stage_b_scientific_or_materialisation_child_start": True,
        "outcome_free_pre_fit_import_probe_exception": {
            "allowed_before_replay": True,
            "scope": ["cpu_child", "gpu_child"],
            "fit_outcome_rows_opened": 0,
            "calibration_rows_opened": 0,
            "heldout_rows_opened": 0,
            "tensor_rows_opened": 0,
            "training_steps": 0,
            "scientific_inference_or_materialisation": False,
        },
        "stage_b_or_stage_c_artifacts_before_gate_forbidden": {
            "prefixes": list(EXECUTION_CORRECTION_FORBIDDEN_BEFORE_REPLAY_PREFIXES),
            "globs": list(EXECUTION_CORRECTION_FORBIDDEN_BEFORE_REPLAY_GLOBS),
            "stage_b_gate_receipt_is_the_only_stage_b_named_exception": True,
        },
    },
    "scientific_replay_digest_policy": (
        "all scientific keyed maps and derangements retain "
        "SCIENTIFIC_AUTHORITY_CONTRACT_SHA256; the amended wrapper contract digest "
        "must not perturb scientific content"
    ),
    "failure_after_correction": "NO_FURTHER_RETRY",
}
EXECUTION_CORRECTION_ALLOWED_CHANGED_PATHS = (
    "lewm/safety/plan_aware_monotone_jepa_cost_v1_contract.py",
    "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py",
    "scripts/materialize_plan_aware_proprio_predictor_substitution_v1.py",
    "lewm/tests/test_plan_aware_monotone_jepa_cost_v1_contract.py",
    "lewm/tests/test_evaluate_plan_aware_monotone_jepa_cost_v1.py",
    "lewm/tests/test_materialize_plan_aware_proprio_predictor_substitution_v1.py",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_preregistration_2026-08-27.md",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_contract.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_output_schema.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_evaluator_fixture.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_source_closure.json",
)
EXECUTION_CORRECTION_REQUIRED_CHANGED_PATHS = (
    "lewm/safety/plan_aware_monotone_jepa_cost_v1_contract.py",
    "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py",
    "scripts/materialize_plan_aware_proprio_predictor_substitution_v1.py",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_preregistration_2026-08-27.md",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_contract.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_output_schema.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_evaluator_fixture.json",
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_source_closure.json",
)

CONDITION_IDS = (
    "KINEMATIC_PLUS_NO_LATENT_RESIDUAL",
    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL",
)
SHARED_BASE_SEED_SUBKEY = "SHARED_BASE_RESIDUAL"
KEYED_SEED_NAMESPACE = "PLAN_AWARE_MONOTONE_JEPA_COST_V1/KEYED_SEED_V1"


def derive_keyed_seed(key: str, *, seed_family: int = RANKER_SEED) -> dict[str, Any]:
    """Derive one deterministic Torch seed from the frozen family and exact key."""

    if not isinstance(key, str) or not key or "\x00" in key:
        raise ValueError("keyed-seed key must be non-empty UTF-8 text without NUL")
    if isinstance(seed_family, bool) or not isinstance(seed_family, int) or not (
        0 <= seed_family < 2**64
    ):
        raise ValueError("keyed-seed family must be an unsigned 64-bit integer")
    preimage = (
        KEYED_SEED_NAMESPACE.encode("utf-8")
        + b"\x00"
        + seed_family.to_bytes(8, "big", signed=False)
        + b"\x00"
        + key.encode("utf-8")
    )
    digest = hashlib.sha256(preimage).digest()
    return {
        "key": key,
        "key_utf8_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        "keyed_seed_sha256": digest.hex(),
        "torch_seed": int.from_bytes(digest[:8], "big", signed=False) & (2**63 - 1),
    }


CONDITION_KEYED_SEEDS = {
    "schema": "plan_aware_condition_keyed_seed_v1",
    "namespace": KEYED_SEED_NAMESPACE,
    "preimage": "namespace UTF-8 + NUL + seed_family uint64 big-endian + NUL + key UTF-8",
    "seed_family": RANKER_SEED,
    "condition_keys": {
        condition: derive_keyed_seed(condition) for condition in CONDITION_IDS
    },
    "shared_base_subkey": derive_keyed_seed(SHARED_BASE_SEED_SUBKEY),
    "initialisation_policy": {
        "both_base_branches": "SHARED_BASE_RESIDUAL torch_seed",
        "latent_only_parameters": (
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL torch_seed"
        ),
        "no_latent_condition_key_use": "custody identity; base uses shared-base subkey",
        "byte_identical_base_initialisation_required": True,
    },
}
CHECKPOINT_SEED_METADATA = {
    condition: {
        "seed_family": RANKER_SEED,
        "condition_id": condition,
        "condition_key_sha256": CONDITION_KEYED_SEEDS["condition_keys"][condition][
            "key_utf8_sha256"
        ],
        "condition_keyed_seed_sha256": CONDITION_KEYED_SEEDS["condition_keys"][
            condition
        ]["keyed_seed_sha256"],
        "condition_torch_seed": CONDITION_KEYED_SEEDS["condition_keys"][condition][
            "torch_seed"
        ],
        "shared_base_subkey": SHARED_BASE_SEED_SUBKEY,
        "shared_base_subkey_sha256": CONDITION_KEYED_SEEDS["shared_base_subkey"][
            "key_utf8_sha256"
        ],
        "shared_base_keyed_seed_sha256": CONDITION_KEYED_SEEDS[
            "shared_base_subkey"
        ]["keyed_seed_sha256"],
        "shared_base_torch_seed": CONDITION_KEYED_SEEDS["shared_base_subkey"][
            "torch_seed"
        ],
    }
    for condition in CONDITION_IDS
}

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "plan_aware_monotone_jepa_cost_v1"
)
TRACKED_PREREGISTRATION_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_preregistration_2026-08-27.md"
)
TRACKED_CONTRACT_PATH = Path("docs/lewm_plan_aware_monotone_jepa_cost_v1_contract.json")
TRACKED_OUTPUT_SCHEMA_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_output_schema.json"
)
TRACKED_FIXTURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_evaluator_fixture.json"
)
TRACKED_SOURCE_CLOSURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_source_closure.json"
)
TRACKED_ROUTE_ROLE_AUTHORITY_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_route_role_authority.json"
)
TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_contract.json"
)
TRACKED_EXECUTION_CORRECTION_PREREGISTRATION_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_preregistration_2026-08-27.md"
)
TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_output_schema.json"
)
TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_evaluator_fixture.json"
)
TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_"
    "execution_correction_amendment_source_closure.json"
)
TRACKED_RESULT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_result_2026-08-27.json"
)
TRACKED_REPORT_PATH = Path(
    "docs/lewm_plan_aware_monotone_jepa_cost_v1_result_2026-08-27.md"
)

ROLE_IDS = ("fit", "calibration", "heldout")
ROLE_STATE_COUNTS = {"fit": 32, "calibration": 8, "heldout": 8}
ROLE_ROW_COUNTS = {role: count * 12 for role, count in ROLE_STATE_COUNTS.items()}
FAMILY_IDS = (
    "large_enclosed_maze",
    "medium_enclosed_maze",
    "small_enclosed_maze",
    "loop_alias_stress",
)
HORIZON_IDS = ("H1", "H2", "H3")
CANDIDATE_INDICES = tuple(range(12))

LATENT_SOURCE_IDS = (
    "TRUE_FUTURE",
    "R1_RGB_ONE_STEP",
    "RR_RGB_ROLLOUT",
    "P1_PROPRIO_ONE_STEP",
    "PR_PROPRIO_ROLLOUT",
)
PREDICTED_SOURCE_IDS = LATENT_SOURCE_IDS[1:]
STAGE_IDS = ("STAGE_A_TRUE_FUTURE", "STAGE_B_PREDICTOR_SUBSTITUTION", "STAGE_C_ATTRIBUTION")

PRIMARY_CLASSIFICATIONS = (
    "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL",
    "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO",
    "KINEMATIC_BASELINE_DOMINANT",
    "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
)
SECONDARY_CLASSIFICATIONS = (
    "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
    "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE",
    "CANDIDATE_SPECIFIC_LATENT_ROUTE_INFORMATION_USED",
    "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL",
    "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS",
    "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION",
    "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED",
    "PROPRIOCEPTIVE_SUBSTITUTION_TENDENCY",
    "PROPRIOCEPTIVE_SUBSTITUTION_NOT_SUPPORTED",
    "MULTIMODAL_ROUTE_DEPENDENCE",
    "VISUAL_ROUTE_DEPENDENCE",
    "ROLLOUT_ROUTE_INFORMATION_SIGNAL",
    "WRONG_PLANNING_READOUT",
    "ENCODER_ROUTE_INFORMATION_INSUFFICIENT",
    "PREDICTOR_ROUTE_GEOMETRY_LOSS",
)
NEXT_EXPERIMENT_IDS = (
    "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1",
    "PLAN_AWARE_PREDICTOR_TRAINING_V1",
    "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
)
NEXT_EXPERIMENT_SPECIFICATIONS = {
    "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1": {
        "status": "SPECIFY_ONLY_NOT_AUTHORISED_IN_THIS_EXPERIMENT",
        "admissibility": "exact Genesis contact and successor viability oracle",
        "route_ranking": "fixed plan-aware JEPA route cost",
        "execution": "execute a short prefix, reobserve, and replan",
        "candidate_generation": (
            "begin with the frozen fixed candidate bank; CEM or MPPI may follow "
            "only after fixed-bank closed-loop success"
        ),
        "separate_outcomes": [
            "contact",
            "successor_viability",
            "route_progress",
            "abstention",
        ],
        "deployment_safety_claim": False,
    },
    "PLAN_AWARE_PREDICTOR_TRAINING_V1": {
        "status": "SPECIFY_ONLY_NOT_AUTHORISED_IN_THIS_EXPERIMENT",
        "predictor_seeds": 1,
        "objective": "route-consistency for the frozen plan-aware route ranking",
        "route_only": True,
        "safety_target_or_model_change": False,
        "protected_contact_scope_change": False,
        "deployment_safety_claim": False,
    },
    "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1": {
        "status": "SPECIFY_ONLY_NOT_AUTHORISED_IN_THIS_EXPERIMENT",
        "task": (
            "a local subgoal whose successful route requires temporary movement "
            "away from the direct goal because of an intervening obstacle"
        ),
        "minimum_geometrically_plausible_route_alternatives": 2,
        "admissibility": "exact Genesis contact and successor viability oracle",
        "topological_memory_initially": False,
        "beacon_layer_initially": False,
        "deployment_safety_claim": False,
    },
}

FIT_STATE_OPTIMIZATION_STATUSES = (
    "CONTRIBUTING",
    "SKIPPED_ZERO_ADMISSIBLE",
    "SKIPPED_SINGLETON_ADMISSIBLE",
)
FIT_OPTIMIZATION_RECEIPT_FIELDS = (
    "fit_states_total",
    "fit_state_ids_total",
    "fit_states_contributing",
    "fit_state_ids_contributing",
    "fit_states_skipped_zero_admissible",
    "fit_state_ids_skipped_zero_admissible",
    "fit_states_skipped_singleton_admissible",
    "fit_state_ids_skipped_singleton_admissible",
    "epoch_average_denominator",
)

ACTIVE_POLICIES = (
    "EVALUATION_FIRST_SINGLE_SEED",
    "ROW_LEVEL_EVIDENCE_PERSISTENCE",
    "DEVELOPMENT_MODE_END_TO_END_EXECUTION",
    "EXPLORATORY_SINGLE_SEED_FIRST",
)

PRESERVED_REQUIREMENTS_CLASSIFICATIONS = (
    "REQUIREMENTS_ACQUISITION_REQUIRED",
    "PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED",
    "SIMULATED_CONTACT_PROXY_SCOPE_ONLY",
    "REPLANNING_INTERFACE_UNRESOLVED",
    "GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING",
)
PRESERVED_PREDECESSOR_FACTS = (
    "RAW_LATENT_GOAL_COST_NO_GO",
    "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
    "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS_NOT_SUPPORTED",
)
PRESERVED_PREDECESSOR_FACT_SCOPE = {
    "authority": "JEPA_LOCAL_WAYPOINT_PLANNING_COST_QUALIFICATION_V1",
    "scope": "predecessor raw token-wise goal-cosine planning-cost assay",
    "immutable_historical_facts": list(PRESERVED_PREDECESSOR_FACTS),
    "successor_plan_aware_result_may_overwrite": False,
    "incremental_not_supported_scope": (
        "JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS_NOT_SUPPORTED applies "
        "to the predecessor raw-cost result only"
    ),
}
PRESERVED_PREDECESSOR_NARRATIVE = (
    "rollout training improves direct counterfactual future fidelity at H1–H4.",
    (
        "rollout training improves selected action-specific retrieval metrics, "
        "strongest at H3–H4."
    ),
    (
        "predicted latents retain partial occupancy information but remain "
        "substantially below true-target occupancy."
    ),
    "the registered proprioception interaction was broadly null.",
    "planning utility was not tested by the predictor qualification assay.",
    (
        "Raw token-wise cosine distance between future V-JEPA latents and the "
        "virtual goal-view latent did not provide useful route ordering, even "
        "with true-future latents and oracle viability."
    ),
    (
        "The virtual goal renderer contained the floor plane but not the maze "
        "walls or landmarks. That result was therefore not a valid visual "
        "wall-avoidance test."
    ),
)

CHECKPOINT_BINDINGS = {
    "R1_RGB_ONE_STEP": {
        "path": (
            "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
            "seed_2026080901/seed_2026080901_rgb_one_step_epoch21.pt"
        ),
        "sha256": "20b6e3fa2a2d3c3ec2c20ea37e524f9c2872fdcfd5226b114822efa26872261a",
        "bytes": 206_534_551,
        "use_proprio": False,
    },
    "RR_RGB_ROLLOUT": {
        "path": (
            "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
            "seed_2026080901/seed_2026080901_rgb_rollout_epoch21.pt"
        ),
        "sha256": "75e7a8f5eb5416100dd91fdd07c6aeae1c8fa2255ef189bfde2a5ce300f881b4",
        "bytes": 206_534_551,
        "use_proprio": False,
    },
    "P1_PROPRIO_ONE_STEP": {
        "path": (
            "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
            "seed_2026080901/seed_2026080901_proprio_one_step_epoch21.pt"
        ),
        "sha256": "41d1c5a48d7adacf2e2b698318782de29c7b95342181bdf5fd5578d35346f1d1",
        "bytes": 206_691_255,
        "use_proprio": True,
    },
    "PR_PROPRIO_ROLLOUT": {
        "path": (
            "/home/andrewknowles/.cache/lewm_go2_temporal_v03/factorial_v1/"
            "seed_2026080901/seed_2026080901_proprio_rollout_epoch21.pt"
        ),
        "sha256": "75ab2a5dd5c48ebb2f33935d962d957c4e62eab3427ce0ad8108d690a1df9218",
        "bytes": 206_691_255,
        "use_proprio": True,
    },
}
ENCODER_BINDING = {
    "path": "/home/andrewknowles/.cache/vjepa2_1_vitl_dist_vitG_384.pt",
    "sha256": "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
    "bytes": 5_151_198_524,
}

PANEL_BINDINGS = {
    "state_manifest": {
        "path": ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json",
        "sha256": "da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37",
        "bytes": 51_066,
    },
    "split": {
        "path": ".generated/safe_local_waypoint_purpose_built_v1/split.json",
        "sha256": "ebef7db828a4c754432375818fd6b1eff0731cc3bc546ff2b69667b03abe56a8",
        "bytes": 975,
    },
    "branch_ledger": {
        "path": ".generated/safe_local_waypoint_purpose_built_v1/branch_labels.jsonl",
        "sha256": "9b25b227c3e4de11e68e4abee454c4251399fafb468458a4e0d65f89bc6cdf7c",
        "bytes": 1_532_376,
    },
    "route_labels": {
        "path": ".generated/safe_local_waypoint_route_intent_v2/route_intent_labels.jsonl",
        "sha256": "e8d33671502f717426836ec9a1039d445558b81e636ae105a81e2113151a8b69",
        "bytes": 558_131,
    },
    "route_data_audit": {
        "path": ".generated/safe_local_waypoint_route_intent_v2/data_audit.json",
        "sha256": "73381d7dc834813286b52f571a6b5d3370d04582fb5bf27beeee9447e5e4fd92",
        "bytes": 5_379,
    },
    "route_v2_result": {
        "path": ".generated/safe_local_waypoint_route_intent_v2/result.json",
        "sha256": "0dd4e3d7d6f10a7693bc51fcb71faf10e9ea89a881c2914787f1fd64c71a83e9",
        "bytes": 35_429,
        "authorised_field": "action_only.per_state[*].{state_id,class} only",
    },
    "true_future_index": {
        "path": ".generated/safe_local_waypoint_route_intent_v2/target_latent_index.json",
        "sha256": "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874",
        "bytes": 937_776,
        "records": 1_728,
        "shape": [768, 1024],
        "dtype": "float16",
    },
}

PREDECESSOR_TENSOR_PACKAGE = {
    "root": (
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
        "jepa_local_waypoint_planning_cost_qualification_v1"
    ),
    "tensor_index": {
        "path": "latents/tensor_index.json",
        "sha256": "2e54e1eac26e4412113410dba1da6de59159a482a5a701fb1f99b2a63fc8aaf3",
        "content_digest": "4505a0077a6591e013d220287978d1e6df5ff9bc8b7d7312ac70b702d722e151",
        "bytes": 2_847_399,
        "logical_records": 5_424,
        "unique_payloads": 5_376,
        "tensor_bytes": 8_456_404_992,
    },
    "batch_manifest": {
        "path": "latents/batch_manifest.json",
        "sha256": "a62e0de2f1c82b59fb6d5dab7924b536867ed8ca52bb0c51aa80e416baf02824",
        "content_digest": "295a4e3a2af115880ddf212c8c4ff89159c5910bc87712591a7d55f5c23fbad8",
        "bytes": 32_787,
    },
    "goal_view_index": {
        "path": "goal_views/index.json",
        "sha256": "6306dbf8fdba718791ba47285e4ada4f0d08ee0add6dde6e0d1845dc669a62ad",
        "content_digest": "91842fd854e738fd9d6d3c03f4e1715926ec4b1a1086dfc658b0f1e145fb85bf",
        "bytes": 110_941,
        "states": 48,
    },
    "context_reconstruction_index": {
        "path": "materialization/context_reconstruction_index.json",
        "sha256": "6026a330bd8e48fbd9ad37aabd808871ea3418e1178709da9d82af045165023e",
        "content_digest": "654848f37e4590cddfdd8a53c01268ba299eb77a61c641045d9a1143b4087faf",
        "bytes": 1_225_908,
        "states": 48,
        "proprio_related_keys": [],
    },
    "persistence_receipt": {
        "path": "receipts/persistence.json",
        "sha256": "bd531c2ad0a2da8c0993c5b964b8d5b37e18d48c3e6e241bea9442d0c06a95fb",
        "content_digest": "c4658c8c4f8e71e7cf8bd3db46a9f60d046180b7a8838bb78a875752fe59de89",
        "bytes": 948_462,
        "nothing_running": True,
    },
    "counts_by_kind": {
        "CONTEXT": 144,
        "CURRENT": 48,
        "GOAL": 48,
        "TRUE_FUTURE": 1_728,
        "ONE_STEP_PREDICTED": 1_728,
        "TWO_STEP_PREDICTED": 1_728,
    },
    "bound_48_state_sources": ["TRUE_FUTURE", "R1_RGB_ONE_STEP", "RR_RGB_ROLLOUT"],
    "unbound_48_state_sources": ["P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"],
    "p1_pr_status": (
        "checkpoints exist, but no P1/PR prediction tensors, prediction indexes, "
        "or proprio-input custody receipts are bound to the 48x12xH1-H3 panel"
    ),
}

PROPRIO_INPUT_BINDINGS = {
    "normalisation": {
        "path": "/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/proprio_norm_stats.json",
        "sha256": "9380b4c6d9b59099e43bba9898e1417c273f88075d1ed122401cbb3272e18f94",
        "stats_sha256": "f5ea58b29d79362d4d814ff1b4225b54a5c97fb95442c866def80b0c2c4c2fab",
        "bytes": 2_329,
    },
    "factorial_manifest": {
        "path": "/home/andrewknowles/.cache/lewm_go2_temporal_v03/proprio_v1/factorial_manifest.json",
        "sha256": "8bf59020d24e02fdb11948f3732220df839aa1c3bc8612392ce6baab6b8d629c",
        "bytes": 2_305_678,
        "panel": "legacy 20-state counterfactual panel; identity witness only",
    },
    "48_state_proprio_context": {
        "status": "MISSING_PROSPECTIVE_BINDING",
        "required_shape_per_state": [3, 5, 30],
        "required_records": 48,
        "future_proprioception": "forbidden",
    },
}

LEGACY_COUNTERFACTUAL_TENSOR_BINDINGS = {
    "status": "INELIGIBLE_IDENTITY_WITNESSES_ONLY",
    "panel": {
        "states": 20,
        "candidates_per_state": 12,
        "horizons": 4,
        "state_identity_prefix": "v12-",
        "reason_ineligible": (
            "not the 48-state purpose-* local-waypoint panel and carries no "
            "frozen Route-Intent V2 goal/role identity"
        ),
    },
    "per_state_prediction_shape": [12, 4, 768, 1024],
    "dtype": "float16",
    "prediction_indexes": {
        "R1_RGB_ONE_STEP": {
            "path": (
                ".generated/go2_counterfactual_fidelity_v1_2/predictor_assay/"
                "prediction_ledgers/seed_2026080901_rgb_one_step/predictions_index.json"
            ),
            "sha256": "207177a509413ba7f9e27921f717b10d3d5ec2c2204d2700d9ad1230ae8bf3a5",
            "predictions_index_digest": "371ddeb09fd6fc447ab285a45c91661002cad25e67277b060a29c8b1d69cd4d9",
            "bytes": 257_217,
        },
        "RR_RGB_ROLLOUT": {
            "path": (
                ".generated/go2_counterfactual_fidelity_v1_2/predictor_assay/"
                "prediction_ledgers/seed_2026080901_rgb_rollout/predictions_index.json"
            ),
            "sha256": "2384734ee8d9130088096baac1ddedae5878c94d93e86f2b84592e365a3fe45d",
            "predictions_index_digest": "7e64ffd8e4ef5e162148fb5160588c6284c23ca9bcd65baf713045d26ddcd299",
            "bytes": 256_696,
        },
        "P1_PROPRIO_ONE_STEP": {
            "path": (
                ".generated/go2_counterfactual_fidelity_v1_2/predictor_assay/"
                "prediction_ledgers/seed_2026080901_proprio_one_step/predictions_index.json"
            ),
            "sha256": "593f265aeb509107e2f08e529b012d122ff87569bb04794235ee6ee1bb8a5b38",
            "predictions_index_digest": "11fda26865f57c3ef602626c00d919f143d87f73b5a6a0c3bef843a47b000b58",
            "bytes": 259_301,
        },
        "PR_PROPRIO_ROLLOUT": {
            "path": (
                ".generated/go2_counterfactual_fidelity_v1_2/predictor_assay/"
                "prediction_ledgers/seed_2026080901_proprio_rollout/predictions_index.json"
            ),
            "sha256": "e369ff60ce7ec231147acf21cdc47979a81754963cb8c2ebe75c43e30d675190",
            "predictions_index_digest": "1c8a13aac3a3351e3bd05b41fab00a524d4ab785babc6faeb1e8c4ca449f0f14",
            "bytes": 258_780,
        },
    },
}

PREDECESSOR_RESULT_BINDINGS = {
    "json": {
        "path": "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_result_2026-08-26.json",
        "sha256": "f5a8959338dcbdde38544e5cc37bba4d6ba0246b7e0abd8f6525280eb881540b",
        "bytes": 75_960,
        "custody": "frozen comparator authority; values excluded from successor design and tuning",
    },
    "markdown": {
        "path": "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_result_2026-08-26.md",
        "sha256": "47f9e698cfbdbc5c05d8d8ec3df2f4db9e744c7ebc0b8904a6d392d398f481ee",
        "bytes": 18_880,
        "custody": "frozen narrative authority; detailed values excluded from successor design and tuning",
    },
}

PREDECESSOR_CANDIDATE_EVIDENCE_BINDING = {
    "path": (
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
        "jepa_local_waypoint_planning_cost_qualification_v1/"
        "evidence/candidate_evidence.jsonl.gz"
    ),
    "sha256": "9084b501f2d47a4d366c5739caf55bbe5f103f36e97023ea7db081103c4efe02",
    "bytes": 603_913,
    "rows": 1_728,
    "schema": "jepa_local_waypoint_candidate_evidence_v1",
    "predecessor_sources": [
        "TRUE_FUTURE",
        "ONE_STEP_PREDICTED",
        "TWO_STEP_PREDICTED",
    ],
    "successor_source_mapping": {
        "RAW_TRUE_FUTURE_GOAL_COSINE": "TRUE_FUTURE",
        "RAW_R1_GOAL_COSINE": "ONE_STEP_PREDICTED",
        "RAW_RR_GOAL_COSINE": "TWO_STEP_PREDICTED",
    },
    "rows_per_source": 576,
    "score_field": "cost_h3",
    "score_transform": "score = -cost_h3 (higher is better)",
    "required_reduction_fields": [
        "schema",
        "state_id",
        "family",
        "role",
        "candidate_index",
        "source",
        "population_membership",
        "cost_h3",
        "oracle_route_fields_h3_primary",
        "immediate_contact_h1",
        "successor_safe_action_count",
        "successor_viable",
        "oracle_viability_admissible",
        "successor_nonviable",
        "stuck",
        "completed",
    ],
    "open_barrier": (
        "read-only only after both final ranker checkpoints are locked and the "
        "evaluation contract is frozen/heldout opening is authorised"
    ),
    "reduction_policy": (
        "apply the successor population-conditioned Borda ordering, tie-aware "
        "rank, and separate progress metrics to -cost_h3; never rerun cosine, "
        "predictor inference, or tensor inference"
    ),
    "historical_aggregate_policy": (
        "retain copied predecessor aggregates separately as historical and "
        "non-comparable; never substitute them for successor-metric re-reduction"
    ),
}
RAW_COST_REREDUCED_SOURCE_IDS = tuple(
    PREDECESSOR_CANDIDATE_EVIDENCE_BINDING["successor_source_mapping"]
)

ROLE_CLASS_MAPPING = {
    "TRANSLATIONAL_PROGRESS_AVAILABLE": "translational",
    "ALIGNMENT_PROGRESS_AVAILABLE": "alignment",
    "SAFE_HOLD_OR_ABSTAIN": "hold_abstain",
    "NO_SAFE_CANDIDATE": "hold_abstain",
}
ROUTE_ROLE_AUTHORITY_BINDING = {
    "path": (
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
        "jepa_local_waypoint_planning_cost_qualification_v1/aggregates/metrics.json"
    ),
    "sha256": "cefc11d4640a5a7ad6475d52b9a525e3d528775da3af206b82b1a1a77d9870d4",
    "content_digest": "c5da5782627446ba479bad4e04ff98468196422895d7d5ceed373fee8b46d864",
    "bytes": 7_281_659,
    "authority_path": (
        "by_source_population.KINEMATIC_ROUTE_BASELINE.ALL_CANDIDATES.per_state"
    ),
    "copied_fields": ["state_id", "family", "role", "route_population_class"],
    "derived_field": "route_role = ROLE_CLASS_MAPPING[route_population_class]",
    "policy": (
        "direct recovery of the already-frozen candidate-invariant role authority; "
        "no route-label outcome inference or class recomputation"
    ),
}
ROUTE_ROLE_RECEIPT_BINDING = {
    "path": str(TRACKED_ROUTE_ROLE_AUTHORITY_PATH),
    "sha256": "fa06b4bcfe10608625ae6dc586a9946755ac11d9f8b69a5895f049a3a5f59cc4",
    "content_digest": "0f338807d55475a6059d742393c7c1a05e47fe4ee86b9588f364dd78888d154d",
    "bytes": 10_110,
    "records": 48,
}
ROLE_ONE_HOT_ORDER = ("translational", "alignment", "hold_abstain")

BASE_FEATURE_LAYOUT = (
    ("waypoint", 6, "dx,dy,distance,desired_rel_heading,sin,cos"),
    ("route_role_one_hot", 3, "translational,alignment,hold_abstain"),
    ("requested_post_slew", 45, "[3,5,3] row-major"),
    ("applied_post_slew", 45, "[3,5,3] row-major"),
    ("previous_applied_command", 3, "vx,vy,yaw"),
    ("observed_control_history", 30, "[3,5,2] row-major"),
    ("deterministic_kinematic", 6, "x_m,y_m,yaw_rad,nominal_p_d,nominal_p_theta,score_anchor"),
)
QUERY_FEATURE_LAYOUT = (
    ("waypoint", 6, "dx,dy,distance,desired_rel_heading,sin,cos"),
    ("route_role_one_hot", 3, "translational,alignment,hold_abstain"),
    ("frozen_predictor_applied_active_plan", 30, "[3,10] row-major"),
    ("previous_active_command", 2, "vx,yaw"),
    ("observed_control_history", 30, "[3,5,2] row-major"),
)
BASE_FEATURE_DIM = sum(width for _, width, _ in BASE_FEATURE_LAYOUT)
QUERY_FEATURE_DIM = sum(width for _, width, _ in QUERY_FEATURE_LAYOUT)
TOKEN_DIM = 1024
TOKENS_PER_TIMEPOINT = 768
TOKEN_GRID_SHAPE = (24, 32)
LATENT_WIDTH = 64
LATENT_SEQUENCE_LENGTH = 4  # current, H1, H2, H3

NO_LATENT_PARAMETER_COUNT = (
    BASE_FEATURE_DIM * 128
    + 128
    + 128 * 64
    + 64
    + 64 * 1
    + 1
)
LATENT_PARAMETER_COUNT = (
    NO_LATENT_PARAMETER_COUNT  # paired kinematic+base residual path
    + 2 * TOKEN_DIM  # shared LayerNorm affine
    + TOKEN_DIM * LATENT_WIDTH
    + LATENT_WIDTH
    + QUERY_FEATURE_DIM * LATENT_WIDTH
    + LATENT_WIDTH
    + (LATENT_SEQUENCE_LENGTH * LATENT_WIDTH + BASE_FEATURE_DIM) * 256
    + 256
    + 256 * 128
    + 128
    + 128
    + 1
)

TRAINING_POLICY = {
    "models": {
        "NO_LATENT": {
            "input": "ordered BASE_FEATURE_LAYOUT only",
            "architecture": [BASE_FEATURE_DIM, 128, 64, 1],
            "activation": "GELU after hidden layers",
            "score": "-kinematic_rank_cost + learned_residual",
            "parameter_count": NO_LATENT_PARAMETER_COUNT,
            "parameter_cap_exclusive": 250_000,
        },
        "LATENT": {
            "base_input": BASE_FEATURE_DIM,
            "token_encoder": (
                "view [768,1024] as [24,32,1024] row-major, flatten the "
                "spatial grid in the same order, then shared token LayerNorm "
                "and shared Linear(1024,64)"
            ),
            "tokens_per_timepoint": TOKENS_PER_TIMEPOINT,
            "token_grid_shape": list(TOKEN_GRID_SHAPE),
            "token_grid_storage_order": "row-major",
            "token_flat_index": "y * 32 + x",
            "query": [QUERY_FEATURE_DIM, 64],
            "attention": (
                "one shared scaled-dot-product query over each of current,H1,H2,H3; "
                "no source- or horizon-specific parameters"
            ),
            "readout_input": LATENT_SEQUENCE_LENGTH * LATENT_WIDTH + BASE_FEATURE_DIM,
            "architecture": [
                LATENT_SEQUENCE_LENGTH * LATENT_WIDTH + BASE_FEATURE_DIM,
                256,
                128,
                1,
            ],
            "activation": "GELU after hidden layers",
            "score": (
                "-kinematic_rank_cost + paired base residual + latent residual"
            ),
            "parameter_count": LATENT_PARAMETER_COUNT,
            "parameter_cap_exclusive": 500_000,
        },
    },
    "seed": RANKER_SEED,
    "keyed_seed_derivation": copy.deepcopy(CONDITION_KEYED_SEEDS),
    "checkpoint_seed_metadata": copy.deepcopy(CHECKPOINT_SEED_METADATA),
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 60,
    "checkpoint": (
        "final epoch only; no early stopping, selection, full-training retry, or "
        "best-checkpoint choice; only the separate validated pretraining "
        "TRAINING_SMOKE correction policy can authorize a wholly fresh attempt"
    ),
    "loss": {
        "pairwise_weight": 1.0,
        "listwise_weight": 0.5,
        "residual_l2_weight": 1e-3,
        "listwise_temperature": 1.0,
        "pairwise": (
            "logistic loss with y_ij = sign(conditioned margin-Borda utility_i "
            "- utility_j) only when abs(delta) > 1e-12; otherwise y_ij = 0"
        ),
        "pairwise_utility_tolerance": PAIRWISE_UTILITY_TOLERANCE,
        "listwise": "cross-entropy from score softmax to margin-Borda target softmax",
        "residual": "mean squared learned residual before the fixed kinematic anchor",
    },
    "target": {
        "name": "EXISTING_LOCAL_WAYPOINT_MARGIN_BORDA_ROUTE_PREFERENCE",
        "ordered_components": ["completed", "p_d", "p_theta_rad"],
        "distance_indifference_margin_m": 0.03,
        "heading_indifference_margin_deg": 5.0,
        "primitive_route_preference_use": (
            "the completion/0.03m/5deg comparator constructs conditioned "
            "margin-Borda utility; it is not a separate pairwise-loss target"
        ),
        "completion_use": (
            "only inside the existing route-preference tuple; no completion head, "
            "completion regression, or completion classification target"
        ),
        "forbidden_targets": [
            "contact",
            "safety",
            "successor viability",
            "stuck",
            "prior aggregate scorer utility",
            "material contact",
        ],
    },
    "population_conditioning": {
        "fit_population": "ORACLE_VIABILITY_ADMISSIBLE",
        "policy": "TRAIN_ONLY_ON_ORACLE_VIABILITY_ADMISSIBLE_CANDIDATE_SETS",
        "fit_state_identities_total": ROLE_STATE_COUNTS["fit"],
        "retain_every_fit_state_identity_and_row": True,
        "minimum_admissible_candidates_for_optimization": 2,
        "fit_state_optimization_statuses": list(FIT_STATE_OPTIMIZATION_STATUSES),
        "zero_or_singleton_state_loss_contribution": {
            "pairwise": 0.0,
            "listwise": 0.0,
            "residual": 0.0,
        },
        "zero_or_singleton_state_optimizer_step": False,
        "skip_policy": (
            "fit states with zero or one ORACLE_VIABILITY_ADMISSIBLE candidate "
            "are retained in row evidence and skipped deterministically from "
            "optimization"
        ),
        "epoch_loss_averaging_denominator": (
            "CONTRIBUTING fit states only (states with at least two "
            "ORACLE_VIABILITY_ADMISSIBLE candidates)"
        ),
        "mask_is_score_input": False,
        "mask_is_route_target": False,
        "mask_must_be_disclosed_row_level": True,
        "interpretation": (
            "prospectively authorised experimental conditioning, not a learned "
            "safety target or JEPA safety claim"
        ),
    },
    "roles": {
        "fit": "training only",
        "calibration": "may be opened only after both final epoch-60 checkpoints are locked",
        "heldout": "may be opened only after the evaluation contract and all gates are frozen",
        "model_or_hyperparameter_selection_from_calibration_or_heldout": False,
    },
}

TRUE_FUTURE_GATE = {
    "population": "ORACLE_VIABILITY_ADMISSIBLE",
    "role": "heldout",
    "ordering_authority": (
        "population-conditioned margin-Borda utility; pairwise deltas greater "
        "than 1e-12 and tie-aware best-set rank metrics"
    ),
    "progress_diagnostics": (
        "route-progress Spearman/Kendall and max-progress normalized regret are "
        "separate from Borda ordering"
    ),
    "pairwise_accuracy_min": 0.75,
    "spearman_rho_min": 0.60,
    "normalized_regret_max": 0.20,
    "best_route_top3_min": 0.75,
    "selected_progress_fraction_of_oracle_min": 0.80,
    "no_family_complete_collapse": True,
    "candidate_derangement_material": True,
}
DERANGEMENT_MATERIALITY = {
    "ordering_authority": (
        "matched and deranged pairwise accuracy use population-conditioned "
        "margin-Borda deltas greater than 1e-12"
    ),
    "pairwise_accuracy_drop_min": 0.05,
    "selected_progress_fraction_drop_min": 0.10,
    "normalized_regret_increase_min": 0.05,
    "pairwise_accuracy_opposing_improvement_min": 0.05,
    "selected_progress_fraction_opposing_improvement_min": 0.10,
    "normalized_regret_opposing_improvement_min": 0.05,
    "required_reported_damage_fields": [
        "pairwise_accuracy_loss",
        "selected_progress_m_loss",
        "selected_progress_ratio_loss",
        "normalized_regret_worsening",
        "best_route_top3_loss",
    ],
    "descriptive_only_damage_fields": [
        "selected_progress_m_loss",
        "best_route_top3_loss",
    ],
    "gate_trigger_fields": [
        "pairwise_accuracy_loss",
        "selected_progress_ratio_loss",
        "normalized_regret_worsening",
    ],
    "combination": (
        "any damage threshold, unless any opposing change reaches its "
        "corresponding material-reversal threshold"
    ),
    "subthreshold_mixed_direction_changes_tolerated": True,
}
TRUE_INCREMENTAL_GATE = {
    "comparators": ["KINEMATIC", "NO_LATENT"],
    "must_pass_against_each_comparator": True,
    "selected_progress_gain": {
        "absolute_m_min": 0.02,
        "or_oracle_progress_fraction_gain_min": 0.05,
    },
    "normalized_regret_reduction_min": 0.03,
    "family_tie_or_improve_min": 3,
    "family_count": 4,
    "maximum_population_progress_loss_fraction": 0.02,
    "all_candidates_contact_selections_no_worse": True,
    "all_candidates_nonviable_selections_no_worse": True,
}
TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD = {
    "schema": "plan_aware_incremental_route_value_gate_v1",
    "thresholds": {
        "selected_progress_gain_m_minimum": 0.02,
        "oracle_normalized_progress_gain_minimum": 0.05,
        "normalized_regret_reduction_minimum": 0.03,
        "families_improved_or_tied_minimum": 3,
        "population_progress_loss_maximum": 0.02,
    },
    "comparisons": {},
    "pass": False,
    "classification": (
        "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_EVALUATED"
    ),
    "status": "NOT_EVALUATED_TRUE_GATE_FAILED",
    "evaluated": False,
    "reason": "TRUE_FUTURE_GATE_FAILED",
}
RR_GATE = {
    "population": "ORACLE_VIABILITY_ADMISSIBLE",
    "ordering_authority": (
        "population-conditioned margin-Borda utility; pairwise and tie-aware "
        "best-set rank metrics"
    ),
    "pairwise_accuracy_min": 0.70,
    "normalized_regret_max": 0.25,
    "best_route_top3_min": 0.75,
    "selected_progress_fraction_of_oracle_min": 0.75,
    "selected_progress_fraction_of_true_min": 0.85,
    "pairwise_accuracy_strictly_above_r1": True,
    "progress_or_regret_improves_over_r1": True,
    "no_family_complete_collapse": True,
    "all_candidates_contact_selections_no_worse_than_r1": True,
    "all_candidates_nonviable_selections_no_worse_than_r1": True,
    "all_candidates_stuck_selections_no_worse_than_r1": True,
}
PREDICTED_SOURCE_ABSOLUTE_PRESERVATION_GATE = {
    "policy": "OUTCOME_BLIND_FROZEN_BEFORE_PREDICTED_SOURCE_EVALUATION",
    "population": "ORACLE_VIABILITY_ADMISSIBLE",
    "ordering_authority": (
        "population-conditioned margin-Borda utility; pairwise and tie-aware "
        "best-set rank metrics"
    ),
    "sources": list(PREDICTED_SOURCE_IDS),
    "pairwise_accuracy_min": 0.70,
    "normalized_regret_max": 0.25,
    "best_route_top3_min": 0.75,
    "selected_progress_fraction_of_oracle_min": 0.75,
    "selected_progress_fraction_of_true_min": 0.85,
    "no_family_complete_collapse": True,
    "all_fail_definition": "no R1/RR/P1/PR source passes every absolute criterion",
    "primary_fail_closed_when_any_passes_but_rr_gate_fails": True,
    "rr_comparative_and_adverse_clauses_included": False,
}
PROPRIO_CONTRIBUTION_GATE = {
    "minimum_criteria": 2,
    "criteria": {
        "pairwise_accuracy_gain_min": 0.05,
        "selected_progress_fraction_gain_min": 0.05,
        "normalized_regret_reduction_min": 0.03,
        "best_route_top3_gain_min": 0.125,
    },
    "paired_cells": [
        ["P1_PROPRIO_ONE_STEP", "R1_RGB_ONE_STEP"],
        ["PR_PROPRIO_ROLLOUT", "RR_RGB_ROLLOUT"],
    ],
    "all_candidates_contact_selections_no_worse": True,
    "all_candidates_nonviable_selections_no_worse": True,
}

# Public aliases consumed by the separately reviewed model/evaluator modules.
ROUTE_COST_SEED = RANKER_SEED
RANDOM_SEED = RANKER_SEED
CPU_WORKERS = 24
NUMERICAL_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
TRAINING = TRAINING_POLICY
GATES = {
    "true_future": TRUE_FUTURE_GATE,
    "derangement_materiality": DERANGEMENT_MATERIALITY,
    "true_incremental": TRUE_INCREMENTAL_GATE,
    "rr": RR_GATE,
    "predicted_source_absolute_preservation": (
        PREDICTED_SOURCE_ABSOLUTE_PRESERVATION_GATE
    ),
    "proprio_contribution": PROPRIO_CONTRIBUTION_GATE,
}
METRIC_CONTRACT = {
    "populations": [
        "ALL_CANDIDATES",
        "ORACLE_CONTACT_FREE",
        "ORACLE_VIABILITY_ADMISSIBLE",
    ],
    "ordering": [
        "route_progress_spearman",
        "kendall_tau_b",
        "pairwise_accuracy",
        "best_route_top1",
        "best_route_top3",
        "mrr",
        "mean_best_route_rank",
        "score_spread",
        "tie_rate",
    ],
    "selected_route": [
        "candidate_index",
        "realised_distance_progress_m",
        "realised_heading_improvement_rad",
        "selected_combined_route_utility",
        "normalized_regret",
        "waypoint_completion",
        "abstention",
    ],
    "separate_postselection_outcomes": [
        "immediate_contact",
        "nonviable_successor",
        "stuck",
    ],
    "levels": ["aggregate", "per-family", "per-state", "per-role"],
    "route_preference": "frozen local-waypoint population-conditioned margin-Borda utility",
    "ordering_authority": {
        "pairwise_target": (
            "sign(population-conditioned margin-Borda utility_i - utility_j) "
            "when abs(delta) > 1e-12; otherwise tied and excluded"
        ),
        "ideal_ranking": (
            "descending population-conditioned margin-Borda utility; candidate "
            "index breaks utility ties only for deterministic representative/order"
        ),
        "oracle_best_set": (
            "every candidate within 1e-12 of the maximum population-conditioned "
            "margin-Borda utility"
        ),
        "topk_mrr_mean_rank": (
            "tie-aware: top-k succeeds when any oracle-best-set member occurs "
            "within k; reciprocal and mean rank use the minimum model rank among "
            "oracle-best-set members"
        ),
        "route_progress_correlation": (
            "Spearman and Kendall compare score with realised distance progress, "
            "separately from the Borda ordering target"
        ),
        "normalized_regret": (
            "selected realised distance-progress regret relative to maximum "
            "population progress and the population progress span; separately "
            "from the Borda ordering target"
        ),
    },
    "selected_combined_route_utility": {
        "per_state_field": "selected_combined_route_utility",
        "definition": (
            "margin-Borda route utility recomputed within the current candidate "
            "population, at the candidate selected by the current source"
        ),
        "aggregate_fields": [
            "selected_combined_route_utility_sum",
            "selected_combined_route_utility_mean",
            "selected_combined_route_utility_count",
        ],
        "sum_abstention_treatment": "no contribution",
        "mean_denominator": (
            "states with a selected candidate and therefore a defined "
            "selected_combined_route_utility; abstaining states are excluded"
        ),
        "abstentions_reported_separately": True,
        "required_for_every_population_and_source": True,
        "reported_separately_from_distance_heading_completion_and_safety": True,
    },
    "family_collapse": {
        "population_level": "per-family",
        "complete_score_tie_is_collapse": True,
        "all_nonabstaining_score_spreads_at_or_below_tolerance_is_collapse": True,
        "score_tie_tolerance": 1e-12,
        "tie_broken_top3_or_progress_cannot_negate_complete_score_tie": True,
    },
    "future_derangement_reporting": {
        "required_damage_fields": copy.deepcopy(
            DERANGEMENT_MATERIALITY["required_reported_damage_fields"]
        ),
        "descriptive_only_fields": copy.deepcopy(
            DERANGEMENT_MATERIALITY["descriptive_only_damage_fields"]
        ),
        "gate_trigger_fields": copy.deepcopy(
            DERANGEMENT_MATERIALITY["gate_trigger_fields"]
        ),
        "new_descriptive_fields_do_not_change_gate": True,
    },
    "descriptive_adverse_downranking": {
        "population": "ALL_CANDIDATES",
        "outcomes": [
            "immediate_contact",
            "successor_nonviable",
            "stuck",
        ],
        "within_state_cross_group_pairs_only": True,
        "credit": {
            "nonadverse_scored_above_adverse": 1.0,
            "score_tie": 0.5,
            "adverse_scored_above_nonadverse": 0.0,
        },
        "reported": ["pair_count", "correct_credit", "pairwise_accuracy"],
        "levels": ["overall", "per-family"],
        "descriptive_only": True,
        "score_or_route_target": False,
        "classification_gate": False,
        "existing_selected_count_no_worse_gates_unchanged": True,
    },
    "safety_fields_in_score_or_route_preference": [],
    "unavailable_metric_policy": {
        "legitimate_causes": [
            "empty evaluation population",
            "complete score tie",
            "constant realised route progress",
            "no Borda-ordered candidate pairs",
        ],
        "evidence_value": None,
        "required_gate_criterion": False,
        "gate_or_classification_exception": False,
        "fail_closed": True,
        "stage_a_primary_on_failed_true_gate": "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        "stage_b_predicted_classification": "TWO_STEP_PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        "proprio_classification": (
            "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED"
        ),
        "substitution_classification": (
            "PROPRIOCEPTIVE_SUBSTITUTION_NOT_SUPPORTED"
        ),
        "descriptive_spearman_kendall_deltas_nullable": True,
        "factorial_contrasts_nullable": True,
    },
}

STAGE_C_ABLATION_INPUTS = {
    "PR_VISUAL_CONTEXT_DERANGED": {
        "feature_id": "visual_context_sequence",
        "predictor_input_component": "visual",
    },
    "PR_PROPRIO_HISTORY_DERANGED": {
        "feature_id": "proprio_history",
        "predictor_input_component": "proprio",
    },
    "PR_CONTROL_HISTORY_DERANGED": {
        "feature_id": "previous_applied_control_history",
        "predictor_input_component": "control",
    },
}

STAGE_POLICY = {
    "STAGE_A_TRUE_FUTURE": {
        "train": ["NO_LATENT", "LATENT_TRUE_FUTURE"],
        "fit_only": True,
        "matched_raw_cost_comparator_rereduction": {
            "sources": list(RAW_COST_REREDUCED_SOURCE_IDS),
            "binding": copy.deepcopy(PREDECESSOR_CANDIDATE_EVIDENCE_BINDING),
            "open_phase": "post-checkpoint and post-evaluation-contract only",
            "score": "-cost_h3",
            "current_metric_contract": True,
            "raw_cosine_rerun": False,
            "copied_historical_aggregates_retained_separately": True,
            "matched_comparisons": {
                "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": (
                    "RAW_TRUE_FUTURE_GOAL_COSINE"
                ),
                "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1": (
                    "RAW_R1_GOAL_COSINE"
                ),
                "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR": (
                    "RAW_RR_GOAL_COSINE"
                ),
            },
            "matched_comparison_metrics": [
                "pairwise_accuracy_gain",
                "spearman_gain",
                "kendall_gain",
                "normalized_regret_reduction",
                "best_route_top3_gain",
                "selected_progress_gain_m",
                "all_candidates_contact_selection_delta",
                "all_candidates_nonviable_selection_delta",
            ],
            "paired_per_state_and_descriptive_bootstrap_required": True,
            "classification_gate": False,
        },
        "stop_if_true_gate_fails": True,
        "on_failure": {
            "primary": "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
            "next": "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
            "true_incremental_route_value": copy.deepcopy(
                TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD
            ),
            "new_predictor_inference": False,
            "closed_loop": False,
        },
    },
    "STAGE_B_PREDICTOR_SUBSTITUTION": {
        "condition": "TRUE_FUTURE_GATE passes",
        "ranker_weights": "fixed LATENT final epoch-60 checkpoint; no refit or recalibration",
        "sources": list(PREDICTED_SOURCE_IDS),
        "r1_rr": "reuse bound 48-state tensors; no predictor inference",
        "p1_pr": (
            "conditional materialisation only after a complete prospective proprio-input "
            "custody receipt; absent today, therefore fail closed rather than use the legacy panel"
        ),
    },
    "STAGE_C_ATTRIBUTION": {
        "condition": "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION is supported",
        "ablation_location": "frozen PR predictor inputs before predictor inference",
        "ablation_condition_ids": list(STAGE_C_ABLATION_INPUTS),
        "ablation_id_to_feature": copy.deepcopy(STAGE_C_ABLATION_INPUTS),
        "donor_constraints": [
            "same frozen split",
            "same frozen maze family",
            "different state",
        ],
        "recipient_fields_unchanged": [
            "candidate action",
            "waypoint",
            "state and candidate identity",
            "route labels",
        ],
        "donor_mapping": {
            "algorithm": (
                "SHA256(contract digest, split, family, state, exact uppercase "
                "ablation condition ID), followed "
                "by an outcome-blind bijective cyclic derangement within each "
                "split/family cohort"
            ),
            "hash_ablation_key": "exact key from ablation_id_to_feature",
            "minimum_cohort_size": 2,
            "singleton_action": "stop Stage C as contract-invalid",
            "persist": ["donor state IDs", "donor input SHA-256 values"],
        },
        "predicted_latent_or_ranker_derangement": False,
        "ranker_refit": False,
    },
}

ACCIDENTAL_EXPOSURES = (
    {
        "actor": "protected_review_commit_audit",
        "command_pattern": (
            "rg -n PLAN_AWARE_MONOTONE_JEPA_COST_V1|plan-aware monotone|"
            "plan_aware_monotone docs lewm scripts .generated"
        ),
        "path": (
            "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
            "result_2026-08-26.json"
        ),
        "exposure": "predecessor aggregate and per-state heldout result values",
        "disposition": "FROZEN_COMPARATOR_AUTHORITY_ONLY_EXCLUDED_FROM_SUCCESSOR_DESIGN_AND_TUNING",
        "values_used_for_successor_design_or_tuning": False,
    },
    {
        "actor": "root",
        "command_pattern": "predecessor detailed result/aggregate source inspection",
        "path": (
            "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
            "result_2026-08-26.json"
        ),
        "exposure": "predecessor detailed result and aggregate values",
        "disposition": "FROZEN_COMPARATOR_AUTHORITY_ONLY_EXCLUDED_FROM_SUCCESSOR_DESIGN_AND_TUNING",
        "values_used_for_successor_design_or_tuning": False,
    },
    {
        "actor": "protected_review_commit_audit",
        "command_pattern": (
            "rg -n -i raw/floor/occupancy/proprio/planning-utility patterns over "
            "dated predecessor documentation"
        ),
        "path": (
            "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_"
            "contract_2026-08-26.json"
        ),
        "exposure": (
            "predecessor contract output included embedded aggregate and per-state "
            "heldout comparator values"
        ),
        "disposition": "FROZEN_COMPARATOR_AUTHORITY_ONLY_EXCLUDED_FROM_SUCCESSOR_DESIGN_AND_TUNING",
        "values_used_for_successor_design_or_tuning": False,
    },
)

CPU_WORKER_BENCHMARK = {
    "classification": "OUTCOME_FREE_PREFREEZE_CPU_FIXTURE_BENCHMARK",
    "fixture_digest": "67c81175a6f610a56567491c6e0a36f5b42daed07ec31ef8338cdbc6b64eb305",
    "threads_per_worker": 1,
    "tasks": 384,
    "configurations": [
        {"workers": 16, "runtime_s": 0.148594584, "tasks_per_s": 2584.21},
        {"workers": 24, "runtime_s": 0.145168053, "tasks_per_s": 2645.21},
        {"workers": 32, "runtime_s": 0.161542391, "tasks_per_s": 2377.09},
    ],
    "selected_workers": 24,
    "selection_rule": "highest measured outcome-free fixture throughput",
    "swap_free_before_bytes": 20_887_203_840,
    "swap_free_after_bytes": 20_887_203_840,
    "memory_pressure_observed": False,
    "scientific_outcomes_read": 0,
    "scientific_semantics_changed": False,
}


class ContractError(ValueError):
    """Raised when prospective custody or a frozen receipt fails closed."""


def classification_vocabularies_are_unique() -> bool:
    """Return whether each frozen decision vocabulary is internally unique."""

    return all(
        len(values) == len(set(values))
        for values in (
            PRIMARY_CLASSIFICATIONS,
            SECONDARY_CLASSIFICATIONS,
            NEXT_EXPERIMENT_IDS,
        )
    )


def duplicate_literal_dict_keys(path: str | Path) -> list[dict[str, Any]]:
    """Return duplicate literal dictionary keys with their source line numbers."""

    source_path = Path(path)
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    duplicates: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen: dict[Any, int] = {}
        for key in node.keys:
            if not isinstance(key, ast.Constant):
                continue
            value = key.value
            try:
                duplicate = value in seen
            except TypeError:
                continue
            if duplicate:
                duplicates.append(
                    {
                        "key_repr": repr(value),
                        "first_line": seen[value],
                        "duplicate_line": int(key.lineno),
                    }
                )
            else:
                seen[value] = int(key.lineno)
    return duplicates


def validate_no_duplicate_literal_dict_keys(path: str | Path) -> None:
    duplicates = duplicate_literal_dict_keys(path)
    if duplicates:
        raise ContractError(f"duplicate literal dictionary keys: {duplicates!r}")


def _validate_json_value(value: Any, location: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{location} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{location} contains a non-string key")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ContractError(f"{location} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes without a trailing newline."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def attach_self_digest(value: Mapping[str, Any], key: str = "content_digest") -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop(key, None)
    payload[key] = canonical_json_sha256(payload)
    return payload


def validate_self_digest(
    value: Mapping[str, Any], key: str = "content_digest"
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    declared = payload.pop(key, None)
    if not isinstance(declared, str) or len(declared) != 64:
        raise ContractError(f"{key} is missing or malformed")
    if canonical_json_sha256(payload) != declared:
        raise ContractError(f"{key} mismatch")
    return copy.deepcopy(dict(value))


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _inventory_row(relative_path: str) -> dict[str, Any]:
    for row in EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS:
        if row["path"] == relative_path:
            return copy.deepcopy(row)
    raise ContractError(f"unknown execution-correction archive path: {relative_path}")


def _binding_with_content_digest(relative_path: str) -> dict[str, Any]:
    row = _inventory_row(relative_path)
    row["content_digest"] = EXECUTION_CORRECTION_RECEIPT_CONTENT_DIGESTS[
        relative_path
    ]
    return row


def _remove_exact_dotted_path(value: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ContractError("scientific projection contains an invalid dotted path")
    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ContractError(
                f"scientific projection exclusion is absent: {dotted_path}"
            )
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise ContractError(
            f"scientific projection exclusion is absent: {dotted_path}"
        )
    del current[parts[-1]]


def scientific_content_projection(
    relative_path: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Drop only the narrow wrapper bindings frozen for correction replay."""

    if relative_path not in EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS:
        raise ContractError(
            f"no execution-correction projection is frozen for {relative_path}"
        )
    if not isinstance(value, Mapping):
        raise ContractError("scientific projection input must be a JSON object")
    projected = copy.deepcopy(dict(value))
    exclusions = EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS[relative_path]
    if len(exclusions) != len(set(exclusions)):
        raise ContractError("scientific projection repeats an exclusion")
    for dotted_path in exclusions:
        _remove_exact_dotted_path(projected, dotted_path)
    return projected


def scientific_content_digest(
    relative_path: str, value: Mapping[str, Any]
) -> str:
    return canonical_json_sha256(scientific_content_projection(relative_path, value))


def build_execution_correction_amendment() -> dict[str, Any]:
    """Build the metric-value-blind, row-value-blind, gate-aware correction."""

    return attach_self_digest(
        {
            "schema": EXECUTION_CORRECTION_AMENDMENT_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "date": DATE,
            "status": "PROSPECTIVE_EXECUTION_ONLY_CORRECTION_NOT_EXECUTED",
            "base_scientific_authority": {
                "freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
                "contract_sha256": SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
                "tracked_files": copy.deepcopy(BASE_SCIENTIFIC_AUTHORITY_BINDINGS),
                "immutability": "BYTE_EXACT",
                "amendment_is_separate_overlay": True,
            },
            "failed_attempt": {
                "archive_path": str(EXECUTION_CORRECTION_FAILED_ARCHIVE),
                "source_freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
                "source_parent_commit": SOURCE_COMMIT,
                "source_commit_subject": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "inventory": copy.deepcopy(EXECUTION_CORRECTION_ARCHIVE_INVENTORY),
                "failure_receipt": _binding_with_content_digest(
                    "receipts/failure.json"
                ),
                "preexecution_receipt": _binding_with_content_digest(
                    "receipts/preexecution.json"
                ),
                "source_closure_snapshot": _binding_with_content_digest(
                    "receipts/source_closure.json"
                ),
                "training_smoke_receipt": _binding_with_content_digest(
                    "receipts/training_smoke.json"
                ),
                "training_receipt": _binding_with_content_digest(
                    "receipts/training.json"
                ),
                "evaluation_contract": _binding_with_content_digest(
                    "receipts/evaluation_contract.json"
                ),
                "stage_b_gate_receipt": _binding_with_content_digest(
                    "receipts/stage_b_gate.json"
                ),
            },
            "barrier_receipt": {
                "phase": "CONDITIONAL_STAGE_B_AND_C",
                "full_training_epochs_completed": 60,
                "calibration_rows_opened": 96,
                "heldout_rows_opened": 96,
                "final_checkpoint_published": True,
                "nothing_running": True,
                "prohibition_counters_all_zero": True,
                "stage_a_true_future_gate": {
                    "classification": "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
                    "pass": True,
                },
                "stage_b_authorised": True,
                "stage_b_child_started": True,
                "stage_b_materialisation_completed": False,
                "stage_c_gate_published": False,
                "stage_c_started": False,
                "canonical_output_published": False,
                "failed_archive_staging_files": 0,
            },
            "environment_correction": copy.deepcopy(
                EXECUTION_CORRECTION_ENVIRONMENT_PROBE
            ),
            "policy": copy.deepcopy(EXECUTION_CORRECTION_POLICY),
            "custody_boundary": {
                "outcome_values_used_to_author_correction": False,
                "metric_values_used_to_author_correction": False,
                "row_outcome_values_used_to_author_correction": False,
                "tensors_opened_to_author_correction": False,
                "binary_gate_status_used_to_author_correction": True,
                "authorised_observations": [
                    "file names, byte sizes and SHA-256 digests",
                    "source and contract identities",
                    "lifecycle barrier counters",
                    "gate pass/classification and conditional-stage entry",
                    "environment import failure and module selection",
                ],
                "scientific_interpretation_changed": False,
                "stage_a_result_used_for_design_or_tuning": False,
            },
            "amended_contract_binding_policy": (
                "the separate amendment contract is bound by its canonical bytes "
                "and enclosing execution-correction freeze commit; original "
                "scientific preregistration/contract/schema/fixture/route-role/"
                "source-closure bytes remain unchanged, and the amendment does not "
                "circularly contain its future commit hash"
            ),
        }
    )


EXECUTION_CORRECTION_AMENDMENT = build_execution_correction_amendment()
EXECUTION_CORRECTION_AMENDMENT_RECEIPT_BYTES = (
    canonical_json_bytes(EXECUTION_CORRECTION_AMENDMENT) + b"\n"
)
EXECUTION_CORRECTION_AMENDMENT_BINDING = {
    "path": str(TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH),
    "sha256": hashlib.sha256(
        EXECUTION_CORRECTION_AMENDMENT_RECEIPT_BYTES
    ).hexdigest(),
    "content_digest": EXECUTION_CORRECTION_AMENDMENT["content_digest"],
    "bytes": len(EXECUTION_CORRECTION_AMENDMENT_RECEIPT_BYTES),
    "schema": EXECUTION_CORRECTION_AMENDMENT_SCHEMA_VERSION,
}


def validate_execution_correction_amendment(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(value)
    if canonical_json_bytes(value) != canonical_json_bytes(
        EXECUTION_CORRECTION_AMENDMENT
    ):
        raise ContractError("execution-correction amendment value drift")
    inventory = value["failed_attempt"]["inventory"]
    if canonical_json_sha256(inventory["rows"]) != inventory["manifest_sha256"]:
        raise ContractError("execution-correction archive manifest digest drift")
    if inventory["files"] != len(inventory["rows"]) or inventory["bytes"] != sum(
        int(row["bytes"]) for row in inventory["rows"]
    ):
        raise ContractError("execution-correction archive inventory cardinality drift")
    return copy.deepcopy(dict(value))


def execution_correction_amendment_receipt_bytes() -> bytes:
    validate_execution_correction_amendment(EXECUTION_CORRECTION_AMENDMENT)
    return EXECUTION_CORRECTION_AMENDMENT_RECEIPT_BYTES


def validate_base_scientific_authorities(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Require every original scientific authority to remain byte-exact."""

    root = Path(repo_root)
    rows: list[dict[str, Any]] = []
    for label, expected in BASE_SCIENTIFIC_AUTHORITY_BINDINGS.items():
        path = root / str(expected["path"])
        if not path.is_file():
            raise ContractError(f"base scientific authority is absent: {label}")
        sha256, size = _sha256_file(path)
        if sha256 != expected["sha256"] or size != expected["bytes"]:
            raise ContractError(f"base scientific authority changed: {label}")
        if "content_digest" in expected:
            try:
                value = json.loads(path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"base scientific authority JSON is invalid: {label}"
                ) from exc
            if not isinstance(value, dict):
                raise ContractError(
                    f"base scientific authority is not an object: {label}"
                )
            digest_key = {
                "contract": "contract_sha256",
                "output_schema": "output_schema_sha256",
                "evaluator_fixture": "fixture_sha256",
            }.get(label, "content_digest")
            if digest_key is None:
                raise ContractError(
                    f"base scientific authority digest is absent: {label}"
                )
            validate_self_digest(value, digest_key)
            observed_digest = value[digest_key]
            if observed_digest != expected["content_digest"]:
                raise ContractError(
                    f"base scientific authority content digest changed: {label}"
                )
        rows.append(
            {
                "label": label,
                "path": str(expected["path"]),
                "sha256": sha256,
                "bytes": size,
            }
        )
    if CONTRACT_SHA256 != SCIENTIFIC_AUTHORITY_CONTRACT_SHA256:
        raise ContractError("live scientific contract payload changed")
    return {
        "freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
        "contract_sha256": SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
        "rows": rows,
        "pass": True,
    }


def validate_execution_correction_archive(
    archive_root: str | Path = EXECUTION_CORRECTION_FAILED_ARCHIVE,
) -> dict[str, Any]:
    """Hash the bound failed archive and validate only lifecycle/gate metadata."""

    root = Path(archive_root)
    if root.resolve() != EXECUTION_CORRECTION_FAILED_ARCHIVE.resolve():
        raise ContractError("execution-correction archive path drift")
    if not root.is_dir():
        raise ContractError("execution-correction failed archive is absent")
    observed_paths = sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )
    expected_paths = [
        str(row["path"]) for row in EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS
    ]
    if observed_paths != expected_paths:
        raise ContractError("execution-correction archive path-set drift")
    observed_rows: list[dict[str, Any]] = []
    for expected in EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS:
        path = root / str(expected["path"])
        sha256, size = _sha256_file(path)
        observed = {"path": expected["path"], "sha256": sha256, "bytes": size}
        if observed != expected:
            raise ContractError(
                f"execution-correction archive binding drift: {expected['path']}"
            )
        observed_rows.append(observed)
    if (
        canonical_json_sha256(observed_rows)
        != EXECUTION_CORRECTION_ARCHIVE_INVENTORY["manifest_sha256"]
    ):
        raise ContractError("execution-correction archive inventory digest drift")
    receipts: dict[str, dict[str, Any]] = {}
    for relative_path, expected_digest in (
        EXECUTION_CORRECTION_RECEIPT_CONTENT_DIGESTS.items()
    ):
        try:
            value = json.loads((root / relative_path).read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"execution-correction receipt is invalid: {relative_path}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError(
                f"execution-correction receipt is not an object: {relative_path}"
            )
        validate_self_digest(value)
        if value.get("content_digest") != expected_digest:
            raise ContractError(
                f"execution-correction receipt content drift: {relative_path}"
            )
        receipts[relative_path] = value
    failure = receipts["receipts/failure.json"]
    expected_failure = {
        "schema": "plan_aware_monotone_jepa_failure_v1",
        "source_freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
        "phase": "CONDITIONAL_STAGE_B_AND_C",
        "error_type": "QualificationError",
        "full_training_epochs_completed": 60,
        "calibration_rows_opened": 96,
        "heldout_rows_opened": 96,
        "final_checkpoint_published": True,
        "nothing_running": True,
    }
    if any(failure.get(key) != expected for key, expected in expected_failure.items()):
        raise ContractError("execution-correction failure barrier drift")
    if any(int(value) != 0 for value in failure["prohibition_counters"].values()):
        raise ContractError("execution-correction failure prohibition drift")
    gate = receipts["receipts/stage_b_gate.json"]
    if (
        gate.get("contract_freeze_commit") != INITIAL_EXECUTION_FREEZE_COMMIT
        or gate.get("contract_sha256") != SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or gate.get("true_future_gate")
        != {"classification": "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL", "pass": True}
        or gate.get("stage_b_authorised") is not True
    ):
        raise ContractError("execution-correction Stage-B gate custody drift")
    staging_files = [path for path in (root / "staging").rglob("*") if path.is_file()]
    if staging_files:
        raise ContractError("failed archive contains reusable staging files")
    return {
        "archive_path": str(root),
        "inventory": copy.deepcopy(EXECUTION_CORRECTION_ARCHIVE_INVENTORY),
        "source_freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
        "failure_receipt": _binding_with_content_digest("receipts/failure.json"),
        "source_closure_snapshot": _binding_with_content_digest(
            "receipts/source_closure.json"
        ),
        "stage_b_gate_receipt": _binding_with_content_digest(
            "receipts/stage_b_gate.json"
        ),
        "files_reused": 0,
        "nothing_running": True,
        "pass": True,
    }


def validate_execution_correction_replay(
    attempt_root: str | Path,
    archive_root: str | Path = EXECUTION_CORRECTION_FAILED_ARCHIVE,
) -> dict[str, Any]:
    """Fail closed unless the fresh attempt reproduces science before Stage B."""

    attempt = Path(attempt_root)
    archive = Path(archive_root)
    validate_execution_correction_archive(archive)
    if not attempt.is_dir() or attempt.resolve() == archive.resolve():
        raise ContractError("execution-correction replay requires a fresh attempt root")
    byte_exact_rows: list[dict[str, Any]] = []
    for relative_path in EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS:
        expected = _inventory_row(relative_path)
        candidate = attempt / relative_path
        if not candidate.is_file():
            raise ContractError(
                f"execution-correction replay artifact is absent: {relative_path}"
            )
        sha256, size = _sha256_file(candidate)
        if sha256 != expected["sha256"] or size != expected["bytes"]:
            raise ContractError(
                f"execution-correction byte replay drift: {relative_path}"
            )
        byte_exact_rows.append(copy.deepcopy(expected))
    normalized_rows: list[dict[str, Any]] = []
    for relative_path, expected_digest in (
        EXECUTION_CORRECTION_NORMALIZED_REPLAY_DIGESTS.items()
    ):
        candidate = attempt / relative_path
        if not candidate.is_file():
            raise ContractError(
                f"execution-correction normalized replay artifact is absent: {relative_path}"
            )
        try:
            value = json.loads(candidate.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"execution-correction normalized replay JSON is invalid: {relative_path}"
            ) from exc
        if not isinstance(value, dict):
            raise ContractError("execution-correction normalized replay must be an object")
        observed_digest = scientific_content_digest(relative_path, value)
        if observed_digest != expected_digest:
            raise ContractError(
                f"execution-correction scientific replay drift: {relative_path}"
            )
        normalized_rows.append(
            {
                "path": relative_path,
                "excluded_paths": list(
                    EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS[
                        relative_path
                    ]
                ),
                "scientific_content_digest": observed_digest,
            }
        )
    observed_attempt_files = [
        str(path.relative_to(attempt))
        for path in attempt.rglob("*")
        if path.is_file()
    ]
    premature = sorted(
        relative_path
        for relative_path in observed_attempt_files
        if any(
            relative_path.startswith(prefix)
            for prefix in EXECUTION_CORRECTION_FORBIDDEN_BEFORE_REPLAY_PREFIXES
        )
        or any(
            Path(relative_path).match(pattern)
            for pattern in EXECUTION_CORRECTION_FORBIDDEN_BEFORE_REPLAY_GLOBS
        )
    )
    if premature:
        raise ContractError(
            f"conditional artifact exists before correction replay gate: {premature}"
        )
    return attach_self_digest(
        {
            "schema": EXECUTION_CORRECTION_REPLAY_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "amendment": copy.deepcopy(EXECUTION_CORRECTION_AMENDMENT_BINDING),
            "failed_archive": str(archive),
            "fresh_attempt": str(attempt),
            "files_reused": 0,
            "byte_exact_replay": byte_exact_rows,
            "normalized_scientific_replay": normalized_rows,
            "stage_b_started_before_replay_gate": False,
            "stage_c_started_before_replay_gate": False,
            "pass": True,
        }
    )


def build_execution_correction_output_schema() -> dict[str, Any]:
    """Build the separate overlay schema without changing base output science."""

    return attach_self_digest(
        {
            "schema": EXECUTION_CORRECTION_OUTPUT_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "base_output_schema": copy.deepcopy(
                BASE_SCIENTIFIC_AUTHORITY_BINDINGS["output_schema"]
            ),
            "amendment": copy.deepcopy(EXECUTION_CORRECTION_AMENDMENT_BINDING),
            "base_result_schema_unchanged": True,
            "required_runtime_artifacts": {
                "conditional_child_environment_preflight": {
                    "path": "receipts/conditional_child_environment_preflight.json",
                    "schema": (
                        "plan_aware_monotone_jepa_cost_v1."
                        "conditional_child_environment_preflight.v1"
                    ),
                    "required_fields": [
                        "schema",
                        "experiment_id",
                        "cpu_child",
                        "gpu_child",
                        "inherited_python_environment_presence",
                        "fit_outcome_rows_opened",
                        "calibration_rows_opened",
                        "heldout_rows_opened",
                        "tensor_rows_opened",
                        "training_steps",
                        "pass",
                        "content_digest",
                    ],
                    "required_zero_fields": [
                        "fit_outcome_rows_opened",
                        "calibration_rows_opened",
                        "heldout_rows_opened",
                        "tensor_rows_opened",
                        "training_steps",
                    ],
                    "environment_authority": copy.deepcopy(
                        EXECUTION_CORRECTION_ENVIRONMENT_PROBE
                    ),
                    "publish_before_fit_or_tensor_open": True,
                },
                "execution_correction_replay": {
                    "path": "receipts/execution_correction_replay.json",
                    "schema": EXECUTION_CORRECTION_REPLAY_SCHEMA_VERSION,
                    "required_fields": [
                        "schema",
                        "experiment_id",
                        "amendment",
                        "failed_archive",
                        "fresh_attempt",
                        "files_reused",
                        "byte_exact_replay",
                        "normalized_scientific_replay",
                        "stage_b_started_before_replay_gate",
                        "stage_c_started_before_replay_gate",
                        "pass",
                        "content_digest",
                    ],
                    "publish_after_stage_b_gate_receipt": True,
                    "publish_before_any_conditional_scientific_or_materialisation_child": True,
                    "outcome_free_pre_fit_cpu_gpu_import_probes_exempt": True,
                    "files_reused_required": 0,
                    "replay_policy": copy.deepcopy(
                        EXECUTION_CORRECTION_POLICY["pre_stage_b_replay_gate"]
                    ),
                },
            },
            "required_custody_field": {
                "name": "execution_correction_custody",
                "required_in": [
                    "preexecution receipt",
                    "persistence receipt",
                    "result.stage_execution",
                ],
                "required_subfields": [
                    "amendment",
                    "amendment_source_closure",
                    "archive_path",
                    "archive_inventory",
                    "failure_receipt",
                    "source_freeze_commit",
                    "files_reused",
                    "conditional_child_environment_preflight",
                    "execution_correction_replay",
                    "pass",
                ],
                "files_reused_required": 0,
            },
            "lifecycle": {
                "fresh_attempt_namespace": True,
                "base_archive_read_only": True,
                "direct_archive_artifact_reuse": False,
                "environment_preflight_before_fit": True,
                "scientific_replay_gate_after_stage_b_gate": True,
                "scientific_replay_gate_before_conditional_scientific_or_materialisation_child": True,
                "outcome_free_pre_fit_cpu_gpu_import_probes_before_replay": True,
                "stage_c_remains_conditioned_on_proprioception_gate": True,
                "failure_after_correction": "NO_FURTHER_RETRY",
            },
        },
        "output_schema_sha256",
    )


EXECUTION_CORRECTION_OUTPUT_SCHEMA = build_execution_correction_output_schema()
EXECUTION_CORRECTION_OUTPUT_SCHEMA_BINDING = {
    "path": str(TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH),
    "sha256": hashlib.sha256(
        canonical_json_bytes(EXECUTION_CORRECTION_OUTPUT_SCHEMA) + b"\n"
    ).hexdigest(),
    "output_schema_sha256": EXECUTION_CORRECTION_OUTPUT_SCHEMA[
        "output_schema_sha256"
    ],
    "bytes": len(canonical_json_bytes(EXECUTION_CORRECTION_OUTPUT_SCHEMA)) + 1,
    "schema": EXECUTION_CORRECTION_OUTPUT_SCHEMA_VERSION,
}


def build_execution_correction_fixture() -> dict[str, Any]:
    exclusions = EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS
    fixture = {
        "schema": EXECUTION_CORRECTION_FIXTURE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "amendment": copy.deepcopy(EXECUTION_CORRECTION_AMENDMENT_BINDING),
        "output_schema": copy.deepcopy(EXECUTION_CORRECTION_OUTPUT_SCHEMA_BINDING),
        "checks": {
            "base_scientific_contract_digest_unchanged": (
                BASE_SCIENTIFIC_AUTHORITY_BINDINGS["contract"]["content_digest"]
                == SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
            ),
            "base_contract_receipt_bytes_unchanged": (
                BASE_SCIENTIFIC_AUTHORITY_BINDINGS["contract"]["sha256"]
                == "f79146ae2183d18d289c691cc41a9326e9ea0c35ff8ae6fcfbf44d0f8edc9604"
                and BASE_SCIENTIFIC_AUTHORITY_BINDINGS["contract"]["bytes"]
                == 38_779
            ),
            "archive_inventory_exact": (
                len(EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS) == 15
                and sum(
                    int(row["bytes"])
                    for row in EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS
                )
                == 4_984_772
                and canonical_json_sha256(
                    list(EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS)
                )
                == EXECUTION_CORRECTION_ARCHIVE_INVENTORY["manifest_sha256"]
            ),
            "replay_path_sets_unique_and_disjoint": (
                len(EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS)
                == len(set(EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS))
                and set(EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS).isdisjoint(
                    exclusions
                )
            ),
            "scientific_contract_digest_never_excluded": all(
                "contract_sha256" not in paths
                and "experiment_contract_digest" not in paths
                for paths in exclusions.values()
            ),
            "normalized_replay_paths_exact": (
                set(exclusions)
                == set(EXECUTION_CORRECTION_NORMALIZED_REPLAY_DIGESTS)
            ),
            "environment_scrub_is_exact": (
                EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
                    "only_authorised_environment_change"
                ]["remove"]
                == ["PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP"]
                and EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
                    "only_authorised_environment_change"
                ]["set_shared"]
                == {"PYTHONNOUSERSITE": "1"}
                and EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
                    "only_authorised_environment_change"
                ]["interpreter_flags"]
                == ["-E", "-s"]
            ),
            "environment_interpreter_mapping_is_exact": (
                EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
                    "only_authorised_environment_change"
                ]["per_interpreter"]
                == {
                    "cpu_child": {
                        "interpreter": (
                            "/home/andrewknowles/Workspace/LeWMQuad-v3/"
                            ".generated/venvs/genesis_render_vulkan/bin/python"
                        ),
                        "VIRTUAL_ENV": (
                            "/home/andrewknowles/Workspace/LeWMQuad-v3/"
                            ".generated/venvs/genesis_render_vulkan"
                        ),
                        "PATH_prepend": (
                            "/home/andrewknowles/Workspace/LeWMQuad-v3/"
                            ".generated/venvs/genesis_render_vulkan/bin"
                        ),
                    },
                    "gpu_child": {
                        "interpreter": (
                            "/home/andrewknowles/TinyQuadJEPA/bin/python"
                        ),
                        "VIRTUAL_ENV": "/home/andrewknowles/TinyQuadJEPA",
                        "PATH_prepend": "/home/andrewknowles/TinyQuadJEPA/bin",
                    },
                }
            ),
            "cpu_and_gpu_probe_bound": (
                set(
                    EXECUTION_CORRECTION_ENVIRONMENT_PROBE["required_preflight"]
                )
                == {"flags", "cpu_child", "gpu_child"}
                and all(
                    EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
                        "required_preflight"
                    ][label]["interpreter"]
                    == EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
                        "only_authorised_environment_change"
                    ]["per_interpreter"][label]["interpreter"]
                    for label in ("cpu_child", "gpu_child")
                )
            ),
            "pre_fit_import_probe_exception_is_outcome_free_and_narrow": (
                EXECUTION_CORRECTION_POLICY["pre_stage_b_replay_gate"][
                    "outcome_free_pre_fit_import_probe_exception"
                ]
                == {
                    "allowed_before_replay": True,
                    "scope": ["cpu_child", "gpu_child"],
                    "fit_outcome_rows_opened": 0,
                    "calibration_rows_opened": 0,
                    "heldout_rows_opened": 0,
                    "tensor_rows_opened": 0,
                    "training_steps": 0,
                    "scientific_inference_or_materialisation": False,
                }
            ),
            "base_post_smoke_retry_remains_forbidden": (
                EXECUTION_RETRY_POLICY["later_failure"]["retry_allowed"] is False
                and EXECUTION_CORRECTION_POLICY["maximum_fresh_attempts"] == 1
            ),
            "zero_reuse_is_mandatory": (
                EXECUTION_CORRECTION_POLICY["failed_archive_files_reused"] == 0
                and EXECUTION_CORRECTION_POLICY["direct_checkpoint_or_ledger_reuse"]
                is False
            ),
        },
    }
    value = attach_self_digest(fixture, "fixture_sha256")
    if not all(value["checks"].values()):
        raise ContractError("execution-correction fixture failed")
    return value


EXECUTION_CORRECTION_FIXTURE = build_execution_correction_fixture()
EXECUTION_CORRECTION_FIXTURE_BINDING = {
    "path": str(TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH),
    "sha256": hashlib.sha256(
        canonical_json_bytes(EXECUTION_CORRECTION_FIXTURE) + b"\n"
    ).hexdigest(),
    "fixture_sha256": EXECUTION_CORRECTION_FIXTURE["fixture_sha256"],
    "bytes": len(canonical_json_bytes(EXECUTION_CORRECTION_FIXTURE)) + 1,
    "schema": EXECUTION_CORRECTION_FIXTURE_SCHEMA_VERSION,
}


def build_execution_correction_preregistration_markdown() -> str:
    return "\n".join(
        [
            "# Plan-aware monotone JEPA cost V1 execution-correction amendment",
            "",
            f"Experiment: `{EXPERIMENT_ID}`.",
            "",
            "This separate, metric-value-blind and row-outcome-value-blind, "
            "gate-status-aware amendment authorises one wholly fresh "
            "execution-only correction for the exact bound failed archive. It "
            "does not alter the original scientific preregistration, contract, "
            "output schema, fixture, route-role authority, source-closure bytes, "
            "model, training, metrics, gates, classifications, or claims.",
            "",
            f"Base execution freeze: `{INITIAL_EXECUTION_FREEZE_COMMIT}`.",
            f"Base scientific contract: `{SCIENTIFIC_AUTHORITY_CONTRACT_SHA256}`.",
            f"Failed archive inventory: {EXECUTION_CORRECTION_ARCHIVE_INVENTORY['files']} "
            f"files, {EXECUTION_CORRECTION_ARCHIVE_INVENTORY['bytes']} bytes, "
            f"manifest `{EXECUTION_CORRECTION_ARCHIVE_INVENTORY['manifest_sha256']}`.",
            "",
            "The sole implementation correction removes inherited `PYTHONPATH`, "
            "`PYTHONHOME`, `PYTHONUSERBASE`, and `PYTHONSTARTUP` from the complete "
            "conditional CPU/GPU child chain; sets the exact child `VIRTUAL_ENV`, "
            "prepends its bin directory to `PATH`, sets `PYTHONNOUSERSITE=1`, and "
            "uses interpreter flags `-E -s`. Exact CPU Genesis and GPU Torch module "
            "paths, hashes, bytes, versions, and `typing_extensions.Sentinel` are "
            "checked before fit outcomes or tensors open.",
            "These two CPU/GPU import probes are outcome-free pre-fit custody "
            "checks and are the only conditional children exempt from the later "
            "scientific replay barrier.",
            "",
            "The corrected execution uses a new attempt namespace and reuses zero "
            "files. Before Stage B starts, checkpoints, training receipts and all "
            "binding-free ledgers must be byte-exact to the failed archive. The "
            "evaluation contract and Stage-A/Stage-B gate receipts must have the "
            "same normalized scientific content; only narrowly enumerated wrapper "
            "provenance and dependent digest fields are excluded. The old scientific "
            "contract digest, derangement maps, model parameter/history digests, "
            "score/outcome rows, gate criteria and booleans remain in the projection.",
            "",
            "No Stage-B/Stage-C scientific or materialisation directory, ledger, "
            "aggregate, helper receipt, or conditional log may exist before the "
            "replay receipt passes. Stage C "
            "remains conditional on the unchanged proprioception gate. Any failure "
            "of the corrected attempt permits no further retry.",
            "",
            f"Amendment digest: `{EXECUTION_CORRECTION_AMENDMENT['content_digest']}`.",
            f"Correction output-schema digest: "
            f"`{EXECUTION_CORRECTION_OUTPUT_SCHEMA['output_schema_sha256']}`.",
            f"Correction fixture digest: "
            f"`{EXECUTION_CORRECTION_FIXTURE['fixture_sha256']}`.",
            "",
        ]
    )


def execution_correction_output_schema_receipt_bytes() -> bytes:
    validate_self_digest(EXECUTION_CORRECTION_OUTPUT_SCHEMA, "output_schema_sha256")
    return canonical_json_bytes(EXECUTION_CORRECTION_OUTPUT_SCHEMA) + b"\n"


def execution_correction_fixture_receipt_bytes() -> bytes:
    validate_self_digest(EXECUTION_CORRECTION_FIXTURE, "fixture_sha256")
    return canonical_json_bytes(EXECUTION_CORRECTION_FIXTURE) + b"\n"


def _validate_hex_commit(value: str, label: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{label} is not a full lowercase Git object id")


def validate_prior_smoke_failure_custody(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the narrow runtime bindings for eligible smoke archives."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ContractError("prior_smoke_failure_custody must be a list")
    observed_archives: set[str] = set()
    observed_freezes: set[str] = set()
    output: list[dict[str, Any]] = []
    binding_fields = {"path", "sha256", "bytes", "content_digest"}
    for index, raw_record in enumerate(value):
        if not isinstance(raw_record, Mapping) or set(raw_record) != set(
            PRIOR_SMOKE_FAILURE_CUSTODY_RECORD_FIELDS
        ):
            raise ContractError(f"prior smoke custody record {index} key-set drift")
        record = dict(raw_record)
        archive_path = record["archive_path"]
        if not isinstance(archive_path, str) or not Path(archive_path).is_absolute():
            raise ContractError(f"prior smoke custody record {index} archive path drift")
        if archive_path in observed_archives:
            raise ContractError("prior smoke custody repeats an archive")
        observed_archives.add(archive_path)

        source_freeze = record["source_freeze_commit"]
        if not isinstance(source_freeze, str):
            raise ContractError("prior smoke source-freeze type drift")
        _validate_hex_commit(source_freeze, "prior smoke source_freeze_commit")
        if source_freeze in observed_freezes:
            raise ContractError("same-source-freeze smoke retry is forbidden")
        observed_freezes.add(source_freeze)

        for label in ("failure_receipt", "source_closure"):
            binding_value = record[label]
            if not isinstance(binding_value, Mapping) or set(binding_value) != binding_fields:
                raise ContractError(f"prior smoke {label} binding drift")
            if (
                not isinstance(binding_value["path"], str)
                or not isinstance(binding_value["sha256"], str)
                or len(binding_value["sha256"]) != 64
                or not isinstance(binding_value["bytes"], int)
                or binding_value["bytes"] < 0
                or not isinstance(binding_value["content_digest"], str)
                or len(binding_value["content_digest"]) != 64
            ):
                raise ContractError(f"prior smoke {label} binding value drift")

        inventory = record["inventory"]
        if not isinstance(inventory, Mapping) or set(inventory) != {
            "files",
            "bytes",
            "manifest_sha256",
        }:
            raise ContractError("prior smoke inventory key-set drift")
        if (
            not isinstance(inventory["files"], int)
            or inventory["files"] < 1
            or not isinstance(inventory["bytes"], int)
            or inventory["bytes"] < 1
            or not isinstance(inventory["manifest_sha256"], str)
            or len(inventory["manifest_sha256"]) != 64
        ):
            raise ContractError("prior smoke inventory value drift")
        if record["files_reused"] != 0:
            raise ContractError("prior smoke artifacts are nonreusable")
        output.append(copy.deepcopy(record))
    return output


def _validate_correction_immutable_authorities(
    repo_root: Path,
    prior_custody: Sequence[Mapping[str, Any]],
) -> None:
    """Require every frozen scientific authority to match each smoke snapshot."""

    for record in prior_custody:
        binding_value = record["source_closure"]
        snapshot_path = Path(str(binding_value["path"]))
        if not snapshot_path.is_file():
            raise ContractError("prior smoke source-closure snapshot is absent")
        observed_sha256, observed_bytes = _sha256_file(snapshot_path)
        if (
            observed_sha256 != binding_value["sha256"]
            or observed_bytes != binding_value["bytes"]
        ):
            raise ContractError("prior smoke source-closure snapshot binding drift")
        try:
            snapshot = json.loads(snapshot_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("prior smoke source-closure snapshot is invalid") from exc
        validate_source_closure(snapshot)
        if snapshot.get("content_digest") != binding_value["content_digest"]:
            raise ContractError("prior smoke source-closure content-digest drift")
        rows = {
            str(row["path"]): row
            for row in snapshot["rows"]
            if isinstance(row, Mapping) and "path" in row
        }
        for relative in CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS:
            expected = rows.get(relative)
            if expected is None:
                raise ContractError(
                    f"prior smoke source closure omits immutable authority {relative}"
                )
            current_path = repo_root / relative
            if not current_path.is_file():
                raise ContractError(f"immutable authority is absent: {relative}")
            current_sha256, current_bytes = _sha256_file(current_path)
            if (
                current_sha256 != expected.get("sha256")
                or current_bytes != expected.get("bytes")
            ):
                raise ContractError(
                    f"post-smoke scientific authority changed: {relative}"
                )


def _validate_correction_mutable_python(
    repo_root: Path,
    prior_custody: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Prove a real implementation/test correction against every smoke archive."""

    mutable_paths = tuple(
        str(path)
        for path in SOURCE_CLOSURE_DEFAULT_PATHS
        if str(path).endswith(".py")
    )
    observed: dict[str, bool] = {}
    for record in prior_custody:
        binding_value = record["source_closure"]
        snapshot_path = Path(str(binding_value["path"]))
        try:
            snapshot = json.loads(snapshot_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("prior smoke source-closure snapshot is invalid") from exc
        validate_source_closure(snapshot)
        rows = {
            str(row["path"]): row
            for row in snapshot["rows"]
            if isinstance(row, Mapping) and "path" in row
        }
        correction_observed = False
        for relative in mutable_paths:
            expected = rows.get(relative)
            if expected is None:
                raise ContractError(
                    f"prior smoke source closure omits mutable Python path {relative}"
                )
            current_path = repo_root / relative
            if not current_path.is_file():
                raise ContractError(f"closure-covered Python path is absent: {relative}")
            current_sha256, current_bytes = _sha256_file(current_path)
            if (
                current_sha256 != expected.get("sha256")
                or current_bytes != expected.get("bytes")
            ):
                correction_observed = True
                break
        archive_path = str(record["archive_path"])
        observed[archive_path] = correction_observed
        if not correction_observed:
            raise ContractError(
                "corrected freeze has no implementation/test Python byte change "
                f"relative to prior smoke archive {archive_path}"
            )
    return observed


def validate_repository_custody(repo_root: str | Path) -> dict[str, Any]:
    """Read-only validation of source HEAD, ancestry, and worktree cleanliness."""

    root = Path(repo_root).resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    if head != SOURCE_COMMIT:
        raise ContractError(f"source HEAD drift: {head}")
    if git("status", "--porcelain=v1"):
        raise ContractError("source worktree is dirty")
    for ancestor in (REQUIRED_REQUIREMENTS_ANCESTOR, PREDECESSOR_EXECUTION_FREEZE_COMMIT):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ContractError(f"required ancestor is absent: {ancestor}")
    return {
        "head": head,
        "clean": True,
        "required_ancestors": [
            REQUIRED_REQUIREMENTS_ANCESTOR,
            PREDECESSOR_EXECUTION_FREEZE_COMMIT,
        ],
        "pass": True,
    }


def validate_execution_freeze_custody(
    repo_root: str | Path,
    *,
    prior_smoke_failure_custody: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Validate the normal freeze or a smoke-authorised linear correction freeze."""

    root = Path(repo_root).resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    prior_custody = validate_prior_smoke_failure_custody(
        prior_smoke_failure_custody
    )
    lineage = git("rev-list", "--parents", "-n", "1", head).split()
    if git("status", "--porcelain=v1"):
        raise ContractError("execution contract-freeze worktree is dirty")
    subject = git("show", "-s", "--format=%s", head)
    if subject != CONTRACT_FREEZE_COMMIT_SUBJECT:
        raise ContractError(
            f"execution HEAD subject {subject!r} != {CONTRACT_FREEZE_COMMIT_SUBJECT!r}"
        )

    if not prior_custody:
        if len(lineage) != 2 or lineage[0] != head or lineage[1] != SOURCE_COMMIT:
            raise ContractError(
                "execution HEAD must be the direct single-parent contract-freeze "
                f"child of {SOURCE_COMMIT} when no prior smoke failure exists"
            )
        custody_mode = "DIRECT_CONTRACT_FREEZE"
        changed_paths: list[str] = []
        mutable_correction_observed_by_archive: dict[str, bool] = {}
    else:
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        ).returncode:
            raise ContractError("corrected freeze does not descend from SOURCE_COMMIT")
        lineage_rows = git(
            "rev-list", "--parents", f"{SOURCE_COMMIT}..{head}"
        ).splitlines()
        if not lineage_rows or any(len(row.split()) != 2 for row in lineage_rows):
            raise ContractError("corrected freeze ancestry is not clean and no-merge linear")
        for record in prior_custody:
            prior_freeze = str(record["source_freeze_commit"])
            if prior_freeze == head or subprocess.run(
                ["git", "merge-base", "--is-ancestor", prior_freeze, head],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            ).returncode:
                raise ContractError(
                    "every prior smoke source freeze must be a strict ancestor of HEAD"
                )
        changed_paths = [
            path
            for path in git("diff", "--name-only", SOURCE_COMMIT, head).splitlines()
            if path
        ]
        allowed_paths = {
            *(
                str(path)
                for path in SOURCE_CLOSURE_DEFAULT_PATHS
                if str(path).endswith(".py")
            ),
            *CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS,
            str(TRACKED_SOURCE_CLOSURE_PATH),
        }
        outside_domain = sorted(set(changed_paths) - allowed_paths)
        if outside_domain:
            raise ContractError(
                "corrected freeze changed paths outside the source-closure/generated-"
                f"authority domain: {outside_domain}"
            )
        _validate_correction_immutable_authorities(root, prior_custody)
        mutable_correction_observed_by_archive = (
            _validate_correction_mutable_python(root, prior_custody)
        )
        custody_mode = "VALIDATED_TRAINING_SMOKE_CORRECTION_FREEZE"
    return {
        "source_commit": SOURCE_COMMIT,
        "source_freeze_commit": head,
        "contract_freeze_commit": head,
        "head_subject": subject,
        "custody_mode": custody_mode,
        "prior_smoke_failure_custody": prior_custody,
        "prior_smoke_failure_count": len(prior_custody),
        "changed_paths": changed_paths,
        "mutable_correction_observed_by_archive": (
            mutable_correction_observed_by_archive
        ),
        "closure_and_authority_exact_validation_required": True,
        "sole_parent": lineage[1] if len(lineage) == 2 else None,
        "single_parent": len(lineage) == 2,
        "single_parent_linear_history": True,
        "clean": True,
        "policy": CONTRACT_FREEZE_ANCESTRY_POLICY,
        "pass": True,
    }


def validate_execution_correction_freeze_custody(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate the one exact post-failure execution-correction freeze."""

    root = Path(repo_root).resolve()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain=v1"):
        raise ContractError("execution-correction freeze worktree is dirty")
    lineage = git("rev-list", "--parents", "-n", "1", head).split()
    if (
        len(lineage) != 2
        or lineage[0] != head
        or lineage[1] != INITIAL_EXECUTION_FREEZE_COMMIT
    ):
        raise ContractError(
            "execution-correction freeze must be the direct single-parent child "
            f"of {INITIAL_EXECUTION_FREEZE_COMMIT}"
        )
    subject = git("show", "-s", "--format=%s", head)
    if subject != EXECUTION_CORRECTION_FREEZE_COMMIT_SUBJECT:
        raise ContractError("execution-correction freeze subject drift")
    for ancestor in (
        SOURCE_COMMIT,
        REQUIRED_REQUIREMENTS_ANCESTOR,
        INITIAL_EXECUTION_FREEZE_COMMIT,
    ):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, head],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise ContractError(
                f"execution-correction required ancestor is absent: {ancestor}"
            )
    changed_paths = tuple(
        path
        for path in git(
            "diff", "--name-only", INITIAL_EXECUTION_FREEZE_COMMIT, head
        ).splitlines()
        if path
    )
    outside = sorted(set(changed_paths) - set(EXECUTION_CORRECTION_ALLOWED_CHANGED_PATHS))
    missing = sorted(set(EXECUTION_CORRECTION_REQUIRED_CHANGED_PATHS) - set(changed_paths))
    if outside:
        raise ContractError(
            f"execution-correction freeze changed an unauthorised path: {outside}"
        )
    if missing:
        raise ContractError(
            f"execution-correction freeze omits a required path: {missing}"
        )
    base_authorities = validate_base_scientific_authorities(root)
    amendment = load_and_validate_execution_correction_amendment(
        root / TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH
    )
    correction_schema = load_and_validate_execution_correction_output_schema(
        root / TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH
    )
    correction_fixture = load_and_validate_execution_correction_fixture(
        root / TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH
    )
    correction_closure = load_and_validate_execution_correction_source_closure(
        root / TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH
    )
    for row in correction_closure["rows"]:
        closure_path = root / str(row["path"])
        if not closure_path.is_file():
            raise ContractError(
                f"execution-correction closure path is absent: {row['path']}"
            )
        closure_sha256, closure_bytes = _sha256_file(closure_path)
        if closure_sha256 != row["sha256"] or closure_bytes != row["bytes"]:
            raise ContractError(
                f"execution-correction closure row drift: {row['path']}"
            )
    correction_closure_path = root / TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH
    correction_closure_sha256, correction_closure_bytes = _sha256_file(
        correction_closure_path
    )
    preregistration_path = root / TRACKED_EXECUTION_CORRECTION_PREREGISTRATION_PATH
    if (
        not preregistration_path.is_file()
        or preregistration_path.read_bytes()
        != build_execution_correction_preregistration_markdown().encode("utf-8")
    ):
        raise ContractError("execution-correction preregistration bytes drift")
    archive_custody = validate_execution_correction_archive()
    parent = OUTPUT_ROOT.parent
    attempt_prefix = f".{OUTPUT_ROOT.name}.attempt-"
    failure_prefix = f".{OUTPUT_ROOT.name}.failed-"
    attempts = sorted(
        str(path)
        for path in parent.iterdir()
        if path.name.startswith(attempt_prefix)
    )
    correction_failures = sorted(
        str(path)
        for path in parent.iterdir()
        if path.name.startswith(failure_prefix)
        and path.resolve() != EXECUTION_CORRECTION_FAILED_ARCHIVE.resolve()
    )
    if OUTPUT_ROOT.exists() or attempts or correction_failures:
        raise ContractError(
            "execution-correction one-attempt namespace is already consumed"
        )
    return {
        "source_commit": SOURCE_COMMIT,
        "source_freeze_commit": head,
        "contract_freeze_commit": head,
        "base_source_freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
        "head_subject": subject,
        "sole_parent": INITIAL_EXECUTION_FREEZE_COMMIT,
        "changed_paths": list(changed_paths),
        "base_scientific_authorities": base_authorities,
        "amendment": copy.deepcopy(EXECUTION_CORRECTION_AMENDMENT_BINDING),
        "amendment_output_schema": copy.deepcopy(
            EXECUTION_CORRECTION_OUTPUT_SCHEMA_BINDING
        ),
        "amendment_fixture": copy.deepcopy(EXECUTION_CORRECTION_FIXTURE_BINDING),
        "amendment_source_closure": {
            "path": str(TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH),
            "sha256": correction_closure_sha256,
            "bytes": correction_closure_bytes,
            "content_digest": correction_closure["content_digest"],
            "rows": correction_closure["row_count"],
        },
        "failed_archive_custody": archive_custody,
        "files_reused": 0,
        "fresh_attempts_authorised": 1,
        "fresh_attempts_already_consumed": 0,
        "scientific_authority_contract_sha256": (
            SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        ),
        "pass": bool(
            amendment and correction_schema and correction_fixture and correction_closure
        ),
    }


def build_role_map(split_payload: Mapping[str, Any]) -> dict[str, str]:
    """Build the frozen state-to-role map without reading any route outcomes."""

    if set(split_payload) != {"fit", "calibration", "heldout", "policy"}:
        raise ContractError("split schema drift")
    role_map: dict[str, str] = {}
    for role in ROLE_IDS:
        state_ids = split_payload[role]
        if not isinstance(state_ids, list) or len(state_ids) != ROLE_STATE_COUNTS[role]:
            raise ContractError(f"{role} state cardinality drift")
        if len(set(state_ids)) != len(state_ids):
            raise ContractError(f"duplicate state in {role}")
        for state_id in state_ids:
            if not isinstance(state_id, str) or not state_id:
                raise ContractError(f"invalid state identity in {role}")
            if state_id in role_map:
                raise ContractError(f"state appears in multiple roles: {state_id}")
            role_map[state_id] = role
    if len(role_map) != 48:
        raise ContractError("split must contain exactly 48 distinct states")
    return role_map


def route_role_from_state_class(state_class: str) -> str:
    try:
        return ROLE_CLASS_MAPPING[state_class]
    except KeyError as exc:
        raise ContractError(f"unrecognised frozen V2 state class: {state_class}") from exc


def _validate_bound_file(repo_root: Path, binding: Mapping[str, Any]) -> Path:
    path = repo_root / str(binding["path"])
    if not path.is_file():
        raise ContractError(f"bound route-role input is missing: {path}")
    sha256, size = _sha256_file(path)
    if sha256 != binding["sha256"] or size != binding["bytes"]:
        raise ContractError(f"bound route-role input drift: {path}")
    return path


def build_route_role_authority(repo_root: str | Path) -> dict[str, Any]:
    """Directly copy the already-frozen candidate-invariant role authority once."""

    root = Path(repo_root).resolve()
    metrics_path = _validate_bound_file(root, ROUTE_ROLE_AUTHORITY_BINDING)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics_payload, Mapping):
        raise ContractError("frozen predecessor metrics are not a JSON object")
    if metrics_payload.get("content_digest") != ROUTE_ROLE_AUTHORITY_BINDING[
        "content_digest"
    ]:
        raise ContractError("frozen predecessor metrics content identity drift")
    validate_self_digest(metrics_payload)
    try:
        source_rows = metrics_payload["by_source_population"][
            "KINEMATIC_ROUTE_BASELINE"
        ]["ALL_CANDIDATES"]["per_state"]
    except (KeyError, TypeError) as exc:
        raise ContractError("frozen route-population-class authority is missing") from exc
    if not isinstance(source_rows, list) or len(source_rows) != 48:
        raise ContractError("frozen route-role authority cardinality drift")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row in source_rows:
        if not isinstance(source_row, Mapping):
            raise ContractError("frozen route-role authority row schema drift")
        copied = {
            key: source_row.get(key)
            for key in ("state_id", "family", "role", "route_population_class")
        }
        state_id = copied["state_id"]
        if not isinstance(state_id, str) or not state_id or state_id in seen:
            raise ContractError("frozen route-role authority state identity drift")
        seen.add(state_id)
        if copied["family"] not in FAMILY_IDS or copied["role"] not in ROLE_IDS:
            raise ContractError("frozen route-role authority family/split drift")
        if copied["route_population_class"] not in ROLE_CLASS_MAPPING:
            raise ContractError("frozen route-role authority class drift")
        records.append(
            {
                **copied,
                "route_role": route_role_from_state_class(
                    str(copied["route_population_class"])
                ),
            }
        )
    return attach_self_digest(
        {
            "schema": ROUTE_ROLE_AUTHORITY_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "source_commit": SOURCE_COMMIT,
            "status": "RECOVERED_PRIOR_FROZEN_CANDIDATE_INVARIANT_INPUT_AUTHORITY",
            "generation_phase": "BEFORE_NEW_RANKER_TRAINING",
            "records": records,
            "record_count": 48,
            "split_state_counts": copy.deepcopy(ROLE_STATE_COUNTS),
            "authority": {
                "source": copy.deepcopy(ROUTE_ROLE_AUTHORITY_BINDING),
                "source_fields_read": [
                    "state_id",
                    "family",
                    "role",
                    "route_population_class",
                ],
                "role_mapping": copy.deepcopy(ROLE_CLASS_MAPPING),
                "class_recomputation": False,
            },
            "input_bindings": {
                "predecessor_metrics": copy.deepcopy(ROUTE_ROLE_AUTHORITY_BINDING)
            },
            "custody": {
                "purpose": (
                    "recover prior frozen state-level route-role input authority, "
                    "not infer route role live during training or evaluation"
                ),
                "new_metric_or_target": False,
                "new_design_or_threshold_tuning": False,
                "route_label_candidate_outcomes_read": False,
                "route_class_recomputed": False,
                "source_metric_fields_read": [],
                "candidate_outcome_fields_available_to_new_ranker": [],
                "calibration_or_heldout_metric_values_used": False,
                "live_role_inference_from_candidate_outcomes_forbidden": True,
            },
        }
    )


def validate_route_role_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value)
    if value.get("content_digest") != ROUTE_ROLE_RECEIPT_BINDING["content_digest"]:
        raise ContractError("route-role authority content identity drift")
    if value.get("schema") != ROUTE_ROLE_AUTHORITY_SCHEMA_VERSION:
        raise ContractError("route-role authority schema drift")
    if value.get("experiment_id") != EXPERIMENT_ID or value.get("source_commit") != SOURCE_COMMIT:
        raise ContractError("route-role authority identity drift")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 48 or value.get("record_count") != 48:
        raise ContractError("route-role authority cardinality drift")
    seen: set[str] = set()
    counts = {role: 0 for role in ROLE_IDS}
    for row in records:
        required = {
            "state_id",
            "family",
            "role",
            "route_population_class",
            "route_role",
        }
        if not isinstance(row, Mapping) or set(row) != required:
            raise ContractError("route-role authority record schema drift")
        state_id = row["state_id"]
        if not isinstance(state_id, str) or state_id in seen:
            raise ContractError("route-role authority state identity drift")
        seen.add(state_id)
        if row["family"] not in FAMILY_IDS or row["role"] not in ROLE_IDS:
            raise ContractError("route-role authority family/split drift")
        counts[row["role"]] += 1
        expected_role = route_role_from_state_class(row["route_population_class"])
        if row["route_role"] != expected_role:
            raise ContractError("route-role authority mapping drift")
    if counts != ROLE_STATE_COUNTS:
        raise ContractError("route-role authority split counts drift")
    authority = value.get("authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("source") != ROUTE_ROLE_AUTHORITY_BINDING
        or authority.get("source_fields_read")
        != ["state_id", "family", "role", "route_population_class"]
        or authority.get("class_recomputation") is not False
    ):
        raise ContractError("route-role direct-copy authority drift")
    custody = value.get("custody")
    if (
        not isinstance(custody, Mapping)
        or custody.get("live_role_inference_from_candidate_outcomes_forbidden")
        is not True
        or custody.get("route_label_candidate_outcomes_read") is not False
        or custody.get("route_class_recomputed") is not False
        or custody.get("source_metric_fields_read") != []
    ):
        raise ContractError("route-role live-inference prohibition drift")
    return copy.deepcopy(dict(value))


def route_role_authority_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    validate_route_role_authority(value)
    return canonical_json_bytes(value) + b"\n"


def write_route_role_authority(
    repo_root: str | Path,
    path: str | Path = TRACKED_ROUTE_ROLE_AUTHORITY_PATH,
) -> Path:
    """Atomically materialise the exact frozen route-role input authority.

    Unlike the immutable preregistration writers, this recovery writer is
    deliberately corrective: a missing or stale destination is replaced from
    the bound predecessor authority before any new ranker training.  The
    prospectively frozen receipt binding is checked before publication.
    """

    root = Path(repo_root).resolve()
    destination = Path(path)
    if not destination.is_absolute():
        destination = root / destination
    value = build_route_role_authority(root)
    payload = route_role_authority_receipt_bytes(value)
    if (
        len(payload) != ROUTE_ROLE_RECEIPT_BINDING["bytes"]
        or hashlib.sha256(payload).hexdigest()
        != ROUTE_ROLE_RECEIPT_BINDING["sha256"]
    ):
        raise ContractError("rebuilt route-role authority binding drift")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    load_and_validate_route_role_authority(destination)
    return destination


def load_and_validate_route_role_authority(
    path: str | Path = TRACKED_ROUTE_ROLE_AUTHORITY_PATH,
) -> dict[str, Any]:
    """Load the frozen role receipt; never reopen candidate outcomes."""

    authority_path = Path(path)
    observed = authority_path.read_bytes()
    if len(observed) != ROUTE_ROLE_RECEIPT_BINDING["bytes"]:
        raise ContractError("route-role authority byte-count drift")
    if hashlib.sha256(observed).hexdigest() != ROUTE_ROLE_RECEIPT_BINDING["sha256"]:
        raise ContractError("route-role authority file SHA-256 drift")
    value = json.loads(observed)
    if not isinstance(value, dict):
        raise ContractError("route-role authority must be a JSON object")
    if observed != canonical_json_bytes(value) + b"\n":
        raise ContractError("route-role authority bytes are not canonical")
    return validate_route_role_authority(value)


def load_route_role_map(
    path: str | Path = TRACKED_ROUTE_ROLE_AUTHORITY_PATH,
) -> dict[str, str]:
    authority = load_and_validate_route_role_authority(path)
    return {
        str(row["state_id"]): str(row["route_role"])
        for row in authority["records"]
    }


def build_stage_c_donor_mapping(
    state_ids: Sequence[str],
    *,
    split: str,
    family: str,
    ablation: str,
    contract_digest: str | None = None,
) -> dict[str, Any]:
    """Freeze an outcome-blind bijective cyclic donor derangement.

    The caller supplies one already-validated same-split/same-family cohort.
    Candidate actions, waypoints, identities and labels are not arguments and
    therefore cannot influence the mapping.
    """

    if split not in ROLE_IDS:
        raise ContractError(f"unknown split for donor mapping: {split}")
    if family not in FAMILY_IDS:
        raise ContractError(f"unknown family for donor mapping: {family}")
    if ablation not in STAGE_C_ABLATION_INPUTS:
        raise ContractError(f"unknown Stage-C ablation: {ablation}")
    if len(state_ids) < 2:
        raise ContractError("Stage-C split/family cohort is singleton")
    if len(set(state_ids)) != len(state_ids) or any(
        not isinstance(state_id, str) or not state_id for state_id in state_ids
    ):
        raise ContractError("invalid or duplicate Stage-C state identity")
    digest = CONTRACT_SHA256 if contract_digest is None else contract_digest
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ContractError("contract digest for donor mapping is malformed")

    def state_key(state_id: str) -> tuple[str, str]:
        payload = "\0".join((digest, split, family, state_id, ablation)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), state_id

    ordered = sorted(state_ids, key=state_key)
    donors = ordered[1:] + ordered[:1]
    rows = [
        {
            "recipient_state_id": recipient,
            "donor_state_id": donor,
            "split": split,
            "family": family,
            "ablation_condition_id": ablation,
            "ablation_feature_id": STAGE_C_ABLATION_INPUTS[ablation]["feature_id"],
        }
        for recipient, donor in zip(ordered, donors, strict=True)
    ]
    if {row["recipient_state_id"] for row in rows} != {
        row["donor_state_id"] for row in rows
    } or any(row["recipient_state_id"] == row["donor_state_id"] for row in rows):
        raise ContractError("Stage-C donor mapping is not a strict bijective derangement")
    return attach_self_digest(
        {
            "schema": "plan_aware_stage_c_donor_mapping_v1",
            "contract_sha256": digest,
            "split": split,
            "family": family,
            "ablation_condition_id": ablation,
            "ablation_feature": copy.deepcopy(STAGE_C_ABLATION_INPUTS[ablation]),
            "hash_preimage_fields": [
                "contract_sha256",
                "split",
                "family",
                "state_id",
                "ablation_condition_id",
            ],
            "rows": rows,
            "row_count": len(rows),
            "outcome_fields_read": [],
            "required_persisted_execution_fields": [
                "recipient_state_id",
                "donor_state_id",
                "donor_input_sha256",
            ],
        }
    )


_ROUTE_ROW_KEYS = {"branch_id", "candidate_index", "family", "horizons", "split", "state_id"}
_ROUTE_HORIZON_KEYS = {
    "completed",
    "heading_error_end_rad",
    "heading_error_start_rad",
    "p_d",
    "p_theta_deg",
    "p_theta_rad",
    "route_heading_world_rad",
    "safe",
}


def extract_route_only_rows(
    route_rows: Iterable[Mapping[str, Any]],
    split_payload: Mapping[str, Any],
    *,
    phase: str,
    final_checkpoints_locked: bool = False,
    evaluation_contract_frozen: bool = False,
    require_complete_role: bool = True,
) -> list[dict[str, Any]]:
    """Extract only the authorised route-preference components for one role.

    ``safe`` is schema-checked but its value is never read or returned.  Contact,
    viability, stuck, prior aggregate-scorer utility, and material-contact
    values are neither accepted as route targets nor emitted by this helper.
    """

    phase_to_role = {
        "TRAINING": "fit",
        "CALIBRATION": "calibration",
        "EVALUATION": "heldout",
    }
    if phase not in phase_to_role:
        raise ContractError(f"unknown route extraction phase: {phase}")
    role = phase_to_role[phase]
    if phase == "CALIBRATION" and not final_checkpoints_locked:
        raise ContractError("calibration cannot open before final checkpoints are locked")
    if phase == "EVALUATION" and not evaluation_contract_frozen:
        raise ContractError("heldout cannot open before the evaluation contract is frozen")

    role_map = build_role_map(split_payload)
    extracted: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    states: set[str] = set()
    for source_row in route_rows:
        if not isinstance(source_row, Mapping) or set(source_row) != _ROUTE_ROW_KEYS:
            raise ContractError("route-label top-level schema drift")
        state_id = source_row["state_id"]
        if state_id not in role_map:
            raise ContractError(f"route row has unknown state: {state_id}")
        authoritative_role = role_map[state_id]
        if source_row["split"] != authoritative_role:
            raise ContractError(f"route row split disagrees with frozen split: {state_id}")
        if authoritative_role != role:
            continue
        candidate_index = source_row["candidate_index"]
        if candidate_index not in CANDIDATE_INDICES:
            raise ContractError("candidate index outside the frozen bank")
        identity = (state_id, candidate_index)
        if identity in seen:
            raise ContractError(f"duplicate route row: {identity}")
        seen.add(identity)
        states.add(state_id)
        family = source_row["family"]
        if family not in FAMILY_IDS:
            raise ContractError(f"unknown family: {family}")
        horizons = source_row["horizons"]
        if not isinstance(horizons, Mapping) or set(horizons) != {"1", "2", "3"}:
            raise ContractError("route horizon schema drift")
        h3 = horizons["3"]
        if not isinstance(h3, Mapping) or set(h3) != _ROUTE_HORIZON_KEYS:
            raise ContractError("route H3 schema drift")
        completed = h3["completed"]
        p_d = h3["p_d"]
        p_theta_rad = h3["p_theta_rad"]
        if not isinstance(completed, bool):
            raise ContractError("completed route component must be bool")
        if isinstance(p_d, bool) or not isinstance(p_d, (int, float)) or not math.isfinite(float(p_d)):
            raise ContractError("p_d route component must be finite")
        if (
            isinstance(p_theta_rad, bool)
            or not isinstance(p_theta_rad, (int, float))
            or not math.isfinite(float(p_theta_rad))
        ):
            raise ContractError("p_theta_rad route component must be finite")
        extracted.append(
            {
                "branch_id": source_row["branch_id"],
                "state_id": state_id,
                "candidate_index": candidate_index,
                "family": family,
                "role": role,
                "route_preference_components": {
                    "completed": completed,
                    "p_d": float(p_d),
                    "p_theta_rad": float(p_theta_rad),
                },
            }
        )
    extracted.sort(key=lambda row: (row["state_id"], row["candidate_index"]))
    if require_complete_role:
        if len(extracted) != ROLE_ROW_COUNTS[role]:
            raise ContractError(f"{role} route-row cardinality drift")
        if len(states) != ROLE_STATE_COUNTS[role]:
            raise ContractError(f"{role} route-state cardinality drift")
        grouped: dict[str, set[int]] = {}
        for row in extracted:
            grouped.setdefault(row["state_id"], set()).add(row["candidate_index"])
        if any(indices != set(CANDIDATE_INDICES) for indices in grouped.values()):
            raise ContractError("a state does not contain the complete frozen candidate bank")
    return extracted


def derangement_is_material(
    *,
    pairwise_accuracy_drop: float | None,
    selected_progress_fraction_drop: float | None,
    normalized_regret_increase: float | None,
) -> bool:
    pairwise = _gate_number(pairwise_accuracy_drop)
    progress = _gate_number(selected_progress_fraction_drop)
    regret = _gate_number(normalized_regret_increase)
    material_opposing_reversal = any(
        (
            pairwise is not None
            and pairwise
            <= -DERANGEMENT_MATERIALITY[
                "pairwise_accuracy_opposing_improvement_min"
            ],
            progress is not None
            and progress
            <= -DERANGEMENT_MATERIALITY[
                "selected_progress_fraction_opposing_improvement_min"
            ],
            regret is not None
            and regret
            <= -DERANGEMENT_MATERIALITY[
                "normalized_regret_opposing_improvement_min"
            ],
        )
    )
    if material_opposing_reversal:
        return False
    return any(
        (
            pairwise is not None
            and pairwise >= DERANGEMENT_MATERIALITY["pairwise_accuracy_drop_min"],
            progress is not None
            and progress
            >= DERANGEMENT_MATERIALITY["selected_progress_fraction_drop_min"],
            regret is not None
            and regret
            >= DERANGEMENT_MATERIALITY["normalized_regret_increase_min"],
        )
    )


def margin_borda_pairwise_target(left_utility: float, right_utility: float) -> int:
    """Return the strict Section-12 target from conditioned Borda utilities."""

    left = float(left_utility)
    right = float(right_utility)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ContractError("margin-Borda pairwise utilities must be finite")
    delta = left - right
    if abs(delta) <= PAIRWISE_UTILITY_TOLERANCE:
        return 0
    return 1 if delta > 0.0 else -1


def fit_state_optimization_status(admissible_candidate_count: int) -> str:
    """Return the frozen fit-state optimizer status without opening outcomes."""

    if (
        isinstance(admissible_candidate_count, bool)
        or not isinstance(admissible_candidate_count, int)
        or not 0 <= admissible_candidate_count <= len(CANDIDATE_INDICES)
    ):
        raise ContractError(
            "admissible candidate count must be an integer in the frozen "
            f"candidate-bank range [0,{len(CANDIDATE_INDICES)}]"
        )
    if admissible_candidate_count == 0:
        return "SKIPPED_ZERO_ADMISSIBLE"
    if admissible_candidate_count == 1:
        return "SKIPPED_SINGLETON_ADMISSIBLE"
    return "CONTRIBUTING"


def _gate_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _gate_minimum(value: Any, threshold: float) -> bool:
    numeric = _gate_number(value)
    return bool(numeric is not None and numeric >= threshold)


def _gate_maximum(value: Any, threshold: float) -> bool:
    numeric = _gate_number(value)
    return bool(numeric is not None and numeric <= threshold)


def _gate_strict_positive(value: Any) -> bool:
    numeric = _gate_number(value)
    return bool(numeric is not None and numeric > 0.0)


def true_future_gate_passes(metrics: Mapping[str, Any]) -> bool:
    required = {
        "pairwise_accuracy",
        "spearman_rho",
        "normalized_regret",
        "best_route_top3",
        "selected_progress_fraction_of_oracle",
        "no_family_complete_collapse",
        "candidate_derangement_material",
    }
    if set(metrics) != required:
        raise ContractError("true-future gate metric schema drift")
    return bool(
        _gate_minimum(
            metrics["pairwise_accuracy"], TRUE_FUTURE_GATE["pairwise_accuracy_min"]
        )
        and _gate_minimum(
            metrics["spearman_rho"], TRUE_FUTURE_GATE["spearman_rho_min"]
        )
        and _gate_maximum(
            metrics["normalized_regret"], TRUE_FUTURE_GATE["normalized_regret_max"]
        )
        and _gate_minimum(
            metrics["best_route_top3"], TRUE_FUTURE_GATE["best_route_top3_min"]
        )
        and _gate_minimum(
            metrics["selected_progress_fraction_of_oracle"],
            TRUE_FUTURE_GATE["selected_progress_fraction_of_oracle_min"],
        )
        and metrics["no_family_complete_collapse"] is True
        and metrics["candidate_derangement_material"] is True
    )


def true_incremental_gate_passes(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Require the frozen incremental criteria against kinematics *and* no-latent."""

    comparator_ids = {"KINEMATIC", "NO_LATENT"}
    if set(comparisons) != comparator_ids:
        raise ContractError("true-incremental comparator schema drift")
    required = {
        "selected_progress_gain_m",
        "oracle_progress_fraction_gain",
        "normalized_regret_reduction",
        "family_tie_or_improve_count",
        "maximum_population_progress_loss_fraction",
        "all_candidates_contact_selections_no_worse",
        "all_candidates_nonviable_selections_no_worse",
    }
    for comparator in sorted(comparator_ids):
        metrics = comparisons[comparator]
        if set(metrics) != required:
            raise ContractError(f"true-incremental metric schema drift: {comparator}")
        progress_pass = bool(
            _gate_minimum(
                metrics["selected_progress_gain_m"],
                TRUE_INCREMENTAL_GATE["selected_progress_gain"]["absolute_m_min"],
            )
            or _gate_minimum(
                metrics["oracle_progress_fraction_gain"],
                TRUE_INCREMENTAL_GATE["selected_progress_gain"][
                    "or_oracle_progress_fraction_gain_min"
                ],
            )
        )
        if not (
            progress_pass
            and _gate_minimum(
                metrics["normalized_regret_reduction"],
                TRUE_INCREMENTAL_GATE["normalized_regret_reduction_min"],
            )
            and _gate_minimum(
                metrics["family_tie_or_improve_count"],
                TRUE_INCREMENTAL_GATE["family_tie_or_improve_min"],
            )
            and _gate_maximum(
                metrics["maximum_population_progress_loss_fraction"],
                TRUE_INCREMENTAL_GATE[
                    "maximum_population_progress_loss_fraction"
                ],
            )
            and metrics["all_candidates_contact_selections_no_worse"] is True
            and metrics["all_candidates_nonviable_selections_no_worse"] is True
        ):
            return False
    return True


def true_incremental_not_evaluated_payload() -> dict[str, Any]:
    """Return the exact fail-closed Stage-A payload after a failed TRUE gate."""

    return copy.deepcopy(TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD)


def rr_gate_passes(metrics: Mapping[str, Any]) -> bool:
    required = {
        "pairwise_accuracy",
        "normalized_regret",
        "best_route_top3",
        "selected_progress_fraction_of_oracle",
        "selected_progress_fraction_of_true",
        "pairwise_accuracy_delta_vs_r1",
        "selected_progress_delta_vs_r1",
        "normalized_regret_reduction_vs_r1",
        "no_family_complete_collapse",
        "all_candidates_contact_selections_no_worse_than_r1",
        "all_candidates_nonviable_selections_no_worse_than_r1",
        "all_candidates_stuck_selections_no_worse_than_r1",
    }
    if set(metrics) != required:
        raise ContractError("RR gate metric schema drift")
    return bool(
        _gate_minimum(metrics["pairwise_accuracy"], RR_GATE["pairwise_accuracy_min"])
        and _gate_maximum(
            metrics["normalized_regret"], RR_GATE["normalized_regret_max"]
        )
        and _gate_minimum(
            metrics["best_route_top3"], RR_GATE["best_route_top3_min"]
        )
        and _gate_minimum(
            metrics["selected_progress_fraction_of_oracle"],
            RR_GATE["selected_progress_fraction_of_oracle_min"],
        )
        and _gate_minimum(
            metrics["selected_progress_fraction_of_true"],
            RR_GATE["selected_progress_fraction_of_true_min"],
        )
        and _gate_strict_positive(metrics["pairwise_accuracy_delta_vs_r1"])
        and (
            _gate_strict_positive(metrics["selected_progress_delta_vs_r1"])
            or _gate_strict_positive(metrics["normalized_regret_reduction_vs_r1"])
        )
        and metrics["no_family_complete_collapse"] is True
        and metrics["all_candidates_contact_selections_no_worse_than_r1"] is True
        and metrics["all_candidates_nonviable_selections_no_worse_than_r1"] is True
        and metrics["all_candidates_stuck_selections_no_worse_than_r1"] is True
    )


def predicted_source_absolute_preservation_passes(
    metrics: Mapping[str, Any],
) -> bool:
    """Apply the same frozen absolute screen independently to one source."""

    required = {
        "pairwise_accuracy",
        "normalized_regret",
        "best_route_top3",
        "selected_progress_fraction_of_oracle",
        "selected_progress_fraction_of_true",
        "no_family_complete_collapse",
    }
    if set(metrics) != required:
        raise ContractError("predicted-source absolute-preservation schema drift")
    gate = PREDICTED_SOURCE_ABSOLUTE_PRESERVATION_GATE
    return bool(
        _gate_minimum(metrics["pairwise_accuracy"], gate["pairwise_accuracy_min"])
        and _gate_maximum(
            metrics["normalized_regret"], gate["normalized_regret_max"]
        )
        and _gate_minimum(metrics["best_route_top3"], gate["best_route_top3_min"])
        and _gate_minimum(
            metrics["selected_progress_fraction_of_oracle"],
            gate["selected_progress_fraction_of_oracle_min"],
        )
        and _gate_minimum(
            metrics["selected_progress_fraction_of_true"],
            gate["selected_progress_fraction_of_true_min"],
        )
        and metrics["no_family_complete_collapse"] is True
    )


def all_predicted_substitutions_fail_materially(
    per_source_metrics: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return true iff none of the four predicted sources passes absolutely."""

    if set(per_source_metrics) != set(PREDICTED_SOURCE_IDS):
        raise ContractError("predicted-source absolute-preservation source drift")
    return not any(
        predicted_source_absolute_preservation_passes(per_source_metrics[source])
        for source in PREDICTED_SOURCE_IDS
    )


def proprio_contribution_passes(metrics: Mapping[str, Any]) -> bool:
    required = {
        "pairwise_accuracy_gain",
        "selected_progress_fraction_gain",
        "normalized_regret_reduction",
        "best_route_top3_gain",
        "all_candidates_contact_selections_no_worse",
        "all_candidates_nonviable_selections_no_worse",
    }
    if set(metrics) != required:
        raise ContractError("proprio contribution metric schema drift")
    criteria = PROPRIO_CONTRIBUTION_GATE["criteria"]
    passes = sum(
        (
            _gate_minimum(
                metrics["pairwise_accuracy_gain"],
                criteria["pairwise_accuracy_gain_min"],
            ),
            _gate_minimum(
                metrics["selected_progress_fraction_gain"],
                criteria["selected_progress_fraction_gain_min"],
            ),
            _gate_minimum(
                metrics["normalized_regret_reduction"],
                criteria["normalized_regret_reduction_min"],
            ),
            _gate_minimum(
                metrics["best_route_top3_gain"],
                criteria["best_route_top3_gain_min"],
            ),
        )
    )
    return bool(
        passes >= PROPRIO_CONTRIBUTION_GATE["minimum_criteria"]
        and metrics["all_candidates_contact_selections_no_worse"] is True
        and metrics["all_candidates_nonviable_selections_no_worse"] is True
    )


def derive_primary_classification(
    *,
    true_gate_pass: bool,
    true_incremental_gate_pass: bool,
    rr_gate_pass: bool,
    all_predicted_substitutions_fail_materially: bool,
) -> str:
    """Apply the frozen, mutually exclusive primary decision tree."""

    if not true_gate_pass:
        return "PLAN_AWARE_JEPA_COST_NO_SIGNAL"
    if not true_incremental_gate_pass:
        return "KINEMATIC_BASELINE_DOMINANT"
    if rr_gate_pass:
        return "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL"
    if all_predicted_substitutions_fail_materially:
        return "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO"
    raise ContractError("primary classification is unresolved by the frozen decision tree")


def next_experiment_for_primary(primary: str) -> str:
    mapping = {
        "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL": "ORACLE_ADMISSIBLE_CLOSED_LOOP_JEPA_MPC_V1",
        "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO": "PLAN_AWARE_PREDICTOR_TRAINING_V1",
        "KINEMATIC_BASELINE_DOMINANT": "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
        "PLAN_AWARE_JEPA_COST_NO_SIGNAL": "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
    }
    try:
        return mapping[primary]
    except KeyError as exc:
        raise ContractError(f"unknown primary classification: {primary}") from exc


def next_experiment_specification_for_primary(primary: str) -> dict[str, Any]:
    """Return the exact specify-only successor payload for one primary result."""

    experiment_id = next_experiment_for_primary(primary)
    try:
        specification = NEXT_EXPERIMENT_SPECIFICATIONS[experiment_id]
    except KeyError as exc:  # guarded by frozen vocabulary tests
        raise ContractError(
            f"missing next-experiment specification: {experiment_id}"
        ) from exc
    return {
        "experiment_id": experiment_id,
        **copy.deepcopy(specification),
    }


def _contract_core() -> dict[str, Any]:
    if not classification_vocabularies_are_unique():
        raise ContractError("primary, secondary, or next-experiment vocabulary repeats")
    return {
        "schema": CONTRACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "date": DATE,
        "status": "PROSPECTIVE_NOT_EXECUTED",
        "claim_boundary": {
            "development_only": True,
            "route_only_ranker": True,
            "deployment_safety_claim": False,
            "learned_safety_model": False,
            "oracle_contact_and_viability": (
                "contact/viability/stuck are never score inputs or route targets; "
                "oracle viability is used only as the prospectively authorised "
                "training-set conditioning mask and evaluation population"
            ),
            "closed_loop_navigation": False,
        },
        "git_custody": {
            "source_commit": SOURCE_COMMIT,
            "required_ancestor": REQUIRED_REQUIREMENTS_ANCESTOR,
            "predecessor_execution_freeze_commit": PREDECESSOR_EXECUTION_FREEZE_COMMIT,
            "contract_freeze_commit": "TO_BE_BOUND_AFTER_ATOMIC_CONTRACT_COMMIT",
            "result_commit_in_tracked_result_json": None,
            "result_commit_binding_policy": RESULT_COMMIT_BINDING_POLICY,
            "contract_freeze_ancestry_policy": CONTRACT_FREEZE_ANCESTRY_POLICY,
            "required_result_ancestry": (
                "result commit descends from the contract-freeze commit, which descends "
                "from source commit and required requirements ancestor"
            ),
        },
        "panel": {
            "name": "SAFE_LOCAL_WAYPOINT_ROUTE_INTENT_V2",
            "states": 48,
            "candidates_per_state": 12,
            "candidate_rows": 576,
            "families": list(FAMILY_IDS),
            "horizons": list(HORIZON_IDS),
            "roles": copy.deepcopy(ROLE_STATE_COUNTS),
            "bindings": copy.deepcopy(PANEL_BINDINGS),
            "independent_heldout_generalisation_claim": False,
        },
        "encoder": copy.deepcopy(ENCODER_BINDING),
        "checkpoints": copy.deepcopy(CHECKPOINT_BINDINGS),
        "predictor_seed": PREDICTOR_SEED,
        "ranker_seed": RANKER_SEED,
        "active_policies": list(ACTIVE_POLICIES),
        "execution_retry_policy": copy.deepcopy(EXECUTION_RETRY_POLICY),
        "predecessor_tensor_package": copy.deepcopy(PREDECESSOR_TENSOR_PACKAGE),
        "predecessor_candidate_evidence": copy.deepcopy(
            PREDECESSOR_CANDIDATE_EVIDENCE_BINDING
        ),
        "proprio_inputs": copy.deepcopy(PROPRIO_INPUT_BINDINGS),
        "legacy_counterfactual_tensors": copy.deepcopy(
            LEGACY_COUNTERFACTUAL_TENSOR_BINDINGS
        ),
        "feature_contract": {
            "base_order": [
                {"name": name, "width": width, "semantics": semantics}
                for name, width, semantics in BASE_FEATURE_LAYOUT
            ],
            "base_dimension": BASE_FEATURE_DIM,
            "query_order": [
                {"name": name, "width": width, "semantics": semantics}
                for name, width, semantics in QUERY_FEATURE_LAYOUT
            ],
            "query_dimension": QUERY_FEATURE_DIM,
            "route_role_mapping": copy.deepcopy(ROLE_CLASS_MAPPING),
            "route_role_authority": copy.deepcopy(ROUTE_ROLE_AUTHORITY_BINDING),
            "route_role_receipt": copy.deepcopy(ROUTE_ROLE_RECEIPT_BINDING),
            "route_role_runtime_policy": (
                "load by state_id from the bound receipt only; live inference from "
                "candidate route/contact outcomes is forbidden"
            ),
            "route_role_one_hot_order": list(ROLE_ONE_HOT_ORDER),
            "score_anchor": "negative deterministic kinematic_rank_cost",
        },
        "training": copy.deepcopy(TRAINING_POLICY),
        "gates": {
            "true_future": copy.deepcopy(TRUE_FUTURE_GATE),
            "derangement_materiality": copy.deepcopy(DERANGEMENT_MATERIALITY),
            "true_incremental": copy.deepcopy(TRUE_INCREMENTAL_GATE),
            "rr": copy.deepcopy(RR_GATE),
            "predicted_source_absolute_preservation": copy.deepcopy(
                PREDICTED_SOURCE_ABSOLUTE_PRESERVATION_GATE
            ),
            "proprio_contribution": copy.deepcopy(PROPRIO_CONTRIBUTION_GATE),
        },
        "stages": copy.deepcopy(STAGE_POLICY),
        "primary_classifications": list(PRIMARY_CLASSIFICATIONS),
        "secondary_classifications": list(SECONDARY_CLASSIFICATIONS),
        "next_experiments": {
            primary: next_experiment_for_primary(primary)
            for primary in PRIMARY_CLASSIFICATIONS
        },
        "next_experiment_specifications": copy.deepcopy(
            NEXT_EXPERIMENT_SPECIFICATIONS
        ),
        "next_experiment_execution_authorised_here": False,
        "preserved_requirements_classifications": list(
            PRESERVED_REQUIREMENTS_CLASSIFICATIONS
        ),
        "preserved_predecessor_facts": list(PRESERVED_PREDECESSOR_FACTS),
        "preserved_predecessor_fact_scope": copy.deepcopy(
            PRESERVED_PREDECESSOR_FACT_SCOPE
        ),
        "preserved_predecessor_narrative": list(PRESERVED_PREDECESSOR_NARRATIVE),
        "requirements_statement": (
            "Deployment hard-contact requirements, consequences and recovery criteria "
            "remain unresolved. No further deployment-safety scope reduction, sensor "
            "qualification or learned hard-safety model is authorised."
        ),
        "accidental_exposures": copy.deepcopy(list(ACCIDENTAL_EXPOSURES)),
        "preexecution_cpu_worker_benchmark": copy.deepcopy(CPU_WORKER_BENCHMARK),
        "outcome_barrier": {
            "design_authority": "user literals, frozen structural metadata, and hashes only",
            "predecessor_detailed_values_used_for_design_or_tuning": False,
            "fit_route_rows": "only after tracked contract/source closure are committed",
            "calibration_route_rows": "only after final checkpoints are locked",
            "heldout_route_rows": "only after evaluation contract and gates are frozen",
        },
        "prohibitions": {
            "safety_model": True,
            "contact_or_viability_target": True,
            "completion_head_or_regression": True,
            "fresh_panel": True,
            "new_sensor_layout": True,
            "protected_contact_scope_change": True,
            "closed_loop_navigation": True,
            "memory_novelty_routing_or_beacon_capture": True,
            "stage_b_before_true_gate": True,
            "stage_c_without_proprio_contribution": True,
        },
        "tracked_paths": {
            "preregistration": str(TRACKED_PREREGISTRATION_PATH),
            "contract": str(TRACKED_CONTRACT_PATH),
            "output_schema": str(TRACKED_OUTPUT_SCHEMA_PATH),
            "fixture": str(TRACKED_FIXTURE_PATH),
            "source_closure": str(TRACKED_SOURCE_CLOSURE_PATH),
            "route_role_authority": str(TRACKED_ROUTE_ROLE_AUTHORITY_PATH),
            "result": str(TRACKED_RESULT_PATH),
            "report": str(TRACKED_REPORT_PATH),
        },
        "output_root": str(OUTPUT_ROOT),
    }


def build_contract() -> dict[str, Any]:
    return attach_self_digest(_contract_core(), "contract_sha256")


def _output_schema_core() -> dict[str, Any]:
    return {
        "schema": OUTPUT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "conditional_stages": {
            "stage_a": "always",
            "stage_b": "only when the true-future gate passes",
            "stage_c": "only when PROPRIOCEPTIVE_ROUTE_CONTRIBUTION is supported",
        },
        "execution_retry_policy": copy.deepcopy(EXECUTION_RETRY_POLICY),
        "artifacts": {
            "route_role_authority": {
                **copy.deepcopy(ROUTE_ROLE_RECEIPT_BINDING),
                "schema": ROUTE_ROLE_AUTHORITY_SCHEMA_VERSION,
                "external_to_output_root": True,
                "required": True,
            },
            "route_only_target_ledger": {
                "path": "ledgers/route_only_targets.jsonl",
                "schema": "plan_aware_route_only_target_row_v1",
                "rows": 576,
                "required_training_conditioning_fields": [
                    "oracle_viability_admissible_conditioning",
                    "fit_state_optimization_status",
                    "used_for_fit",
                    "conditioned_margin_borda_utility",
                ],
                "fit_state_status_null_outside_fit_role": True,
                "used_for_fit_definition": (
                    "fit-role row whose state status is CONTRIBUTING and whose "
                    "candidate is oracle-viability-admissible"
                ),
                "all_fit_state_rows_persisted_including_skipped_states": True,
                "conditioning_policy": (
                    "TRAIN_ONLY_ON_ORACLE_VIABILITY_ADMISSIBLE_CANDIDATE_SETS"
                ),
                "forbidden_fields": [
                    "safe",
                    "contact",
                    "successor_viability",
                    "stuck",
                    "prior_aggregate_scorer_utility",
                    "material_contact",
                ],
            },
            "no_latent_checkpoint": {
                "path": "checkpoints/no_latent_final_epoch_060.pt",
                "schema": "plan_aware_ranker_checkpoint_receipt_v1",
                "required": True,
                "condition": "KINEMATIC_PLUS_NO_LATENT_RESIDUAL",
                "required_root_fields": [
                    "schema",
                    "condition",
                    "seed_family",
                    "seed_metadata",
                    "epoch",
                    "final_epoch_only",
                    "parameter_count",
                    "model_contract",
                    "fit_optimization",
                    "state_dict",
                    "parameter_digest",
                    "optimizer_state_persisted",
                    "training_history",
                ],
                "required_seed_metadata_fields": [
                    "seed_family",
                    "condition_id",
                    "condition_key_sha256",
                    "condition_keyed_seed_sha256",
                    "condition_torch_seed",
                    "shared_base_subkey",
                    "shared_base_subkey_sha256",
                    "shared_base_keyed_seed_sha256",
                    "shared_base_torch_seed",
                ],
                "seed_metadata_binding": copy.deepcopy(
                    CHECKPOINT_SEED_METADATA["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"]
                ),
                "model_contract_location": "root.model_contract",
                "fit_optimization_location": "root.fit_optimization",
                "required_fit_optimization_fields": list(
                    FIT_OPTIMIZATION_RECEIPT_FIELDS
                ),
            },
            "latent_checkpoint": {
                "path": "checkpoints/latent_true_future_final_epoch_060.pt",
                "schema": "plan_aware_ranker_checkpoint_receipt_v1",
                "required": True,
                "condition": "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL",
                "required_root_fields": [
                    "schema",
                    "condition",
                    "seed_family",
                    "seed_metadata",
                    "epoch",
                    "final_epoch_only",
                    "parameter_count",
                    "model_contract",
                    "fit_optimization",
                    "state_dict",
                    "parameter_digest",
                    "optimizer_state_persisted",
                    "training_history",
                ],
                "required_seed_metadata_fields": [
                    "seed_family",
                    "condition_id",
                    "condition_key_sha256",
                    "condition_keyed_seed_sha256",
                    "condition_torch_seed",
                    "shared_base_subkey",
                    "shared_base_subkey_sha256",
                    "shared_base_keyed_seed_sha256",
                    "shared_base_torch_seed",
                ],
                "seed_metadata_binding": copy.deepcopy(
                    CHECKPOINT_SEED_METADATA["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"]
                ),
                "model_contract_location": "root.model_contract",
                "fit_optimization_location": "root.fit_optimization",
                "required_fit_optimization_fields": list(
                    FIT_OPTIMIZATION_RECEIPT_FIELDS
                ),
            },
            "training_ledger": {
                "path": "ledgers/training_epochs.jsonl",
                "schema": "plan_aware_training_epoch_row_v1",
                "rows": 120,
                "required_state_accounting_fields": [
                    "fit_states_total",
                    "fit_states_contributing",
                    "fit_states_skipped_zero_admissible",
                    "fit_states_skipped_singleton_admissible",
                    "epoch_average_denominator",
                ],
                "epoch_average_denominator_semantics": (
                    "integer equal to fit_states_contributing"
                ),
            },
            "stage_a_rows": {
                "path": "ledgers/stage_a_true_future.jsonl",
                "schema": "plan_aware_evaluation_row_v1",
                "rows": 576,
                "required_merged_raw_score_fields": list(
                    RAW_COST_REREDUCED_SOURCE_IDS
                ),
                "raw_cost_scores_merged_after_authorised_barrier_open": True,
            },
            "stage_a_raw_cost_rereduced_rows": {
                "path": "ledgers/stage_a_raw_cost_rereduced.jsonl",
                "schema": "plan_aware_raw_cost_rereduced_row_v1",
                "rows": 1_728,
                "source_ids": list(RAW_COST_REREDUCED_SOURCE_IDS),
                "rows_per_source": 576,
                "input_binding": copy.deepcopy(
                    PREDECESSOR_CANDIDATE_EVIDENCE_BINDING
                ),
                "score_field": "negative_cost_h3",
                "required_fields": [
                    "schema",
                    "split",
                    "state_id",
                    "family",
                    "candidate_index",
                    "source_id",
                    "predecessor_source",
                    "cost_h3",
                    "score",
                    "population_membership",
                ],
                "successor_metric_contract_required": True,
                "no_cosine_or_predictor_inference": True,
                "also_merged_into_stage_a_score_maps_and_summaries": True,
            },
            "stage_b_rows": {
                "path": "ledgers/stage_b_predictor_substitution.jsonl",
                "schema": "plan_aware_evaluation_row_v1",
                "rows_if_complete": 2_304,
                "conditional": True,
            },
            "stage_c_rows": {
                "path": "ledgers/stage_c_attribution.jsonl",
                "schema": "plan_aware_attribution_row_v1",
                "rows_if_complete": 1_728,
                "conditional": True,
            },
            "stage_c_direct_fidelity_rows": {
                "path": "ledgers/stage_c_direct_fidelity.jsonl",
                "schema": "plan_aware_stage_c_direct_fidelity_row_v1",
                "rows_if_complete": 6_912,
                "conditional": True,
            },
            "stage_c_candidate_action_sensitivity_rows": {
                "path": "ledgers/stage_c_candidate_action_sensitivity.jsonl",
                "schema": "plan_aware_candidate_action_sensitivity_row_v1",
                "rows_if_complete": 576,
                "conditional": True,
            },
            "metrics": {
                "path": "aggregates/metrics.json",
                "schema": "plan_aware_monotone_jepa_cost_metrics_v1",
                "self_digest_key": "content_digest",
            },
            "preexecution_receipt": {
                "path": "receipts/preexecution.json",
                "schema": "plan_aware_preexecution_receipt_v1",
                "required_fields": [
                    "prior_smoke_failure_custody",
                    "source_closure_snapshot",
                ],
                "prior_smoke_failure_custody_schema": copy.deepcopy(
                    EXECUTION_RETRY_POLICY["prior_smoke_failure_custody"]
                ),
            },
            "pre_smoke_source_closure_snapshot": {
                "path": "receipts/source_closure.json",
                "schema": SOURCE_CLOSURE_SCHEMA_VERSION,
                "required": True,
                "write_phase": "PREEXECUTION_BEFORE_FIT_OR_TRAINING_SMOKE",
                "byte_exact_copy_of_tracked_source_closure": True,
                "preexecution_binding_field": "source_closure_snapshot",
                "purpose": (
                    "immutable scientific-authority witness for any subsequently "
                    "authorised smoke-only correction refreeze"
                ),
            },
            "failed_attempt_archive": {
                "external_to_successful_output_root": True,
                "schema": "plan_aware_monotone_jepa_failure_v1",
                "eligible_retry_phase": "TRAINING_SMOKE",
                "required_retry_eligibility_fields": [
                    "phase",
                    "full_training_epochs_completed",
                    "calibration_rows_opened",
                    "heldout_rows_opened",
                    "final_checkpoint_published",
                    "partial_artifacts_reusable",
                    "nothing_running",
                ],
                "required_eligible_values": copy.deepcopy(
                    EXECUTION_RETRY_POLICY["eligible_failure_receipt_exact"]
                ),
            },
            "training_receipt": {
                "path": "receipts/training.json",
                "schema": "plan_aware_training_receipt_v1",
                "required_root_fields": ["fit_optimization"],
                "fit_optimization_location": "root.fit_optimization",
                "required_fit_optimization_fields": list(
                    FIT_OPTIMIZATION_RECEIPT_FIELDS
                ),
                "fit_states_total_value": ROLE_STATE_COUNTS["fit"],
                "epoch_average_denominator_semantics": (
                    "integer equal to fit_states_contributing"
                ),
                "id_order": "numeric_state_key ascending",
                "partition_required": True,
            },
            "evaluation_receipt": {
                "path": "receipts/evaluation.json",
                "schema": "plan_aware_evaluation_receipt_v1",
            },
            "persistence_receipt": {
                "path": "receipts/persistence.json",
                "schema": "plan_aware_persistence_receipt_v1",
                "required_fields": [
                    "artifact_manifest",
                    "row_counts",
                    "row_to_aggregate_reproduction",
                    "source_commit",
                    "source_freeze_commit",
                    "contract_freeze_commit",
                    "result_commit",
                    "result_commit_binding_policy",
                    "ancestry_validation",
                    "nothing_running",
                    "prohibition_counters",
                    "prior_smoke_failure_custody",
                    "content_digest",
                ],
            },
        },
        "result_required_fields": [
            "schema",
            "experiment_id",
            "source_commit",
            "source_freeze_commit",
            "contract_freeze_commit",
            "result_commit",
            "result_commit_binding_policy",
            "ancestry_validation",
            "contract_sha256",
            "output_schema_sha256",
            "panel_bindings",
            "checkpoint_bindings",
            "stage_execution",
            "metrics",
            "primary_classification",
            "secondary_classifications",
            "next_experiment",
            "next_experiment_specification",
            "requirements_workstream",
            "predecessor_fact_authority",
            "predecessor_narrative_authority",
            "prior_smoke_failure_custody",
            "prohibition_counters",
            "runtime_and_storage",
            "nothing_running",
            "content_digest",
        ],
        "common_row_identity": [
            "state_id",
            "candidate_index",
            "family",
            "split",
            "latent_source",
            "population_membership",
        ],
        "persistence": {
            "all_rows_required": True,
            "aggregate_reproduction_without_training_or_inference": True,
            "all_json_receipts_self_digest": True,
            "all_binary_artifacts_sha256_and_bytes": True,
        },
        "required_metric_fields": {
            "stage_a_failed_true_gate_incremental_payload": copy.deepcopy(
                TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD
            ),
            "per_state": ["selected_combined_route_utility"],
            "aggregate_per_population_source": [
                "selected_combined_route_utility_sum",
                "selected_combined_route_utility_mean",
                "selected_combined_route_utility_count",
            ],
            "descriptive_per_source": ["descriptive_adverse_downranking"],
            "stage_a_matched_raw_cost_rereduction": {
                "field": "stage_a_raw_cost_rereduced",
                "source_ids": list(RAW_COST_REREDUCED_SOURCE_IDS),
                "metric_contract": "current successor METRIC_CONTRACT",
                "paired_comparator_mapping": {
                    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": (
                        "RAW_TRUE_FUTURE_GOAL_COSINE"
                    ),
                    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1": (
                        "RAW_R1_GOAL_COSINE"
                    ),
                    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR": (
                        "RAW_RR_GOAL_COSINE"
                    ),
                },
            },
            "stage_a_matched_raw_cost_comparisons": {
                "field": "stage_a_raw_cost_matched_comparisons",
                "paired_per_state": True,
                "descriptive_bootstrap": True,
                "classification_gate": False,
                "comparison_mapping": copy.deepcopy(
                    STAGE_POLICY["STAGE_A_TRUE_FUTURE"][
                        "matched_raw_cost_comparator_rereduction"
                    ]["matched_comparisons"]
                ),
                "metrics": copy.deepcopy(
                    STAGE_POLICY["STAGE_A_TRUE_FUTURE"][
                        "matched_raw_cost_comparator_rereduction"
                    ]["matched_comparison_metrics"]
                ),
            },
            "historical_raw_aggregate_context": {
                "field": "historical_raw_latent_goal_cosine_comparators",
                "comparable_to_successor_metrics": False,
                "retained_separately": True,
            },
        },
        "required_predecessor_narrative_authority": list(
            PRESERVED_PREDECESSOR_NARRATIVE
        ),
        "required_predecessor_fact_authority": list(PRESERVED_PREDECESSOR_FACTS),
        "required_next_experiment_specifications": copy.deepcopy(
            NEXT_EXPERIMENT_SPECIFICATIONS
        ),
    }


def build_output_schema() -> dict[str, Any]:
    return attach_self_digest(_output_schema_core(), "output_schema_sha256")


def _fixture_core() -> dict[str, Any]:
    true_pass = {
        "pairwise_accuracy": 0.75,
        "spearman_rho": 0.60,
        "normalized_regret": 0.20,
        "best_route_top3": 0.75,
        "selected_progress_fraction_of_oracle": 0.80,
        "no_family_complete_collapse": True,
        "candidate_derangement_material": True,
    }
    rr_pass = {
        "pairwise_accuracy": 0.70,
        "normalized_regret": 0.25,
        "best_route_top3": 0.75,
        "selected_progress_fraction_of_oracle": 0.75,
        "selected_progress_fraction_of_true": 0.85,
        "pairwise_accuracy_delta_vs_r1": 1e-12,
        "selected_progress_delta_vs_r1": 1e-12,
        "normalized_regret_reduction_vs_r1": 0.0,
        "no_family_complete_collapse": True,
        "all_candidates_contact_selections_no_worse_than_r1": True,
        "all_candidates_nonviable_selections_no_worse_than_r1": True,
        "all_candidates_stuck_selections_no_worse_than_r1": True,
    }
    return {
        "schema": FIXTURE_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "synthetic_only": True,
        "heldout_rows_read": 0,
        "checks": {
            "base_feature_dimension": BASE_FEATURE_DIM == 138,
            "query_feature_dimension": QUERY_FEATURE_DIM == 71,
            "no_latent_parameter_cap": NO_LATENT_PARAMETER_COUNT < 250_000,
            "latent_parameter_cap": LATENT_PARAMETER_COUNT < 500_000,
            "token_grid_contract_exact": (
                TOKEN_GRID_SHAPE == (24, 32)
                and TOKEN_GRID_SHAPE[0] * TOKEN_GRID_SHAPE[1]
                == TOKENS_PER_TIMEPOINT
                and TRAINING_POLICY["models"]["LATENT"]["token_flat_index"]
                == "y * 32 + x"
            ),
            "classification_vocabularies_unique": (
                classification_vocabularies_are_unique()
            ),
            "pairwise_utility_above_tolerance_orders": (
                margin_borda_pairwise_target(
                    2.0 * PAIRWISE_UTILITY_TOLERANCE, 0.0
                )
                == 1
            ),
            "pairwise_utility_at_tolerance_ties": (
                margin_borda_pairwise_target(PAIRWISE_UTILITY_TOLERANCE, 0.0)
                == 0
            ),
            "evaluation_ordering_authority_is_borda_and_tie_aware": (
                "population-conditioned margin-Borda"
                in METRIC_CONTRACT["ordering_authority"]["pairwise_target"]
                and "oracle-best-set"
                in METRIC_CONTRACT["ordering_authority"]["topk_mrr_mean_rank"]
                and "maximum population progress"
                in METRIC_CONTRACT["ordering_authority"]["normalized_regret"]
            ),
            "fit_state_zero_is_skipped": (
                fit_state_optimization_status(0) == "SKIPPED_ZERO_ADMISSIBLE"
            ),
            "fit_state_singleton_is_skipped": (
                fit_state_optimization_status(1)
                == "SKIPPED_SINGLETON_ADMISSIBLE"
            ),
            "fit_state_two_or_more_contributes": (
                fit_state_optimization_status(2) == "CONTRIBUTING"
                and fit_state_optimization_status(len(CANDIDATE_INDICES))
                == "CONTRIBUTING"
            ),
            "next_experiment_specifications_complete": (
                set(NEXT_EXPERIMENT_SPECIFICATIONS) == set(NEXT_EXPERIMENT_IDS)
                and all(
                    next_experiment_specification_for_primary(primary)[
                        "experiment_id"
                    ]
                    == next_experiment_for_primary(primary)
                    for primary in PRIMARY_CLASSIFICATIONS
                )
            ),
            "raw_cost_comparator_rereduction_is_bound_and_no_inference": (
                PREDECESSOR_CANDIDATE_EVIDENCE_BINDING["rows"] == 1_728
                and PREDECESSOR_CANDIDATE_EVIDENCE_BINDING["rows_per_source"]
                == 576
                and RAW_COST_REREDUCED_SOURCE_IDS
                == (
                    "RAW_TRUE_FUTURE_GOAL_COSINE",
                    "RAW_R1_GOAL_COSINE",
                    "RAW_RR_GOAL_COSINE",
                )
                and "never rerun cosine"
                in PREDECESSOR_CANDIDATE_EVIDENCE_BINDING["reduction_policy"]
            ),
            "shared_base_keyed_seed_exact": CONDITION_KEYED_SEEDS[
                "shared_base_subkey"
            ]["keyed_seed_sha256"]
            == "b52ee5677368a328cb0f1dea45f236c991280568628e7f685f321876d4e02978",
            "no_latent_condition_keyed_seed_exact": CONDITION_KEYED_SEEDS[
                "condition_keys"
            ]["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"]["keyed_seed_sha256"]
            == "cf4b986fac761239d9849c3e532788f135cb49e95a44107485f8f70c23e549d2",
            "latent_condition_keyed_seed_exact": CONDITION_KEYED_SEEDS[
                "condition_keys"
            ]["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"]["keyed_seed_sha256"]
            == "b9dada5d0d1f4628b4110e3a09997b47395b0d84bc8c03d056c10a52c87fb8ea",
            "true_gate_inclusive_boundaries": true_future_gate_passes(true_pass),
            "true_gate_unavailable_metric_fails_closed": not true_future_gate_passes(
                {**true_pass, "spearman_rho": None}
            ),
            "rr_gate_strict_r1_pairwise": rr_gate_passes(rr_pass),
            "derangement_any_threshold": derangement_is_material(
                pairwise_accuracy_drop=0.05,
                selected_progress_fraction_drop=0.0,
                normalized_regret_increase=0.0,
            ),
            "derangement_submaterial_opposing_change_allowed": derangement_is_material(
                pairwise_accuracy_drop=0.05,
                selected_progress_fraction_drop=-0.099,
                normalized_regret_increase=0.0,
            ),
            "derangement_material_reversal_fails": not derangement_is_material(
                pairwise_accuracy_drop=1.0,
                selected_progress_fraction_drop=-0.10,
                normalized_regret_increase=1.0,
            ),
            "stage_a_fail_classification": derive_primary_classification(
                true_gate_pass=False,
                true_incremental_gate_pass=False,
                rr_gate_pass=False,
                all_predicted_substitutions_fail_materially=True,
            )
            == "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
            "stage_a_failed_true_incremental_payload_exact": (
                true_incremental_not_evaluated_payload()
                == TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD
                and TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD["evaluated"] is False
                and TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD["pass"] is False
                and TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD["comparisons"] == {}
            ),
            "retry_policy_is_smoke_only_and_nonreusing": (
                EXECUTION_RETRY_POLICY["default"] == "NO_RETRY"
                and EXECUTION_RETRY_POLICY["eligible_failure_phase"]
                == "TRAINING_SMOKE"
                and EXECUTION_RETRY_POLICY["eligible_failure_receipt_exact"]
                ["full_training_epochs_completed"]
                == 0
                and EXECUTION_RETRY_POLICY["eligible_failure_receipt_exact"]
                ["final_checkpoint_published"]
                is False
                and EXECUTION_RETRY_POLICY["prior_artifact_reuse"] is False
                and EXECUTION_RETRY_POLICY["later_failure"]["retry_allowed"]
                is False
                and validate_prior_smoke_failure_custody([]) == []
            ),
        },
    }


def build_evaluator_fixture() -> dict[str, Any]:
    fixture = attach_self_digest(_fixture_core(), "fixture_sha256")
    if not all(fixture["checks"].values()):
        raise ContractError("synthetic evaluator fixture failed")
    return fixture


CONTRACT = build_contract()
CONTRACT_SHA256 = CONTRACT["contract_sha256"]
OUTPUT_SCHEMA = build_output_schema()
OUTPUT_SCHEMA_SHA256 = OUTPUT_SCHEMA["output_schema_sha256"]
EVALUATOR_FIXTURE = build_evaluator_fixture()
EVALUATOR_FIXTURE_SHA256 = EVALUATOR_FIXTURE["fixture_sha256"]


def build_preregistration_markdown() -> str:
    """Build a compact deterministic human-readable preregistration."""

    lines = [
        "# Plan-aware monotone JEPA cost V1 preregistration",
        "",
        f"Experiment: `{EXPERIMENT_ID}`.",
        "",
        "This is a development-only, route-only ranker experiment. It supports no "
        "deployment-safety, material-hazard, learned-assurance, or closed-loop "
        "navigation claim.",
        "",
        "## Preserved predecessor authority",
        "",
        *[f"- `{value}`" for value in PRESERVED_PREDECESSOR_FACTS],
        "- These IDs remain scoped to the predecessor raw token-wise goal-cosine "
        "assay; a successor plan-aware finding does not overwrite them.",
        "",
        *[f"- {statement}" for statement in PRESERVED_PREDECESSOR_NARRATIVE],
        "",
        "## Active policies",
        "",
        *[f"- `{policy}`" for policy in ACTIVE_POLICIES],
        "",
        "## Frozen custody",
        "",
        f"- Source commit: `{SOURCE_COMMIT}`.",
        f"- Required requirements ancestor: `{REQUIRED_REQUIREMENTS_ANCESTOR}`.",
        "- Panel: 48 frozen states, 12 candidates/state, H1-H3; 32 fit, 8 "
        "calibration, 8 observed development-heldout.",
        "- TRUE/R1/RR tensors are already bound. P1/PR checkpoints exist, but "
        "48-state P1/PR tensors and proprio-input custody are not yet bound.",
        "- Route role is loaded from the bound 48-state candidate-invariant "
        "authority receipt; it is never inferred live from candidate outcomes.",
        f"- Contract-freeze commit subject: `{CONTRACT_FREEZE_COMMIT_SUBJECT}`.",
        "",
        "## Rankers",
        "",
        "Two final-only rankers are trained on fit rows: a 138-D no-latent "
        "kinematic/base MLP and a latent ranker with shared 1024-to-64 token "
        "projection, one 71-to-64 query, shared attention over current/H1/H2/H3, "
        "and a 394-to-256-to-128-to-1 readout. Both are residuals around the "
        "negative kinematic rank-cost anchor.",
        "Every 768-by-1024 timepoint tensor is explicitly viewed as a 24-by-32-"
        "by-1024 spatial grid in row-major order (`flat_index = y * 32 + x`) and "
        "flattened in that same order before shared token LayerNorm, projection, "
        "and attention. This view/flatten contract does not permute token values.",
        "",
        "AdamW uses lr 1e-3, weight decay 1e-4, 60 epochs and only the final "
        "checkpoint. Loss weights are pairwise 1, listwise 0.5 and residual L2 "
        "1e-3; listwise temperature is 1.",
        "Pairwise targets are the sign of conditioned margin-Borda utility "
        "differences only when the absolute difference exceeds 1e-12; otherwise "
        "they are zero. The completion/0.03 m/5 degree direct comparator only "
        "constructs that Borda utility and is not a separate pairwise target.",
        "Evaluation uses the same population-conditioned Borda authority for "
        "pairwise accuracy and ideal top-k/MRR/mean-rank metrics. Every candidate "
        "within 1e-12 of maximum Borda utility belongs to the oracle-best set; "
        "top-k succeeds for any member and rank uses the earliest model-ranked "
        "member. Candidate index resolves only deterministic representative/order "
        "ties. Route-progress Spearman/Kendall and maximum-progress normalized "
        "regret remain separate progress diagnostics.",
        "A legitimately unavailable metric from an empty population, complete "
        "score tie, constant progress, or absence of Borda-ordered pairs remains "
        "`null` evidence. Any required gate criterion using it is false, never an "
        "execution exception. Stage A therefore classifies an all-tied true scorer "
        "as `PLAN_AWARE_JEPA_COST_NO_SIGNAL`; predicted, proprioception, and "
        "substitution gates likewise fail closed while preserving nullable "
        "descriptive deltas and factorial contrasts.",
        "Condition-keyed seeds use the frozen SHA-256 derivation. Both models' "
        "base branches use the shared-base subkey for byte-identical initialisation; "
        "latent-only parameters use the exact latent-condition key.",
        "Execution is no-retry by default. Only an archived failure in phase "
        "`TRAINING_SMOKE`, with zero completed full-training epochs, zero opened "
        "calibration/heldout rows, no final checkpoint, nonreusable partial "
        "artifacts, and nothing running, may authorize a wholly fresh corrected "
        "attempt. It must use a clean no-merge descendant correction freeze, bind "
        "every prior smoke archive in `prior_smoke_failure_custody`, reuse no "
        "artifact, and validate exact closure/authority. Because fit rows were "
        "opened before smoke, every archived pre-smoke source-closure snapshot "
        "must prove byte-identical preregistration, contract, output schema, "
        "fixture, and route-role authority; only closure-covered Python "
        "implementation/tests and the refreshed source-closure receipt may change. "
        "An empty or closure-only descendant is forbidden: at least one covered "
        "Python implementation/test byte must differ from every archived smoke "
        "snapshot. Same-freeze and every post-smoke retry are forbidden.",
        "",
        "The target is the existing local-waypoint margin-Borda preference. "
        "Completion appears only in that ordering tuple. The prospectively frozen "
        "local route margin-Borda listwise target is the only utility target; there "
        "is no completion, safety, contact, viability, stuck, material-contact, or "
        "prior aggregate-scorer-utility head or target, and no unrestricted "
        "aggregate-utility head.",
        "Training uses only the prospectively authorised oracle-viability-"
        "admissible candidate sets. Oracle viability is a disclosed row-level "
        "conditioning mask, never a score input or route target.",
        "All 32 fit-state identities and every fit row remain in the evidence "
        "ledger. A fit state contributes to optimization only when at least two "
        "oracle-viability-admissible candidates exist. Zero- and singleton-"
        "admissible states are deterministically classified as "
        "`SKIPPED_ZERO_ADMISSIBLE` or `SKIPPED_SINGLETON_ADMISSIBLE`, contribute "
        "no pairwise, listwise, or residual loss, receive no optimizer step, and "
        "are excluded from the epoch-loss denominator. Epoch averages use "
        "`CONTRIBUTING` fit states only.",
        "For every source and candidate population, evaluation separately reports "
        "the selected candidate's population-conditioned margin-Borda route utility "
        "per state and its aggregate sum and mean.",
        "After both final checkpoints are locked and the evaluation/heldout barrier "
        "is open, the byte-bound predecessor 1,728-row candidate-evidence ledger is "
        "read once to re-reduce `RAW_TRUE_FUTURE_GOAL_COSINE`, "
        "`RAW_R1_GOAL_COSINE`, and `RAW_RR_GOAL_COSINE` from `-cost_h3` "
        "under these same successor Borda/rank/progress metrics. No cosine or "
        "predictor inference is rerun. Copied predecessor aggregates remain a "
        "separate historical, non-comparable context rather than matched evidence.",
        "",
        "## Stages and stop rules",
        "",
        "Stage A evaluates true-future candidate-specific route information. A "
        "failed true-future gate stops before new predictor inference and yields "
        "`PLAN_AWARE_JEPA_COST_NO_SIGNAL`; its incremental gate is persisted as "
        "classification `TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_EVALUATED`, "
        "status `NOT_EVALUATED_TRUE_GATE_FAILED`, evaluated/pass false, reason "
        "`TRUE_FUTURE_GATE_FAILED`, and empty comparisons. Stage B is permitted only after that "
        "gate passes and substitutes R1/RR/P1/PR into the fixed latent ranker. "
        "Stage C runs only after `PROPRIOCEPTIVE_ROUTE_CONTRIBUTION` and uses "
        "strict substitutions/derangements without refitting.",
        "The future-derangement result reports pairwise-accuracy loss, selected-"
        "progress loss in metres, selected-progress-ratio loss, normalized-regret "
        "worsening, and best-route-top-3 loss. The metre and top-3 losses are "
        "descriptive only; the frozen materiality triggers remain pairwise loss, "
        "progress-ratio loss, or regret worsening with no material principal "
        "reversal.",
        "R1, RR, P1 and PR each receive the same frozen absolute preservation "
        "screen. `all predicted substitutions fail materially` means none passes. "
        "If any source passes while the complete RR gate fails, the primary "
        "classification remains unresolved and execution fails closed.",
        "The complete RR gate additionally requires ALL_CANDIDATES selected "
        "immediate-contact, successor-nonviable, and stuck counts all to be no "
        "worse than R1. The incremental and proprioception gates retain their "
        "separate contact-plus-nonviability adverse checks.",
        "The result persists the complete specify-only payload for its selected "
        "next experiment: oracle-admissible fixed-bank closed-loop MPC; one-seed "
        "route-consistency predictor training without a safety or protected-scope "
        "change; or a non-greedy obstacle-mediated local-subgoal assay with at "
        "least two geometrically plausible alternatives, oracle admissibility, "
        "and initially no topological memory or beacon layer.",
        "",
        "All thresholds, primary classifications, next decisions, hashes, role "
        "barriers and output schemas are normative in the accompanying canonical "
        "JSON contract.",
        "",
        f"Contract SHA-256: `{CONTRACT_SHA256}`.",
        f"Output-schema SHA-256: `{OUTPUT_SCHEMA_SHA256}`.",
        f"Evaluator-fixture SHA-256: `{EVALUATOR_FIXTURE_SHA256}`.",
        "",
    ]
    return "\n".join(lines)


def _receipt_bytes(value: Mapping[str, Any], digest_key: str) -> bytes:
    validate_self_digest(value, digest_key)
    return canonical_json_bytes(value) + b"\n"


def contract_receipt_bytes() -> bytes:
    return _receipt_bytes(CONTRACT, "contract_sha256")


def output_schema_receipt_bytes() -> bytes:
    return _receipt_bytes(OUTPUT_SCHEMA, "output_schema_sha256")


def evaluator_fixture_receipt_bytes() -> bytes:
    return _receipt_bytes(EVALUATOR_FIXTURE, "fixture_sha256")


SOURCE_CLOSURE_DEFAULT_PATHS = (
    "lewm/__init__.py",
    "lewm/safety/__init__.py",
    "lewm/safety/plan_aware_monotone_jepa_cost_v1_contract.py",
    "lewm/tests/test_plan_aware_monotone_jepa_cost_v1_contract.py",
    "lewm/safety/plan_aware_monotone_jepa_cost_v1.py",
    "lewm/safety/plan_aware_monotone_jepa_cost_metrics_v1.py",
    "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py",
    "scripts/materialize_plan_aware_proprio_predictor_substitution_v1.py",
    "lewm/tests/test_plan_aware_monotone_jepa_cost_v1.py",
    "lewm/tests/test_plan_aware_monotone_jepa_cost_metrics_v1.py",
    "lewm/tests/test_evaluate_plan_aware_monotone_jepa_cost_v1.py",
    "lewm/tests/test_materialize_plan_aware_proprio_predictor_substitution_v1.py",
    str(TRACKED_PREREGISTRATION_PATH),
    str(TRACKED_CONTRACT_PATH),
    str(TRACKED_OUTPUT_SCHEMA_PATH),
    str(TRACKED_FIXTURE_PATH),
    str(TRACKED_ROUTE_ROLE_AUTHORITY_PATH),
    "lewm/safety/jepa_local_waypoint_planning_cost_qualification_v1_contract.py",
    "lewm/safety/jepa_local_waypoint_planning_cost_metrics_v1.py",
    "scripts/dev_proprio_predictor_v1.py",
    "scripts/build_dev_v03_proprio_action_manifest_v1.py",
    "scripts/dev_action_slew_reconstruction_v1.py",
    "scripts/run_dev_v03_temporal_action_jepa_v1.py",
    "scripts/materialize_deployment_valid_dense_proprioception_v1.py",
    "scripts/run_go2_oracle_branch_pilot_v1.py",
    "scripts/run_go2_oracle_branch_pilot_v1_2.py",
    "scripts/run_safe_local_waypoint_route_intent_v2.py",
    "scripts/dev_checkpoint_v1.py",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "scripts/materialize_dense_route_intent_true_future_v1.py",
    "scripts/replay_safe_local_waypoint_route_intent_v2.py",
    "scripts/render_replay_v03.py",
    "scripts/analyze_go2_closed_loop_quality.py",
    "lewm/models/__init__.py",
    "lewm/models/direct_egocentric_bev_state_jepa_v1.py",
    "lewm/models/encoders.py",
    "lewm/models/lewm.py",
    "lewm/models/phase2d_spatial_lewm.py",
    "lewm/models/predictor.py",
    "lewm/models/primitive_affordance.py",
    "lewm/models/sigreg.py",
    "lewm/models/source_action_utility.py",
    "lewm/models/spatial_lewm.py",
    "lewm/models/spatial_predictor.py",
    "lewm/oracle/__init__.py",
    "lewm/oracle/go2_branch_oracle_v1_2.py",
    "lewm/oracle/go2_textured_v03_renderer.py",
    "lewm_genesis/lewm_genesis/__init__.py",
    "lewm_genesis/lewm_genesis/batch_renderer.py",
    "lewm_genesis/lewm_genesis/camera_safety.py",
    "lewm_genesis/lewm_genesis/collectors/__init__.py",
    "lewm_genesis/lewm_genesis/collectors/base.py",
    "lewm_genesis/lewm_genesis/collectors/frontier.py",
    "lewm_genesis/lewm_genesis/collectors/ou_noise.py",
    "lewm_genesis/lewm_genesis/collectors/primitive_curriculum.py",
    "lewm_genesis/lewm_genesis/collectors/recovery.py",
    "lewm_genesis/lewm_genesis/collectors/route_teacher.py",
    "lewm_genesis/lewm_genesis/go2_adapter.py",
    "lewm_genesis/lewm_genesis/lewm_contract.py",
    "lewm_genesis/lewm_genesis/parity_checks.py",
    "lewm_genesis/lewm_genesis/render_replay.py",
    "lewm_genesis/lewm_genesis/rollout.py",
    "lewm_genesis/lewm_genesis/ros_msg_adapter.py",
    "lewm_genesis/lewm_genesis/scene_builder.py",
    "lewm_genesis/lewm_genesis/scene_loader.py",
    "lewm_genesis/lewm_genesis/textures.py",
    "lewm_worlds/lewm_worlds/__init__.py",
    "lewm_worlds/lewm_worlds/corpus.py",
    "lewm_worlds/lewm_worlds/exporters/__init__.py",
    "lewm_worlds/lewm_worlds/exporters/to_gazebo_sdf.py",
    "lewm_worlds/lewm_worlds/exporters/to_genesis.py",
    "lewm_worlds/lewm_worlds/families.py",
    "lewm_worlds/lewm_worlds/labels/__init__.py",
    "lewm_worlds/lewm_worlds/labels/derived.py",
    "lewm_worlds/lewm_worlds/labels/topology.py",
    "lewm_worlds/lewm_worlds/manifest.py",
    "lewm_worlds/lewm_worlds/planning_grid.py",
    "lewm_worlds/lewm_worlds/randomization.py",
    "lewm_worlds/lewm_worlds/scene_graph.py",
    "lewm_worlds/lewm_worlds/scene_validation.py",
    "lewm_worlds/lewm_worlds/splits.py",
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_contract_2026-08-26.json",
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_output_schema_2026-08-26.json",
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_source_closure_2026-08-26.json",
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_result_2026-08-26.json",
    "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_result_2026-08-26.md",
)

EXECUTION_CORRECTION_SOURCE_CLOSURE_DEFAULT_PATHS = (
    *SOURCE_CLOSURE_DEFAULT_PATHS,
    str(TRACKED_SOURCE_CLOSURE_PATH),
    str(TRACKED_EXECUTION_CORRECTION_PREREGISTRATION_PATH),
    str(TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH),
    str(TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH),
    str(TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH),
)


def build_source_closure(
    repo_root: str | Path,
    *,
    paths: Iterable[str | Path] | None = None,
    additional_paths: Iterable[str | Path] = (),
    require_complete: bool = True,
) -> dict[str, Any]:
    """Hash an explicit source list without traversing caches or result roots."""

    root = Path(repo_root).resolve()
    selected = [Path(value) for value in (SOURCE_CLOSURE_DEFAULT_PATHS if paths is None else paths)]
    selected.extend(Path(value) for value in additional_paths)
    if len({str(path) for path in selected}) != len(selected):
        raise ContractError("source closure contains a duplicate path")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in selected:
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"source-closure path must be repository-relative: {relative}")
        absolute = root / relative
        if not absolute.is_file():
            missing.append(str(relative))
            continue
        sha256, size = _sha256_file(absolute)
        rows.append({"path": str(relative), "sha256": sha256, "bytes": size})
    if require_complete and missing:
        raise ContractError(f"source closure is incomplete: {missing!r}")
    return attach_self_digest(
        {
            "schema": SOURCE_CLOSURE_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "source_commit": SOURCE_COMMIT,
            "declared_paths": [str(path) for path in selected],
            "rows": rows,
            "row_count": len(rows),
            "missing_paths": missing,
            "complete": not missing,
            "outcome_or_result_payloads_parsed": [],
            "custody_only_predecessor_results_hashed_without_parsing": [
                str(PREDECESSOR_RESULT_BINDINGS["json"]["path"]),
                str(PREDECESSOR_RESULT_BINDINGS["markdown"]["path"]),
            ],
            "generated_cache_paths_traversed": [],
            "accidental_exposures": copy.deepcopy(list(ACCIDENTAL_EXPOSURES)),
        }
    )


def source_closure_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    return _receipt_bytes(value, "content_digest")


def build_execution_correction_source_closure(
    repo_root: str | Path,
    *,
    paths: Iterable[str | Path] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Hash current correction code plus immutable base and amendment authorities."""

    root = Path(repo_root).resolve()
    selected = [
        Path(value)
        for value in (
            EXECUTION_CORRECTION_SOURCE_CLOSURE_DEFAULT_PATHS
            if paths is None
            else paths
        )
    ]
    if len({str(path) for path in selected}) != len(selected):
        raise ContractError("execution-correction source closure repeats a path")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for relative in selected:
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(
                "execution-correction source-closure paths must be repository-relative"
            )
        absolute = root / relative
        if not absolute.is_file():
            missing.append(str(relative))
            continue
        sha256, size = _sha256_file(absolute)
        rows.append({"path": str(relative), "sha256": sha256, "bytes": size})
    if require_complete and missing:
        raise ContractError(
            f"execution-correction source closure is incomplete: {missing}"
        )
    return attach_self_digest(
        {
            "schema": EXECUTION_CORRECTION_SOURCE_CLOSURE_SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "base_source_freeze_commit": INITIAL_EXECUTION_FREEZE_COMMIT,
            "scientific_authority_contract_sha256": (
                SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
            ),
            "amendment": copy.deepcopy(EXECUTION_CORRECTION_AMENDMENT_BINDING),
            "amendment_output_schema": copy.deepcopy(
                EXECUTION_CORRECTION_OUTPUT_SCHEMA_BINDING
            ),
            "amendment_fixture": copy.deepcopy(
                EXECUTION_CORRECTION_FIXTURE_BINDING
            ),
            "correction_freeze_commit_binding_policy": (
                "bound by the enclosing Git correction-freeze commit after "
                "byte-finalization; no circular commit hash is embedded"
            ),
            "declared_paths": [str(path) for path in selected],
            "rows": rows,
            "row_count": len(rows),
            "missing_paths": missing,
            "complete": not missing,
            "self_path_excluded_to_avoid_circular_digest": str(
                TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH
            ),
            "base_scientific_authorities_immutable": copy.deepcopy(
                BASE_SCIENTIFIC_AUTHORITY_BINDINGS
            ),
            "outcome_values_parsed": False,
            "tensors_opened": False,
            "failed_archive_hashed_only": True,
        }
    )


def validate_execution_correction_source_closure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    validate_self_digest(value)
    if value.get("schema") != EXECUTION_CORRECTION_SOURCE_CLOSURE_SCHEMA_VERSION:
        raise ContractError("execution-correction source closure schema drift")
    if value.get("base_source_freeze_commit") != INITIAL_EXECUTION_FREEZE_COMMIT:
        raise ContractError("execution-correction base freeze drift")
    if (
        value.get("scientific_authority_contract_sha256")
        != SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
    ):
        raise ContractError("execution-correction scientific digest drift")
    if value.get("complete") is not True or value.get("missing_paths") != []:
        raise ContractError("execution-correction source closure is incomplete")
    declared = value.get("declared_paths")
    rows = value.get("rows")
    if (
        not isinstance(declared, list)
        or declared != list(EXECUTION_CORRECTION_SOURCE_CLOSURE_DEFAULT_PATHS)
        or len(declared) != len(set(declared))
        or not isinstance(rows, list)
        or value.get("row_count") != len(rows)
        or [row.get("path") for row in rows] != declared
    ):
        raise ContractError("execution-correction source closure path-set drift")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256", "bytes"}
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
        ):
            raise ContractError("execution-correction source closure row drift")
    if value.get("base_scientific_authorities_immutable") != (
        BASE_SCIENTIFIC_AUTHORITY_BINDINGS
    ):
        raise ContractError("execution-correction base-authority binding drift")
    if value.get("outcome_values_parsed") is not False or value.get(
        "tensors_opened"
    ) is not False:
        raise ContractError("execution-correction source closure crossed outcome barrier")
    return copy.deepcopy(dict(value))


def execution_correction_source_closure_receipt_bytes(
    value: Mapping[str, Any],
) -> bytes:
    validate_execution_correction_source_closure(value)
    return canonical_json_bytes(value) + b"\n"


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value, "contract_sha256")
    if canonical_json_bytes(value) != canonical_json_bytes(CONTRACT):
        raise ContractError("contract value drift")
    return copy.deepcopy(dict(value))


def validate_output_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value, "output_schema_sha256")
    if canonical_json_bytes(value) != canonical_json_bytes(OUTPUT_SCHEMA):
        raise ContractError("output schema value drift")
    return copy.deepcopy(dict(value))


def validate_evaluator_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value, "fixture_sha256")
    if canonical_json_bytes(value) != canonical_json_bytes(EVALUATOR_FIXTURE):
        raise ContractError("evaluator fixture value drift")
    if not all(value["checks"].values()):
        raise ContractError("evaluator fixture check failed")
    return copy.deepcopy(dict(value))


def validate_source_closure(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_self_digest(value, "content_digest")
    if value.get("schema") != SOURCE_CLOSURE_SCHEMA_VERSION:
        raise ContractError("source closure schema drift")
    if value.get("source_commit") != SOURCE_COMMIT:
        raise ContractError("source closure commit drift")
    if value.get("complete") is not True or value.get("missing_paths") != []:
        raise ContractError("source closure is incomplete")
    if value.get("row_count") != len(value.get("rows", [])):
        raise ContractError("source closure cardinality drift")
    declared = value.get("declared_paths")
    rows = value.get("rows")
    missing = value.get("missing_paths")
    if (
        not isinstance(declared, list)
        or any(not isinstance(path, str) for path in declared)
        or len(declared) != len(set(declared))
    ):
        raise ContractError("source closure declared-path drift")
    if not isinstance(rows, list) or not isinstance(missing, list):
        raise ContractError("source closure row schema drift")
    observed_paths: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "bytes"}:
            raise ContractError("source closure row schema drift")
        if (
            not isinstance(row["path"], str)
            or not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or not isinstance(row["bytes"], int)
            or row["bytes"] < 0
        ):
            raise ContractError("source closure row value drift")
        observed_paths.append(row["path"])
    if len(missing) != len(set(missing)) or any(
        not isinstance(path, str) for path in missing
    ):
        raise ContractError("source closure missing-path drift")
    expected_observed = [path for path in declared if path not in set(missing)]
    if observed_paths != expected_observed or set(declared) != set(observed_paths) | set(missing):
        raise ContractError("source closure declared/observed path drift")
    return copy.deepcopy(dict(value))


def validate_result_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate result structure and decisions, but never calculate a metric."""

    validate_self_digest(value, "content_digest")
    required = set(OUTPUT_SCHEMA["result_required_fields"])
    if set(value) != required:
        raise ContractError("result receipt key-set drift")
    if value["schema"] != "plan_aware_monotone_jepa_cost_v1.result.v1":
        raise ContractError("result schema drift")
    if value["experiment_id"] != EXPERIMENT_ID or value["source_commit"] != SOURCE_COMMIT:
        raise ContractError("result identity drift")
    if value["contract_sha256"] != CONTRACT_SHA256:
        raise ContractError("result contract binding drift")
    if value["output_schema_sha256"] != OUTPUT_SCHEMA_SHA256:
        raise ContractError("result output-schema binding drift")
    _validate_hex_commit(value["source_freeze_commit"], "source_freeze_commit")
    _validate_hex_commit(value["contract_freeze_commit"], "contract_freeze_commit")
    if value["source_freeze_commit"] != value["contract_freeze_commit"]:
        raise ContractError("source-freeze and contract-freeze commit drift")
    if value["result_commit"] is not None:
        raise ContractError("tracked result_commit must be null to avoid circular binding")
    if value["result_commit_binding_policy"] != RESULT_COMMIT_BINDING_POLICY:
        raise ContractError("result commit binding policy drift")
    ancestry = value["ancestry_validation"]
    if not isinstance(ancestry, Mapping) or set(ancestry) != {
        "requirements_ancestor_to_source",
        "source_to_contract_freeze",
        "result_commit_pending",
    }:
        raise ContractError("result ancestry validation failed")
    if (
        ancestry["requirements_ancestor_to_source"] is not True
        or ancestry["source_to_contract_freeze"] is not True
        or ancestry["result_commit_pending"] is not True
    ):
        raise ContractError("result ancestry validation failed")
    primary = value["primary_classification"]
    if primary not in PRIMARY_CLASSIFICATIONS:
        raise ContractError("unknown primary classification")
    secondaries = value["secondary_classifications"]
    if not isinstance(secondaries, list) or len(secondaries) != len(set(secondaries)):
        raise ContractError("secondary classifications must be unique")
    if any(item not in SECONDARY_CLASSIFICATIONS for item in secondaries):
        raise ContractError("unknown secondary classification")
    if value["next_experiment"] != next_experiment_for_primary(primary):
        raise ContractError("next experiment disagrees with primary classification")
    if value["next_experiment_specification"] != (
        next_experiment_specification_for_primary(primary)
    ):
        raise ContractError(
            "next-experiment specification disagrees with primary classification"
        )
    if value["predecessor_fact_authority"] != list(PRESERVED_PREDECESSOR_FACTS):
        raise ContractError("predecessor fact authority drift")
    if value["predecessor_narrative_authority"] != list(
        PRESERVED_PREDECESSOR_NARRATIVE
    ):
        raise ContractError("predecessor narrative authority drift")
    validate_prior_smoke_failure_custody(value["prior_smoke_failure_custody"])
    if value["nothing_running"] is not True:
        raise ContractError("result does not confirm nothing running")
    return copy.deepcopy(dict(value))


def _write_immutable(path: Path, payload: bytes, label: str) -> Path:
    if path.exists():
        if path.read_bytes() != payload:
            raise ContractError(f"refusing to overwrite a different {label}: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def write_preregistration(path: str | Path = TRACKED_PREREGISTRATION_PATH) -> Path:
    validate_no_duplicate_literal_dict_keys(Path(__file__))
    return _write_immutable(Path(path), build_preregistration_markdown().encode("utf-8"), "preregistration")


def write_contract(path: str | Path = TRACKED_CONTRACT_PATH) -> Path:
    return _write_immutable(Path(path), contract_receipt_bytes(), "contract")


def write_output_schema(path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH) -> Path:
    return _write_immutable(Path(path), output_schema_receipt_bytes(), "output schema")


def write_evaluator_fixture(path: str | Path = TRACKED_FIXTURE_PATH) -> Path:
    return _write_immutable(Path(path), evaluator_fixture_receipt_bytes(), "evaluator fixture")


def write_source_closure(
    value: Mapping[str, Any], path: str | Path = TRACKED_SOURCE_CLOSURE_PATH
) -> Path:
    validate_source_closure(value)
    return _write_immutable(Path(path), source_closure_receipt_bytes(value), "source closure")


def write_execution_correction_amendment(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH,
) -> Path:
    return _write_immutable(
        Path(path),
        execution_correction_amendment_receipt_bytes(),
        "execution-correction amendment",
    )


def write_execution_correction_output_schema(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH,
) -> Path:
    return _write_immutable(
        Path(path),
        execution_correction_output_schema_receipt_bytes(),
        "execution-correction output schema",
    )


def write_execution_correction_fixture(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH,
) -> Path:
    return _write_immutable(
        Path(path),
        execution_correction_fixture_receipt_bytes(),
        "execution-correction fixture",
    )


def write_execution_correction_preregistration(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_PREREGISTRATION_PATH,
) -> Path:
    return _write_immutable(
        Path(path),
        build_execution_correction_preregistration_markdown().encode("utf-8"),
        "execution-correction preregistration",
    )


def write_execution_correction_source_closure(
    value: Mapping[str, Any],
    path: str | Path = TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH,
) -> Path:
    return _write_immutable(
        Path(path),
        execution_correction_source_closure_receipt_bytes(value),
        "execution-correction source closure",
    )


def write_execution_correction_authorities(repo_root: str | Path) -> dict[str, Path]:
    """Write only the separate amendment suite, never base scientific files."""

    root = Path(repo_root)
    paths = {
        "preregistration": write_execution_correction_preregistration(
            root / TRACKED_EXECUTION_CORRECTION_PREREGISTRATION_PATH
        ),
        "amendment": write_execution_correction_amendment(
            root / TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH
        ),
        "output_schema": write_execution_correction_output_schema(
            root / TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH
        ),
        "fixture": write_execution_correction_fixture(
            root / TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH
        ),
    }
    closure = build_execution_correction_source_closure(root, require_complete=True)
    paths["source_closure"] = write_execution_correction_source_closure(
        closure, root / TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH
    )
    validate_base_scientific_authorities(root)
    return paths


def _load_exact(path: Path, expected: bytes, label: str) -> dict[str, Any]:
    observed = path.read_bytes()
    if observed != expected:
        raise ContractError(f"{label} bytes drift: {path}")
    value = json.loads(observed)
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object")
    return value


def load_and_validate_contract(path: str | Path = TRACKED_CONTRACT_PATH) -> dict[str, Any]:
    return validate_contract(_load_exact(Path(path), contract_receipt_bytes(), "contract"))


def load_and_validate_output_schema(
    path: str | Path = TRACKED_OUTPUT_SCHEMA_PATH,
) -> dict[str, Any]:
    return validate_output_schema(
        _load_exact(Path(path), output_schema_receipt_bytes(), "output schema")
    )


def load_and_validate_evaluator_fixture(
    path: str | Path = TRACKED_FIXTURE_PATH,
) -> dict[str, Any]:
    return validate_evaluator_fixture(
        _load_exact(Path(path), evaluator_fixture_receipt_bytes(), "evaluator fixture")
    )


def load_and_validate_source_closure(
    path: str | Path = TRACKED_SOURCE_CLOSURE_PATH,
) -> dict[str, Any]:
    observed = Path(path).read_bytes()
    value = json.loads(observed)
    if not isinstance(value, dict):
        raise ContractError("source closure must be a JSON object")
    if observed != source_closure_receipt_bytes(value):
        raise ContractError("source closure bytes are not canonical")
    validated = validate_source_closure(value)
    if validated["declared_paths"] != list(SOURCE_CLOSURE_DEFAULT_PATHS):
        raise ContractError("tracked source closure path-set drift")
    return validated


def load_and_validate_execution_correction_amendment(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH,
) -> dict[str, Any]:
    value = _load_exact(
        Path(path),
        execution_correction_amendment_receipt_bytes(),
        "execution-correction amendment",
    )
    return validate_execution_correction_amendment(value)


def load_and_validate_execution_correction_output_schema(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH,
) -> dict[str, Any]:
    value = _load_exact(
        Path(path),
        execution_correction_output_schema_receipt_bytes(),
        "execution-correction output schema",
    )
    validate_self_digest(value, "output_schema_sha256")
    if canonical_json_bytes(value) != canonical_json_bytes(
        EXECUTION_CORRECTION_OUTPUT_SCHEMA
    ):
        raise ContractError("execution-correction output schema value drift")
    return value


def load_and_validate_execution_correction_fixture(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH,
) -> dict[str, Any]:
    value = _load_exact(
        Path(path),
        execution_correction_fixture_receipt_bytes(),
        "execution-correction fixture",
    )
    validate_self_digest(value, "fixture_sha256")
    if canonical_json_bytes(value) != canonical_json_bytes(
        EXECUTION_CORRECTION_FIXTURE
    ) or not all(value["checks"].values()):
        raise ContractError("execution-correction fixture value drift")
    return value


def load_and_validate_execution_correction_source_closure(
    path: str | Path = TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH,
) -> dict[str, Any]:
    observed = Path(path).read_bytes()
    try:
        value = json.loads(observed)
    except json.JSONDecodeError as exc:
        raise ContractError("execution-correction source closure is invalid") from exc
    if not isinstance(value, dict):
        raise ContractError("execution-correction source closure must be an object")
    if observed != execution_correction_source_closure_receipt_bytes(value):
        raise ContractError("execution-correction source closure bytes are not canonical")
    return validate_execution_correction_source_closure(value)


__all__ = [
    "ACTIVE_POLICIES",
    "ACCIDENTAL_EXPOSURES",
    "BASE_FEATURE_DIM",
    "BASE_FEATURE_LAYOUT",
    "CANDIDATE_INDICES",
    "CHECKPOINT_BINDINGS",
    "CHECKPOINT_SEED_METADATA",
    "CONDITION_IDS",
    "CONDITION_KEYED_SEEDS",
    "CONTRACT",
    "CONTRACT_SHA256",
    "CONTRACT_SCHEMA_VERSION",
    "CONTRACT_FREEZE_ANCESTRY_POLICY",
    "CONTRACT_FREEZE_COMMIT_SUBJECT",
    "CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS",
    "ContractError",
    "CPU_WORKER_BENCHMARK",
    "CPU_WORKERS",
    "DATE",
    "DERANGEMENT_MATERIALITY",
    "ENCODER_BINDING",
    "EVALUATOR_FIXTURE",
    "EVALUATOR_FIXTURE_SHA256",
    "BASE_SCIENTIFIC_AUTHORITY_BINDINGS",
    "EXECUTION_CORRECTION_ALLOWED_CHANGED_PATHS",
    "EXECUTION_CORRECTION_AMENDMENT",
    "EXECUTION_CORRECTION_AMENDMENT_BINDING",
    "EXECUTION_CORRECTION_AMENDMENT_SCHEMA_VERSION",
    "EXECUTION_CORRECTION_ARCHIVE_INVENTORY",
    "EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS",
    "EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS",
    "EXECUTION_CORRECTION_ENVIRONMENT_PROBE",
    "EXECUTION_CORRECTION_FAILED_ARCHIVE",
    "EXECUTION_CORRECTION_FIXTURE",
    "EXECUTION_CORRECTION_FIXTURE_BINDING",
    "EXECUTION_CORRECTION_FREEZE_COMMIT_SUBJECT",
    "EXECUTION_CORRECTION_NORMALIZED_REPLAY_DIGESTS",
    "EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS",
    "EXECUTION_CORRECTION_OUTPUT_SCHEMA",
    "EXECUTION_CORRECTION_OUTPUT_SCHEMA_BINDING",
    "EXECUTION_CORRECTION_POLICY",
    "EXECUTION_CORRECTION_REPLAY_SCHEMA_VERSION",
    "EXECUTION_CORRECTION_REQUIRED_CHANGED_PATHS",
    "EXECUTION_CORRECTION_SOURCE_CLOSURE_DEFAULT_PATHS",
    "EXECUTION_CORRECTION_SOURCE_CLOSURE_SCHEMA_VERSION",
    "EXPERIMENT_ID",
    "EXECUTION_RETRY_POLICY",
    "FAMILY_IDS",
    "FIT_OPTIMIZATION_RECEIPT_FIELDS",
    "FIT_STATE_OPTIMIZATION_STATUSES",
    "GATES",
    "LATENT_PARAMETER_COUNT",
    "LATENT_SOURCE_IDS",
    "LEGACY_COUNTERFACTUAL_TENSOR_BINDINGS",
    "KEYED_SEED_NAMESPACE",
    "METRIC_CONTRACT",
    "NEXT_EXPERIMENT_IDS",
    "NEXT_EXPERIMENT_SPECIFICATIONS",
    "NO_LATENT_PARAMETER_COUNT",
    "OUTPUT_ROOT",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_SHA256",
    "PANEL_BINDINGS",
    "PAIRWISE_UTILITY_TOLERANCE",
    "PREDECESSOR_CANDIDATE_EVIDENCE_BINDING",
    "PREDECESSOR_TENSOR_PACKAGE",
    "PREDICTED_SOURCE_IDS",
    "PRIOR_SMOKE_FAILURE_CUSTODY_RECORD_FIELDS",
    "RAW_COST_REREDUCED_SOURCE_IDS",
    "PREDICTED_SOURCE_ABSOLUTE_PRESERVATION_GATE",
    "PRIMARY_CLASSIFICATIONS",
    "PRESERVED_PREDECESSOR_FACTS",
    "PRESERVED_PREDECESSOR_FACT_SCOPE",
    "PRESERVED_PREDECESSOR_NARRATIVE",
    "PRESERVED_REQUIREMENTS_CLASSIFICATIONS",
    "PROPRIO_CONTRIBUTION_GATE",
    "PROPRIO_INPUT_BINDINGS",
    "QUERY_FEATURE_DIM",
    "QUERY_FEATURE_LAYOUT",
    "RANDOM_SEED",
    "RANKER_SEED",
    "REQUIRED_REQUIREMENTS_ANCESTOR",
    "RESULT_COMMIT_BINDING_POLICY",
    "ROLE_CLASS_MAPPING",
    "ROUTE_ROLE_AUTHORITY_BINDING",
    "ROUTE_ROLE_AUTHORITY_SCHEMA_VERSION",
    "ROUTE_ROLE_RECEIPT_BINDING",
    "ROLE_IDS",
    "ROLE_ROW_COUNTS",
    "ROLE_STATE_COUNTS",
    "ROUTE_COST_SEED",
    "RR_GATE",
    "SECONDARY_CLASSIFICATIONS",
    "SOURCE_CLOSURE_DEFAULT_PATHS",
    "SOURCE_COMMIT",
    "INITIAL_EXECUTION_FREEZE_COMMIT",
    "SCIENTIFIC_AUTHORITY_CONTRACT_SHA256",
    "SHARED_BASE_SEED_SUBKEY",
    "STAGE_POLICY",
    "STAGE_C_ABLATION_INPUTS",
    "TRACKED_CONTRACT_PATH",
    "TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH",
    "TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH",
    "TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH",
    "TRACKED_EXECUTION_CORRECTION_PREREGISTRATION_PATH",
    "TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH",
    "TRACKED_FIXTURE_PATH",
    "TRACKED_OUTPUT_SCHEMA_PATH",
    "TRACKED_PREREGISTRATION_PATH",
    "TRACKED_REPORT_PATH",
    "TRACKED_RESULT_PATH",
    "TRACKED_ROUTE_ROLE_AUTHORITY_PATH",
    "TRACKED_SOURCE_CLOSURE_PATH",
    "TRAINING",
    "TRAINING_POLICY",
    "TRUE_FUTURE_GATE",
    "TRUE_INCREMENTAL_GATE",
    "TRUE_INCREMENTAL_NOT_EVALUATED_PAYLOAD",
    "TOKEN_GRID_SHAPE",
    "TOKENS_PER_TIMEPOINT",
    "NUMERICAL_THREAD_ENV",
    "attach_self_digest",
    "all_predicted_substitutions_fail_materially",
    "build_contract",
    "build_execution_correction_amendment",
    "build_execution_correction_fixture",
    "build_execution_correction_output_schema",
    "build_execution_correction_preregistration_markdown",
    "build_execution_correction_source_closure",
    "build_evaluator_fixture",
    "build_output_schema",
    "build_preregistration_markdown",
    "build_route_role_authority",
    "build_role_map",
    "build_source_closure",
    "build_stage_c_donor_mapping",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "classification_vocabularies_are_unique",
    "contract_receipt_bytes",
    "derangement_is_material",
    "derive_primary_classification",
    "derive_keyed_seed",
    "duplicate_literal_dict_keys",
    "evaluator_fixture_receipt_bytes",
    "execution_correction_amendment_receipt_bytes",
    "execution_correction_fixture_receipt_bytes",
    "execution_correction_output_schema_receipt_bytes",
    "execution_correction_source_closure_receipt_bytes",
    "extract_route_only_rows",
    "fit_state_optimization_status",
    "load_and_validate_contract",
    "load_and_validate_execution_correction_amendment",
    "load_and_validate_execution_correction_fixture",
    "load_and_validate_execution_correction_output_schema",
    "load_and_validate_execution_correction_source_closure",
    "load_and_validate_evaluator_fixture",
    "load_and_validate_output_schema",
    "load_and_validate_route_role_authority",
    "load_and_validate_source_closure",
    "load_route_role_map",
    "margin_borda_pairwise_target",
    "next_experiment_for_primary",
    "next_experiment_specification_for_primary",
    "output_schema_receipt_bytes",
    "proprio_contribution_passes",
    "predicted_source_absolute_preservation_passes",
    "route_role_from_state_class",
    "route_role_authority_receipt_bytes",
    "rr_gate_passes",
    "source_closure_receipt_bytes",
    "scientific_content_digest",
    "scientific_content_projection",
    "true_future_gate_passes",
    "true_incremental_gate_passes",
    "true_incremental_not_evaluated_payload",
    "validate_contract",
    "validate_execution_freeze_custody",
    "validate_execution_correction_amendment",
    "validate_execution_correction_archive",
    "validate_execution_correction_freeze_custody",
    "validate_execution_correction_replay",
    "validate_execution_correction_source_closure",
    "validate_base_scientific_authorities",
    "validate_no_duplicate_literal_dict_keys",
    "validate_evaluator_fixture",
    "validate_output_schema",
    "validate_prior_smoke_failure_custody",
    "validate_repository_custody",
    "validate_result_receipt",
    "validate_route_role_authority",
    "validate_self_digest",
    "validate_source_closure",
    "write_contract",
    "write_execution_correction_amendment",
    "write_execution_correction_authorities",
    "write_execution_correction_fixture",
    "write_execution_correction_output_schema",
    "write_execution_correction_preregistration",
    "write_execution_correction_source_closure",
    "write_evaluator_fixture",
    "write_output_schema",
    "write_preregistration",
    "write_route_role_authority",
    "write_source_closure",
]
