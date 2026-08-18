"""Contract for the one-shot four-step predictor evaluation successor.

This contract deliberately binds the predictor/occupancy evaluator closure and
does not treat the unrelated utility-scorer contract as evaluator code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

BASE_SOURCE_COMMIT = "dc94fdfd0e8d29f65643a34981f901cc7dcd5bcb"
PREDECESSOR_RUNTIME = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
    "four_step_rollout_v1_input_custody_successor_2"
)
RUNTIME_ROOT = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
    "four_step_rollout_v1_evaluation_successor_v2"
)
PREVIOUS_SUCCESSOR_RUNTIME = Path(
    "/home/andrewknowles/.cache/lewm_go2_temporal_v03/"
    "four_step_rollout_v1_evaluation_successor_v1"
)
PREVIOUS_SUCCESSOR_TERMINAL_DIGEST = "ffceac739ebbfaaff62f98be2afbff6978741e3bfcdf54b1078d71abbb65eb88"
PREVIOUS_SUCCESSOR_TERMINAL_RAW_SHA256 = "3b01a23388a7447d06c06cb6ba91376ab18991211fbed214a88a43a00bec6808"
PREDECESSOR_CONTRACT_DIGEST = (
    "823a722dffc2a13843bd2c5936bd46d5bf7de4399d1323d691e7f778d12d5100"
)
PREDECESSOR_TERMINAL_DIGEST = (
    "7d754e0d84564184607bf7e0738eff285db8e47f72fa15bc662732538ef42786"
)
PREDECESSOR_TERMINAL_RAW_SHA256 = (
    "8b8c81b1597008a6f74fc166196f266889200a61cd63fbbd926d67d2ca330061"
)
SCORER_CONTRACT_PATH = "lewm/oracle/go2_scorer_contract_v1_2.py"
STAGE_A_BASE_COMMIT = "ee47b47e7964c16360f265c4cfbe7f8181d16402"
STAGE_A_BASE_SHA256 = "cd84ae922bd25e3576a1d3466a9d126e92992230180cd75faccaab047a144b16"
CURRENT_SCORER_SHA256 = "a916c20fdc92762a7aeefc773823bf4870e671528b2aea8bed0df69cfa1c0664"
SCORER_DIFF_SHA256 = "3dd599d3b27fb1c7d0456ae16ce796981e7e873516cca4e5bf3004b4ff125a8d3"
SCORER_DIFF_ADDED_LINES = 712
SCORER_DIFF_REMOVED_LINES = 25
ENCODER_PATH = "scripts/dev_frozen_dense_representation_encoders_v1.py"
ENCODER_BASE_SHA256 = "9fa9780376416cc0181d2e37980d5af8a1bb632dec82ec802e53e33991f74efb"
ENCODER_CURRENT_SHA256 = "c5bb12ddc4711071dbdbac8c2ad6cc4b7528dd8ceb263b752fd539bd954aa9e2"
ENCODER_DIFF_SHA256 = "48aff8e48959464d3c349d2865c5431a49e5be9dad33b6097e94b4c40698d1e6"
CLASSIFICATION = "OVERBROAD_SOURCE_CLOSURE_REFUSAL_PRE_EVALUATION"
CHECKPOINTS = {
    "2026080901": "de815f01df7dde9a776bfe80f388a3da674e8b5ea29b1c9ef8ef44be670e44f8",
    "2026080902": "e4eb0159f8ab91b9bc1f6e21dda06bd9b0b2f82c1357abe3470c3e60fd7a3f0e",
    "2026080903": "f7db658c56d45374a52874ccaa11f42663db182aff5c2d34b4b08da5f0e47cf9",
    "2026080904": "50c737cc3701ead84823805c49a7f69a1400c766cbe626d934e06cdebac84359",
    "2026080905": "77179f89249002241ec186cfd5520e837a06737978cc339a79bb92f3c6652a67",
    "2026080906": "640e158bf8e3e778d77300d8f1a324200ef7763d44416693db9a3d36a3c9d29b",
    "2026080907": "d064a95a7a8097bc463810821f70caf8efa44e7f5155c17ee632f48a4c5be671",
    "2026080908": "89c9fed1befb16fb4064ed1994740511ae8c0eb007197584ada2b00de7de93de",
}

REPORT = {
    "schema": "go2_four_step_predictor_evaluation_dependency_report_v1",
    "evaluation_entry_point": (
        "scripts/run_go2_rgb_control_history_four_step_autoregressive_v1.py::evaluate_stage"
    ),
    "direct_evaluator_modules": [
        "scripts.analyze_go2_counterfactual_predictor_qualification_v1_2",
        "scripts.run_go2_counterfactual_occupancy_assay_v1_2",
        "scripts.dev_proprio_predictor_v1",
        "scripts.dev_checkpoint_v1",
        "scripts.run_dev_v03_temporal_action_jepa_v1",
        "scripts.run_dev_proprio_factorial_driver_v1",
    ],
    "executed_metric_functions": [
        "Q.validate_stage_a_metadata",
        "Q.validate_stage_a_latent_shards",
        "Q.predict_state",
        "Q.score_state_predictions",
        "Q.aggregate_records",
        "O.load_stage_a",
        "O.load_labels",
        "O.load_probe",
        "O.aggregate_prediction",
    ],
    "scientific_inputs_affected_by_changed_file": {
        "branch_identity": False,
        "target_latent": False,
        "changed_token_mask": False,
        "predictor_input": False,
        "fidelity_metric": False,
        "retrieval_metric": False,
        "occupancy_metric": False,
        "aggregation": False,
    },
    "changed_file": {
        "path": SCORER_CONTRACT_PATH,
        "stage_a_base_commit": STAGE_A_BASE_COMMIT,
        "stage_a_base_sha256": STAGE_A_BASE_SHA256,
        "current_sha256": CURRENT_SCORER_SHA256,
        "unified_diff_sha256": SCORER_DIFF_SHA256,
        "added_lines": SCORER_DIFF_ADDED_LINES,
        "removed_lines": SCORER_DIFF_REMOVED_LINES,
        "changed_symbols": [
            "CORPUS_SELECTION_CONTRACT",
            "source_bindings",
            "issue_contract",
            "interruption/performance lineage validation",
            "state-selector and candidate-allocation successor bindings",
        ],
        "role": "scorer/utility and corpus-selection lineage only",
    },
    "additional_changed_files": [{
        "path": ENCODER_PATH,
        "stage_a_base_commit": STAGE_A_BASE_COMMIT,
        "stage_a_base_sha256": ENCODER_BASE_SHA256,
        "current_sha256": ENCODER_CURRENT_SHA256,
        "unified_diff_sha256": ENCODER_DIFF_SHA256,
        "added_lines": 87,
        "removed_lines": 2,
        "changed_symbols": [
            "drop_path_compat_v1", "DropPathCompatV1",
            "scoped_timm_drop_path_shim_v1", "VJepa21Arm.load_backbone",
        ],
        "role": "target encoding compatibility only; cached targets are frozen",
    }],
    "closure_finding": {
        "imported_by_predictor_evaluator": False,
        "executed_by_predictor_evaluator": False,
        "all_changed_stage_a_bindings_are_evaluator_unused": True,
        "transitively_imported_but_unused": False,
        "used_only_by_scorer_or_utility_code": True,
        "stage_a_manifest_binding_only": True,
        "classification": CLASSIFICATION,
    },
    "custody": {
        "model_forward_during_audit": False,
        "checkpoint_load_during_audit": False,
        "evaluation_corpus_opened_during_audit": False,
        "predictor_utility_shards_opened_during_audit": False,
    },
}


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def contract_payload(source_commit: str, source_digest: str) -> dict[str, Any]:
    return {
        "schema": "go2_four_step_predictor_evaluation_successor_contract_v1",
        "status": "EVALUATION_ONLY_SUCCESSOR",
        "classification": CLASSIFICATION,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "source_commit": source_commit,
        "source_closure_digest": source_digest,
        "predecessor_contract_digest": PREDECESSOR_CONTRACT_DIGEST,
        "predecessor_terminal_digest": PREDECESSOR_TERMINAL_DIGEST,
        "predecessor_terminal_raw_sha256": PREDECESSOR_TERMINAL_RAW_SHA256,
        "previous_successor_terminal_digest": PREVIOUS_SUCCESSOR_TERMINAL_DIGEST,
        "previous_successor_terminal_raw_sha256": PREVIOUS_SUCCESSOR_TERMINAL_RAW_SHA256,
        "checkpoint_digests": CHECKPOINTS,
        "dependency_report": REPORT,
        "model_forwards_authorized": 8 * 20,
        "historical_comparator_forwards": 0,
        "training_authorized": False,
        "selection_rows": 240,
        "utility_scorer_authorized": False,
        "final_corpus_authorized": False,
        "replacement_number": 2,
        "maximum_evaluation_successors": 2,
    }


def seal(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["contract_digest"] = digest(body)
    return body
