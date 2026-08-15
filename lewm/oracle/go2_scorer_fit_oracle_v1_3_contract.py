"""Frozen scorer-fit oracle-v1.3 contract and selector-integrity amendment.

The contract is intentionally a narrow successor to the protected scorer-fit
V2 corpus.  It freezes the graph-boundary label rule, the exact eighteen-row
replay allowlist, the disposition of the historical calibration states, the
outcome-blind fresh-calibration selector, the unchanged scorer/qualification
budget, and closed output/stage authority.  A separate selector-only integrity
replacement binds the pre-manifest selector crash without changing that
scientific contract.  Importing this module reads or writes nothing and opens
no model checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lewm.oracle import go2_branch_oracle_v1_3 as ORACLE


CONTRACT_SCHEMA = "go2_scorer_fit_oracle_v1_3_contract_v1"
ORIGINAL_PREREGISTRATION_SCHEMA = "go2_scorer_fit_oracle_v1_3_preregistration_v1"
PREREGISTRATION_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_selector_integrity_replacement_v1_"
    "preregistration_v1"
)
CONTRACT_STATUS = "FROZEN_PREOUTCOME_DEVELOPMENT_CONTRACT"
ORIGINAL_PREREGISTRATION_STATUS = "FROZEN_BEFORE_V1_3_BRANCH_EXECUTION"
PREREGISTRATION_STATUS = (
    "FROZEN_AFTER_PREMANIFEST_SELECTOR_SIGSEGV_BEFORE_INTEGRITY_REPLACEMENT"
)

ORIGINAL_PREREGISTRATION_PATH = Path(
    "docs/lewm_go2_scorer_fit_oracle_v1_3_preregistration_2026-08-15.json"
)
PREREGISTRATION_PATH = Path(
    "docs/go2_scorer_fit_oracle_v1_3_selector_integrity_replacement_v1_"
    "preregistration_2026-08-15.json"
)
ORIGINAL_PREREGISTRATION_BINDING = {
    "path": str(ORIGINAL_PREREGISTRATION_PATH),
    "schema": ORIGINAL_PREREGISTRATION_SCHEMA,
    "status": ORIGINAL_PREREGISTRATION_STATUS,
    "self_digest":
        "6ec652931e2ab8085c391d88d4c199de3fd1db9061b824891a748821eb0119a7",
    "raw_sha256":
        "5ede05ca2cd25caaa055adafcc57ead2a868977e5a829fa2941b11e465d778a3",
    "byte_count": 40749,
}
GENERATED_ROOT = Path(".generated/go2_scorer_fit_oracle_v1_3")
REGISTERED_GENERATED_TARGET_ROOT = Path(
    "/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2/"
    "active/go2_scorer_fit_oracle_v1_3"
)
SUPERSEDED_PREATTEMPT_ARCHIVE_ROOT = (
    REGISTERED_GENERATED_TARGET_ROOT.parent
    / "go2_scorer_fit_oracle_v1_3_superseded_snapshot_gate_c4f28d67"
)
GENESIS_RUNTIME_CONTRACT = {
    "interpreter_relative_path":
        ".generated/venvs/genesis_render_vulkan/bin/python",
    "pyvenv_config_relative_path":
        ".generated/venvs/genesis_render_vulkan/pyvenv.cfg",
    "pyvenv_config_sha256":
        "41c6a8f52f3404bd3b7fcd805519c1976e2f4194ef9aa8eccf2f0919383386a9",
    "pyvenv_config_byte_count": 219,
    "python_version": "3.12.3",
    "genesis_version": "0.3.14",
    "numpy_version": "2.4.6",
    "torch_version": "2.12.0+cu130",
    "backend": "cpu",
    "gstaichi_version": "4.6.0",
}
SUPERSEDED_PREATTEMPT_EXECUTION_AUTHORITY = {
    "source_commit": "c4f28d67a97646eaddfcf268432cce793bbdf126",
    "authority": {
        "self_digest":
            "e8bbecc4820897e4a9854a323ec3b2e38e0bae0e726845a32544d33ce64e0703",
        "raw_sha256":
            "cd906f464975c755ad5b6257ad60e3df70dcc22a14d8b4d7e296f0bb07b7cc25",
        "byte_count": 51976,
    },
    "replay_plan": {
        "self_digest":
            "14228b2c2a2a39dcf4f51f79a29baa820f17641b94220a73ea62ff22ed5ea37e",
        "raw_sha256":
            "434660c6aa862675f74517123693b75420ab539ceb62e7c5b9c3bc3ecef153cd",
        "byte_count": 122301,
    },
    "equivalence_receipt": {
        "self_digest":
            "1bb88a95ef4694136a106ff1ef9b3240232a94853682888d030b21f284443074",
        "raw_sha256":
            "b0bfad652c9244220179b68d397eb2b5ac99c6d7b9d545c62660bf3b450d4ed2",
        "byte_count": 644424,
    },
    "replay_attempt_markers": 0,
    "replay_overlays": 0,
    "candidate_branch_execution_started": False,
    "archive_root": str(SUPERSEDED_PREATTEMPT_ARCHIVE_ROOT),
    "disposition": "SUPERSEDED_PREATTEMPT_IMPLEMENTATION_CORRECTION",
    "reason": (
        "the in-process snapshot digest includes unpersisted process-global RNG "
        "history and is not a cross-run physical-state fingerprint"
    ),
}
AUTHORITY_PATH = GENERATED_ROOT / "authority.json"
SUPERSEDED_PREATTEMPT_TRANSITION_PATH = (
    GENERATED_ROOT / "superseded_preattempt_transition.json"
)
REPLAY_PLAN_PATH = GENERATED_ROOT / "replay_plan.json"
EQUIVALENCE_RECEIPT_PATH = GENERATED_ROOT / "equivalence_receipt.json"
REPLAY_ATTEMPTS_ROOT = GENERATED_ROOT / "replay_attempts"
REPLAY_OVERLAYS_ROOT = GENERATED_ROOT / "replay_overlays"
REPLAY_OVERLAY_MANIFEST_PATH = GENERATED_ROOT / "replay_overlay_manifest.json"
FRESH_CALIBRATION_ROOT = GENERATED_ROOT / "fresh_calibration"
FRESH_CALIBRATION_STATE_MANIFEST_PATH = FRESH_CALIBRATION_ROOT / "state_manifest.json"
FRESH_CALIBRATION_ROWS_ROOT = FRESH_CALIBRATION_ROOT / "rows"
FRESH_CALIBRATION_FRAMES_ROOT = FRESH_CALIBRATION_ROOT / "frames"
FRESH_CALIBRATION_CORPUS_RECEIPT_PATH = (
    FRESH_CALIBRATION_ROOT / "corpus_receipt.json"
)
SELECTOR_INTEGRITY_REPLACEMENT_AUTHORITY_PATH = (
    FRESH_CALIBRATION_ROOT / "selector_integrity_replacement_v1.json"
)
FRESH_SELECTION_ATTEMPT_PATH = FRESH_CALIBRATION_ROOT / "selection_attempt.json"
FRESH_SELECTION_TASKS_ROOT = FRESH_CALIBRATION_ROOT / "selection_tasks"
FRESH_SELECTION_RESULTS_ROOT = FRESH_CALIBRATION_ROOT / "selection_results"
FRESH_SELECTION_TERMINAL_PATH = FRESH_CALIBRATION_ROOT / "selection_terminal.json"
TRAINING_VIEW_PATH = GENERATED_ROOT / "training_view.json"

ENCODED_TRAINING_VIEW_ROOT = GENERATED_ROOT / "encoded_training_view"
LATENT_INDEX_PATH = ENCODED_TRAINING_VIEW_ROOT / "latent_index.json"
ENCODING_RECEIPT_PATH = ENCODED_TRAINING_VIEW_ROOT / "encoding_receipt.json"
HORIZON_LATENTS_ROOT = ENCODED_TRAINING_VIEW_ROOT / "latents/horizon"
SCORER_ROOT = GENERATED_ROOT / "scorer"
QUALIFICATION_PATH = SCORER_ROOT / "qualification.json"
TRAINING_EXECUTION_AUTHORISATION_PATH = (
    SCORER_ROOT / "training_execution_authorisation.json"
)
QUALIFICATION_EVALUATION_AUTHORISATION_PATH = (
    SCORER_ROOT / "qualification_evaluation_authorisation.json"
)
SCORER_PACKAGE_PATH = SCORER_ROOT / "scorer_package.pt"
SCORER_PACKAGE_RECEIPT_PATH = SCORER_ROOT / "scorer_package_receipt.json"
NO_LATENT_BASELINE_PATH = SCORER_ROOT / "no_latent_baseline.pt"
NO_LATENT_BASELINE_RECEIPT_PATH = SCORER_ROOT / "no_latent_baseline_receipt.json"
FAILED_SCORER_PATH = SCORER_ROOT / "failed_scorer.pt"
REGISTERED_INITIALISATIONS_ROOT = SCORER_ROOT / "initialisations"
TRAINING_CHECKPOINTS_ROOT = SCORER_ROOT / "training"

TRACE_SCHEMA = ORACLE.TRACE_SCHEMA
REPLAY_OVERLAY_SCHEMA = "go2_scorer_fit_oracle_v1_3_replay_overlay_v1"
LATENT_INDEX_SCHEMA = "go2_scorer_fit_oracle_v1_3_latent_index_v1"
ENCODING_RECEIPT_SCHEMA = "go2_scorer_fit_oracle_v1_3_encoding_receipt_v1"
QUALIFICATION_SCHEMA = "go2_utility_scorer_oracle_v1_3_qualification_v1"
TRAINING_EXECUTION_AUTHORISATION_SCHEMA = (
    "go2_utility_scorer_oracle_v1_3_training_execution_authorisation_v1"
)
QUALIFICATION_EVALUATION_AUTHORISATION_SCHEMA = (
    "go2_utility_scorer_oracle_v1_3_qualification_evaluation_authorisation_v1"
)
SCORER_PACKAGE_SCHEMA = "go2_utility_scorer_oracle_v1_3_package_v1"
SCORER_PACKAGE_RECEIPT_SCHEMA = (
    "go2_utility_scorer_oracle_v1_3_package_receipt_v1"
)
NO_LATENT_BASELINE_SCHEMA = "go2_utility_scorer_oracle_v1_3_no_latent_baseline_v1"
NO_LATENT_BASELINE_RECEIPT_SCHEMA = (
    "go2_utility_scorer_oracle_v1_3_no_latent_baseline_receipt_v1"
)
SELECTOR_INTEGRITY_REPLACEMENT_SCHEMA = (
    "go2_scorer_fit_oracle_v1_3_selector_integrity_replacement_authority_v1"
)
SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY = (
    "selector_integrity_replacement_authority_digest"
)

SOURCE_BASE_COMMIT = "1c07df778b7225301b9be80ebd2f6162bfe16965"
AUDITED_CORPUS_SOURCE_COMMIT = "5c67135ad83b9206e6520e507f1ecaf980fd3d8d"
FROZEN_CORPUS_DIGEST = (
    "5216e2182a4e165a673714fcccbd6b769d01fa565a69a466b3cab066ab01ccc3"
)
FROZEN_BRANCH_ROWS_SHA256 = (
    "e7cabd8734e1e5b1776a5ad0de3eb093f6222169103e0b1c39e8ef9b2be60036"
)
FROZEN_STATE_MANIFEST_DIGEST = (
    "db79efce49d949522832d920b23a38292a491dc9e6fb2cbf2b8e0a5176fb062e"
)
FROZEN_ASSIGNMENT_MANIFEST_DIGEST = (
    "a91d6d211f5b07270df5a66262ce4ba218e8a3925ae5f8aba196b8c10f4959f4"
)
FROZEN_BRANCH_IDENTITY_SET_DIGEST = (
    "d9330a4d9102011c616abcb6d38bb8644e8bbf9f497aa0cb176bf184ad7acdf3"
)
DIAGNOSTIC_PATH = Path(
    "docs/lewm_go2_scorer_fit_v2_graph_label_failure_diagnostic_2026-08-15.json"
)
DIAGNOSTIC_FILE_SHA256 = (
    "26ba643ff311acebbeba6fb59941bb56cc80fd29b5c9d52ee2c2acacae1c5d43"
)
DIAGNOSTIC_AUDIT_DIGEST = (
    "90dda36b7e85a650a75d1efb5d21faf3f3ed40f0860f3bdb3f6a4e69b8bd3741"
)
V1_2_PROGRESS_DIGEST = (
    "840328d918f446bad1a5855e72f13f8937fc9a42eafd87818bf8cd94305e2c3d"
)
V1_2_SAFETY_DIGEST = (
    "5cf4572be2490c1b6f748abc704fff3a3c15fb1ea8dc060e49314e2bbaf01e0f"
)
V1_2_ORACLE_DIGEST = (
    "3ffbe1a87f7975c97e7ff42e50a6a00ca0f47d8840a434d0ff215c303bf6f0e4"
)
V1_2_SCORER_CONTRACT_DIGEST = (
    "6cd0916954d3ad22adc19e66c0bf873ff0016cc685f95ce47d6355dd55450e10"
)
V2_SUCCESSOR_SCORER_CONTRACT_DIGEST = (
    "8fc0edae875cba6487ff1a1a771f96b0157da1474ac00de4186ecdb41b66d5df"
)
CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)

FIT_IDENTITY_PROJECTION_DIGEST = (
    "c81b031af848149f5b263129285c1ea78a79b3c3fb136f44309f9ad673049f48"
)
HISTORICAL_CALIBRATION_IDENTITY_PROJECTION_DIGEST = (
    "143f736376c7ef03b0f943670de79fcf69cc2b198c601b13188efc487d35b65f"
)
ALL_HISTORICAL_IDENTITY_PROJECTION_DIGEST = (
    "8aeb6db77f92d7564b4202f5523bc5ec623e20e931b264d947726ba99b6cedf6"
)


@dataclass(frozen=True)
class FailedBranchIdentity:
    branch_identity_digest: str
    state_identity_digest: str
    state_id: str
    scene_id: str
    split_role: str
    candidate_index: int
    candidate: str
    diagnostic_category: str


FAILED_BRANCH_IDENTITIES = (
    FailedBranchIdentity("33de555bbc5b913511a5b0a2e8cca854f80abd2ea0bb4921afd125eb46cba095", "dec61a0ec069783511a34d5f3744bf8e6f5ce3068e2063b887aa946432a740a6", "scorer_fit-local_composite_motifs-completion_enriched-03", "local_composite_motifs_2a01daa53a72", "fit", 10, "reverse_then_turn", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("76d449932f11663ffee4d73d9399b3a0d7027cd978acf28c1346d87c37a2d614", "b84f201bcf8b8d913760c2e64436ca4034b43c6d4476b24105a6fe962be966d8", "scorer_fit-medium_enclosed_maze-general-04", "medium_enclosed_maze_1124574f7392", "fit", 5, "turn_left_sustained", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("2c52803b44470dff7a3e4b89f83dd5b0fdaaf7404d14e8bae9b5a2a9b29faab7", "b84f201bcf8b8d913760c2e64436ca4034b43c6d4476b24105a6fe962be966d8", "scorer_fit-medium_enclosed_maze-general-04", "medium_enclosed_maze_1124574f7392", "fit", 6, "turn_right_sustained", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("893c3293aecf7f27050bc5471f0f3844347836a04078133bf5b338867289646c", "b84f201bcf8b8d913760c2e64436ca4034b43c6d4476b24105a6fe962be966d8", "scorer_fit-medium_enclosed_maze-general-04", "medium_enclosed_maze_1124574f7392", "fit", 10, "reverse_then_turn", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("52a516e559ecf88ad27e91ad4e4c66e670fc715297956ff391c9a72272f9d77c", "2e1b7f43e028eee7b162cf4fbbaf1981b85e9266c7b91bc218c456ca3c60ed74", "scorer_fit-open_obstacle_field-general-00", "open_obstacle_field_03fdc12540a1", "calibration", 0, "straight_fast", "OFF_NAVIGABLE_GRAPH_OUTCOME"),
    FailedBranchIdentity("e175015f9e54321b64f88bde4cd59771046136d2cc54ecec31c41e02509c12b7", "2e1b7f43e028eee7b162cf4fbbaf1981b85e9266c7b91bc218c456ca3c60ed74", "scorer_fit-open_obstacle_field-general-00", "open_obstacle_field_03fdc12540a1", "calibration", 1, "straight_medium", "OFF_NAVIGABLE_GRAPH_OUTCOME"),
    FailedBranchIdentity("69f9315951c67db1ebf9ccc00272ef05cd17f025431842bafc526d30bddd5f5e", "c5bd2d361a1a8f394f4e12097efac68cbd00767179c8823d6f8efa106ea61b41", "scorer_fit-open_obstacle_field-general-02", "open_obstacle_field_060b48609180", "fit", 0, "straight_fast", "OFF_NAVIGABLE_GRAPH_OUTCOME"),
    FailedBranchIdentity("6f2d34186896a1d1ff75b936a7b1bd7599181baaf54a1b01efa3f51f5e616eef", "c5bd2d361a1a8f394f4e12097efac68cbd00767179c8823d6f8efa106ea61b41", "scorer_fit-open_obstacle_field-general-02", "open_obstacle_field_060b48609180", "fit", 1, "straight_medium", "OFF_NAVIGABLE_GRAPH_OUTCOME"),
    FailedBranchIdentity("bb26b919f04a3b26cd8852192a18fa8a050f0e179a442b449b1064021c4e5051", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 0, "straight_fast", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("6d3900adf9e67e1e15bbd7017b6ab6bcc1378cc41278b15333174ea2070f7832", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 1, "straight_medium", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("1c571b29a26c4922d89b258d9c7f6f4a0902c7ac6edb17788c814b836875e6ec", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 2, "straight_slow", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("9b5333525d0d86a8039d1d53cd302fcb915cf3b1e64e94d45e8a8827e2bc82ca", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 3, "arc_left_sustained", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("b446e9f2746590e4237952086cba55ac51e38e55e3fb28e7d43be333dfaa588d", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 4, "arc_right_sustained", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("36330c334efe461cfbfa0beae5b46d11aff10940d661a0fc1b457927b4adfa74", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 5, "turn_left_sustained", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("682ac99be9f989602607b6d34ae1d64484e22cb1665d4c832a989aa0fb2ad305", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 7, "turn_left_then_go", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("2ebefcf80c8229b2fcc828977342f1998a74259714f4ea5c3c0de85ab962c6c1", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 8, "turn_right_then_go", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("82cd0c2c647a71327b58c0a1e7d0dc3b6c4a2d46ce738887af48e944782c5200", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 9, "go_then_turn_left", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
    FailedBranchIdentity("6021ecc564f728d49420c28bcf18d2d64eaea535f9b488593b36f20679913bdb", "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "calibration", 11, "hold_all", "LOCATABLE_GOAL_UNREACHABLE_OUTCOME"),
)
FAILED_BRANCH_ALLOWLIST = frozenset(
    row.branch_identity_digest for row in FAILED_BRANCH_IDENTITIES
)


@dataclass(frozen=True)
class HistoricalCalibrationIdentity:
    state_identity_digest: str
    state_id: str
    scene_id: str
    family: str
    stratum: str


HISTORICAL_CALIBRATION_STATES = (
    HistoricalCalibrationIdentity("9237d8db4c775c0a03b0f0391df05a50eb9518ee0a93c7b15bb67305f7e9a830", "scorer_fit-large_enclosed_maze-general-00", "large_enclosed_maze_0294e4b379e1", "large_enclosed_maze", "general"),
    HistoricalCalibrationIdentity("7a0d0c4886f9e59c827be19b2e35d70c92ff233e47543fb9e999e31299e10bb5", "scorer_fit-large_enclosed_maze-safety_enriched-00", "large_enclosed_maze_196aa71822f2", "large_enclosed_maze", "safety_enriched"),
    HistoricalCalibrationIdentity("451efe30e767df2f0d2d5cb8cc0b7813c36b4f66aff5fb6c3a9fc3a8284dd015", "scorer_fit-large_enclosed_maze-completion_enriched-00", "large_enclosed_maze_2fab4b1c16a6", "large_enclosed_maze", "completion_enriched"),
    HistoricalCalibrationIdentity("3ca257fb07956a88a0101a01cf67549be02f415df3ab676b4a5b06260258991a", "scorer_fit-local_composite_motifs-general-00", "local_composite_motifs_0302989c334d", "local_composite_motifs", "general"),
    HistoricalCalibrationIdentity("5ba6a01d358195db42fb265c4e406bc61347077f6e95179de9479435126fb441", "scorer_fit-local_composite_motifs-safety_enriched-00", "local_composite_motifs_15175bca8bdb", "local_composite_motifs", "safety_enriched"),
    HistoricalCalibrationIdentity("24a28ab7ffeca55088cdac2f7eb2b9b79313c4706f9852c5f8c954d4dbbdc0e3", "scorer_fit-local_composite_motifs-completion_enriched-00", "local_composite_motifs_15385f6ed7d6", "local_composite_motifs", "completion_enriched"),
    HistoricalCalibrationIdentity("8d9287576881e9b006b88cc6c4ab4dc4ae0152d7cc7c33d7acff2cd99791651c", "scorer_fit-loop_alias_stress-general-00", "loop_alias_stress_02bc8ab4058a", "loop_alias_stress", "general"),
    HistoricalCalibrationIdentity("2d4d0a270a3db9baf12f621b08090bb1281c24dfbef063009f365064e44fa659", "scorer_fit-loop_alias_stress-safety_enriched-00", "loop_alias_stress_2ea7bc8d2198", "loop_alias_stress", "safety_enriched"),
    HistoricalCalibrationIdentity("17cdc4f171251f9728d8f12a3fcf1f60966fdfe7d0669c91a5476952afc0093b", "scorer_fit-loop_alias_stress-completion_enriched-00", "loop_alias_stress_39316161a20c", "loop_alias_stress", "completion_enriched"),
    HistoricalCalibrationIdentity("be7401dcbf89c27cc9766a131ebae6d77019b190819e0bb901aae379d6eb5ebb", "scorer_fit-medium_enclosed_maze-general-00", "medium_enclosed_maze_0011a1affc54", "medium_enclosed_maze", "general"),
    HistoricalCalibrationIdentity("24676131f0c9c03e7e8496b7481d216d8b7bf5a2462fb380729628daf20d876d", "scorer_fit-medium_enclosed_maze-safety_enriched-00", "medium_enclosed_maze_137e1f03ca2d", "medium_enclosed_maze", "safety_enriched"),
    HistoricalCalibrationIdentity("9c3940c1a512f4ddaa4f29d7a4f05b210bdf6dbed212d26f94ced72301ed3235", "scorer_fit-medium_enclosed_maze-completion_enriched-00", "medium_enclosed_maze_2167bdab4e49", "medium_enclosed_maze", "completion_enriched"),
    HistoricalCalibrationIdentity("2e1b7f43e028eee7b162cf4fbbaf1981b85e9266c7b91bc218c456ca3c60ed74", "scorer_fit-open_obstacle_field-general-00", "open_obstacle_field_03fdc12540a1", "open_obstacle_field", "general"),
    HistoricalCalibrationIdentity("fbce3ad81c19ffb1a6a57573403077a86b3cf9162da931064cb2b5c221ce45b2", "scorer_fit-open_obstacle_field-safety_enriched-00", "open_obstacle_field_14592a4488b0", "open_obstacle_field", "safety_enriched"),
    HistoricalCalibrationIdentity("220525522bfd3054dbcdde8c5338d6f1459d46443457798e27682f85343c3b8f", "scorer_fit-open_obstacle_field-completion_enriched-00", "open_obstacle_field_165a5b0554e1", "open_obstacle_field", "completion_enriched"),
    HistoricalCalibrationIdentity("18108441dd1a56a4e76f59c9b151ac71726a7a149bbde97553b78dd9091878f4", "scorer_fit-rough_local_dynamics-general-00", "rough_local_dynamics_088e9f856dda", "rough_local_dynamics", "general"),
    HistoricalCalibrationIdentity("5cce57d3f1e709e02beadadb351c18aeefb13d1b49a41260246ba1a57c8fd80e", "scorer_fit-rough_local_dynamics-safety_enriched-00", "rough_local_dynamics_249e04b2e81a", "rough_local_dynamics", "safety_enriched"),
    HistoricalCalibrationIdentity("aebe821f5d1361977bae8e99d816a04a38ef08cb2b0bcf5ce8621ce27f321b5d", "scorer_fit-rough_local_dynamics-completion_enriched-00", "rough_local_dynamics_1a072465e2d2", "rough_local_dynamics", "completion_enriched"),
    HistoricalCalibrationIdentity("0b9a6eb9a429145dc441bf15d0a8dd38e25cff1c501c4420cddb1da1a83296cc", "scorer_fit-small_enclosed_maze-general-00", "small_enclosed_maze_035bc78f0849", "small_enclosed_maze", "general"),
    HistoricalCalibrationIdentity("1cfccd331434a6d8350dfae42c71abfddf3c38d1a5f817e0483ab90345170e6c", "scorer_fit-small_enclosed_maze-safety_enriched-00", "small_enclosed_maze_0a705161a37f", "small_enclosed_maze", "safety_enriched"),
    HistoricalCalibrationIdentity("38a1bbf11b5aab346790d5f3973b4a8f45f6094ba87bbdd861349e0ebe41a05e", "scorer_fit-small_enclosed_maze-completion_enriched-00", "small_enclosed_maze_fd68c85e4fec", "small_enclosed_maze", "completion_enriched"),
    HistoricalCalibrationIdentity("b8c1cd6e866610bdef0af28f8b444823d891ff28a417000c8c300d77827f24c9", "scorer_fit-visual_sensor_stress-general-00", "visual_sensor_stress_02a4094c2467", "visual_sensor_stress", "general"),
    HistoricalCalibrationIdentity("95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976", "scorer_fit-visual_sensor_stress-safety_enriched-00", "visual_sensor_stress_2301db82bb24", "visual_sensor_stress", "safety_enriched"),
    HistoricalCalibrationIdentity("7750a982f03e5ef237230d1ebae5a96ac284f5b6b47476404812b89b348c565e", "scorer_fit-visual_sensor_stress-completion_enriched-00", "visual_sensor_stress_07b6ccbe4132", "visual_sensor_stress", "completion_enriched"),
)
OLD_CALIBRATION_STATES = HISTORICAL_CALIBRATION_STATES

FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
STRATA = ("general", "safety_enriched", "completion_enriched")

FRESH_CALIBRATION_SELECTOR = {
    "schema": "go2_scorer_fit_oracle_v1_3_fresh_calibration_selector_v1",
    "count": 24,
    "allocation": "exact Cartesian product of eight families and three strata",
    "families": list(FAMILIES),
    "strata": list(STRATA),
    "states_per_family_stratum": 1,
    "one_state_per_scene": True,
    "scene_order": "ascending lexical scene_id within each family",
    "capture_rule": "first eligible boundary under the inherited selector",
    "warmup_blocks_inclusive": [40, 120],
    "drive_seed_rule": "20260811 XOR crc32(scene_id)",
    "exclusions": {
        "all_120_historical_scorer_fit_scenes": True,
        "source_state_manifest_digest": FROZEN_STATE_MANIFEST_DIGEST,
        "historical_identity_projection_digest":
            ALL_HISTORICAL_IDENTITY_PROJECTION_DIGEST,
        "all_predecessor_selector_exclusions": True,
        "selected_fresh_scenes_are_pairwise_distinct": True,
        "future_final_evaluation_must_exclude_fresh_scenes": True,
    },
    "forbidden_selection_inputs": [
        "candidate identity",
        "candidate outcome",
        "oracle label",
        "latent",
        "scorer output",
        "qualification metric",
    ],
    "worker_or_infrastructure_failure": "terminal; never skip to a replacement scene",
    "post_selection_replacement": False,
    "manifest_frozen_before_candidate_branch_generation": True,
}

SELECTOR_INTEGRITY_REPLACEMENT = {
    "schema": (
        "go2_scorer_fit_oracle_v1_3_selector_integrity_replacement_contract_v1"
    ),
    "status": "FROZEN_SELECTOR_ONLY_INTEGRITY_REPLACEMENT",
    "scope": "fresh_calibration_state_selection_only",
    "predecessor_authority_digest": (
        "aa7c536059017d4eb7953f6ebb42bf0d83f7ca4c6303f4c52d3631637a0fe48f"
    ),
    "predecessor_source_commit": (
        "cfe087c36b2129510f4c75af36c1a2631b59f6ac"
    ),
    "predecessor_contract_digest": (
        "93532f22a0cbc0e57ccdab3d5c01419cd824bc402d637738c5004eb621c23a89"
    ),
    "predecessor_source_files_digest": (
        "12578e82ad25b5a1c49f0c81b0c490f280ff5b86da692ab52249409c952e9997"
    ),
    "selector_digest": (
        "041607943d7246ee657584130e4fe35d7565944f933f3e07a9ee4ffc576981cb"
    ),
    "successor_preregistration_path": str(PREREGISTRATION_PATH),
    "successor_preregistration_digest_key": "preregistration_digest",
    "incident": {
        "stage": "select-calibration",
        "process_id": 1822672,
        "observed_at_local": "2026-08-15T12:45:50+01:00",
        "observed_at_utc": "2026-08-15T11:45:50Z",
        "evidence_source": "kernel journal",
        "signal": "SIGSEGV",
        "fault_address": "0x1088",
        "page_fault_error_code": 4,
        "shared_object": "libc.so.6",
        "shared_object_offset": "0xa00d4",
        "likely_cpu": 8,
        "process_gone": True,
        "phase": "before any fresh-calibration manifest or durable output",
    },
    "preserved_predecessor_artifacts": {
        "authority": {
            "path": str(AUTHORITY_PATH),
            "self_key": "authority_digest",
            "self_digest": (
                "aa7c536059017d4eb7953f6ebb42bf0d83f7ca4c6303f4c52d3631637a0fe48f"
            ),
            "raw_sha256": (
                "669c6847c97e04b319ba8de12a5df72957cbf225796605f9567626e26fa6bb4e"
            ),
            "byte_count": 55477,
            "file_mode": "0600",
        },
        "replay_plan": {
            "path": str(REPLAY_PLAN_PATH),
            "self_key": "replay_plan_digest",
            "self_digest": (
                "39a75ba2abe545b30adf27222bf9bc03283c6036fad37bd6f14f1bd5d36a6c03"
            ),
            "raw_sha256": (
                "beebf4c65a4e8d2633f6e33ad836c45a24aa504fa2f56334baf2e154665ef0ec"
            ),
            "byte_count": 460892,
            "file_mode": "0600",
            "entry_count": 18,
        },
        "equivalence_receipt": {
            "path": str(EQUIVALENCE_RECEIPT_PATH),
            "self_key": "equivalence_receipt_digest",
            "self_digest": (
                "49a4efb800b8eae5fe4a4a7cfc47833bfda8c5994acc02f5736a2f08ec76992e"
            ),
            "raw_sha256": (
                "f023724313f6f85d2fa60abbf432a54652a7de79a0941be1515e1e54d896c779"
            ),
            "byte_count": 644424,
            "file_mode": "0600",
            "compared_count": 1422,
            "mismatch_count": 0,
        },
        "replay_overlay_manifest": {
            "path": str(REPLAY_OVERLAY_MANIFEST_PATH),
            "self_key": "replay_overlay_manifest_digest",
            "self_digest": (
                "b4906f7eff24a63e3d1c102187678f9dcacc835aa55bc0c69f66f14b16022594"
            ),
            "raw_sha256": (
                "3c5548b51f090f879ba3095f66d08f65704459e90913fd84eda44e8c351abf53"
            ),
            "byte_count": 8697,
            "file_mode": "0600",
            "overlay_count": 18,
            "fit_overlay_count": 6,
            "historical_calibration_overlay_count": 12,
        },
        "replay_attempt_markers": {
            "root": str(REPLAY_ATTEMPTS_ROOT),
            "count": 18,
            "file_mode": "0600",
            "file_binding_set_digest": (
                "baf2cf718367118f6b05d30365fa93405c9e1ef139e5a9d34cb2053e9f80e2cf"
            ),
        },
        "replay_overlays": {
            "root": str(REPLAY_OVERLAYS_ROOT),
            "count": 18,
            "file_mode": "0600",
            "file_binding_set_digest": (
                "5233bca4c560c844324f3e937469faef850d4fba4e6588ae07945a6484872b44"
            ),
        },
        "file_binding_set_rule": (
            "canonical digest of ascending-path records containing path, sha256, "
            "and byte_count"
        ),
        "mutation_authorised": False,
    },
    "failed_selector_outputs": {
        "fresh_calibration_root_exists": False,
        "state_manifest_exists": False,
        "selection_attempt_exists": False,
        "selection_task_count": 0,
        "selection_result_count": 0,
        "selected_state_count": 0,
        "fresh_branch_attempt_count": 0,
        "fresh_branch_outcome_count": 0,
    },
    "one_shot_selector_attempt": True,
    "selector_attempt_count": 1,
    "same_outcome_blind_selector_required": True,
    "per_scene_process_isolation_required": True,
    "exclusive_per_task_launch_marker_required": True,
    "launch_marker_path_rule": (
        "selection_tasks/<task_digest>.launch.json"
    ),
    "worker_refuses_existing_terminal_result_or_launch_marker": True,
    "task_result_atomic_publication_required": True,
    "parent_exact_validation_before_merge_required": True,
    "resume_only_from_valid_task_result": True,
    "unresolved_child_is_terminal": True,
    "valid_result_survives_nonzero_teardown": True,
    "nonzero_teardown_survival_condition": (
        "child result was atomically published before teardown and passes exact "
        "parent validation"
    ),
    "scene_skip_or_replacement": False,
    "post_selection_replacement": False,
    "manifest_publication": (
        "only after exactly 24 valid isolated results satisfy the unchanged selector"
    ),
    "paths": {
        "replacement_authority":
            str(SELECTOR_INTEGRITY_REPLACEMENT_AUTHORITY_PATH),
        "selection_attempt": str(FRESH_SELECTION_ATTEMPT_PATH),
        "selection_tasks_root": str(FRESH_SELECTION_TASKS_ROOT),
        "selection_results_root": str(FRESH_SELECTION_RESULTS_ROOT),
        "selection_terminal": str(FRESH_SELECTION_TERMINAL_PATH),
        "state_manifest": str(FRESH_CALIBRATION_STATE_MANIFEST_PATH),
    },
    "authority": {
        "selector_recomputation": True,
        "fresh_calibration_state_manifest_publication": True,
        "candidate_branch_execution": False,
        "branch_retry_or_replacement": False,
        "latent_encoding": False,
        "scorer_training": False,
        "qualification": False,
        "predictor_checkpoint_or_utility_shard_open": False,
        "final_benchmark_generation": False,
    },
    "preserved_predecessor_authority_continuation": {
        "authorising_artifact": str(AUTHORITY_PATH),
        "authorising_artifact_digest": (
            "aa7c536059017d4eb7953f6ebb42bf0d83f7ca4c6303f4c52d3631637a0fe48f"
        ),
        "replacement_grants_new_candidate_branch_authority": False,
        "predecessor_candidate_branch_authority_preserved": True,
        "continuation_requires_valid_selector_terminal": True,
        "continuation_requires_exact_24_state_manifest": True,
        "continuation_requires_current_successor_preregistration_binding": True,
        "fresh_branch_attempts_before_replacement": 0,
        "fresh_branch_budget": 288,
        "branch_attempt_policy": "AT_MOST_ONCE_PER_EXACT_IDENTITY_NO_REPLACEMENT",
    },
}

STAGE_COUNTS = {
    "historical_states": 120,
    "historical_fit_states": 96,
    "historical_calibration_states_development_only": 24,
    "historical_branch_rows": 1440,
    "legacy_valid_adoptions": 1422,
    "legacy_valid_fit_adoptions": 1146,
    "legacy_valid_historical_calibration_adoptions": 276,
    "exact_failed_branch_replays": 18,
    "failed_fit_branch_replays": 6,
    "failed_historical_calibration_branch_replays": 12,
    "fresh_calibration_states": 24,
    "fresh_calibration_branches": 288,
    "v1_3_simulator_branch_executions": 306,
    "complete_fit_rows": 1152,
    "complete_fresh_calibration_rows": 288,
    "training_view_states": 120,
    "training_view_rows": 1440,
    "encoded_horizon_latent_shards": 1440,
    "shared_scorer_training_runs": 1,
    "no_latent_baseline_training_runs": 1,
    "qualification_evaluations": 1,
    "predictor_utility_shards_opened_before_qualification_pass": 0,
    "final_benchmark_states_generated": 0,
}

SCORER_TRAINING_CONTRACT = {
    "protected_predecessor_scorer_contract_digest": V1_2_SCORER_CONTRACT_DIGEST,
    "protected_v2_successor_contract_digest": V2_SUCCESSOR_SCORER_CONTRACT_DIGEST,
    "architecture": "unchanged protected shared three-head scorer",
    "heads": ["progress", "safety", "completion"],
    "utility": "1.0*progress - 2.0*safety + 0.5*completion",
    "training": {
        "epochs": 60,
        "batch": 64,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "optimiser": "AdamW",
        "seed": 20260811,
    },
    "updates_per_epoch": 18,
    "updates_per_model": 1080,
    "weight_selection": "final epoch only; no best-epoch selection",
    "final_epoch_only": True,
    "fit_calibration_split": "by scene",
    "shared_scorer_runs": 1,
    "required_no_latent_baseline_runs": 1,
    "paired_no_latent_baseline_required": True,
    "retry_or_hyperparameter_tuning_after_outcomes": False,
    "missing_label_policy": "stop before encoding/training",
    "degenerate_label_policy": "stop before training or fail qualification",
}

QUALIFICATION_THRESHOLDS = {
    "progress_spearman_min": 0.50,
    "safety_roc_auc_min": 0.75,
    "safety_ece_max": 0.10,
    "completion_roc_auc_min": 0.75,
    "completion_ece_max": 0.10,
    "composite_within_state_pairwise_accuracy_min": 0.65,
    "no_latent_pairwise_margin_min": 0.05,
    "completion_nondegenerate_in_fit_and_calibration": True,
    "conjunction_required": True,
    "tie_tolerance": 0.02,
    "evaluation_count": 1,
    "failure_is_terminal": True,
}

OUTPUT_PATHS = {
    "authority": str(AUTHORITY_PATH),
    "superseded_preattempt_transition":
        str(SUPERSEDED_PREATTEMPT_TRANSITION_PATH),
    "replay_plan": str(REPLAY_PLAN_PATH),
    "equivalence_receipt": str(EQUIVALENCE_RECEIPT_PATH),
    "replay_attempts_root": str(REPLAY_ATTEMPTS_ROOT),
    "replay_overlays_root": str(REPLAY_OVERLAYS_ROOT),
    "replay_overlay_manifest": str(REPLAY_OVERLAY_MANIFEST_PATH),
    "fresh_calibration_state_manifest": str(FRESH_CALIBRATION_STATE_MANIFEST_PATH),
    "fresh_calibration_rows_root": str(FRESH_CALIBRATION_ROWS_ROOT),
    "fresh_calibration_frames_root": str(FRESH_CALIBRATION_FRAMES_ROOT),
    "fresh_calibration_corpus_receipt": str(FRESH_CALIBRATION_CORPUS_RECEIPT_PATH),
    "training_view": str(TRAINING_VIEW_PATH),
    "encoded_training_view_root": str(ENCODED_TRAINING_VIEW_ROOT),
    "latent_index": str(LATENT_INDEX_PATH),
    "encoding_receipt": str(ENCODING_RECEIPT_PATH),
    "horizon_latents_root": str(HORIZON_LATENTS_ROOT),
    "scorer_root": str(SCORER_ROOT),
    "qualification": str(QUALIFICATION_PATH),
    "training_execution_authorisation": str(TRAINING_EXECUTION_AUTHORISATION_PATH),
    "qualification_evaluation_authorisation": str(
        QUALIFICATION_EVALUATION_AUTHORISATION_PATH
    ),
    "scorer_package": str(SCORER_PACKAGE_PATH),
    "scorer_package_receipt": str(SCORER_PACKAGE_RECEIPT_PATH),
    "no_latent_baseline": str(NO_LATENT_BASELINE_PATH),
    "no_latent_baseline_receipt": str(NO_LATENT_BASELINE_RECEIPT_PATH),
    "failed_scorer": str(FAILED_SCORER_PATH),
    "registered_initialisations_root": str(REGISTERED_INITIALISATIONS_ROOT),
    "training_checkpoints_root": str(TRAINING_CHECKPOINTS_ROOT),
}

SOURCE_CLOSURE_PATHS = (
    "lewm/oracle/go2_branch_oracle_v1_3.py",
    "lewm/oracle/go2_scorer_fit_oracle_v1_3_contract.py",
    "scripts/run_go2_scorer_fit_oracle_v1_3.py",
    "scripts/encode_go2_scorer_fit_oracle_v1_3.py",
    "scripts/train_go2_utility_scorer_v1_3.py",
    "lewm/oracle/go2_branch_oracle_v1_2.py",
    "scripts/run_go2_oracle_branch_pilot_v1.py",
    "scripts/run_go2_oracle_branch_pilot_v1_2.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "lewm_worlds/lewm_worlds/scene_graph.py",
    "lewm_worlds/lewm_worlds/labels/derived.py",
    "lewm_genesis/lewm_genesis/rollout.py",
    "lewm/oracle/go2_scorer_contract_v1_2.py",
    "scripts/encode_go2_branch_corpus_v1_2.py",
    "scripts/train_go2_utility_scorer_v1_2.py",
    "scripts/dev_action_slew_reconstruction_v1.py",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "config/go2_platform_manifest.yaml",
    "config/go2_primitive_registry.yaml",
)
TEST_CLOSURE_PATHS = (
    "lewm/tests/test_go2_branch_oracle_v1_3.py",
    "lewm/tests/test_go2_scorer_fit_oracle_v1_3_contract.py",
    "lewm/tests/test_run_go2_scorer_fit_oracle_v1_3.py",
    "lewm/tests/test_encode_go2_scorer_fit_oracle_v1_3.py",
    "lewm/tests/test_train_go2_utility_scorer_v1_3.py",
)


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _failure_rows() -> list[dict[str, Any]]:
    return [asdict(row) for row in FAILED_BRANCH_IDENTITIES]


def _historical_calibration_rows() -> list[dict[str, Any]]:
    return [asdict(row) for row in HISTORICAL_CALIBRATION_STATES]


def failed_branch_allowlist_digest() -> str:
    return canonical_digest(_failure_rows())


def historical_calibration_disposition_digest() -> str:
    return canonical_digest({
        "states": _historical_calibration_rows(),
        "status": "DEVELOPMENT_ONLY",
        "qualification_eligible": False,
        "discarded": False,
        "replaced": False,
    })


def fresh_calibration_selector_digest() -> str:
    return canonical_digest(FRESH_CALIBRATION_SELECTOR)


def selector_integrity_replacement_digest() -> str:
    return canonical_digest(SELECTOR_INTEGRITY_REPLACEMENT)


def contract() -> dict[str, Any]:
    inherited = ORACLE.oracle_contract()
    if (
        inherited.get("supersedes_oracle_v1_2_digest") != V1_2_ORACLE_DIGEST
        or inherited.get("inherited_progress_digest") != V1_2_PROGRESS_DIGEST
        or inherited.get("inherited_safety_digest") != V1_2_SAFETY_DIGEST
    ):
        raise RuntimeError("protected oracle-v1.2 lineage changed")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "oracle_contract": inherited,
        "oracle_v1_3_digest": ORACLE.oracle_digest(),
        "lineage": {
            "source_base_commit": SOURCE_BASE_COMMIT,
            "audited_corpus_source_commit": AUDITED_CORPUS_SOURCE_COMMIT,
            "frozen_corpus_digest": FROZEN_CORPUS_DIGEST,
            "frozen_branch_rows_sha256": FROZEN_BRANCH_ROWS_SHA256,
            "frozen_state_manifest_digest": FROZEN_STATE_MANIFEST_DIGEST,
            "frozen_assignment_manifest_digest": FROZEN_ASSIGNMENT_MANIFEST_DIGEST,
            "frozen_branch_identity_set_digest": FROZEN_BRANCH_IDENTITY_SET_DIGEST,
            "diagnostic_path": str(DIAGNOSTIC_PATH),
            "diagnostic_file_sha256": DIAGNOSTIC_FILE_SHA256,
            "diagnostic_audit_digest": DIAGNOSTIC_AUDIT_DIGEST,
            "v1_2_progress_digest": V1_2_PROGRESS_DIGEST,
            "v1_2_safety_digest": V1_2_SAFETY_DIGEST,
            "v1_2_oracle_digest": V1_2_ORACLE_DIGEST,
            "candidate_bank_digest": CANDIDATE_BANK_DIGEST,
        },
        "exact_replay_allowlist": _failure_rows(),
        "exact_replay_allowlist_digest": failed_branch_allowlist_digest(),
        "replay_policy": {
            "attempts": 18,
            "exact_source_state_redrive_and_candidate_action": True,
            "source_snapshot_digest_preserved_as_lineage": True,
            "source_snapshot_digest_equality_required": False,
            "source_snapshot_digest_limitation": (
                "snapshot bytes and original process-global RNG history were not "
                "persisted"
            ),
            "prebranch_physical_witness": (
                "three context poses plus persisted proprio/control/action context "
                "and previous applied command"
            ),
            "context_pose_atol": 1e-5,
            "horizon_pose_atol": 1e-5,
            "action_atol": 1e-6,
            "four_horizon_pose_checks_required": True,
            "new_overlay_only": True,
            "predecessor_rows_or_frames_overwritten": False,
            "retry_or_replacement": False,
            "all_twenty_ticks_required": True,
        },
        "genesis_runtime": dict(GENESIS_RUNTIME_CONTRACT),
        "superseded_preattempt_execution_authority":
            dict(SUPERSEDED_PREATTEMPT_EXECUTION_AUTHORITY),
        "legacy_valid_equivalence": {
            "row_count": 1422,
            "mode": "exact v1.2 label adoption plus deterministic recomputation",
            "requires_exact_row_digest_allowlist": True,
            "simulator_rerun_count": 0,
            "legacy_label_keys": list(ORACLE.LEGACY_LABEL_KEYS),
        },
        "historical_calibration_disposition": {
            "states": _historical_calibration_rows(),
            "state_count": 24,
            "branch_count": 288,
            "identity_projection_digest":
                HISTORICAL_CALIBRATION_IDENTITY_PROJECTION_DIGEST,
            "status": "DEVELOPMENT_ONLY",
            "qualification_eligible": False,
            "discarded": False,
            "replaced": False,
        },
        "historical_calibration_disposition_digest":
            historical_calibration_disposition_digest(),
        "fit_identity_projection_digest": FIT_IDENTITY_PROJECTION_DIGEST,
        "all_historical_identity_projection_digest":
            ALL_HISTORICAL_IDENTITY_PROJECTION_DIGEST,
        "fresh_calibration_selector": FRESH_CALIBRATION_SELECTOR,
        "fresh_calibration_selector_digest": fresh_calibration_selector_digest(),
        "stage_counts": STAGE_COUNTS,
        "scorer_training": SCORER_TRAINING_CONTRACT,
        "qualification": QUALIFICATION_THRESHOLDS,
        "output_paths": OUTPUT_PATHS,
        "managed_output_storage": {
            "logical_root": str(GENERATED_ROOT),
            "registered_physical_target":
                str(REGISTERED_GENERATED_TARGET_ROOT),
            "only_the_logical_root_may_be_a_symlink": True,
        },
        "output_schemas": {
            "trace": TRACE_SCHEMA,
            "replay_overlay": REPLAY_OVERLAY_SCHEMA,
            "latent_index": LATENT_INDEX_SCHEMA,
            "encoding_receipt": ENCODING_RECEIPT_SCHEMA,
            "qualification": QUALIFICATION_SCHEMA,
            "training_execution_authorisation":
                TRAINING_EXECUTION_AUTHORISATION_SCHEMA,
            "qualification_evaluation_authorisation":
                QUALIFICATION_EVALUATION_AUTHORISATION_SCHEMA,
            "scorer_package": SCORER_PACKAGE_SCHEMA,
            "scorer_package_receipt": SCORER_PACKAGE_RECEIPT_SCHEMA,
            "no_latent_baseline": NO_LATENT_BASELINE_SCHEMA,
            "no_latent_baseline_receipt": NO_LATENT_BASELINE_RECEIPT_SCHEMA,
        },
        "source_closure_paths": list(SOURCE_CLOSURE_PATHS),
        "test_closure_paths": list(TEST_CLOSURE_PATHS),
        "prohibitions": {
            "replace_predecessor_branch_state_candidate_or_frame": True,
            "train_with_any_missing_label": True,
            "use_historical_calibration_for_qualification": True,
            "open_predictor_utility_shards_before_qualification_pass": True,
            "generate_final_200_state_benchmark": True,
            "retry_or_tune_after_training_or_qualification_outcomes": True,
        },
        "qualification_pass_authorises_predictor_open_in_this_workflow": False,
        "final_200_state_benchmark_authorised": False,
    }


def contract_digest() -> str:
    return canonical_digest(contract())


def _file_binding(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise RuntimeError(f"required source-closure path is absent: {relative_path}")
    data = path.read_bytes()
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def source_bindings(
    root: Path,
    *,
    paths: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected = tuple(SOURCE_CLOSURE_PATHS + TEST_CLOSURE_PATHS) if paths is None \
        else tuple(paths)
    if len(selected) != len(set(selected)):
        raise RuntimeError("source closure contains duplicate paths")
    return [_file_binding(root, path) for path in selected]


def build_preregistration(
    *,
    source_bindings_value: Sequence[Mapping[str, Any]],
    source_freeze_commit: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "status": PREREGISTRATION_STATUS,
        "complete": True,
        "contract": contract(),
        "contract_digest": contract_digest(),
        "selector_integrity_replacement": SELECTOR_INTEGRITY_REPLACEMENT,
        "selector_integrity_replacement_digest":
            selector_integrity_replacement_digest(),
        "original_preregistration_binding": ORIGINAL_PREREGISTRATION_BINDING,
        "source_repository_base_commit":
            SELECTOR_INTEGRITY_REPLACEMENT["predecessor_source_commit"],
        "source_freeze_commit": source_freeze_commit,
        "source_bindings": [dict(row) for row in source_bindings_value],
        "legacy_equivalence_completed": True,
        "legacy_equivalence_compared_count": 1422,
        "legacy_equivalence_mismatch_count": 0,
        "exact_failed_branch_replay_completed": True,
        "exact_failed_branch_replay_count": 18,
        "failed_selector_process_observed": True,
        "fresh_calibration_state_manifest_published": False,
        "selector_integrity_replacement_started": False,
        "fresh_calibration_branch_execution_started": False,
        "latent_encoding_started": False,
        "scorer_training_started": False,
        "qualification_started": False,
        "predictor_utility_shards_opened": False,
        "final_200_state_benchmark_generated": False,
    }
    payload["preregistration_digest"] = canonical_digest(payload)
    return payload


def validate_original_preregistration(
    value: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    artifact = dict(value)
    digest = artifact.pop("preregistration_digest", None)
    if digest != canonical_digest(artifact):
        raise RuntimeError("original v1.3 preregistration self digest changed")
    artifact["preregistration_digest"] = digest
    if (
        digest != ORIGINAL_PREREGISTRATION_BINDING["self_digest"]
        or artifact.get("schema") != ORIGINAL_PREREGISTRATION_SCHEMA
        or artifact.get("status") != ORIGINAL_PREREGISTRATION_STATUS
        or artifact.get("complete") is not True
        or artifact.get("contract") != contract()
        or artifact.get("contract_digest") != contract_digest()
    ):
        raise RuntimeError("original v1.3 preregistration contract changed")
    if root is not None:
        path = root / ORIGINAL_PREREGISTRATION_PATH
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("original v1.3 preregistration file is absent")
        data = path.read_bytes()
        if (
            len(data) != ORIGINAL_PREREGISTRATION_BINDING["byte_count"]
            or hashlib.sha256(data).hexdigest()
            != ORIGINAL_PREREGISTRATION_BINDING["raw_sha256"]
        ):
            raise RuntimeError("original v1.3 preregistration bytes changed")
    return artifact


def validate_preregistration(
    value: Mapping[str, Any],
    *,
    root: Path | None = None,
    require_complete_source_closure: bool = True,
) -> dict[str, Any]:
    artifact = dict(value)
    digest = artifact.pop("preregistration_digest", None)
    if digest != canonical_digest(artifact):
        raise RuntimeError("v1.3 preregistration self digest changed")
    artifact["preregistration_digest"] = digest
    if (
        artifact.get("schema") != PREREGISTRATION_SCHEMA
        or artifact.get("status") != PREREGISTRATION_STATUS
        or artifact.get("complete") is not True
        or artifact.get("contract") != contract()
        or artifact.get("contract_digest") != contract_digest()
        or artifact.get("selector_integrity_replacement")
        != SELECTOR_INTEGRITY_REPLACEMENT
        or artifact.get("selector_integrity_replacement_digest")
        != selector_integrity_replacement_digest()
        or artifact.get("original_preregistration_binding")
        != ORIGINAL_PREREGISTRATION_BINDING
        or artifact.get("source_repository_base_commit")
        != SELECTOR_INTEGRITY_REPLACEMENT["predecessor_source_commit"]
    ):
        raise RuntimeError("v1.3 preregistration contract changed")
    required_state = {
        "legacy_equivalence_completed": True,
        "legacy_equivalence_compared_count": 1422,
        "legacy_equivalence_mismatch_count": 0,
        "exact_failed_branch_replay_completed": True,
        "exact_failed_branch_replay_count": 18,
        "failed_selector_process_observed": True,
        "fresh_calibration_state_manifest_published": False,
        "selector_integrity_replacement_started": False,
        "fresh_calibration_branch_execution_started": False,
        "latent_encoding_started": False,
        "scorer_training_started": False,
        "qualification_started": False,
        "predictor_utility_shards_opened": False,
        "final_200_state_benchmark_generated": False,
    }
    for key, expected_value in required_state.items():
        if artifact.get(key) != expected_value:
            raise RuntimeError(f"selector-integrity preregistration state changed: {key}")
    bindings = artifact.get("source_bindings")
    if not isinstance(bindings, list) or any(
        not isinstance(row, Mapping) for row in bindings
    ):
        raise RuntimeError("v1.3 preregistration source bindings are malformed")
    paths = [str(row.get("path")) for row in bindings]
    required = set(SOURCE_CLOSURE_PATHS + TEST_CLOSURE_PATHS)
    if require_complete_source_closure and set(paths) != required:
        raise RuntimeError("v1.3 preregistration source closure is not exact")
    if len(paths) != len(set(paths)):
        raise RuntimeError("v1.3 preregistration repeats a source binding")
    if root is not None:
        expected = source_bindings(root, paths=paths)
        if bindings != expected:
            raise RuntimeError("v1.3 live source bytes differ from preregistration")
        original = json.loads((root / ORIGINAL_PREREGISTRATION_PATH).read_text())
        validate_original_preregistration(original, root=root)
        diagnostic = root / DIAGNOSTIC_PATH
        if (
            not diagnostic.is_file()
            or hashlib.sha256(diagnostic.read_bytes()).hexdigest()
            != DIAGNOSTIC_FILE_SHA256
        ):
            raise RuntimeError("v1.3 diagnostic lineage file changed")
    return artifact


__all__ = [
    "ALL_HISTORICAL_IDENTITY_PROJECTION_DIGEST",
    "AUTHORITY_PATH",
    "CONTRACT_SCHEMA",
    "CONTRACT_STATUS",
    "ENCODED_TRAINING_VIEW_ROOT",
    "ENCODING_RECEIPT_PATH",
    "ENCODING_RECEIPT_SCHEMA",
    "EQUIVALENCE_RECEIPT_PATH",
    "FAILED_BRANCH_ALLOWLIST",
    "FAILED_BRANCH_IDENTITIES",
    "FAMILIES",
    "GENESIS_RUNTIME_CONTRACT",
    "FIT_IDENTITY_PROJECTION_DIGEST",
    "FRESH_CALIBRATION_CORPUS_RECEIPT_PATH",
    "FRESH_CALIBRATION_FRAMES_ROOT",
    "FRESH_CALIBRATION_ROOT",
    "FRESH_CALIBRATION_ROWS_ROOT",
    "FRESH_CALIBRATION_SELECTOR",
    "FRESH_CALIBRATION_STATE_MANIFEST_PATH",
    "FRESH_SELECTION_ATTEMPT_PATH",
    "FRESH_SELECTION_RESULTS_ROOT",
    "FRESH_SELECTION_TASKS_ROOT",
    "FRESH_SELECTION_TERMINAL_PATH",
    "GENERATED_ROOT",
    "HISTORICAL_CALIBRATION_STATES",
    "HORIZON_LATENTS_ROOT",
    "LATENT_INDEX_PATH",
    "LATENT_INDEX_SCHEMA",
    "NO_LATENT_BASELINE_PATH",
    "NO_LATENT_BASELINE_RECEIPT_PATH",
    "NO_LATENT_BASELINE_RECEIPT_SCHEMA",
    "NO_LATENT_BASELINE_SCHEMA",
    "OLD_CALIBRATION_STATES",
    "ORIGINAL_PREREGISTRATION_BINDING",
    "ORIGINAL_PREREGISTRATION_PATH",
    "ORIGINAL_PREREGISTRATION_SCHEMA",
    "ORIGINAL_PREREGISTRATION_STATUS",
    "OUTPUT_PATHS",
    "PREREGISTRATION_PATH",
    "PREREGISTRATION_SCHEMA",
    "QUALIFICATION_PATH",
    "QUALIFICATION_EVALUATION_AUTHORISATION_PATH",
    "QUALIFICATION_EVALUATION_AUTHORISATION_SCHEMA",
    "QUALIFICATION_SCHEMA",
    "QUALIFICATION_THRESHOLDS",
    "REGISTERED_INITIALISATIONS_ROOT",
    "REGISTERED_GENERATED_TARGET_ROOT",
    "REPLAY_ATTEMPTS_ROOT",
    "REPLAY_OVERLAYS_ROOT",
    "REPLAY_OVERLAY_MANIFEST_PATH",
    "REPLAY_OVERLAY_SCHEMA",
    "REPLAY_PLAN_PATH",
    "SCORER_PACKAGE_PATH",
    "SCORER_PACKAGE_RECEIPT_PATH",
    "SCORER_PACKAGE_RECEIPT_SCHEMA",
    "SCORER_PACKAGE_SCHEMA",
    "SCORER_ROOT",
    "SCORER_TRAINING_CONTRACT",
    "SELECTOR_INTEGRITY_REPLACEMENT",
    "SELECTOR_INTEGRITY_REPLACEMENT_AUTHORITY_PATH",
    "SELECTOR_INTEGRITY_REPLACEMENT_SCHEMA",
    "SELECTOR_INTEGRITY_REPLACEMENT_SELF_KEY",
    "SOURCE_CLOSURE_PATHS",
    "STAGE_COUNTS",
    "STRATA",
    "SUPERSEDED_PREATTEMPT_EXECUTION_AUTHORITY",
    "SUPERSEDED_PREATTEMPT_ARCHIVE_ROOT",
    "SUPERSEDED_PREATTEMPT_TRANSITION_PATH",
    "TEST_CLOSURE_PATHS",
    "TRACE_SCHEMA",
    "TRAINING_CHECKPOINTS_ROOT",
    "TRAINING_EXECUTION_AUTHORISATION_PATH",
    "TRAINING_EXECUTION_AUTHORISATION_SCHEMA",
    "TRAINING_VIEW_PATH",
    "build_preregistration",
    "canonical_digest",
    "contract",
    "contract_digest",
    "failed_branch_allowlist_digest",
    "fresh_calibration_selector_digest",
    "historical_calibration_disposition_digest",
    "selector_integrity_replacement_digest",
    "source_bindings",
    "validate_original_preregistration",
    "validate_preregistration",
]
