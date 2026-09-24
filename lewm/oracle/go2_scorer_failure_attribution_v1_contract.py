"""Frozen exploratory contract for scorer-failure attribution.

This module is deliberately source-only.  It freezes the already examined
ViT-L evidence, the diagnostic transformations, and the sole permitted
final-layer attentive readout.  It neither opens scientific artefacts nor
issues a runtime contract.  A runner must pass a clean, committed eight-file
source closure to :func:`build_contract` before doing any diagnostic work.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping, Sequence


STATUS = "EXPLORATORY_SCORER_FAILURE_ATTRIBUTION"
CONTRACT_SCHEMA = "go2_scorer_failure_attribution_v1_contract_v1"
CONTRACT_SELF_KEY = "diagnostic_contract_digest"
SOURCE_CLOSURE_SCHEMA = "go2_scorer_failure_attribution_v1_source_closure_v1"
SOURCE_CLOSURE_SELF_KEY = "source_closure_digest"

GENERATED_ROOT = Path(".generated/go2_scorer_failure_attribution_v1")
REGISTERED_GENERATED_TARGET_ROOT = Path(
    "/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2/active/"
    "go2_scorer_failure_attribution_v1"
)
FROZEN_BRANCH_CORPUS_ROOT = Path(".generated/go2_branch_corpus_v1_2/scorer_fit")
SAFETY_OBSERVABILITY_ROOT = GENERATED_ROOT / "safety_observability"
PLAN_PATH = SAFETY_OBSERVABILITY_ROOT / "plan.json"
TRACE_ROWS_PATH = SAFETY_OBSERVABILITY_ROOT / "trace_rows.jsonl"
ATTEMPTS_ROOT = SAFETY_OBSERVABILITY_ROOT / "attempts"
TERMINAL_PATH = SAFETY_OBSERVABILITY_ROOT / "terminal.json"
AUDIT_PATH = SAFETY_OBSERVABILITY_ROOT / "audit.json"

PLAN_SCHEMA = "go2_scorer_failure_attribution_v1_safety_plan_v1"
PLAN_SELF_KEY = "safety_plan_digest"
TRACE_ROW_SCHEMA = "go2_scorer_failure_attribution_v1_tick_trace_v1"
TRACE_ROW_SELF_KEY = "trace_row_digest"
ATTEMPT_SCHEMA = "go2_scorer_failure_attribution_v1_replay_attempt_v1"
ATTEMPT_SELF_KEY = "replay_attempt_digest"
TERMINAL_SCHEMA = "go2_scorer_failure_attribution_v1_safety_terminal_v1"
TERMINAL_SELF_KEY = "safety_terminal_digest"
AUDIT_SCHEMA = "go2_scorer_failure_attribution_v1_safety_audit_v1"
AUDIT_SELF_KEY = "safety_audit_digest"

SOURCE_CLOSURE_PATHS = (
    "lewm/oracle/go2_scorer_failure_attribution_v1_contract.py",
    "lewm/tests/test_go2_scorer_failure_attribution_v1_contract.py",
    "scripts/run_go2_safety_observability_diagnostic_v1.py",
    "lewm/tests/test_run_go2_safety_observability_diagnostic_v1.py",
    "scripts/diagnose_go2_scorer_v1_3_latent_dependence_v1.py",
    "lewm/tests/test_diagnose_go2_scorer_v1_3_latent_dependence_v1.py",
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_v1.py",
    "lewm/tests/test_train_go2_utility_scorer_v1_3_attentive_readout_v1.py",
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Frozen predecessor and corpus lineage.
SOURCE_BASE_COMMIT = "8d36aeea09d1dc069d53dfb48675da560ea0c343"
SOURCE_COMMIT = "5c67135ad83b9206e6520e507f1ecaf980fd3d8d"
FROZEN_VITL_SCORER_SOURCE_LINEAGE = (
    "20aa87496f237b0769486d3e558e833bd6aa03ab"
)
FROZEN_VITG_SOURCE_HEAD = "8d36aeea09d1dc069d53dfb48675da560ea0c343"
FROZEN_VITG_RESULT_DIGEST = (
    "b8b98bb7f5ae607d023a20876107cead59c3bdfa0a858955ea0d760ea5973f0a"
)
FROZEN_VITG_CONCLUSION = "NO_SCALING_SIGNAL"

FROZEN_ORACLE_V1_3_DIGEST = (
    "0592876e7768a627198f1154da64b4ed492237fe68196e011fcbfcfef7706e63"
)
FROZEN_ORACLE_V1_3_CONTRACT_DIGEST = (
    "93532f22a0cbc0e57ccdab3d5c01419cd824bc402d637738c5004eb621c23a89"
)
FROZEN_PROGRESS_DIGEST = (
    "840328d918f446bad1a5855e72f13f8937fc9a42eafd87818bf8cd94305e2c3d"
)
FROZEN_SAFETY_DIGEST = (
    "5cf4572be2490c1b6f748abc704fff3a3c15fb1ea8dc060e49314e2bbaf01e0f"
)
FROZEN_COMPLETION_DIGEST = (
    "40913aa993358d99446b38d1f18c766540331d8584da0e7eeb6415806119357f"
)
FROZEN_CANDIDATE_BANK_DIGEST = (
    "85471e44a0fe8f3c59fff258e9b23933e306f69b6d590c832e2b8da1f34a8cd9"
)
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
FROZEN_STATE_TRACE_BANK_DIGEST = (
    "f5a3ccb27c8b3a38de2d1bb5e12fd7164072d4379972925bcbd0433fef97955e"
)
FROZEN_TRAINING_VIEW_DIGEST = (
    "9eefff24953fdfc1eb7718ff6067a9bc06f5f8bd321f62769521234d6393291c"
)
FROZEN_TARGET_ENCODER_DIGEST = (
    "15ff78a0205ba138a740f12f6eb9bb3f78bce9c5ba8c2849f7e83489a6b2b6a5"
)
FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256 = (
    "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
)
FROZEN_LATENT_INDEX_DIGEST = (
    "25bbd7731fc2e3026063544c64d31abff2c0ded43991504eab4d11938401b758"
)
FROZEN_FIT_STATE_IDENTITY_SET_DIGEST = (
    "858ad55b14d0079ea11c49a1c79b2245c7adb71846493c449e7eb3cf1d16900a"
)
FROZEN_FRESH_CALIBRATION_STATE_IDENTITY_SET_DIGEST = (
    "730e4a4835f748ad28f1fae9422c8613d8fd56a2afe0135720842c7203c04b7c"
)
FROZEN_FRESH_CALIBRATION_MANIFEST_DIGEST = (
    "9eaebfd78da51b2d072c34e8c725ad9a75a73f2888537f347a1e682a0a57b30b"
)
FROZEN_HISTORICAL_CALIBRATION_DISPOSITION_DIGEST = (
    "8e8b7aba9f55c62ec1fbefffafc324794df564234d348ed6a8f35e6afb3d072a"
)
FROZEN_HISTORICAL_CALIBRATION_IDENTITY_PROJECTION_DIGEST = (
    "143f736376c7ef03b0f943670de79fcf69cc2b198c601b13188efc487d35b65f"
)

FROZEN_VITL_FINAL_CHECKPOINT_SHA256 = (
    "83a57e61808aa6a23b5a56ba428b7dc926932ea14067c27995ebfc365ed7ec8b"
)
FROZEN_VITL_FINAL_STATE_DIGEST = (
    "bb0f947cca8c724961f3bf98a2d717854e038e2625297019bc1b0957e4896874"
)
FROZEN_VITL_FAILURE_ARTIFACT_SHA256 = (
    "c9959582a0c8f266133623d143a6679a6502cad7a06fc9604d0a1a75ade174ef"
)
FROZEN_VITL_QUALIFICATION_TERMINAL_DIGEST = (
    "441f52d4199ba152825f30a9f5422b80537f68b9f7a3633f4e01610f964de419"
)
FROZEN_BASELINE_CHECKPOINT_SHA256 = (
    "cfd07d2ad739ef884f3d8ebc3faa01a0b807ef6f19049874eb7fc6ecc9c418ca"
)
FROZEN_BASELINE_STATE_DIGEST = (
    "33e7bcffbfab16371fb8e7e233490c33c442336edac823c19733214fa87d91d1"
)
FROZEN_BASELINE_RECEIPT_DIGEST = (
    "454bc81c3077d62cac661a4ccac7212b3eb3860eda3177f9b8879f27632abc25"
)

EXPECTED_STATES = 24
EXPECTED_BRANCHES = 288
ADOPTED_TRACES = 0
REPLAY_TRACES = 288
POLICY_TICKS = tuple(range(20))
HORIZON_SAMPLE_TICKS = (4, 9, 14, 19)
FIT_STATES = 96
FIT_ROWS = 1_152
FRESH_CALIBRATION_STATES = 24
FRESH_CALIBRATION_ROWS = 288
DATA_ORDER_SEED = 20_260_811
TOKENS = 768
TOKEN_DIM = 1024
HORIZONS = 4
HIDDEN_DIM = 512

HISTORICAL_CALIBRATION_STATE_IDENTITY_DIGESTS = (
    "0b9a6eb9a429145dc441bf15d0a8dd38e25cff1c501c4420cddb1da1a83296cc",
    "17cdc4f171251f9728d8f12a3fcf1f60966fdfe7d0669c91a5476952afc0093b",
    "18108441dd1a56a4e76f59c9b151ac71726a7a149bbde97553b78dd9091878f4",
    "1cfccd331434a6d8350dfae42c71abfddf3c38d1a5f817e0483ab90345170e6c",
    "220525522bfd3054dbcdde8c5338d6f1459d46443457798e27682f85343c3b8f",
    "24676131f0c9c03e7e8496b7481d216d8b7bf5a2462fb380729628daf20d876d",
    "24a28ab7ffeca55088cdac2f7eb2b9b79313c4706f9852c5f8c954d4dbbdc0e3",
    "2d4d0a270a3db9baf12f621b08090bb1281c24dfbef063009f365064e44fa659",
    "2e1b7f43e028eee7b162cf4fbbaf1981b85e9266c7b91bc218c456ca3c60ed74",
    "38a1bbf11b5aab346790d5f3973b4a8f45f6094ba87bbdd861349e0ebe41a05e",
    "3ca257fb07956a88a0101a01cf67549be02f415df3ab676b4a5b06260258991a",
    "451efe30e767df2f0d2d5cb8cc0b7813c36b4f66aff5fb6c3a9fc3a8284dd015",
    "5ba6a01d358195db42fb265c4e406bc61347077f6e95179de9479435126fb441",
    "5cce57d3f1e709e02beadadb351c18aeefb13d1b49a41260246ba1a57c8fd80e",
    "7750a982f03e5ef237230d1ebae5a96ac284f5b6b47476404812b89b348c565e",
    "7a0d0c4886f9e59c827be19b2e35d70c92ff233e47543fb9e999e31299e10bb5",
    "8d9287576881e9b006b88cc6c4ab4dc4ae0152d7cc7c33d7acff2cd99791651c",
    "9237d8db4c775c0a03b0f0391df05a50eb9518ee0a93c7b15bb67305f7e9a830",
    "95e8fb1b34724192fec7a3e2cb9a700cf4d472ba62ae348de7b34f3cec87a976",
    "9c3940c1a512f4ddaa4f29d7a4f05b210bdf6dbed212d26f94ced72301ed3235",
    "aebe821f5d1361977bae8e99d816a04a38ef08cb2b0bcf5ce8621ce27f321b5d",
    "b8c1cd6e866610bdef0af28f8b444823d891ff28a417000c8c300d77827f24c9",
    "be7401dcbf89c27cc9766a131ebae6d77019b190819e0bb901aae379d6eb5ebb",
    "fbce3ad81c19ffb1a6a57573403077a86b3cf9162da931064cb2b5c221ce45b2",
)
FROZEN_HISTORICAL_CALIBRATION_STATE_IDENTITY_SET_DIGEST = (
    "577a105272e64e62030ec12e40ed38027afa6bbdf4dfb327a9e699154dd6b89a"
)


class ScorerFailureAttributionContractError(ValueError):
    """The source closure or frozen diagnostic contract is malformed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScorerFailureAttributionContractError(message)


def _require_digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
             f"{label} is not a lowercase SHA-256 digest")
    return str(value)


CURRENT_SCORER_ARCHITECTURE = {
    "classification": "GLOBAL_OR_FIXED_POOLING",
    "raw_latent_shape": ["batch", 4, 768, 1024],
    "model_latent_shape": ["batch", 4, 1024],
    "spatial_aggregation": (
        "arithmetic mean over 768 tokens before every learned module"
    ),
    "spatial_token_order_preserved": False,
    "per_horizon_projection": [1024, 512, 512],
    "per_horizon_activation": "SiLU",
    "temporal_aggregation": (
        "shared scalar softmax attention followed by weighted sum"
    ),
    "horizon_order_explicit": False,
    "horizon_permutation_invariant": True,
    "self_attention": False,
    "cross_attention": False,
    "action_input": "40 exact post-slew values",
    "goal_input": "3-vector [sin(bearing), cos(bearing), range_m]",
    "context_projection": [43, 512, 512],
    "fusion": [1024, 512],
    "shared_component_representation": True,
    "heads": {
        "progress": "linear identity output",
        "safety": "linear logit; sigmoid outside model",
        "completion": "linear logit; sigmoid outside model",
    },
    "parameter_counts": {
        "per_horizon": 787_456,
        "temporal_attention": 513,
        "context": 285_184,
        "fusion": 524_800,
        "progress": 513,
        "safety": 513,
        "completion": 513,
        "total": 1_599_492,
    },
    "duplicate_equivalent_to_permitted_attentive_readout": False,
}

OFFICIAL_ATTENTIVE_POOLER_BINDING = {
    "repository": "facebookresearch/vjepa2",
    "commit": "204698b45b3712590f06245fbfba32d3be539812",
    "binding_digest": (
        "f436439c72e725bfd7f3caab517f2b7c870cac1cf4060623fe0c1f6da63591e6"
    ),
    "files": {
        "src/models/attentive_pooler.py": {
            "sha256": "9be7047d6bfce50575956a57e36d87a37bf63ae84ec92a9ba8649bf1ab7d5feb",
            "byte_count": 4_372,
        },
        "src/models/utils/modules.py": {
            "sha256": "b93f6c7e0747deb216419c000c2878f11a9189024a9adeacfd437e172396dff0",
            "byte_count": 23_001,
        },
        "src/utils/tensors.py": {
            "sha256": "782b58bd2af456e184750e5318ab773105108383f61b280fe4c7a90f46add2c8",
            "byte_count": 1_832,
        },
        "configs/eval_2_1/vitl-384/in1k.yaml": {
            "sha256": "c9e378792ae3437ca77d3c9d6f7ff3f448128312cca34c25b4718a1365937129",
            "byte_count": 3_735,
        },
        "evals/image_classification_frozen/eval.py": {
            "sha256": "ff35b2729d45fc6b212275bec580704673b69058b00064f6e54b90e01e1a50e0",
            "byte_count": 15_577,
        },
    },
    "config": {
        "embed_dim": 512,
        "depth": 4,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "norm_layer": "torch.nn.LayerNorm",
        "norm_eps": 1e-5,
        "activation": "GELU",
        "qkv_bias": True,
        "complete_block": True,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "drop_path": 0.0,
        "init_std": 0.02,
        "use_activation_checkpointing": True,
    },
    "rectangular_sequence_compatible": True,
    "grid_assumption": None,
    "dependency_compatibility": {
        "missing_dependency": "timm",
        "official_import_used": "timm.models.layers.drop_path",
        "permitted_shim": "standard timm drop_path semantics only",
        "reason": (
            "the pinned configuration fixes drop_path=0.0, so every official "
            "block must install torch.nn.Identity and the shim is unreachable"
        ),
        "runtime_assert_every_official_block_drop_path_is_identity": True,
        "shim_execution_expected": False,
        "attention_or_model_mathematics_patch_permitted": False,
        "environment_dependency_mutation_permitted": False,
    },
}


def official_attentive_pooler_binding_payload(
        binding: Mapping[str, Any] = OFFICIAL_ATTENTIVE_POOLER_BINDING,
        ) -> dict[str, Any]:
    """Return the exact digest preimage for the pinned official pooler.

    Compatibility and adaptation statements are contract assertions about
    how the source is consumed; they are deliberately not part of the source
    and configuration identity preimage.
    """

    _require(isinstance(binding, Mapping),
             "official attentive-pooler binding is not a mapping")
    _require(set(binding) == {
        "repository", "commit", "binding_digest", "files", "config",
        "rectangular_sequence_compatible", "grid_assumption",
        "dependency_compatibility",
    }, "official attentive-pooler binding schema changed")
    files = binding["files"]
    config = binding["config"]
    _require(isinstance(files, Mapping) and isinstance(config, Mapping),
             "official attentive-pooler files or config changed type")
    payload = {
        "repository": binding["repository"],
        "commit": binding["commit"],
        "files": {
            str(path): dict(value) for path, value in files.items()
        },
        "config": dict(config),
    }
    _require_digest(binding["binding_digest"],
                    "official attentive-pooler binding digest")
    _require(canonical_digest(payload) == binding["binding_digest"],
             "official attentive-pooler digest preimage changed")
    return payload


OFFICIAL_ATTENTIVE_POOLER_BINDING_PAYLOAD = (
    official_attentive_pooler_binding_payload()
)

DERANGEMENT_NAMESPACE = (
    "EXPLORATORY_SCORER_FAILURE_ATTRIBUTION|B|within-state-derangement|v1"
)
SPATIAL_PERMUTATION_NAMESPACE = (
    "EXPLORATORY_SCORER_FAILURE_ATTRIBUTION|D|spatial-token-permutation|768|v1"
)


def within_state_candidate_derangement(
        state_identity_digest: str,
        candidate_identity_digests: Sequence[str],
        ) -> dict[str, str]:
    """Return the sole hash-ordered cyclic derangement for one 12-row state."""

    _require_digest(state_identity_digest, "state identity")
    candidates = list(candidate_identity_digests)
    _require(len(candidates) == 12 and len(set(candidates)) == 12,
             "a candidate derangement requires twelve unique identities")
    for index, value in enumerate(candidates):
        _require_digest(value, f"candidate identity {index}")
    ordered = sorted(candidates, key=lambda value: (
        hashlib.sha256(canonical_bytes({
            "namespace": DERANGEMENT_NAMESPACE,
            "state_identity_digest": state_identity_digest,
            "candidate_identity_digest": value,
        })).digest(), value))
    return {value: ordered[(index + 1) % len(ordered)]
            for index, value in enumerate(ordered)}


def _spatial_token_permutation() -> tuple[int, ...]:
    return tuple(sorted(range(TOKENS), key=lambda index: (
        hashlib.sha256(canonical_bytes({
            "namespace": SPATIAL_PERMUTATION_NAMESPACE,
            "token_index": index,
        })).digest(), index)))


SPATIAL_TOKEN_PERMUTATION = _spatial_token_permutation()
SPATIAL_TOKEN_PERMUTATION_DIGEST = canonical_digest(
    list(SPATIAL_TOKEN_PERMUTATION))
FROZEN_SPATIAL_TOKEN_PERMUTATION_DIGEST = (
    "4585b86cd8978197298b4d865bc7e29cbb9b8d99c9cab54bf8d5851e00cb340a"
)
_require(SPATIAL_TOKEN_PERMUTATION_DIGEST
         == FROZEN_SPATIAL_TOKEN_PERMUTATION_DIGEST,
         "frozen spatial-token permutation changed")

FIT_MEAN_TRAJECTORY_CONTRACT = {
    "source_split": "fit",
    "states": FIT_STATES,
    "rows": FIT_ROWS,
    "calibration_statistics_used": False,
    "input_shape_per_row": [4, 768, 1024],
    "input_storage_dtype": "float16",
    "canonical_row_order": (
        "ascending (state_id, candidate_index, branch_identity_digest)"
    ),
    "accumulation_dtype": "float64",
    "division_count": FIT_ROWS,
    "materialised_dtype": "float32",
    "output_shape": [4, 768, 1024],
}

TRANSFORMATION_SUITE = {
    "A_MATCHED": {
        "latent": "the branch's exact H1-H4 trajectory",
        "action_goal_unchanged": True,
    },
    "B_WITHIN_STATE_CANDIDATE_DERANGEMENT": {
        "algorithm": (
            "hash-order twelve branch identities using the frozen namespace; "
            "map each ordered identity to its cyclic successor"
        ),
        "namespace": DERANGEMENT_NAMESPACE,
        "fixed_points": 0,
        "state_family_goal_and_action_inputs_unchanged": True,
        "action_goal_unchanged": True,
    },
    "C_HORIZON_REVERSAL": {
        "source_horizons": [4, 3, 2, 1],
        "action_goal_unchanged": True,
    },
    "D_FIXED_SPATIAL_TOKEN_PERMUTATION": {
        "token_count": TOKENS,
        "same_permutation_for_all_rows_and_horizons": True,
        "namespace": SPATIAL_PERMUTATION_NAMESPACE,
        "permutation_digest": SPATIAL_TOKEN_PERMUTATION_DIGEST,
        "action_goal_unchanged": True,
    },
    "E_SPATIAL_MEAN_REPEATED": {
        "operation": (
            "at each horizon compute the arithmetic mean of 768 tokens in "
            "float32 and repeat it at all 768 positions"
        ),
        "action_goal_unchanged": True,
    },
    "F_FIT_SET_MEAN_TRAJECTORY": {
        "statistic": FIT_MEAN_TRAJECTORY_CONTRACT,
        "same_trajectory_for_every_evaluation_row": True,
        "action_goal_unchanged": True,
    },
    "G_SINGLE_HORIZON": {
        "conditions": ["H1", "H2", "H3", "H4"],
        "observed_slot": "the matching branch horizon",
        "other_slots": "their corresponding fit-set mean horizon",
        "absence_representation_used": False,
        "action_goal_unchanged": True,
    },
}

DIAGNOSTIC_METRICS = (
    "progress_mae", "progress_rmse", "progress_spearman",
    "safety_mae", "safety_rmse", "safety_auc", "safety_ece",
    "completion_prevalence", "completion_mae", "completion_brier",
    "completion_auc", "completion_ece", "absolute_rank_regret",
    "normalized_rank_regret", "realised_selected_utility",
    "pairwise_ordering_accuracy", "ranking_spearman", "top_1_recovery",
    "top_3_recovery", "candidate_score_spread", "top_score_tie_rate",
    "all_pair_tie_rate", "per_family_values",
)

SAFETY_OBSERVABILITY_CONTRACT = {
    "states": EXPECTED_STATES,
    "candidates_per_state": 12,
    "branches": EXPECTED_BRANCHES,
    "policy_ticks": list(POLICY_TICKS),
    "sample_ticks": list(HORIZON_SAMPLE_TICKS),
    "prior_trace_references_compared_as_lineage": 12,
    "adopt_complete_preserved_traces": ADOPTED_TRACES,
    "diagnostic_replays": REPLAY_TRACES,
    "state_replacement": False,
    "candidate_replacement": False,
    "target_latent_encoding": False,
    "required_tick_fields": [
        "contact_indicator", "contact_type", "clearance_m",
        "normalized_clearance_deficit", "stuck", "fall_or_unsafe_termination",
        "completion",
    ],
    "safety_mass_attribution": {
        "base": "(C_contact + C_clearance + C_stuck) / 3",
        "contact": "C_contact / 3",
        "clearance": "C_clearance / 3",
        "stuck": "C_stuck / 3",
        "fall": "max(0, F - base)",
        "sum": "contact + clearance + stuck + fall = S",
    },
    "report_by": ["overall", "family", "state_stratum"],
}


def safety_mass_attribution(*, contact: float, clearance: float,
                            stuck: float, fall: float,
                            safety: float) -> dict[str, float]:
    """Partition the frozen outer-max safety target without double counting."""

    values = (contact, clearance, stuck, fall, safety)
    _require(all(isinstance(value, (int, float)) and math.isfinite(value)
                 and 0.0 <= float(value) <= 1.0 for value in values),
             "safety components must be finite values in [0,1]")
    base = (float(contact) + float(clearance) + float(stuck)) / 3.0
    expected = max(float(fall), base)
    _require(abs(expected - float(safety)) <= 1e-12,
             "component inputs do not reproduce the frozen safety target")
    result = {
        "contact": float(contact) / 3.0,
        "clearance": float(clearance) / 3.0,
        "stuck": float(stuck) / 3.0,
        "fall": max(0.0, float(fall) - base),
    }
    _require(abs(sum(result.values()) - float(safety)) <= 1e-12,
             "safety attribution does not sum to S")
    return result


ATTENTIVE_SEED_KEY = {
    "namespace": "go2_final_layer_attentive_readout_v1",
    "base_registered_seed": 20_260_811,
    "input_shape": [4, 768, 1024],
    "token_projection": [1024, 512],
    "component_queries": ["progress", "safety", "completion"],
    "official_pooler_binding_digest": OFFICIAL_ATTENTIVE_POOLER_BINDING[
        "binding_digest"],
}
ATTENTIVE_SEED_KEY_DIGEST = canonical_digest(ATTENTIVE_SEED_KEY)
ATTENTIVE_SEED = int(ATTENTIVE_SEED_KEY_DIGEST[:16], 16) % (2 ** 31)
FROZEN_ATTENTIVE_SEED_KEY_DIGEST = (
    "29b8e09b3f63487485abbee3f5b71f1c71a84f9ec5a67fa2b7eb93e9acf5363b"
)
FROZEN_ATTENTIVE_SEED = 1_063_471_220
_require(ATTENTIVE_SEED_KEY_DIGEST == FROZEN_ATTENTIVE_SEED_KEY_DIGEST
         and ATTENTIVE_SEED == FROZEN_ATTENTIVE_SEED,
         "architecture-keyed attentive seed changed")


def horizon_embedding_float32_bytes() -> bytes:
    """Canonical non-trainable H1-H4 sinusoid, row-major little-endian f32."""

    values = bytearray()
    for horizon_index in range(HORIZONS):
        for pair in range(HIDDEN_DIM // 2):
            scale = 10_000.0 ** (2.0 * pair / HIDDEN_DIM)
            angle = horizon_index / scale
            values.extend(struct.pack("<f", math.sin(angle)))
            values.extend(struct.pack("<f", math.cos(angle)))
    return bytes(values)


HORIZON_EMBEDDING_SHA256 = hashlib.sha256(
    horizon_embedding_float32_bytes()).hexdigest()
FROZEN_HORIZON_EMBEDDING_SHA256 = (
    "aea9cfadd234a5b4ed1ce151d7c65fa0f5733cc1df246f81848057d895de25aa"
)
_require(HORIZON_EMBEDDING_SHA256 == FROZEN_HORIZON_EMBEDDING_SHA256,
         "frozen horizon embedding bytes changed")

ATTENTIVE_READOUT_ARCHITECTURE = {
    "label": "EXPLORATORY_FINAL_LAYER_ATTENTIVE_READOUT",
    "input": "existing final-layer ViT-L dense tokens only",
    "input_shape": ["batch", 4, 768, 1024],
    "token_projection": [1024, 512],
    "token_projection_bias": True,
    "spatial_token_order": "existing 24x32 horizon-local order",
    "horizon_embedding": {
        "shape": [4, 512],
        "formula": (
            "position=0..3; even[2i]=sin(position/10000^(2i/512)); "
            "odd[2i+1]=cos(position/10000^(2i/512))"
        ),
        "dtype": "float32",
        "trainable": False,
        "row_major_little_endian_sha256": HORIZON_EMBEDDING_SHA256,
    },
    "flattened_shape": ["batch", 3072, 512],
    "pooler": OFFICIAL_ATTENTIVE_POOLER_BINDING,
    "component_queries": ["progress", "safety", "completion"],
    "component_query_count": 3,
    "query_outputs_shape": ["batch", 3, 512],
    "context": "unchanged 43->512->512 SiLU action-and-goal MLP",
    "fusion": (
        "apply the unchanged shared 1024->512 SiLU fusion independently to "
        "each component query concatenated with the common context"
    ),
    "heads": CURRENT_SCORER_ARCHITECTURE["heads"],
    "utility_weights": {"progress": 1.0, "safety": -2.0,
                        "completion": 0.5},
    "intermediate_encoder_layers": False,
    "encoder_fine_tuning": False,
    "spatial_position_table_added": False,
    "official_pooler_parameter_count": 12_348_416,
    "trainable_parameter_count": 13_684_739,
    "registered_seed": ATTENTIVE_SEED,
    "registered_seed_key_digest": ATTENTIVE_SEED_KEY_DIGEST,
    "initialisation": {
        "construction_seed": ATTENTIVE_SEED,
        "scope": "every trainable parameter in the attentive scorer",
        "official_pooler": "official init_weights under the construction seed",
        "remaining_modules": (
            "deterministic PyTorch construction under the same seed"
        ),
        "frozen_vitl_initial_artefact": "lineage binding only",
        "copy_frozen_vitl_parameter_tensors": False,
        "mixed_initialisation": False,
    },
}

DATA_ORDER_CONTRACT = {
    "base_order": "state_id_then_candidate_index",
    "base_training_view_row_digest_sequence_digest": (
        "c862d0814efb0cbac179eedf9835d869a4dd3588e66c2df668feb44e469e1296"
    ),
    "generator": "torch.Generator(device='cpu')",
    "seed": DATA_ORDER_SEED,
    "algorithm": "torch.randperm(1152, generator=generator)",
    "rows": FIT_ROWS,
    "epochs": 60,
    "batch_size": 64,
    "updates_per_epoch": 18,
    "permutation_plan_digest": (
        "8e0f2c195f57fa3b883bb8830a4067f95e7965716c851be31b369d5e997c255d"
    ),
    "row_presentation_plan_digest": (
        "85b1b96ad3aab1442c71a90e6afdbb3e3dc87e8115cb0f9c127953531f7efefb"
    ),
    "recomputation_boundary": (
        "recompute from the exact fit rows using the frozen V1.3 witness "
        "function and require every field above before attempt creation"
    ),
}

TRAINING_CONTRACT = {
    "fit_states": FIT_STATES,
    "fit_rows": FIT_ROWS,
    "calibration_states": FRESH_CALIBRATION_STATES,
    "calibration_rows": FRESH_CALIBRATION_ROWS,
    "epochs": 60,
    "batch_size": 64,
    "effective_batch_size": 64,
    "microbatch_size": 4,
    "gradient_accumulation_steps": 16,
    "loss_reduction": "sum per microbatch divided by 64",
    "optimizer_step": (
        "one gradient clip followed by one AdamW step per effective batch"
    ),
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "learning_rate_schedule": "constant",
    "optimizer_updates_per_epoch": 18,
    "optimizer_updates": 1_080,
    "example_presentations": 69_120,
    "registered_seed": ATTENTIVE_SEED,
    "model_construction_seed": ATTENTIVE_SEED,
    "data_order_seed": DATA_ORDER_SEED,
    "data_order": DATA_ORDER_CONTRACT,
    "attempts": 1,
    "final_epoch_only": True,
    "best_epoch_selection": False,
    "calibration_access_during_training": False,
    "no_latent_baseline_retrained": False,
}

DIAGNOSTIC_PREREQUISITES = {
    "execution_order": [
        "safety_observability",
        "latent_dependence",
        "attentive_readout",
    ],
    "safety_observability": {
        "terminal_path": str(SAFETY_OBSERVABILITY_ROOT / "terminal.json"),
        "terminal_schema":
            "go2_safety_observability_diagnostic_v1_trace_manifest_v1",
        "terminal_self_key": "diagnostic_trace_manifest_digest",
        "audit_path": str(SAFETY_OBSERVABILITY_ROOT / "audit.json"),
        "audit_schema": "go2_safety_observability_diagnostic_v1_audit_v1",
        "audit_self_key": "safety_observability_audit_digest",
        "required_states": EXPECTED_STATES,
        "required_branches": EXPECTED_BRANCHES,
        "required_complete_tick_rows": EXPECTED_BRANCHES * len(POLICY_TICKS),
    },
    "latent_dependence": {
        "result_path": str(GENERATED_ROOT / "latent_dependence/result.json"),
        "result_schema":
            "go2_scorer_v1_3_latent_dependence_diagnostic_result_v1",
        "result_self_key": "latent_dependence_result_digest",
        "required_variants": [
            "A_matched", "B_within_state_candidate_derangement",
            "C_horizon_reversed", "D_fixed_token_permutation",
            "E_spatial_mean_repeated", "F_fit_mean_trajectory",
            "G_H1_only", "G_H2_only", "G_H3_only", "G_H4_only",
        ],
        "required_calibration_rows_per_variant": FRESH_CALIBRATION_ROWS,
    },
    "validation_boundary": (
        "both completed self-digest-valid diagnostics must bind this exact "
        "diagnostic contract and source closure before attentive model "
        "construction, attempt creation, training, or calibration evaluation"
    ),
}

INTERPRETATION_RULES = {
    "primary_quantities": [
        "safety_auc", "latent_over_baseline_pairwise_ordering_gain",
    ],
    "strong_requires": [
        "both original primary thresholds met",
        "both primary quantities improve globally over existing ViT-L",
        "no inconsistent per-family primary improvement",
    ],
    "per_family_consistency": {
        "comparison": "attentive minus existing ViT-L",
        "pairwise_gain_reference": "the unchanged no-latent baseline",
        "evaluable_value": "both compared values are finite",
        "inconsistent_if": (
            "any evaluable family has a strictly negative delta for either "
            "safety AUC or latent-over-baseline pairwise ordering gain"
        ),
        "post_hoc_tolerance": None,
        "undefined_family_values": "report explicitly and exclude from sign test",
    },
}

ORIGINAL_GATE_REPLAY = {
    "progress_spearman_min": 0.50,
    "safety_auc_min": 0.75,
    "safety_ece_max": 0.10,
    "completion_auc_min": 0.75,
    "completion_ece_max": 0.10,
    "completion_nondegenerate_fit_and_calibration": True,
    "composite_pairwise_accuracy_min": 0.65,
    "latent_over_baseline_pairwise_gain_min": 0.05,
    "tie_tolerance": 0.02,
    "result_is_exploratory_not_qualification": True,
}

STOPPING_RULES = {
    "current_readout_equivalent": (
        "run safety observability and latent dependence only; do not train"
    ),
    "official_pooler_rectangular_incompatible": "stop before training",
    "attentive_training_attempts": 1,
    "attentive_evaluations": 1,
    "additional_probe_architectures": False,
    "new_qualification_data": False,
    "predictor_checkpoint_or_utility_path": False,
    "predictor_retraining": False,
    "vitg_or_vitG_training": False,
    "final_200_state_corpus": False,
    "oracle_change": False,
}


def validate_source_closure(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, but never discover or open, the runner-supplied closure."""

    expected_keys = {
        "schema", "source_repository_commit", "source_repository_clean",
        "git_status_porcelain_v1", "files", SOURCE_CLOSURE_SELF_KEY,
    }
    _require(isinstance(value, Mapping) and set(value) == expected_keys,
             "source closure schema is not closed")
    closure = dict(value)
    _require(closure["schema"] == SOURCE_CLOSURE_SCHEMA,
             "source closure schema changed")
    _require(isinstance(closure["source_repository_commit"], str)
             and HEX40.fullmatch(closure["source_repository_commit"]) is not None,
             "source closure commit is malformed")
    _require(closure["source_repository_clean"] is True
             and closure["git_status_porcelain_v1"] == "",
             "diagnostic source must be clean and committed")
    files = closure["files"]
    _require(isinstance(files, Mapping)
             and set(files) == set(SOURCE_CLOSURE_PATHS),
             "source closure is not the exact eight-path allowlist")
    normalized: dict[str, dict[str, Any]] = {}
    for path in SOURCE_CLOSURE_PATHS:
        parts = Path(path).parts
        _require("sealed" not in parts
                 and not any(part.startswith("sealed_") for part in parts)
                 and "predictor" not in Path(path).name.lower(),
                 f"forbidden custody path in source closure: {path}")
        binding = files[path]
        _require(isinstance(binding, Mapping)
                 and set(binding) == {"path", "sha256", "byte_count"},
                 f"source binding schema changed: {path}")
        _require(binding["path"] == path,
                 f"source binding path mismatch: {path}")
        _require_digest(binding["sha256"], f"source SHA-256 {path}")
        _require(type(binding["byte_count"]) is int
                 and binding["byte_count"] > 0,
                 f"source byte count is invalid: {path}")
        normalized[path] = dict(binding)
    unsigned = {key: closure[key] for key in expected_keys
                if key != SOURCE_CLOSURE_SELF_KEY}
    _require(closure[SOURCE_CLOSURE_SELF_KEY] == canonical_digest(unsigned),
             "source closure self digest changed")
    closure["files"] = normalized
    return closure


def _static_contract() -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "status": STATUS,
        "scientific_claim_status": "exploratory_only_already_examined_data",
        "frozen_lineage": {
            "source_base_commit": SOURCE_BASE_COMMIT,
            "historical_v2_source_commit": SOURCE_COMMIT,
            "vitl_scorer_source_lineage": FROZEN_VITL_SCORER_SOURCE_LINEAGE,
            "oracle_v1_3_digest": FROZEN_ORACLE_V1_3_DIGEST,
            "oracle_v1_3_contract_digest": FROZEN_ORACLE_V1_3_CONTRACT_DIGEST,
            "progress_digest": FROZEN_PROGRESS_DIGEST,
            "safety_digest": FROZEN_SAFETY_DIGEST,
            "completion_digest": FROZEN_COMPLETION_DIGEST,
            "candidate_bank_digest": FROZEN_CANDIDATE_BANK_DIGEST,
            "corpus_digest": FROZEN_CORPUS_DIGEST,
            "branch_rows_sha256": FROZEN_BRANCH_ROWS_SHA256,
            "state_manifest_digest": FROZEN_STATE_MANIFEST_DIGEST,
            "assignment_manifest_digest": FROZEN_ASSIGNMENT_MANIFEST_DIGEST,
            "branch_identity_set_digest": FROZEN_BRANCH_IDENTITY_SET_DIGEST,
            "state_trace_bank_digest": FROZEN_STATE_TRACE_BANK_DIGEST,
            "training_view_digest": FROZEN_TRAINING_VIEW_DIGEST,
            "target_encoder_digest": FROZEN_TARGET_ENCODER_DIGEST,
            "target_encoder_checkpoint_sha256":
                FROZEN_TARGET_ENCODER_CHECKPOINT_SHA256,
            "latent_index_digest": FROZEN_LATENT_INDEX_DIGEST,
        },
        "identity_sets": {
            "fit": {
                "state_count": FIT_STATES,
                "identity_set_digest": FROZEN_FIT_STATE_IDENTITY_SET_DIGEST,
            },
            "fresh_calibration": {
                "state_count": FRESH_CALIBRATION_STATES,
                "identity_set_digest":
                    FROZEN_FRESH_CALIBRATION_STATE_IDENTITY_SET_DIGEST,
                "manifest_digest": FROZEN_FRESH_CALIBRATION_MANIFEST_DIGEST,
                "already_examined": True,
            },
            "historical_development_only": {
                "state_count": EXPECTED_STATES,
                "state_identity_digests":
                    list(HISTORICAL_CALIBRATION_STATE_IDENTITY_DIGESTS),
                "identity_set_digest":
                    FROZEN_HISTORICAL_CALIBRATION_STATE_IDENTITY_SET_DIGEST,
                "identity_projection_digest":
                    FROZEN_HISTORICAL_CALIBRATION_IDENTITY_PROJECTION_DIGEST,
                "disposition_digest":
                    FROZEN_HISTORICAL_CALIBRATION_DISPOSITION_DIGEST,
                "qualification_eligible": False,
                "discarded": False,
            },
        },
        "frozen_scorers": {
            "vitl": {
                "checkpoint_sha256": FROZEN_VITL_FINAL_CHECKPOINT_SHA256,
                "state_digest": FROZEN_VITL_FINAL_STATE_DIGEST,
                "failure_artifact_sha256":
                    FROZEN_VITL_FAILURE_ARTIFACT_SHA256,
                "terminal_digest":
                    FROZEN_VITL_QUALIFICATION_TERMINAL_DIGEST,
                "safety_auc": 0.7043234199,
                "latent_over_baseline_pairwise_gain": 0.0317880795,
                "terminal": "valid scientific qualification failure",
            },
            "vitg_scale_ablation": {
                "source_head": FROZEN_VITG_SOURCE_HEAD,
                "result_digest": FROZEN_VITG_RESULT_DIGEST,
                "safety_auc": 0.6332379770,
                "latent_over_baseline_pairwise_gain": 0.0019867550,
                "conclusion": FROZEN_VITG_CONCLUSION,
            },
            "no_latent_baseline": {
                "checkpoint_sha256": FROZEN_BASELINE_CHECKPOINT_SHA256,
                "state_digest": FROZEN_BASELINE_STATE_DIGEST,
                "receipt_digest": FROZEN_BASELINE_RECEIPT_DIGEST,
                "retrain": False,
            },
        },
        "current_scorer_architecture": CURRENT_SCORER_ARCHITECTURE,
        "safety_observability": SAFETY_OBSERVABILITY_CONTRACT,
        "transformation_suite": TRANSFORMATION_SUITE,
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "official_attentive_pooler": OFFICIAL_ATTENTIVE_POOLER_BINDING,
        "attentive_readout_architecture": ATTENTIVE_READOUT_ARCHITECTURE,
        "attentive_training": TRAINING_CONTRACT,
        "diagnostic_prerequisites": DIAGNOSTIC_PREREQUISITES,
        "interpretation_rules": INTERPRETATION_RULES,
        "original_gate_replay": ORIGINAL_GATE_REPLAY,
        "stopping_rules": STOPPING_RULES,
        "source_closure_paths": list(SOURCE_CLOSURE_PATHS),
        "output_root": str(GENERATED_ROOT),
        "registered_output_target": str(REGISTERED_GENERATED_TARGET_ROOT),
    }


def build_contract(source_closure: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the frozen science to an externally checked clean source closure."""

    payload = _static_contract()
    payload["source_closure"] = validate_source_closure(source_closure)
    payload[CONTRACT_SELF_KEY] = canonical_digest(payload)
    return payload


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "diagnostic contract is not a mapping")
    payload = dict(value)
    _require(CONTRACT_SELF_KEY in payload, "diagnostic contract digest is absent")
    observed = payload.pop(CONTRACT_SELF_KEY)
    _require_digest(observed, "diagnostic contract digest")
    _require(observed == canonical_digest(payload),
             "diagnostic contract self digest changed")
    source = validate_source_closure(payload.get("source_closure", {}))
    expected = build_contract(source)
    _require(dict(value) == expected, "diagnostic contract fields changed")
    return expected


def contract(source_closure: Mapping[str, Any]) -> dict[str, Any]:
    return build_contract(source_closure)


def contract_digest(source_closure: Mapping[str, Any]) -> str:
    return build_contract(source_closure)[CONTRACT_SELF_KEY]


__all__ = [name for name in globals() if name.isupper()] + [
    "ScorerFailureAttributionContractError", "canonical_bytes",
    "canonical_digest", "within_state_candidate_derangement",
    "safety_mass_attribution", "horizon_embedding_float32_bytes",
    "validate_source_closure", "build_contract", "validate_contract",
    "contract", "contract_digest",
]
