"""Frozen contract for the bounded attentive-gradient localisation diagnostic.

This is an additive diagnostic authority only.  It preserves the failed
production smoke byte-for-byte, permits forward/backward localisation without
an optimiser step, and cannot authorise a scorer repair or scientific training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as BASE


ROOT = Path(__file__).resolve().parents[2]
STATUS = "EXPLORATORY_ATTENTIVE_GRADIENT_LOCALISATION"
CONTRACT_SCHEMA = "go2_attentive_readout_gradient_localisation_v1_contract_v1"
CONTRACT_SELF_KEY = "gradient_localisation_contract_digest"
SOURCE_CLOSURE_SCHEMA = (
    "go2_attentive_readout_gradient_localisation_v1_source_closure_v1")
SOURCE_CLOSURE_SELF_KEY = "gradient_localisation_source_closure_digest"
BASE_SOURCE_COMMIT = "121ad99ce0f0321fcf7b71efbc4932ed84163eb3"

NEW_SOURCE_PATHS = (
    "lewm/oracle/go2_attentive_readout_gradient_localisation_v1_contract.py",
    "lewm/tests/test_go2_attentive_readout_gradient_localisation_v1_contract.py",
    "scripts/diagnose_go2_attentive_readout_gradient_localisation_v1.py",
    "lewm/tests/test_diagnose_go2_attentive_readout_gradient_localisation_v1.py",
)
FROZEN_DEPENDENCY_FILES = {
    "lewm/oracle/go2_scorer_failure_attribution_v1_contract.py": (
        "675de988d1f10e1a46676d7a8b89f0502e95fe680cbe51126aec40ba49361ef6",
        39_358),
    "lewm/oracle/go2_scorer_failure_attribution_v1_prerequisite_amendment.py": (
        "9d83ff5bc16a49a74c5ed7625ff91e3ec6540eba3ca2efb1e4eeb023483f9a95",
        28_662),
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_v1.py": (
        "c7f2bd4945a0d39264ac369469a0102caa09d3dc3d5b8fa32021bda040fcb597",
        62_036),
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py": (
        "4eb907fe53da00324a5e8f95181f991038faa4b0455c8db63040dbe4b0ac0a6f",
        90_695),
}

GENERATED_ROOT = BASE.GENERATED_ROOT
REGISTERED_GENERATED_TARGET_ROOT = BASE.REGISTERED_GENERATED_TARGET_ROOT
AMENDMENT_RUNTIME_ROOT = GENERATED_ROOT / "attentive_readout_amendment_v1"
DIAGNOSTIC_RUNTIME_ROOT = AMENDMENT_RUNTIME_ROOT / "gradient_localisation_v1"
CONTRACT_PATH = DIAGNOSTIC_RUNTIME_ROOT / "contract.json"

AMENDMENT_ARTIFACT_RELATIVE = (
    AMENDMENT_RUNTIME_ROOT / "prerequisite_amendment.json")
SMOKE_FAILURE_RELATIVE = AMENDMENT_RUNTIME_ROOT / "production_smoke_failure.json"
SMOKE_WORK_DIRECTORY_RELATIVE = AMENDMENT_RUNTIME_ROOT / "production_smoke"
AMENDMENT_ARTIFACT_DIGEST = (
    "fb6dae0c7363b766d7ab48688fadf14cac8748de766f86d350ff089a5ddb2180")
AMENDMENT_ARTIFACT_SHA256 = (
    "556fd86c94072077c21703fee514c2e5a7d8050f32df5e0d7ad5d212db6179ad")
AMENDMENT_ARTIFACT_BYTE_COUNT = 214_681
AMENDMENT_SOURCE_CLOSURE_DIGEST = (
    "1f233b696a03704603eeba5e24b20cc81f2778859dfef47c32f0135377ca7ab2")
SMOKE_FAILURE_DIGEST = (
    "230e1510edd0ee2d268a82d77c3a128b05e258c5904f33f027a676994c1d40ef")
SMOKE_FAILURE_SHA256 = (
    "55a7d0c9c2a635030f7378602b25d85f6483352ffedee7eb2717bf7cb74284a0")
SMOKE_FAILURE_BYTE_COUNT = 1_289
SMOKE_TRACEBACK_SHA256 = (
    "50f6afcdb2a4da2b7fe0b8fc0ba4f917859816183f027265e232b767be9edc67")
SMOKE_EXCEPTION_BINDING_SHA256 = (
    "01f6f353aedde06912527aa6b638e4ce6c6ee3f46e608043f5655779ccbb176c")

FROZEN_ARCHITECTURE_DIGEST = (
    "0c5edc716e8bfba944d2ca89de918ca05ff571748df2b8f64f59eeea285df20d")
FROZEN_FIXTURE_DIGEST = (
    "017e14d40a291380f54cd94e36f99d03970161425fb82c1efeeac1db34536888")
FROZEN_ATTENTIVE_SEED = 1_063_471_220
FROZEN_INITIAL_STATE_DIGEST = (
    "02a30a879ec2cc775bd552dc4c0889a97818feadd9cd35c2c25a1a68882fa36f")
FROZEN_PARAMETER_INVENTORY_DIGEST = (
    "c81e6d154e78671329be4af747374a1464e5307033f9811f1dd13b119eefb5e6")
TRAINABLE_PARAMETER_COUNT = 13_684_739
TRAINABLE_PARAMETER_TENSOR_COUNT = 63

OFFICIAL_REPOSITORY_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
OFFICIAL_POOLER_BINDING_DIGEST = (
    "f436439c72e725bfd7f3caab517f2b7c870cac1cf4060623fe0c1f6da63591e6")
OFFICIAL_FILES = {
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
}
OFFICIAL_CALCULATION = {
    "repository_commit": OFFICIAL_REPOSITORY_COMMIT,
    "pooler_binding_digest": OFFICIAL_POOLER_BINDING_DIGEST,
    "files": OFFICIAL_FILES,
    "pooler": {
        "depth": 4, "self_attention_blocks": 3, "heads": 16,
        "mlp_ratio": 4.0, "norm": "LayerNorm", "norm_eps": 1e-5,
        "activation": "GELU", "qkv_bias": True, "complete_block": True,
        "dropout": 0.0, "attention_dropout": 0.0, "drop_path": 0.0,
        "init_std": 0.02, "activation_checkpointing": True,
    },
    "expected_sdpa_invocations_per_forward_backward": {
        "initial_forward_self": 3, "initial_forward_cross": 1,
        "backward_checkpoint_recompute_self": 3, "total": 7,
    },
}

EXECUTION_ENVIRONMENT = {
    "python": ".generated/venvs/genesis_rocm_0_4_6_v1/bin/python",
    "torch_version": "2.12.0+rocm7.2",
    "torch_hip_version": "7.2.53211",
    "torch_distribution_environment":
        ".generated/venvs/world_model_rocm_7_2_1_v1",
    "device": "cuda:0",
    "device_name": "AMD Radeon AI PRO R9700",
    "device_architecture": "gfx1201",
    "device_capability": [12, 0],
    "visible_hip_device_count": 2,
    "production_ambient": {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
        "sdpa_flash_enabled": True,
        "sdpa_memory_efficient_enabled": True,
        "sdpa_math_enabled": True,
        "sdpa_cudnn_enabled": True,
        "sdpa_priority_order": [1, 2, 0, 3, 4],
        "rocm_preferred_flash_attention_library": "AOTriton",
        "deterministic_algorithms_after_fresh_model_state": True,
    },
    "deterministic_algorithms_at_process_entry": False,
}

FORWARD_EQUIVALENCE = {
    "a_b": "exact_tensor_bytes_and_gradient_tensor_bytes",
    "a_c_d_absolute_tolerance": 1e-5,
    "a_c_d_relative_tolerance": 1.3e-6,
    "relative_denominator_floor": 1e-12,
    "element_rule": "abs(candidate-reference)<=atol+rtol*abs(reference)",
    "immutable_after_execution": True,
    "optional_c_d_gradient_diagnostic": {
        "relative_l2_max": 1e-4,
        "cosine_similarity_min": 0.999999,
        "repair_gate": False,
    },
}

BACKEND_MATRIX = {
    "A": {
        "name": "EXACT_PRODUCTION",
        "parameter_dtype": "float32",
        "latent_dtype": "float32",
        "loss_dtype": "float32",
        "autocast": False,
        "sdpa": "production default selected independently per invocation",
    },
    "B": {
        "name": "EXPLICIT_FULL_FP32",
        "parameter_dtype": "float32",
        "latent_dtype": "float32",
        "attention_layernorm_mlp_dtype": "float32",
        "loss_dtype": "float32",
        "autocast": False,
        "sdpa": "production default selected independently per invocation",
        "semantic_relation_to_A": "identical; exact equality required",
    },
    "C": {
        "name": "FULL_FP32_FORCED_MATH_SDPA",
        "parameter_dtype": "float32",
        "latent_dtype": "float32",
        "attention_layernorm_mlp_dtype": "float32",
        "loss_dtype": "float32",
        "autocast": False,
        "sdpa": "force math-only inside every actual SDPA invocation",
    },
    "D": {
        "name": "PRODUCTION_OUTER_OFFICIAL_EXPLICIT_FP32_REDUCTIONS",
        "outer_dtype": "float32",
        "autocast": False,
        "attention": (
            "official non-SDPA calculation with logits, softmax, and weighted "
            "reduction in float32, returning outer dtype"),
        "layernorm": (
            "float32 inputs/parameters/reductions returning outer dtype"),
        "new_scientific_operation": False,
    },
}

LOSS_ISOLATION_PASSES = (
    "progress_only", "safety_only", "completion_only", "frozen_summed_loss")
LOSS_CONTRACT = {
    "effective_batch_denominator": 64,
    "progress": "raw mse_loss(reduction=sum), isolated loss raw_progress/64",
    "safety": (
        "raw binary_cross_entropy_with_logits(reduction=sum), isolated loss "
        "raw_safety/64"),
    "completion": (
        "raw binary_cross_entropy_with_logits(reduction=sum), isolated loss "
        "raw_completion/64"),
    "frozen_summed_loss": (
        "(raw_progress+raw_safety+raw_completion)/64 with one final FP32 "
        "division after the two ordered additions"),
}
ALL_PASS_INVARIANTS = {
    "fixture_digest": FROZEN_FIXTURE_DIGEST,
    "initial_state_digest": FROZEN_INITIAL_STATE_DIGEST,
    "model_mode": "train",
    "activation_checkpointing": True,
    "effective_batch_denominator": 64,
    "fresh_model_for_every_pass": True,
    "model_or_input_mutation_between_passes": False,
    "optimiser_constructed": True,
    "optimizer_state_before_and_after_backward": "empty and unchanged",
    "optimizer_zero_grad_before_forward": True,
    "gradient_clip": False,
    "optimizer_step": False,
    "exact_byte_equality_group": [
        "exact_reproduction", "hook_inventory_frozen_summed_loss",
        "loss_isolation_frozen_summed_loss", "backend_matrix_A",
        "backend_matrix_B",
    ],
    "exact_byte_equality_payload": [
        "fixture_input_tensors", "component_outputs", "component_losses", "total_loss",
        "every_named_parameter_gradient",
    ],
    "mismatch_terminal": "INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP",
    "internal_activation_finiteness_evidence": (
        "the temporary-hook summed-loss pass covers all frozen internal "
        "boundaries and must be exact-equal to the uninstrumented reproduction; "
        "non-hook passes cover component outputs only"),
}
EXECUTION_COUNTS = {
    "exact_reproduction": 1,
    "hook_inventory": 1,
    "loss_isolation": 4,
    "backend_matrix": 4,
    "fresh_model_constructions": 10,
    "forwards": 10,
    "backwards": 10,
    "optimizer_constructions": 10,
    "optimizer_steps": 0,
    "gradient_clips": 0,
    "fixture_validation_row_record_opens": 4,
    "fixture_validation_latent_shard_opens": 4,
    "unique_fit_row_record_files": 4,
    "unique_fit_latent_shard_files": 4,
    "pass_latent_shard_loads": 40,
    "batch_presentations": 10,
    "examples_presented": 40,
}

PARAMETER_INVENTORY_FIELDS = (
    "fully_qualified_name", "module_path", "module_type", "shape",
    "parameter_dtype", "gradient_dtype", "gradient_is_none", "finite_count",
    "nan_count", "positive_infinity_count", "negative_infinity_count",
    "maximum_absolute_finite_value", "finite_only_l2_norm",
    "first_nonfinite_flat_index", "first_nonfinite_multi_index",
    "gradient_tensor_digest",
)
HOOK_INVENTORY_FIELDS = (
    "finite_input", "finite_output", "input_maximum_absolute_finite_value",
    "output_maximum_absolute_finite_value", "finite_gradient_output",
    "finite_gradient_input", "gradient_output_maximum_absolute_finite_value",
    "gradient_input_maximum_absolute_finite_value",
    "first_nonfinite_tensor_path", "first_nonfinite_multi_index",
)
BACKEND_RESULT_FIELDS = (
    "forward_maximum_absolute_difference_from_A",
    "forward_maximum_relative_difference_from_A", "component_losses",
    "complete_gradient_verdict", "offending_parameter_set", "peak_vram_bytes",
    "wall_time_seconds", "actual_sdpa_backend_per_invocation",
)
PRE_BACKWARD_FINITE_FIELDS = (
    "all_model_parameters_finite", "all_inputs_finite",
    "all_activations_finite", "all_targets_finite",
    "all_component_losses_finite", "total_loss_finite",
    "parameter_offenders", "input_offenders", "activation_offenders",
    "target_offenders", "component_loss_offenders",
)
TERMINAL_OFFENDER_FIELDS = (
    "first_reverse_module_with_finite_downstream_and_nonfinite_upstream",
    "first_nonfinite_parameter_gradient", "all_nonfinite_parameter_gradients",
)

HOOK_TARGETS = (
    {"path": "token_projection", "role": "token_projection",
     "module_type": "torch.nn.modules.linear.Linear"},
    {"path": "pooler", "role": "attentive_pooler",
     "module_type": "src.models.attentive_pooler.AttentivePooler"},
    {"path": "pooler.blocks.0", "role": "self_attention_block",
     "module_type": "src.models.utils.modules.Block"},
    {"path": "pooler.blocks.1", "role": "self_attention_block",
     "module_type": "src.models.utils.modules.Block"},
    {"path": "pooler.blocks.2", "role": "self_attention_block",
     "module_type": "src.models.utils.modules.Block"},
    {"path": "pooler.blocks.0.attn", "role": "self_attention_kernel",
     "module_type": "src.models.utils.modules.Attention"},
    {"path": "pooler.blocks.1.attn", "role": "self_attention_kernel",
     "module_type": "src.models.utils.modules.Attention"},
    {"path": "pooler.blocks.2.attn", "role": "self_attention_kernel",
     "module_type": "src.models.utils.modules.Attention"},
    {"path": "pooler.cross_attention_block", "role": "cross_attention_block",
     "module_type": "src.models.utils.modules.CrossAttentionBlock"},
    {"path": "pooler.cross_attention_block.xattn", "role": "cross_attention_kernel",
     "module_type": "src.models.utils.modules.CrossAttention"},
    *tuple(
        {"path": path, "role": "layer_norm",
         "module_type": "torch.nn.modules.normalization.LayerNorm"}
        for path in (
            "pooler.blocks.0.norm1", "pooler.blocks.0.norm2",
            "pooler.blocks.1.norm1", "pooler.blocks.1.norm2",
            "pooler.blocks.2.norm1", "pooler.blocks.2.norm2",
            "pooler.cross_attention_block.norm1",
            "pooler.cross_attention_block.norm2")),
    *tuple(
        {"path": path, "role": "mlp",
         "module_type": "src.models.utils.modules.MLP"}
        for path in (
            "pooler.blocks.0.mlp", "pooler.blocks.1.mlp",
            "pooler.blocks.2.mlp", "pooler.cross_attention_block.mlp")),
    *tuple(
        {"path": f"pooler.blocks.{block}.attn.{leaf}",
         "role": f"self_attention_{leaf}_boundary",
         "module_type": "torch.nn.modules.linear.Linear"}
        for block in range(3) for leaf in ("qkv", "proj")),
    *tuple(
        {"path": f"pooler.blocks.{block}.mlp.{leaf}",
         "role": f"mlp_{leaf}_boundary",
         "module_type": ("torch.nn.modules.activation.GELU" if leaf == "act"
                         else "torch.nn.modules.linear.Linear")}
        for block in range(3) for leaf in ("fc1", "act", "fc2")),
    *tuple(
        {"path": f"pooler.cross_attention_block.xattn.{leaf}",
         "role": f"cross_attention_{leaf}_boundary",
         "module_type": "torch.nn.modules.linear.Linear"}
        for leaf in ("q", "kv")),
    *tuple(
        {"path": f"pooler.cross_attention_block.mlp.{leaf}",
         "role": f"cross_mlp_{leaf}_boundary",
         "module_type": ("torch.nn.modules.activation.GELU" if leaf == "act"
                         else "torch.nn.modules.linear.Linear")}
        for leaf in ("fc1", "act", "fc2")),
    {"path": "context", "role": "action_goal_encoder",
     "module_type": "torch.nn.modules.container.Sequential"},
    {"path": "fuse", "role": "component_fusion_mlp",
     "module_type": "torch.nn.modules.container.Sequential"},
    {"path": "progress", "role": "progress_output_head",
     "module_type": "torch.nn.modules.linear.Linear"},
    {"path": "safety", "role": "safety_output_head",
     "module_type": "torch.nn.modules.linear.Linear"},
    {"path": "completion", "role": "completion_output_head",
     "module_type": "torch.nn.modules.linear.Linear"},
    {"path": "horizon_embeddings", "role": "virtual_horizon_embedding_buffer",
     "module_type": "torch.Tensor"},
    {"path": "pooler.query_tokens", "role": "virtual_component_queries",
     "module_type": "torch.nn.Parameter"},
)

MECHANISM_CLASSIFICATIONS = (
    "BACKEND_NUMERICAL_DEFECT_CONTRACT_PRESERVING",
    "IMPLEMENTATION_DEFECT_CONTRACT_PRESERVING",
    "ARCHITECTURE_OR_OBJECTIVE_CHANGE_REQUIRED",
)
CLASSIFICATION_RULE = {
    "preconditions": [
        "exact reproduction observed the original nonfinite-gradient failure",
        "A and B have exact output, loss, and every named gradient tensor bytes",
        "reproduction, hook summed, isolation summed, A, and B have exact "
        "output and loss bytes",
        "all four backend-matrix passes completed at the exact four-row shape",
    ],
    "precedence": [
        "BACKEND_NUMERICAL_DEFECT_CONTRACT_PRESERVING",
        "IMPLEMENTATION_DEFECT_CONTRACT_PRESERVING",
        "ARCHITECTURE_OR_OBJECTIVE_CHANGE_REQUIRED",
    ],
    "backend_numerical_defect": {
        "A_has_at_least_one_non_math_sdpa_dispatch": True,
        "C_all_gradients_finite": True,
        "A_C_forward_equivalent": True,
        "C_all_seven_actual_sdpa_dispatches_math": True,
    },
    "implementation_defect": {
        "applies_only_if_backend_rule_failed": True,
        "D_all_gradients_finite": True,
        "A_D_forward_equivalent": True,
        "D_exact_official_manual_attention_and_layernorm_audit": True,
    },
    "architecture_or_objective_change_required": (
        "applies when neither preceding rule passes"),
    "both_C_and_D_pass": (
        "BACKEND_NUMERICAL_DEFECT_CONTRACT_PRESERVING"),
    "nonreproduction_or_harness_failure_mechanism": None,
}
STOP_RULES = {
    "nonreproduction": (
        "publish NONREPRODUCTION_TECHNICAL_STOP with null mechanism and stop "
        "after the exact reproduction pass"),
    "harness_disagreement": (
        "publish INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP and stop"),
    "architecture_or_objective_change_required": (
        "publish the immutable classification and stop; no repair is allowed"),
    "repair_gate": (
        "only either contract-preserving classification may support a later, "
        "separately authorised repair decision"),
    "automatic_repair_or_training": False,
}

HEX64 = re.compile(r"[0-9a-f]{64}")


class GradientLocalisationContractError(RuntimeError):
    """The frozen source, failed smoke, or diagnostic contract changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GradientLocalisationContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"{label} is absent or non-regular")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GradientLocalisationContractError(f"{label} is invalid") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(key, None)
    require(isinstance(recorded, str) and HEX64.fullmatch(recorded) is not None
            and recorded == digest(result), f"{label} self digest changed")
    result[key] = recorded
    return result


def signed(value: Mapping[str, Any], key: str = CONTRACT_SELF_KEY,
           ) -> dict[str, Any]:
    result = dict(value)
    require(key not in result, f"{key} already exists")
    result[key] = digest(result)
    return result


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GradientLocalisationContractError(
            f"cannot bind diagnostic source: {exc}") from exc


def source_closure(root: Path = ROOT) -> dict[str, Any]:
    require(_git(root, "status", "--porcelain=v1") == "",
            "diagnostic source must be clean and committed")
    head = _git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(ancestor.returncode == 0,
            "diagnostic source does not descend from its frozen base")
    changed = tuple(sorted(filter(None, _git(
        root, "diff", "--name-only", f"{BASE_SOURCE_COMMIT}..{head}"
    ).splitlines())))
    require(changed == tuple(sorted(NEW_SOURCE_PATHS)),
            "committed diagnostic diff is not exactly the four additive paths")
    frozen = {}
    for relative, (expected_sha, expected_bytes) in FROZEN_DEPENDENCY_FILES.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == expected_bytes
                and file_sha256(path) == expected_sha,
                f"frozen dependency changed at {relative}")
        frozen[relative] = {"path": relative, "sha256": expected_sha,
                            "byte_count": expected_bytes}
    additive = {}
    for relative in NEW_SOURCE_PATHS:
        path = root / relative
        require(path.is_file() and not path.is_symlink(),
                f"additive diagnostic source is absent: {relative}")
        additive[relative] = {"path": relative, "sha256": file_sha256(path),
                              "byte_count": path.stat().st_size}
    payload = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": head,
        "source_repository_clean": True,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "exact_committed_additive_path_diff": list(changed),
        "frozen_dependency_files": frozen,
        "additive_files": additive,
    }
    return {**payload, SOURCE_CLOSURE_SELF_KEY: digest(payload)}


def runtime_root(root: Path = ROOT) -> Path:
    return root / DIAGNOSTIC_RUNTIME_ROOT


def contract_path(root: Path = ROOT) -> Path:
    return root / CONTRACT_PATH


def failed_smoke_lineage(root: Path = ROOT) -> dict[str, Any]:
    amendment_path = root / AMENDMENT_ARTIFACT_RELATIVE
    failure_path = root / SMOKE_FAILURE_RELATIVE
    amendment = validate_signed(
        read_json(amendment_path, "prerequisite amendment"),
        "prerequisite_amendment_digest", "prerequisite amendment")
    failure = validate_signed(
        read_json(failure_path, "production smoke failure"),
        "technical_failure_digest", "production smoke failure")
    traceback_value = str(failure.get("traceback", ""))
    exception_bytes = (
        str(failure.get("exception_type", "")) + "\0"
        + str(failure.get("exception_message", "")) + "\0"
        + traceback_value).encode("utf-8")
    require(amendment["prerequisite_amendment_digest"]
            == AMENDMENT_ARTIFACT_DIGEST
            and amendment.get("source_closure", {}).get(
                "amendment_source_closure_digest")
            == AMENDMENT_SOURCE_CLOSURE_DIGEST
            and amendment_path.stat().st_size == AMENDMENT_ARTIFACT_BYTE_COUNT
            and file_sha256(amendment_path) == AMENDMENT_ARTIFACT_SHA256,
            "prerequisite amendment lineage changed")
    require(failure["technical_failure_digest"] == SMOKE_FAILURE_DIGEST
            and failure.get("schema")
            == "go2_v1_3_final_layer_attentive_readout_amendment_v1_technical_failure_v1"
            and failure.get("status")
            == "INVALID_TECHNICAL_ATTENTIVE_AMENDMENT_EXECUTION"
            and failure.get("stage") == "fit_only_smoke_update"
            and failure.get("exception_type") == "AttentiveAmendmentError"
            and failure.get("exception_message") == "smoke gradient is non-finite"
            and failure.get("completed_optimizer_updates") == 0
            and failure.get("retry_resume_or_replacement_authorised") is False
            and failure_path.stat().st_size == SMOKE_FAILURE_BYTE_COUNT
            and file_sha256(failure_path) == SMOKE_FAILURE_SHA256
            and hashlib.sha256(traceback_value.encode("utf-8")).hexdigest()
            == SMOKE_TRACEBACK_SHA256
            and hashlib.sha256(exception_bytes).hexdigest()
            == SMOKE_EXCEPTION_BINDING_SHA256,
            "production smoke failure lineage changed")
    amendment_root = root / AMENDMENT_RUNTIME_ROOT
    smoke_work = root / SMOKE_WORK_DIRECTORY_RELATIVE
    require(smoke_work.is_dir() and not smoke_work.is_symlink()
            and tuple(smoke_work.iterdir()) == (),
            "failed smoke work directory is not the preserved empty directory")
    absent = (
        amendment_root / "production_smoke.json",
        amendment_root / "initialisation.pt",
        amendment_root / "training",
        amendment_root / "evaluation_authorisation.json",
        amendment_root / "calibration_evidence.json",
        amendment_root / "exploratory_result.json",
        amendment_root / "technical_failure.json",
    )
    require(all(not path.exists() and not path.is_symlink() for path in absent),
            "failed smoke unexpectedly published later scientific artifacts")
    return {
        "prerequisite_amendment_digest": AMENDMENT_ARTIFACT_DIGEST,
        "prerequisite_amendment_sha256": AMENDMENT_ARTIFACT_SHA256,
        "prerequisite_amendment_byte_count": AMENDMENT_ARTIFACT_BYTE_COUNT,
        "amendment_source_closure_digest": AMENDMENT_SOURCE_CLOSURE_DIGEST,
        "production_smoke_failure_digest": SMOKE_FAILURE_DIGEST,
        "production_smoke_failure_sha256": SMOKE_FAILURE_SHA256,
        "production_smoke_failure_byte_count": SMOKE_FAILURE_BYTE_COUNT,
        "traceback_sha256": SMOKE_TRACEBACK_SHA256,
        "exception_binding_sha256": SMOKE_EXCEPTION_BINDING_SHA256,
        "completed_optimizer_updates": 0,
        "checkpoint_published": False,
        "scientific_attempt_started": False,
        "production_smoke_work_directory_present": True,
        "production_smoke_work_directory_empty": True,
        "preserved_artifacts_mutable": False,
    }


def static_contract() -> dict[str, Any]:
    return {
        "status": STATUS,
        "scientific_claim_status": "exploratory_technical_diagnostic_only",
        "frozen_architecture_digest": FROZEN_ARCHITECTURE_DIGEST,
        "frozen_fixture_digest": FROZEN_FIXTURE_DIGEST,
        "frozen_attentive_seed": FROZEN_ATTENTIVE_SEED,
        "frozen_initial_state_digest": FROZEN_INITIAL_STATE_DIGEST,
        "frozen_parameter_inventory_digest": FROZEN_PARAMETER_INVENTORY_DIGEST,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "trainable_parameter_tensor_count": TRAINABLE_PARAMETER_TENSOR_COUNT,
        "official_calculation": OFFICIAL_CALCULATION,
        "execution_environment": EXECUTION_ENVIRONMENT,
        "forward_equivalence": FORWARD_EQUIVALENCE,
        "backend_matrix": BACKEND_MATRIX,
        "loss_isolation_passes": list(LOSS_ISOLATION_PASSES),
        "loss_contract": LOSS_CONTRACT,
        "all_pass_invariants": ALL_PASS_INVARIANTS,
        "execution_counts": EXECUTION_COUNTS,
        "parameter_inventory_fields": list(PARAMETER_INVENTORY_FIELDS),
        "hook_inventory_fields": list(HOOK_INVENTORY_FIELDS),
        "backend_result_fields": list(BACKEND_RESULT_FIELDS),
        "pre_backward_finiteness_fields": list(PRE_BACKWARD_FINITE_FIELDS),
        "terminal_offender_fields": list(TERMINAL_OFFENDER_FIELDS),
        "hook_targets": list(HOOK_TARGETS),
        "mechanism_classifications": list(MECHANISM_CLASSIFICATIONS),
        "classification_rule": CLASSIFICATION_RULE,
        "stop_rules": STOP_RULES,
        "exact_reproduction": {
            "fixture_digest": FROZEN_FIXTURE_DIGEST,
            "latent_action_goal_target_and_loss": "identical to failed smoke",
            "model_seed": FROZEN_ATTENTIVE_SEED,
            "precision": "float32",
            "autocast": False,
            "backend": "production default",
            "operations": ["forward", "frozen_summed_loss", "backward"],
            "gradient_clip": False,
            "optimizer_constructed": True,
            "optimizer": (
                "frozen AdamW settings; empty state; zero_grad before forward"),
            "optimizer_step": False,
            "required_original_failure": "at least one non-finite parameter gradient",
        },
        "authority": {
            "runtime_attempts": 1,
            "maximum_runtime_attempts": 1,
            "repair_authorised": False,
            "training_authorised": False,
            "scorer_checkpoint_authorised": False,
            "predictor_access_authorised": False,
            "calibration_access_authorised": False,
            "new_data_authorised": False,
        },
    }


def build_contract(source: Mapping[str, Any], lineage: Mapping[str, Any],
                   ) -> dict[str, Any]:
    require(source.get("schema") == SOURCE_CLOSURE_SCHEMA
            and source.get(SOURCE_CLOSURE_SELF_KEY)
            == digest({key: value for key, value in source.items()
                       if key != SOURCE_CLOSURE_SELF_KEY}),
            "source closure is invalid")
    require(lineage.get("production_smoke_failure_digest")
            == SMOKE_FAILURE_DIGEST
            and lineage.get("prerequisite_amendment_digest")
            == AMENDMENT_ARTIFACT_DIGEST,
            "failed-smoke lineage is invalid")
    return signed({
        "schema": CONTRACT_SCHEMA,
        **static_contract(),
        "source_closure": dict(source),
        "failed_smoke_lineage": dict(lineage),
    })


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = validate_signed(value, CONTRACT_SELF_KEY,
                             "gradient-localisation contract")
    expected_static = static_contract()
    require(result.get("schema") == CONTRACT_SCHEMA
            and all(result.get(key) == expected for key, expected in
                    expected_static.items()),
            "gradient-localisation contract changed")
    source = result.get("source_closure")
    lineage = result.get("failed_smoke_lineage")
    require(isinstance(source, Mapping) and isinstance(lineage, Mapping)
            and result == build_contract(source, lineage),
            "gradient-localisation contract bindings changed")
    return result


def publish_json_once(path: Path, value: Mapping[str, Any], label: str) -> None:
    require(not path.exists() and not path.is_symlink(), f"{label} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists() and not temporary.is_symlink(),
            f"{label} temporary path exists")
    with temporary.open("xb") as sink:
        sink.write(encoded)
        sink.flush()
    temporary.replace(path)
    path.chmod(0o444)


def issue_contract(root: Path = ROOT) -> dict[str, Any]:
    logical = root / GENERATED_ROOT
    if root.resolve() == ROOT.resolve():
        require(logical.is_symlink()
                and logical.resolve() == REGISTERED_GENERATED_TARGET_ROOT,
                "registered generated-output alias changed")
    else:
        logical.mkdir(parents=True, exist_ok=True)
    source = source_closure(root)
    lineage = failed_smoke_lineage(root)
    value = build_contract(source, lineage)
    path = contract_path(root)
    if path.exists() or path.is_symlink():
        installed = validate_contract(read_json(path, "installed contract"))
        require(installed == value, "installed contract belongs to other source")
        return installed
    root_path = runtime_root(root)
    require(not root_path.exists() and not root_path.is_symlink(),
            "gradient-localisation namespace was consumed")
    publish_json_once(path, value, "gradient-localisation contract")
    return value


__all__ = [name for name in globals() if name.isupper()] + [
    "GradientLocalisationContractError", "build_contract", "canonical_bytes",
    "contract_path", "digest", "failed_smoke_lineage", "file_sha256",
    "issue_contract", "read_json", "runtime_root", "signed",
    "source_closure", "static_contract", "validate_contract",
    "validate_signed",
]
