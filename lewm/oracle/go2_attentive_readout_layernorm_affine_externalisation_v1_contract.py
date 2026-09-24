"""Frozen contract for the final LayerNorm-affine equivalence diagnostic.

This is a one-session diagnostic and conditional production-path smoke only.
It preserves the completed gradient-localisation result and cannot itself
authorise scorer training, qualification, predictor access, or a repair.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as BASE
from lewm.oracle import (
    go2_attentive_readout_gradient_localisation_v1_contract as PREDECESSOR,
)
from lewm.oracle import (
    go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment
    as CONSUMER,
)


ROOT = Path(__file__).resolve().parents[2]
STATUS = "EXPLORATORY_LAYERNORM_AFFINE_EXTERNALISATION"
CONTRACT_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_contract_v1"
CONTRACT_SELF_KEY = "layernorm_affine_externalisation_contract_digest"
SOURCE_CLOSURE_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_source_closure_v1"
SOURCE_CLOSURE_SELF_KEY = "layernorm_affine_externalisation_source_closure_digest"
BASE_SOURCE_COMMIT = "3a00b9819926e8552dadaab89ea59fdb3aeffe96"

NEW_SOURCE_PATHS = (
    "lewm/oracle/go2_attentive_readout_layernorm_affine_externalisation_v1_contract.py",
    "lewm/tests/test_go2_attentive_readout_layernorm_affine_externalisation_v1_contract.py",
    "scripts/diagnose_go2_attentive_readout_layernorm_affine_externalisation_v1.py",
    "lewm/tests/test_diagnose_go2_attentive_readout_layernorm_affine_externalisation_v1.py",
)
FROZEN_DEPENDENCY_FILES = {
    "lewm/oracle/go2_attentive_readout_gradient_localisation_v1_contract.py": (
        "c6e3b08017faa09edbeb77e82ddbc7a1c972dda968913cd7962278ced0faa913", 32_896),
    "lewm/tests/test_go2_attentive_readout_gradient_localisation_v1_contract.py": (
        "0fa93373ea10a661ef5161ac122a68b85f43ce1e7ddce143cb63e4460add544d", 8_330),
    "scripts/diagnose_go2_attentive_readout_gradient_localisation_v1.py": (
        "17ad299d2694403ef7dbf92fd7ddf015645a6e68d289030cb2e357d8836b9b25", 122_185),
    "lewm/tests/test_diagnose_go2_attentive_readout_gradient_localisation_v1.py": (
        "bb9daae4b9fdcd0dfe138ab7d78d774d40b654b66fbb8bed101da9faf9ceb5f1", 8_568),
    "lewm/oracle/go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment.py": (
        "395d8f8b246286e668d233274510206e80775792ef93578a1b4f1eb197e0727b", 19_387),
    "lewm/tests/test_go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment.py": (
        "ff33037f7a8e44c28c70255cd220cef2f4bad496b45dfda5e75c894d8b8a3221", 6_873),
}

GENERATED_PARENT = BASE.GENERATED_ROOT
REGISTERED_PARENT = BASE.REGISTERED_GENERATED_TARGET_ROOT
RUNTIME_RELATIVE = (
    GENERATED_PARENT / "attentive_readout_layernorm_affine_externalisation_v1")
CONTRACT_PATH = RUNTIME_RELATIVE / "contract.json"

PREDECESSOR_SOURCE_CLOSURE_DIGEST = (
    "985318cb7687311a877f57c4cde7715a4004c67c02628e8b57c39bef512c3ecf")
PREDECESSOR_CONSUMER_RECEIPT_DIGEST = (
    "1f9e9b3dd8c67f145d4c1285278fc9861e45e6d3eb08473d580c9cdbbc4e810b")
PREDECESSOR_TERMINAL_DIGEST = (
    "7ec0c9d5cd01c965568f38ca7c5e119e0f7fb74b65dc0f909bdba09f98b26187")
PREDECESSOR_CLASSIFICATION = "ARCHITECTURE_OR_OBJECTIVE_CHANGE_REQUIRED"
PREDECESSOR_CONTRACT_DIGEST = (
    "bc10101d8cd989b61fcdbcc235db0470bf978fe44e9f3cdd4408ae18fc7c8b71")
PREDECESSOR_REPRODUCTION_DIGEST = (
    "ef926ed8b4d8d346b4a7ca69cdb0d4979545a0b6ed72037052e29c70e1c036e4")
FROZEN_DIAGNOSTIC_SOURCE_COMMIT = (
    "ccdb4de735a71760cd2683e491ce221240bcf6e4")
PREDECESSOR_ATTEMPT_DIGEST = (
    "3ab590d43085113a6709a01e4745462673f8a80ae0f9436b245beb63479fcee8")
PREDECESSOR_TERMINAL_FILE_SHA256 = (
    "0afe02dd08baea3fcc1657fbb672153d0c8b07a778856565428cdaac262aa196")
PREDECESSOR_ARTIFACT_SET_DIGEST = (
    "00e97586982c8ccd0382ded8a23388958c56735234e23069a04ffc445032bb17")
PREDECESSOR_RUNTIME_ARTIFACTS = {
    name: {
        "sha256": value["sha256"], "byte_count": value["byte_count"],
        "self_digest": value["self_digest"],
    }
    for name, value in CONSUMER.FROZEN_ARTIFACTS.items()
}

FROZEN_ARCHITECTURE_DIGEST = PREDECESSOR.FROZEN_ARCHITECTURE_DIGEST
FROZEN_FIXTURE_DIGEST = PREDECESSOR.FROZEN_FIXTURE_DIGEST
FROZEN_INITIAL_STATE_DIGEST = PREDECESSOR.FROZEN_INITIAL_STATE_DIGEST
FROZEN_PARAMETER_INVENTORY_DIGEST = (
    PREDECESSOR.FROZEN_PARAMETER_INVENTORY_DIGEST)
FROZEN_ATTENTIVE_SEED = PREDECESSOR.FROZEN_ATTENTIVE_SEED
TRAINABLE_PARAMETER_COUNT = PREDECESSOR.TRAINABLE_PARAMETER_COUNT
TRAINABLE_PARAMETER_TENSOR_COUNT = PREDECESSOR.TRAINABLE_PARAMETER_TENSOR_COUNT
FEATURE_WIDTH = 512
LAYER_NORM_PATHS = (
    "pooler.cross_attention_block.norm1",
    "pooler.blocks.0.norm1", "pooler.blocks.0.norm2",
    "pooler.blocks.1.norm1", "pooler.blocks.1.norm2",
    "pooler.blocks.2.norm1", "pooler.blocks.2.norm2",
)
NATIVE_NONFINITE_PARAMETER_NAMES = tuple(
    f"{path}.{suffix}" for path in LAYER_NORM_PATHS
    for suffix in ("weight", "bias"))
NEGATIVE_CONTROL_PATH = "pooler.cross_attention_block.norm2"
EXACT_REPRODUCTION_GATE = {
    "fixture_digest": FROZEN_FIXTURE_DIGEST,
    "initial_state_digest": FROZEN_INITIAL_STATE_DIGEST,
    "registered_seed": FROZEN_ATTENTIVE_SEED,
    "action_goal_target_and_batch_tensor_digests":
        "exactly equal frozen predecessor reproduction",
    "loss_association": PREDECESSOR.LOSS_CONTRACT["frozen_summed_loss"],
    "model_and_inputs_dtype": "float32", "autocast": False,
    "backend": "current production efficient SDPA selection",
    "exact_predecessor_output_loss_and_named_gradient_receipt": True,
    "exact_nonfinite_parameter_names": list(NATIVE_NONFINITE_PARAMETER_NAMES),
    "each_native_nonfinite_affine_gradient": {
        "shape": [512], "finite_count": 256, "nan_count": 256,
        "positive_infinity_count": 0, "negative_infinity_count": 0,
    },
    "negative_control_affine_gradients_finite": NEGATIVE_CONTROL_PATH,
    "gradient_clip": False, "optimizer_step": False,
    "calibration_rows_or_latents_opened": 0,
}

IMPLEMENTATION_NAME = "IMPLEMENTATION_SUCCESSOR_LAYER_NORM_AFFINE_EXTERNALISATION"
IMPLEMENTATION_CONTRACT = {
    "name": IMPLEMENTATION_NAME,
    "changed_forward_paths": list(LAYER_NORM_PATHS),
    "unchanged_eighth_layernorm": "pooler.cross_attention_block.norm2",
    "normalized_shape": [FEATURE_WIDTH],
    "eps": 1e-5,
    "native_formula": (
        "torch.nn.functional.layer_norm(x,(512,),weight,bias,eps)"),
    "externalised_formula": (
        "normalized=torch.nn.functional.layer_norm(x,(512,),weight=None,"
        "bias=None,eps=eps); y=normalized*weight+bias"),
    "parameter_objects_shapes_and_state_dict_keys_unchanged": True,
    "mathematical_layernorm_normalisation_unchanged": True,
    "affine_weight_and_bias_remain_trainable": True,
    "only_changed_implementation_boundary": (
        "externalise affine multiply/add from the ROCm native LayerNorm affine "
        "backward reduction while preserving the same forward function"),
    "architecture_digest_preserved": FROZEN_ARCHITECTURE_DIGEST,
    "trainable_parameter_count_preserved": TRAINABLE_PARAMETER_COUNT,
    "trainable_parameter_tensor_count_preserved": TRAINABLE_PARAMETER_TENSOR_COUNT,
    "new_normalisation_or_clamp": False,
    "new_epsilon_loss_weight_label_or_data": False,
    "rmsnorm_substitution": False,
    "affine_parameter_freezing": False,
    "nan_replacement": False,
    "gradient_sanitisation": False,
}
IMPLEMENTATION_DIGEST = hashlib.sha256(json.dumps(
    IMPLEMENTATION_CONTRACT, sort_keys=True, separators=(",", ":"),
    ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()

TOLERANCES = {
    "forward": {"absolute": 2e-6, "relative": 2e-5},
    "input_gradient": {"absolute": 1e-5, "relative": 1e-4},
    "weight_gradient": {"absolute": 1e-5, "relative": 1e-4},
    "bias_gradient": {"absolute": 1e-5, "relative": 1e-4},
    "relative_denominator_floor": 1e-12,
    "element_rule": "abs(candidate-reference)<=atol+rtol*abs(reference)",
}
LOCAL_CASES = {
    "CPU_NATIVE": {
        "device": "cpu", "dtype": "float32",
        "formula": IMPLEMENTATION_CONTRACT["native_formula"],
        "backward": "y.backward(exact_captured_finite_upstream_gradient)",
    },
    "GPU_NATIVE": {
        "device": "cuda:0", "dtype": "float32",
        "formula": IMPLEMENTATION_CONTRACT["native_formula"],
        "backward": "y.backward(exact_captured_finite_upstream_gradient)",
    },
    "GPU_EXPLICIT_AFFINE": {
        "device": "cuda:0", "dtype": "float32",
        "formula": IMPLEMENTATION_CONTRACT["externalised_formula"],
        "backward": "y.backward(exact_captured_finite_upstream_gradient)",
    },
}
CAPTURE_CONTRACT = {
    "paths": list(LAYER_NORM_PATHS), "logical_layernorms": 7,
    "full_pass_call_ledger": {
        "checkpointed_path_calls": 12, "cross_norm1_calls": 1,
        "total_calls": 13,
    },
    "pairing": (
        "pair every forward input with the output-gradient hook that fires; "
        "require exactly one backward-active paired occurrence per logical "
        "LayerNorm and bind its local native affine gradients to the reproduced "
        "whole-model affine gradients"),
    "checkpointed_initial_recompute_input_output_digests_equal": True,
    "fields": [
        "path", "eps", "normalized_shape", "dtype", "shape", "stride",
        "storage_offset", "contiguous", "forward_call_count",
        "paired_upstream_count", "input_digest", "upstream_gradient_digest",
        "weight_digest", "bias_digest", "input_finite",
        "upstream_gradient_finite", "weight_finite", "bias_finite",
        "initial_recompute_input_digests",
        "initial_recompute_output_digests",
        "input_layout", "upstream_gradient_layout", "weight_layout",
        "bias_layout",
        "native_whole_model_weight_gradient",
        "native_whole_model_bias_gradient",
    ],
    "persisted_captured_tensor_values": False,
    "persisted_activations_or_gradients": False,
    "ephemeral_tensors_destroyed_before_terminal": True,
}
LOCAL_RESULT_FIELDS = [
    "case", "path", "device", "formula", "dtype", "shape", "stride",
    "eps", "forward", "input_gradient", "weight_gradient", "bias_gradient",
    "forward_comparison_to_cpu", "input_gradient_comparison_to_cpu",
    "weight_gradient_comparison_to_cpu", "bias_gradient_comparison_to_cpu",
]
TENSOR_RESULT_FIELDS = [
    "dtype", "shape", "stride", "storage_offset", "contiguous",
    "tensor_digest", "finite_count", "nan_count",
    "positive_infinity_count", "negative_infinity_count",
    "maximum_absolute_finite_value",
]
COMPARISON_RESULT_FIELDS = [
    "both_finite", "equivalent", "maximum_absolute_difference",
    "maximum_relative_difference", "common_finite_count",
    "maximum_tolerance_excess",
    "absolute_tolerance", "relative_tolerance",
]
LOCAL_SUCCESS_CLASSIFICATION = (
    "BACKEND_LAYERNORM_AFFINE_GRADIENT_DEFECT_CONTRACT_PRESERVING")
LOCAL_CLASSIFICATION_RULE = {
    "all_seven_captured_inputs_and_upstream_gradients_finite": True,
    "all_shapes_end_in_512": True,
    "cpu_native_all_input_weight_bias_gradients_finite": True,
    "cpu_native_forward_finite": True,
    "gpu_native_forward_and_input_gradient_finite": True,
    "gpu_native_each_weight_and_bias_gradient": {
        "finite_count": 256, "nan_count": 256,
        "positive_infinity_count": 0, "negative_infinity_count": 0,
    },
    "gpu_explicit_affine_all_input_weight_bias_gradients_finite": True,
    "gpu_explicit_affine_forward_finite": True,
    "gpu_explicit_affine_forward_and_all_three_gradients_agree_with_cpu":
        TOLERANCES,
    "sole_success_classification": LOCAL_SUCCESS_CLASSIFICATION,
    "otherwise_mechanism_classification": None,
}

SMOKE_CONTRACT = {
    "conditional_on_local_success": True,
    "fixture_digest": FROZEN_FIXTURE_DIGEST,
    "calibration_rows_or_latents_opened": 0,
    "fresh_registered_initial_state": FROZEN_INITIAL_STATE_DIGEST,
    "model_mode": "train",
    "activation_checkpointing": True,
    "production_backend_and_fp32_ambient": PREDECESSOR.EXECUTION_ENVIRONMENT,
    "implementation_name": IMPLEMENTATION_NAME,
    "implementation_digest": IMPLEMENTATION_DIGEST,
    "externalised_paths": list(LAYER_NORM_PATHS),
    "optimizer": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01,
    "gradient_clip_max_norm": 1.0,
    "optimizer_updates": 1,
    "loss": PREDECESSOR.LOSS_CONTRACT["frozen_summed_loss"],
    "requirements": [
        "finite component outputs and loss",
        "component outputs and loss equivalent to exact native reproduction "
        "at forward atol=2e-6 rtol=2e-5",
        "finite complete parameter gradients before clipping",
        "nonzero finite token-projection gradient",
        "nonzero finite gradients in all four attention modules",
        "three finite nonzero and pairwise-distinct component-query gradients",
        "finite affine gradients at all seven externalised LayerNorm paths",
        "finite native affine gradients at unchanged cross-attention norm2",
        "finite clip norm and finite clipped gradients",
        "exactly one AdamW step",
        "finite model and optimizer state after step",
        "checkpoint receipt reload reproduces model and optimizer digests",
    ],
    "checkpoint_is_diagnostic_only": True,
    "scorer_training_started": False,
}

TERMINAL_KINDS = (
    "NONREPRODUCTION_STOP", "LOCAL_EQUIVALENCE_FAILURE_STOP",
    "CONDITIONAL_SMOKE_FAILURE_STOP", "SUCCESS_CLASSIFICATION")
SUCCESSOR_RULE = {
    "success": (
        "local success classification plus passed conditional whole-model smoke"),
    "local_failure": (
        "null mechanism and no successor eligibility"),
    "smoke_failure": (
        "retain local mechanism but set successor eligibility false"),
    "successor_eligibility_label": IMPLEMENTATION_NAME,
    "separately_committed_successor_required": True,
    "training_authorised_by_this_contract": False,
    "any_non_success_closes_current_attentive_readout_line": True,
    "further_implementation_test_automatically_authorised": False,
}

EXECUTION_LIMITS = {
    "run_session_scope_excludes_read_only_consumer_validation": True,
    "sessions": 1, "exact_reproductions": 1,
    "captured_layernorms": 7, "local_cases": 21,
    "conditional_whole_model_smokes": 1,
    "maximum_fresh_model_constructions": 3,
    "conditional_checkpoint_reload_model_constructions": 1,
    "maximum_whole_model_forwards": 2,
    "maximum_whole_model_backwards": 2,
    "maximum_optimizer_constructions": 3,
    "conditional_checkpoint_reload_optimizer_constructions": 1,
    "validator_only_cpu_model_constructions": 1,
    "validator_only_cpu_optimizer_constructions": 1,
    "maximum_optimizer_steps": 1,
    "maximum_gradient_clips": 1,
    "maximum_checkpoint_writes": 1,
    "calibration_rows_opened": 0, "predictor_checkpoints_opened": 0,
}
AUTHORITY = {
    "diagnostic_model_execution": True,
    "ephemeral_captured_tensor_use": True,
    "local_layernorm_forward_backward": True,
    "conditional_one_update_smoke": True,
    "diagnostic_checkpoint_write_reload": True,
    "scientific_training": False, "scorer_qualification": False,
    "repair_implementation": False, "successor_implementation": False,
    "predictor_access": False, "calibration_access": False,
    "corpus_generation": False, "target_encoding": False,
}

HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LayerNormAffineContractError(RuntimeError):
    """The frozen source, predecessor, storage, or contract changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerNormAffineContractError(message)


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
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LayerNormAffineContractError(f"{label} is invalid") from exc
    require(isinstance(result, dict), f"{label} is not an object")
    return result


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(key, None)
    require(isinstance(recorded, str) and HEX64.fullmatch(recorded) is not None
            and recorded == digest(result), f"{label} self digest changed")
    result[key] = recorded
    return result


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LayerNormAffineContractError(f"cannot bind source: {exc}") from exc


def source_closure(root: Path = ROOT) -> dict[str, Any]:
    require(_git(root, "status", "--porcelain=v1") == "",
            "source must be clean and committed")
    head = _git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(ancestor.returncode == 0, "source does not descend from frozen base")
    changed = tuple(sorted(filter(None, _git(
        root, "diff", "--name-only", f"{BASE_SOURCE_COMMIT}..{head}"
    ).splitlines())))
    require(changed == tuple(sorted(NEW_SOURCE_PATHS)),
            "committed diff is not exactly four additive paths")
    frozen = {}
    for relative, (expected_sha, expected_bytes) in FROZEN_DEPENDENCY_FILES.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == expected_bytes
                and file_sha256(path) == expected_sha,
                f"frozen predecessor source changed at {relative}")
        frozen[relative] = {"sha256": expected_sha,
                            "byte_count": expected_bytes}
    additive = {}
    for relative in NEW_SOURCE_PATHS:
        path = root / relative
        require(path.is_file() and not path.is_symlink(),
                f"additive source is absent: {relative}")
        additive[relative] = {"sha256": file_sha256(path),
                              "byte_count": path.stat().st_size}
    payload = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": head,
        "source_repository_clean": True,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "exact_committed_additive_path_diff": list(changed),
        "frozen_dependency_files": frozen, "additive_files": additive,
    }
    return {**payload, SOURCE_CLOSURE_SELF_KEY: digest(payload)}


def storage_binding(root: Path = ROOT) -> dict[str, Any]:
    logical = root / GENERATED_PARENT
    require(logical.is_symlink() and logical.resolve() == REGISTERED_PARENT,
            "registered generated-parent symlink changed")
    runtime = runtime_root(root)
    require(not runtime.exists() and not runtime.is_symlink(),
            "one-shot runtime namespace already exists")
    stat = REGISTERED_PARENT.stat()
    return {
        "logical_parent": str(GENERATED_PARENT),
        "registered_parent": str(REGISTERED_PARENT),
        "logical_parent_is_symlink": True,
        "resolved_parent": str(logical.resolve()),
        "registered_parent_device": int(stat.st_dev),
        "runtime_relative": str(RUNTIME_RELATIVE),
        "runtime_namespace_absent_before_issue": True,
    }


def predecessor_binding(root: Path = ROOT) -> dict[str, Any]:
    terminal = CONSUMER.validate_completed_terminal(root)
    require(terminal[CONSUMER.RUNNER.TERMINAL_SELF_KEY]
            == PREDECESSOR_TERMINAL_DIGEST
            and terminal["mechanism_classification"]
            == PREDECESSOR_CLASSIFICATION,
            "completed predecessor terminal changed")
    return {
        "source_commit": BASE_SOURCE_COMMIT,
        "frozen_diagnostic_source_commit": FROZEN_DIAGNOSTIC_SOURCE_COMMIT,
        "source_closure_digest": PREDECESSOR_SOURCE_CLOSURE_DIGEST,
        "consumer_receipt_digest": PREDECESSOR_CONSUMER_RECEIPT_DIGEST,
        "installed_contract_digest": PREDECESSOR_CONTRACT_DIGEST,
        "terminal_digest": PREDECESSOR_TERMINAL_DIGEST,
        "terminal_classification": PREDECESSOR_CLASSIFICATION,
        "reproduction_digest": PREDECESSOR_REPRODUCTION_DIGEST,
        "attempt_digest": PREDECESSOR_ATTEMPT_DIGEST,
        "terminal_file_sha256": PREDECESSOR_TERMINAL_FILE_SHA256,
        "artifact_set_digest": PREDECESSOR_ARTIFACT_SET_DIGEST,
        "runtime_artifacts": PREDECESSOR_RUNTIME_ARTIFACTS,
    }


def static_contract() -> dict[str, Any]:
    return {
        "status": STATUS,
        "frozen_architecture_digest": FROZEN_ARCHITECTURE_DIGEST,
        "frozen_fixture_digest": FROZEN_FIXTURE_DIGEST,
        "frozen_initial_state_digest": FROZEN_INITIAL_STATE_DIGEST,
        "frozen_attentive_seed": FROZEN_ATTENTIVE_SEED,
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "trainable_parameter_tensor_count": TRAINABLE_PARAMETER_TENSOR_COUNT,
        "layernorm_paths": list(LAYER_NORM_PATHS),
        "negative_control_path": NEGATIVE_CONTROL_PATH,
        "native_nonfinite_parameter_names": list(
            NATIVE_NONFINITE_PARAMETER_NAMES),
        "exact_reproduction_gate": EXACT_REPRODUCTION_GATE,
        "implementation_contract": IMPLEMENTATION_CONTRACT,
        "implementation_digest": IMPLEMENTATION_DIGEST,
        "tolerances": TOLERANCES, "capture_contract": CAPTURE_CONTRACT,
        "local_cases": LOCAL_CASES, "local_result_fields": LOCAL_RESULT_FIELDS,
        "tensor_result_fields": TENSOR_RESULT_FIELDS,
        "comparison_result_fields": COMPARISON_RESULT_FIELDS,
        "local_classification_rule": LOCAL_CLASSIFICATION_RULE,
        "smoke_contract": SMOKE_CONTRACT,
        "terminal_kinds": list(TERMINAL_KINDS),
        "successor_rule": SUCCESSOR_RULE,
        "execution_limits": EXECUTION_LIMITS, "authority": AUTHORITY,
    }


def build_contract(source: Mapping[str, Any], predecessor: Mapping[str, Any],
                   storage: Mapping[str, Any]) -> dict[str, Any]:
    require(source.get("schema") == SOURCE_CLOSURE_SCHEMA
            and source.get(SOURCE_CLOSURE_SELF_KEY)
            == digest({key: value for key, value in source.items()
                       if key != SOURCE_CLOSURE_SELF_KEY}),
            "source closure is invalid")
    expected_predecessor = {
        "source_commit": BASE_SOURCE_COMMIT,
        "frozen_diagnostic_source_commit": FROZEN_DIAGNOSTIC_SOURCE_COMMIT,
        "source_closure_digest": PREDECESSOR_SOURCE_CLOSURE_DIGEST,
        "consumer_receipt_digest": PREDECESSOR_CONSUMER_RECEIPT_DIGEST,
        "installed_contract_digest": PREDECESSOR_CONTRACT_DIGEST,
        "terminal_digest": PREDECESSOR_TERMINAL_DIGEST,
        "terminal_classification": PREDECESSOR_CLASSIFICATION,
        "reproduction_digest": PREDECESSOR_REPRODUCTION_DIGEST,
        "attempt_digest": PREDECESSOR_ATTEMPT_DIGEST,
        "terminal_file_sha256": PREDECESSOR_TERMINAL_FILE_SHA256,
        "artifact_set_digest": PREDECESSOR_ARTIFACT_SET_DIGEST,
        "runtime_artifacts": PREDECESSOR_RUNTIME_ARTIFACTS,
    }
    require(dict(predecessor) == expected_predecessor,
            "predecessor binding changed")
    require(storage.get("logical_parent_is_symlink") is True
            and storage.get("logical_parent") == str(GENERATED_PARENT)
            and storage.get("registered_parent") == str(REGISTERED_PARENT)
            and storage.get("resolved_parent") == str(REGISTERED_PARENT)
            and storage.get("runtime_relative") == str(RUNTIME_RELATIVE)
            and storage.get("runtime_namespace_absent_before_issue") is True,
            "storage binding changed")
    payload = {
        "schema": CONTRACT_SCHEMA, "source_closure": dict(source),
        "predecessor": dict(predecessor), "storage": dict(storage),
        **static_contract(),
    }
    return {**payload, CONTRACT_SELF_KEY: digest(payload)}


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = validate_signed(value, CONTRACT_SELF_KEY, "contract")
    expected = build_contract(result["source_closure"], result["predecessor"],
                              result["storage"])
    require(result == expected, "contract changed")
    return result


def runtime_root(root: Path = ROOT) -> Path:
    return root / RUNTIME_RELATIVE


def contract_path(root: Path = ROOT) -> Path:
    return root / CONTRACT_PATH


__all__ = [name for name in globals() if name.isupper()] + [
    "LayerNormAffineContractError", "build_contract", "canonical_bytes",
    "contract_path", "digest", "file_sha256", "predecessor_binding",
    "read_json", "require", "runtime_root", "source_closure",
    "static_contract", "storage_binding", "validate_contract",
    "validate_signed",
]
