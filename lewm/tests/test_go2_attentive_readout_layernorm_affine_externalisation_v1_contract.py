"""Focused source-only tests for the LayerNorm-affine contract."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.oracle import (
    go2_attentive_readout_layernorm_affine_externalisation_v1_contract as C,
)


ROOT = Path(__file__).resolve().parents[2]


def _source() -> dict:
    payload = {
        "schema": C.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": "a" * 40,
        "source_repository_clean": True,
        "base_source_commit": C.BASE_SOURCE_COMMIT,
        "exact_committed_additive_path_diff": list(C.NEW_SOURCE_PATHS),
        "frozen_dependency_files": {}, "additive_files": {},
    }
    return {**payload, C.SOURCE_CLOSURE_SELF_KEY: C.digest(payload)}


def _predecessor() -> dict:
    return {
        "source_commit": C.BASE_SOURCE_COMMIT,
        "frozen_diagnostic_source_commit": C.FROZEN_DIAGNOSTIC_SOURCE_COMMIT,
        "source_closure_digest": C.PREDECESSOR_SOURCE_CLOSURE_DIGEST,
        "consumer_receipt_digest": C.PREDECESSOR_CONSUMER_RECEIPT_DIGEST,
        "installed_contract_digest": C.PREDECESSOR_CONTRACT_DIGEST,
        "terminal_digest": C.PREDECESSOR_TERMINAL_DIGEST,
        "terminal_classification": C.PREDECESSOR_CLASSIFICATION,
        "reproduction_digest": C.PREDECESSOR_REPRODUCTION_DIGEST,
        "attempt_digest": C.PREDECESSOR_ATTEMPT_DIGEST,
        "terminal_file_sha256": C.PREDECESSOR_TERMINAL_FILE_SHA256,
        "artifact_set_digest": C.PREDECESSOR_ARTIFACT_SET_DIGEST,
        "runtime_artifacts": C.PREDECESSOR_RUNTIME_ARTIFACTS,
    }


def _storage() -> dict:
    return {
        "logical_parent": str(C.GENERATED_PARENT),
        "registered_parent": str(C.REGISTERED_PARENT),
        "logical_parent_is_symlink": True,
        "resolved_parent": str(C.REGISTERED_PARENT),
        "registered_parent_device": 1,
        "runtime_relative": str(C.RUNTIME_RELATIVE),
        "runtime_namespace_absent_before_issue": True,
    }


def test_build_validate_freezes_exact_contract() -> None:
    value = C.build_contract(_source(), _predecessor(), _storage())
    assert C.validate_contract(value) == value
    changed = copy.deepcopy(value)
    changed["tolerances"]["forward"]["absolute"] = 3e-6
    unsigned = dict(changed)
    unsigned.pop(C.CONTRACT_SELF_KEY)
    changed[C.CONTRACT_SELF_KEY] = C.digest(unsigned)
    with pytest.raises(C.LayerNormAffineContractError, match="contract changed"):
        C.validate_contract(changed)


def test_seven_paths_and_literal_externalised_formula_are_exact() -> None:
    assert C.LAYER_NORM_PATHS == (
        "pooler.cross_attention_block.norm1",
        "pooler.blocks.0.norm1", "pooler.blocks.0.norm2",
        "pooler.blocks.1.norm1", "pooler.blocks.1.norm2",
        "pooler.blocks.2.norm1", "pooler.blocks.2.norm2")
    assert C.NEGATIVE_CONTROL_PATH == "pooler.cross_attention_block.norm2"
    assert C.IMPLEMENTATION_CONTRACT["externalised_formula"] == (
        "normalized=torch.nn.functional.layer_norm(x,(512,),weight=None,"
        "bias=None,eps=eps); y=normalized*weight+bias")
    assert C.IMPLEMENTATION_CONTRACT[
        "parameter_objects_shapes_and_state_dict_keys_unchanged"]
    assert C.IMPLEMENTATION_CONTRACT[
        "trainable_parameter_count_preserved"] == 13_684_739


def test_reproduction_capture_and_local_case_gates_are_fail_closed() -> None:
    gate = C.EXACT_REPRODUCTION_GATE
    assert gate["exact_nonfinite_parameter_names"] == list(
        C.NATIVE_NONFINITE_PARAMETER_NAMES)
    assert gate["each_native_nonfinite_affine_gradient"] == {
        "shape": [512], "finite_count": 256, "nan_count": 256,
        "positive_infinity_count": 0, "negative_infinity_count": 0}
    assert gate["optimizer_step"] is False and gate["gradient_clip"] is False
    assert C.CAPTURE_CONTRACT["full_pass_call_ledger"]["total_calls"] == 13
    assert C.CAPTURE_CONTRACT["persisted_captured_tensor_values"] is False
    assert set(C.LOCAL_CASES) == {
        "CPU_NATIVE", "GPU_NATIVE", "GPU_EXPLICIT_AFFINE"}
    assert C.TOLERANCES["forward"] == {"absolute": 2e-6,
                                      "relative": 2e-5}
    assert C.TOLERANCES["weight_gradient"] == {
        "absolute": 1e-5, "relative": 1e-4}


def test_conditional_smoke_is_one_update_but_not_training_authority() -> None:
    smoke = C.SMOKE_CONTRACT
    assert smoke["optimizer"] == "AdamW"
    assert smoke["learning_rate"] == 3e-4
    assert smoke["weight_decay"] == 0.01
    assert smoke["gradient_clip_max_norm"] == 1.0
    assert smoke["optimizer_updates"] == 1
    assert smoke["calibration_rows_or_latents_opened"] == 0
    assert C.AUTHORITY["conditional_one_update_smoke"] is True
    assert C.AUTHORITY["scientific_training"] is False
    assert C.AUTHORITY["successor_implementation"] is False
    assert C.SUCCESSOR_RULE["separately_committed_successor_required"] is True
    assert C.SUCCESSOR_RULE["training_authorised_by_this_contract"] is False
    assert C.EXECUTION_LIMITS["maximum_fresh_model_constructions"] == 3
    assert C.EXECUTION_LIMITS["maximum_optimizer_constructions"] == 3


def test_lineage_and_storage_namespace_are_unambiguous() -> None:
    assert C.BASE_SOURCE_COMMIT == (
        "3a00b9819926e8552dadaab89ea59fdb3aeffe96")
    assert C.FROZEN_DIAGNOSTIC_SOURCE_COMMIT == (
        "ccdb4de735a71760cd2683e491ce221240bcf6e4")
    assert C.PREDECESSOR_ATTEMPT_DIGEST.startswith("3ab590d4")
    assert C.PREDECESSOR_TERMINAL_FILE_SHA256.startswith("0afe02dd")
    assert C.PREDECESSOR_ARTIFACT_SET_DIGEST.startswith("00e97586")
    assert C.RUNTIME_RELATIVE == C.GENERATED_PARENT / (
        "attentive_readout_layernorm_affine_externalisation_v1")
    assert len(C.NEW_SOURCE_PATHS) == 4
    assert Path(__file__).relative_to(ROOT).as_posix() in C.NEW_SOURCE_PATHS
