"""Focused CPU/source tests for the LayerNorm-affine diagnostic runner."""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from lewm.oracle import (
    go2_attentive_readout_layernorm_affine_externalisation_v1_contract as C,
)
from scripts import (
    diagnose_go2_attentive_readout_layernorm_affine_externalisation_v1 as D,
)


ROOT = Path(__file__).resolve().parents[2]


def test_comparison_uses_predeclared_elementwise_tolerance() -> None:
    reference = torch.tensor([1.0, -2.0], dtype=torch.float32)
    candidate = reference + torch.tensor([1e-6, 1e-5])
    result = D.comparison(reference, candidate, C.TOLERANCES["forward"])
    assert result["both_finite"] and result["equivalent"]
    changed = D.comparison(reference, reference + 1e-2,
                           C.TOLERANCES["forward"])
    assert not changed["equivalent"]


def test_literal_explicit_affine_matches_native_forward_and_gradients_cpu() -> None:
    generator = torch.Generator().manual_seed(4)
    captured = {
        "input": torch.randn(2, 3, 512, generator=generator),
        "upstream": torch.randn(2, 3, 512, generator=generator),
        "weight": torch.randn(512, generator=generator),
        "bias": torch.randn(512, generator=generator), "eps": 1e-5,
    }
    native_receipt, native = D.local_case(
        captured, case="CPU_NATIVE", device=torch.device("cpu"))
    # Exercise the exact C formula without requiring a GPU in source tests.
    x = captured["input"].clone().requires_grad_(True)
    weight = captured["weight"].clone().requires_grad_(True)
    bias = captured["bias"].clone().requires_grad_(True)
    normalized = F.layer_norm(x, (512,), weight=None, bias=None, eps=1e-5)
    output = normalized * weight + bias
    output.backward(captured["upstream"])
    explicit = {"forward": output.detach(), "input_gradient": x.grad,
                "weight_gradient": weight.grad, "bias_gradient": bias.grad}
    assert native_receipt["formula"] == C.IMPLEMENTATION_CONTRACT["native_formula"]
    for key, tolerance_key in (
            ("forward", "forward"), ("input_gradient", "input_gradient"),
            ("weight_gradient", "weight_gradient"),
            ("bias_gradient", "bias_gradient")):
        assert D.comparison(native[key], explicit[key],
                            C.TOLERANCES[tolerance_key])["equivalent"]


def test_tensor_receipt_records_nan_inf_and_layout_without_values() -> None:
    value = torch.tensor([[1.0, float("nan")],
                          [float("inf"), float("-inf")]])
    receipt = D.tensor_receipt(value)
    assert receipt["shape"] == [2, 2] and receipt["stride"] == [2, 1]
    assert receipt["finite_count"] == 1 and receipt["nan_count"] == 1
    assert receipt["positive_infinity_count"] == 1
    assert receipt["negative_infinity_count"] == 1
    assert "values" not in receipt


def test_local_success_recomputation_requires_all_seven_paths() -> None:
    local = {"path_count": 7, "case_count": 21,
             "paths": [{"path": path, "passed": True}
                       for path in C.LAYER_NORM_PATHS]}
    assert D.recompute_local_success(local)
    local["paths"][3]["passed"] = False
    assert not D.recompute_local_success(local)
    local["paths"] = local["paths"][:-1]
    with pytest.raises(D.LayerNormAffineError, match="cardinality changed"):
        D.recompute_local_success(local)


def test_externalisation_context_patches_exactly_seven_and_restores() -> None:
    class Holder(torch.nn.Module):
        pass

    model = Holder()
    model.pooler = Holder()
    model.pooler.cross_attention_block = Holder()
    model.pooler.cross_attention_block.norm1 = torch.nn.LayerNorm(512)
    model.pooler.cross_attention_block.norm2 = torch.nn.LayerNorm(512)
    model.pooler.blocks = torch.nn.ModuleList()
    for _ in range(3):
        block = Holder()
        block.norm1 = torch.nn.LayerNorm(512)
        block.norm2 = torch.nn.LayerNorm(512)
        model.pooler.blocks.append(block)
    modules = dict(model.named_modules())
    originals = {path: modules[path].forward for path in C.LAYER_NORM_PATHS}
    negative = modules[C.NEGATIVE_CONTROL_PATH].forward
    with D.externalised_layernorms(model):
        assert all(modules[path].forward != originals[path]
                   for path in C.LAYER_NORM_PATHS)
        assert modules[C.NEGATIVE_CONTROL_PATH].forward == negative
    assert all(modules[path].forward == originals[path]
               for path in C.LAYER_NORM_PATHS)


def test_source_contains_no_scientific_training_or_predictor_route() -> None:
    source = inspect.getsource(D)
    assert "optimizer.step()" in source  # one conditional smoke update
    assert "train_once(" not in source
    assert "_evaluate_streaming(" not in source
    assert "calibration_evidence" not in source
    assert "predictor checkpoint" not in source.lower()
    assert "scientific_training_authorised\": True" not in source
    assert Path(__file__).relative_to(ROOT).as_posix() in C.NEW_SOURCE_PATHS


def _gradient_row(index: int) -> dict:
    return {
        "fully_qualified_name": f"parameter.{index}", "module_path": "parameter",
        "module_type": "torch.nn.Parameter", "shape": [1],
        "parameter_dtype": "torch.float32", "gradient_dtype": "torch.float32",
        "gradient_is_none": False, "finite_count": 1, "nan_count": 0,
        "positive_infinity_count": 0, "negative_infinity_count": 0,
        "maximum_absolute_finite_value": 1.0, "finite_only_l2_norm": 1.0,
        "first_nonfinite_flat_index": None,
        "first_nonfinite_multi_index": None,
        "gradient_tensor_digest": f"{index:064x}",
    }


def _sdpa_audit() -> dict:
    specifications = [
        ("initial_forward", "self", 3072),
        ("initial_forward", "self", 3072),
        ("initial_forward", "self", 3072),
        ("initial_forward", "cross", 3),
        ("backward_checkpoint_recompute", "self", 3072),
        ("backward_checkpoint_recompute", "self", 3072),
        ("backward_checkpoint_recompute", "self", 3072),
    ]
    rows = [{
        "invocation_index": index, "phase": phase, "kind": kind,
        "query_shape": [4, 16, query, 32],
        "key_shape": [4, 16, 3072, 32],
        "value_shape": [4, 16, 3072, 32], "dtype": "torch.float32",
        "selected_backend_inside_effective_context": "EFFICIENT_ATTENTION",
        "dropout_p": 0.0, "is_causal": False,
    } for index, (phase, kind, query) in enumerate(specifications)]
    return {"invocations": rows,
            "backend_sequence": ["EFFICIENT_ATTENTION"] * 7}


def _passing_comparison() -> dict:
    return {"both_finite": True, "equivalent": True,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "common_finite_count": 4, "maximum_tolerance_excess": -2e-6,
            "absolute_tolerance": 2e-6, "relative_tolerance": 2e-5}


def test_smoke_semantics_survive_json_round_trip_and_reject_mutation(
        monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = [_gradient_row(index) for index in range(63)]
    attention_paths = ["pooler.blocks.0.attn", "pooler.blocks.1.attn",
                       "pooler.blocks.2.attn",
                       "pooler.cross_attention_block.xattn"]
    required_names = (["token_projection.weight", "pooler.query_tokens"]
                      + [f"{path}.weight" for path in attention_paths]
                      + [f"{path}.{suffix}"
                         for path in (*C.LAYER_NORM_PATHS,
                                      C.NEGATIVE_CONTROL_PATH)
                         for suffix in ("weight", "bias")])
    for row, name in zip(inventory, required_names, strict=False):
        row["fully_qualified_name"] = name
    monkeypatch.setattr(C, "FROZEN_PARAMETER_INVENTORY_DIGEST", C.digest([
        [row["fully_qualified_name"], row["shape"], row["parameter_dtype"]]
        for row in inventory]))
    verdict = D.D.gradient_verdict(inventory)
    smoke = {
        "schema": D.SMOKE_SCHEMA, "status": C.STATUS,
        "contract_digest": "c" * 64,
        "implementation_name": C.IMPLEMENTATION_NAME,
        "implementation_digest": C.IMPLEMENTATION_DIGEST,
        "fixture_digest": C.FROZEN_FIXTURE_DIGEST,
        "initial_state_digest": C.FROZEN_INITIAL_STATE_DIGEST,
        "all_outputs_and_loss_finite": True,
        "output_comparisons_to_native": {
            name: _passing_comparison() for name in D.D.COMPONENTS},
        "loss_comparison_to_native": _passing_comparison(),
        "all_parameter_gradients_finite": True,
        "parameter_gradient_inventory": inventory,
        "complete_gradient_verdict": verdict,
        "gradient_evidence": {
            "token_projection_finite_nonzero": True,
            "token_projection_gradient_digest": inventory[0][
                "gradient_tensor_digest"],
            "token_projection_gradient_l2": 1.0,
            "all_four_attention_modules_finite_nonzero": True,
            "attention_modules": [{"path": path, "finite_nonzero": True,
                                   "gradient_l2": 1.0}
                                  for path in attention_paths],
            "component_queries_finite_nonzero_pairwise_distinct": True,
            "component_query_gradient_digest": inventory[1][
                "gradient_tensor_digest"],
            "component_query_rows": [
                {"index": index, "gradient_l2": 1.0,
                 "gradient_digest": f"q{index}"}
                for index in range(3)],
        },
        "layernorm_affine_gradients_finite": {
            path: True for path in (*C.LAYER_NORM_PATHS,
                                    C.NEGATIVE_CONTROL_PATH)},
        "gradient_clip_max_norm": 1.0, "preclip_total_norm": 2.0,
        "all_clipped_gradients_finite": True,
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 0.01},
        "completed_optimizer_updates": 1, "fresh_model_constructions": 2,
        "optimizer_constructions": 2, "whole_model_forwards": 1,
        "whole_model_backwards": 1,
        "post_step_model_and_optimizer_finite": True,
        "sdpa_audit": _sdpa_audit(),
        "checkpoint": {"strict_model_optimizer_reload_passed": True},
        "calibration_rows_opened": 0, "predictor_checkpoints_opened": 0,
        "scorer_training_started": False,
        "wall_time_seconds": 1.0, "peak_vram_bytes": 1,
    }
    round_tripped = json.loads(json.dumps(smoke, sort_keys=True))
    assert D.recompute_smoke_success(round_tripped)
    round_tripped["gradient_evidence"][
        "component_queries_finite_nonzero_pairwise_distinct"] = False
    assert not D.recompute_smoke_success(round_tripped)
    round_tripped = json.loads(json.dumps(smoke, sort_keys=True))
    round_tripped["whole_model_backwards"] = 0
    assert not D.recompute_smoke_success(round_tripped)
    round_tripped = json.loads(json.dumps(smoke, sort_keys=True))
    round_tripped["parameter_gradient_inventory"][0][
        "fully_qualified_name"] = "wrong.parameter"
    assert not D.recompute_smoke_success(round_tripped)
    round_tripped = json.loads(json.dumps(smoke, sort_keys=True))
    round_tripped["wall_time_seconds"] = 0.0
    assert not D.recompute_smoke_success(round_tripped)


def _tensor_row(shape: list[int], *, nan_count: int = 0,
                digest: str = "d" * 64) -> dict:
    elements = math.prod(shape)
    return {"dtype": "torch.float32", "shape": shape,
            "stride": ([shape[-1], 1] if len(shape) == 2 else [1]),
            "storage_offset": 0, "contiguous": True,
            "tensor_digest": digest, "finite_count": elements - nan_count,
            "nan_count": nan_count, "positive_infinity_count": 0,
            "negative_infinity_count": 0,
            "maximum_absolute_finite_value": 1.0}


def _local_comparison(tolerance_key: str, *, finite: bool = True) -> dict:
    tolerance = C.TOLERANCES[tolerance_key]
    return {"both_finite": finite, "equivalent": finite,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "common_finite_count": 512,
            "maximum_tolerance_excess": -tolerance["absolute"],
            "absolute_tolerance": tolerance["absolute"],
            "relative_tolerance": tolerance["relative"]}


def test_local_semantic_replay_rejects_resigned_style_tolerance_mutation() -> None:
    captures = {}
    paths = []
    for path in C.LAYER_NORM_PATHS:
        shape = [4, 3, 512]
        capture_weight = _tensor_row([512], nan_count=256, digest="a" * 64)
        capture_bias = _tensor_row([512], nan_count=256, digest="b" * 64)
        captures[path] = {
            "dtype": "torch.float32", "shape": shape,
            "stride": [1536, 512, 1], "eps": 1e-5,
            "native_whole_model_weight_gradient": capture_weight,
            "native_whole_model_bias_gradient": capture_bias,
        }
        cases = {}
        for case_name in C.LOCAL_CASES:
            explicit = case_name == "GPU_EXPLICIT_AFFINE"
            native = case_name == "GPU_NATIVE"
            case = {"case": case_name, "path": path,
                    "device": "cpu" if case_name == "CPU_NATIVE" else "cuda:0",
                    "formula": C.IMPLEMENTATION_CONTRACT[
                        "externalised_formula" if explicit else "native_formula"],
                    "dtype": "torch.float32", "shape": shape,
                    "stride": [1536, 512, 1], "eps": 1e-5,
                    "captured_layout_equal": True}
            for key in ("forward", "input_gradient"):
                case[key] = _tensor_row(shape)
                case[f"{key}_comparison_to_cpu"] = _local_comparison(key)
            for key, digest in (("weight_gradient", "a" * 64),
                                ("bias_gradient", "b" * 64)):
                case[key] = _tensor_row(
                    [512], nan_count=256 if native else 0, digest=digest)
                case[f"{key}_comparison_to_cpu"] = _local_comparison(
                    key, finite=not native)
                if native:
                    case[f"{key}_comparison_to_cpu"]["common_finite_count"] = 256
            if native:
                case["exact_whole_model_native_affine_gradient_match"] = {
                    "weight": True, "bias": True}
            cases[case_name] = case
        paths.append({"path": path, "passed": True, "cases": cases})
    reproduction = {"contract_digest": "c" * 64, "captures": captures}
    local = {"schema": D.LOCAL_SCHEMA, "status": C.STATUS,
             "contract_digest": "c" * 64, "path_count": 7, "case_count": 21,
             "paths": paths, "classification_rule": C.LOCAL_CLASSIFICATION_RULE,
             "all_paths_pass": True,
             "mechanism_classification": C.LOCAL_SUCCESS_CLASSIFICATION,
             "captured_tensor_values_persisted": False,
             "calibration_rows_opened": 0,
             "wall_time_seconds": 1.0, "peak_vram_bytes": 1}
    round_tripped = json.loads(json.dumps(local, sort_keys=True))
    assert D.validate_local_semantics(round_tripped, reproduction)
    comparison_row = round_tripped["paths"][0]["cases"][
        "GPU_EXPLICIT_AFFINE"]["forward_comparison_to_cpu"]
    comparison_row["maximum_tolerance_excess"] = 1.0
    with pytest.raises(D.LayerNormAffineError, match="local path predicate"):
        D.validate_local_semantics(round_tripped, reproduction)


def test_failure_validator_rejects_authority_mutation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = C.runtime_root(tmp_path)
    runtime.mkdir(parents=True)
    contract_path = D.path(tmp_path, "contract")
    failure_path = D.path(tmp_path, "failure")
    contract_path.write_text("{}", encoding="utf-8")
    failure_path.write_text("{}", encoding="utf-8")
    contract_path.chmod(0o444)
    failure_path.chmod(0o444)
    contract_digest = "c" * 64
    failure = {
        "schema": D.FAILURE_SCHEMA, "status": C.STATUS, "complete": True,
        "contract_digest": contract_digest, "attempt_digest": None,
        "stage": "device_preflight",
        "retry_resume_or_replacement_authorised": False,
        "training_authorised": False, "repair_authorised": False,
        "calibration_rows_opened": 0, "predictor_checkpoints_opened": 0,
        "completed_optimizer_updates": 0, "completed_gradient_clips": 0,
        "preserved_artifacts": {
            "contract": {"sha256": C.file_sha256(contract_path),
                         "byte_count": contract_path.stat().st_size,
                         "self_digest": "d" * 64}},
    }
    monkeypatch.setattr(D, "load_contract",
                        lambda _root: {C.CONTRACT_SELF_KEY: contract_digest})

    def fake_read(_root: Path, name: str, key: str, _label: str) -> dict:
        if name == "failure":
            return failure
        return {key: "d" * 64}

    monkeypatch.setattr(D, "read_artifact", fake_read)
    assert D.validate_failure(tmp_path) is failure
    failure["training_authorised"] = True
    with pytest.raises(D.LayerNormAffineError, match="technical failure changed"):
        D.validate_failure(tmp_path)


def test_nonfinite_smoke_gradient_evidence_remains_canonical_json_safe() -> None:
    class Holder(torch.nn.Module):
        pass

    model = Holder()
    model.token_projection = torch.nn.Linear(1, 1, bias=False)
    model.pooler = Holder()
    model.pooler.query_tokens = torch.nn.Parameter(torch.zeros(1, 3, 512))
    model.pooler.blocks = torch.nn.ModuleList()
    for _ in range(3):
        block = Holder()
        block.attn = torch.nn.Linear(1, 1, bias=False)
        model.pooler.blocks.append(block)
    model.pooler.cross_attention_block = Holder()
    model.pooler.cross_attention_block.xattn = torch.nn.Linear(
        1, 1, bias=False)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    evidence = D.attention_and_query_evidence(model)
    assert evidence["token_projection_gradient_l2"] is None
    assert all(row["gradient_l2"] is None
               for row in evidence["attention_modules"])
    assert all(row["gradient_l2"] is None
               for row in evidence["component_query_rows"])
    json.dumps(evidence, sort_keys=True, allow_nan=False)
