#!/usr/bin/env python3
"""Run the one-session LayerNorm-affine equivalence diagnostic and smoke."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import (  # noqa: E402
    go2_attentive_readout_layernorm_affine_externalisation_v1_contract as C,
)
from lewm.oracle import (  # noqa: E402
    go2_attentive_readout_gradient_localisation_v1_terminal_consumer_amendment
    as CONSUMER,
)
from scripts import diagnose_go2_attentive_readout_gradient_localisation_v1 as D  # noqa: E402
from scripts import train_go2_utility_scorer_v1_2 as FROZEN  # noqa: E402
from scripts import train_go2_utility_scorer_v1_3_attentive_readout_v1 as ATTENTIVE  # noqa: E402
from scripts import train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1 as AMENDED  # noqa: E402


ATTEMPT_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_attempt_v1"
REPRODUCTION_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_reproduction_v1"
LOCAL_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_local_cases_v1"
SMOKE_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_smoke_v1"
CHECKPOINT_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_smoke_checkpoint_v1"
TERMINAL_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_terminal_v1"
FAILURE_SCHEMA = "go2_attentive_readout_layernorm_affine_externalisation_v1_technical_failure_v1"
ATTEMPT_KEY = "layernorm_affine_attempt_digest"
REPRODUCTION_KEY = "layernorm_affine_reproduction_digest"
LOCAL_KEY = "layernorm_affine_local_cases_digest"
SMOKE_KEY = "layernorm_affine_smoke_digest"
TERMINAL_KEY = "layernorm_affine_terminal_digest"
FAILURE_KEY = "layernorm_affine_technical_failure_digest"

ARTIFACTS = {
    "contract": "contract.json", "attempt": "attempt.json",
    "reproduction": "reproduction.json", "local": "local_cases.json",
    "smoke": "conditional_whole_model_smoke.json",
    "checkpoint": "conditional_whole_model_smoke_checkpoint.pt",
    "terminal": "terminal.json", "failure": "technical_failure.json",
}
RUNTIME_COUNTERS = {"optimizer_updates": 0, "gradient_clips": 0,
                    "calibration_rows_opened": 0,
                    "predictor_checkpoints_opened": 0}


class LayerNormAffineError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LayerNormAffineError(message)


def signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    require(key not in result, f"duplicate self key {key}")
    result[key] = C.digest(result)
    return result


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    try:
        return C.validate_signed(value, key, label)
    except C.LayerNormAffineContractError as exc:
        raise LayerNormAffineError(str(exc)) from exc


def path(root: Path, name: str) -> Path:
    return C.runtime_root(root) / ARTIFACTS[name]


def publish_json_once(target: Path, value: Mapping[str, Any], label: str) -> None:
    require(not target.exists() and not target.is_symlink(), f"{label} exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, target)
        target.chmod(0o444)
    finally:
        if temporary.exists():
            temporary.unlink()


def artifact_binding(target: Path, value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {"path": str(target.relative_to(ROOT)), "sha256": C.file_sha256(target),
            "byte_count": target.stat().st_size, "self_digest": value[key]}


def issue_contract(root: Path = ROOT) -> dict[str, Any]:
    source = C.source_closure(root)
    predecessor = C.predecessor_binding(root)
    storage = C.storage_binding(root)
    contract = C.build_contract(source, predecessor, storage)
    runtime = C.runtime_root(root)
    runtime.mkdir(parents=False, exist_ok=False)
    publish_json_once(path(root, "contract"), contract, "contract")
    return contract


def load_contract(root: Path = ROOT) -> dict[str, Any]:
    contract = C.validate_contract(C.read_json(path(root, "contract"), "contract"))
    require(contract["source_closure"] == C.source_closure(root)
            and contract["predecessor"] == C.predecessor_binding(root),
            "installed contract does not bind live frozen source/predecessor")
    logical = root / C.GENERATED_PARENT
    require(logical.is_symlink() and logical.resolve() == C.REGISTERED_PARENT,
            "registered generated-parent symlink changed")
    return contract


def tensor_receipt(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    stats = D.tensor_numeric_stats(detached)
    return {
        "dtype": str(detached.dtype), "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "storage_offset": int(detached.storage_offset()),
        "contiguous": bool(detached.is_contiguous()),
        "tensor_digest": FROZEN.tensor_digest(detached.cpu()),
        "finite_count": stats["finite_count"], "nan_count": stats["nan_count"],
        "positive_infinity_count": stats["positive_infinity_count"],
        "negative_infinity_count": stats["negative_infinity_count"],
        "maximum_absolute_finite_value": stats["maximum_absolute_finite_value"],
    }


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def comparison(reference: torch.Tensor, candidate: torch.Tensor,
               tolerance: Mapping[str, float]) -> dict[str, Any]:
    require(reference.shape == candidate.shape and reference.dtype == candidate.dtype,
            "comparison tensor shape/dtype changed")
    common = torch.isfinite(reference) & torch.isfinite(candidate)
    common_count = int(common.sum().item())
    if common_count:
        difference = (candidate[common] - reference[common]).abs()
        denominator = reference[common].abs().clamp_min(
            float(C.TOLERANCES["relative_denominator_floor"]))
        maximum_absolute = float(difference.max().item())
        maximum_relative = float((difference / denominator).max().item())
        threshold = (float(tolerance["absolute"])
                     + float(tolerance["relative"])
                     * reference[common].abs())
        maximum_excess = float((difference - threshold).max().item())
    else:
        maximum_absolute = maximum_relative = maximum_excess = None
    both_finite = finite(reference) and finite(candidate)
    equivalent = bool(both_finite and maximum_excess is not None
                      and maximum_excess <= 0.0)
    return {
        "both_finite": both_finite, "equivalent": equivalent,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "common_finite_count": common_count,
        "maximum_tolerance_excess": maximum_excess,
        "absolute_tolerance": tolerance["absolute"],
        "relative_tolerance": tolerance["relative"],
    }


class LayerNormCapture:
    """Hold seven exact inputs/upstream gradients ephemerally for local replay."""

    def __init__(self, model: nn.Module) -> None:
        modules = dict(model.named_modules())
        self.modules = {name: modules[name] for name in C.LAYER_NORM_PATHS}
        require(all(isinstance(module, nn.LayerNorm)
                    and tuple(module.normalized_shape) == (C.FEATURE_WIDTH,)
                    and module.eps == 1e-5 for module in self.modules.values()),
                "target LayerNorm module contract changed")
        self.events: dict[str, list[dict[str, Any]]] = {
            name: [] for name in C.LAYER_NORM_PATHS}
        self.handles = []
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...],
                    output: torch.Tensor) -> None:
            require(len(inputs) == 1, f"LayerNorm input arity changed at {name}")
            row: dict[str, Any] = {
                "input": inputs[0], "output": output, "upstream": None}
            self.events[name].append(row)
            if output.requires_grad:
                output.register_hook(
                    lambda gradient, target=row: target.__setitem__(
                        "upstream", gradient))
        return capture

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def export(self, model: nn.Module) -> tuple[dict[str, Any], dict[str, Any]]:
        modules = dict(model.named_modules())
        receipts: dict[str, Any] = {}
        tensors: dict[str, Any] = {}
        total_calls = 0
        for name in C.LAYER_NORM_PATHS:
            events = self.events[name]
            expected_calls = 1 if name == C.LAYER_NORM_PATHS[0] else 2
            require(len(events) == expected_calls,
                    f"LayerNorm call ledger changed at {name}")
            total_calls += len(events)
            input_digests = [FROZEN.tensor_digest(
                event["input"].detach().cpu()) for event in events]
            output_digests = [FROZEN.tensor_digest(
                event["output"].detach().cpu()) for event in events]
            require(len(set(input_digests)) == 1 and len(set(output_digests)) == 1,
                    f"checkpoint initial/recompute values changed at {name}")
            paired = [event for event in events if event["upstream"] is not None]
            require(len(paired) == 1,
                    f"backward-active LayerNorm occurrence changed at {name}")
            event = paired[0]
            module = modules[name]
            x = event["input"].detach().clone(
                memory_format=torch.preserve_format).cpu()
            dy = event["upstream"].detach().clone(
                memory_format=torch.preserve_format).cpu()
            weight = module.weight.detach().clone().cpu()
            bias = module.bias.detach().clone().cpu()
            require(x.dtype == dy.dtype == weight.dtype == bias.dtype == torch.float32
                    and x.shape == dy.shape and x.shape[-1] == C.FEATURE_WIDTH
                    and all(finite(value) for value in (x, dy, weight, bias)),
                    f"captured LayerNorm tensor contract changed at {name}")
            tensors[name] = {
                "input": x, "upstream": dy, "weight": weight, "bias": bias,
                "native_weight_gradient": module.weight.grad.detach().clone().cpu(),
                "native_bias_gradient": module.bias.grad.detach().clone().cpu(),
                "eps": float(module.eps),
            }
            receipts[name] = {
                "path": name, "eps": float(module.eps),
                "normalized_shape": list(module.normalized_shape),
                "dtype": str(x.dtype), "shape": list(x.shape),
                "stride": list(x.stride()), "storage_offset": int(x.storage_offset()),
                "contiguous": bool(x.is_contiguous()),
                "forward_call_count": len(events), "paired_upstream_count": 1,
                "initial_recompute_input_digests": input_digests,
                "initial_recompute_output_digests": output_digests,
                "input_digest": FROZEN.tensor_digest(x),
                "upstream_gradient_digest": FROZEN.tensor_digest(dy),
                "weight_digest": FROZEN.tensor_digest(weight),
                "bias_digest": FROZEN.tensor_digest(bias),
                "input_layout": {"dtype": str(x.dtype), "shape": list(x.shape),
                                 "stride": list(x.stride())},
                "upstream_gradient_layout": {
                    "dtype": str(dy.dtype), "shape": list(dy.shape),
                    "stride": list(dy.stride())},
                "weight_layout": {"dtype": str(weight.dtype),
                                  "shape": list(weight.shape),
                                  "stride": list(weight.stride())},
                "bias_layout": {"dtype": str(bias.dtype),
                                "shape": list(bias.shape),
                                "stride": list(bias.stride())},
                "native_whole_model_weight_gradient": tensor_receipt(
                    module.weight.grad.detach()),
                "native_whole_model_bias_gradient": tensor_receipt(
                    module.bias.grad.detach()),
                "input_finite": True, "upstream_gradient_finite": True,
                "weight_finite": True, "bias_finite": True,
            }
        require(total_calls == 13, "LayerNorm full-pass call count changed")
        return receipts, tensors


def fresh_model(device: torch.device) -> tuple[nn.Module, str]:
    model, state, state_digest = AMENDED._fresh_model_state()
    require(state_digest == C.FROZEN_INITIAL_STATE_DIGEST
            and sum(parameter.numel() for parameter in model.parameters())
            == C.TRAINABLE_PARAMETER_COUNT
            and sum(1 for _ in model.parameters())
            == C.TRAINABLE_PARAMETER_TENSOR_COUNT
            and model.training and model.pooler.use_activation_checkpointing,
            "fresh attentive model contract changed")
    model.to(device)
    return model, FROZEN.state_dict_digest(state)


def fixture(root: Path) -> dict[str, Any]:
    rows, store, binding = AMENDED._fit_only_smoke_fixture(root)
    require(binding["fixture_digest"] == C.FROZEN_FIXTURE_DIGEST
            and len(rows) == 4
            and binding["calibration_label_rows_opened"] == 0
            and binding["calibration_latent_shards_opened"] == 0,
            "fit-only fixture changed")
    return {"rows": rows, "store": store, "binding": binding}


def run_reproduction(fit: Mapping[str, Any], device: torch.device,
                     root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model, initial_digest = fresh_model(device)
    budget = ATTENTIVE.frozen_budget()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(budget["lr"]),
                                  weight_decay=float(budget["weight_decay"]))
    require(optimizer.state_dict()["state"] == {}, "initial AdamW state changed")
    action_goal, targets = ATTENTIVE._small_features(fit["rows"], device)
    batch_cpu = torch.arange(4, dtype=torch.int64)
    batch = batch_cpu.to(device)
    tokens = ATTENTIVE._token_batch(
        fit["rows"], fit["store"], batch_cpu.tolist(), device)
    require(tokens.dtype == action_goal.dtype == torch.float32
            and not torch.is_autocast_enabled("cuda"),
            "exact reproduction precision changed")
    capture = LayerNormCapture(model)
    sdpa = D.SDPAInventory("production")
    optimizer.zero_grad(set_to_none=True)
    with sdpa:
        outputs = model(tokens, action_goal[batch])
        raw = D._raw_component_losses(outputs, targets, batch)
        total = D._pass_loss(raw, "frozen_summed_loss")
        require(finite(total), "exact reproduction loss became nonfinite")
        sdpa.backward_started = True
        total.backward()
    capture.close()
    gradients = D.named_parameter_inventory(model)
    verdict = D.gradient_verdict(gradients)
    losses = {key: value / 64 for key, value in raw.items()}
    receipt = {
        "schema": REPRODUCTION_SCHEMA, "status": C.STATUS,
        "contract_digest": load_contract(root)[C.CONTRACT_SELF_KEY],
        "fixture_binding": fit["binding"], "initial_state_digest": initial_digest,
        "registered_seed": C.FROZEN_ATTENTIVE_SEED,
        "outputs": D._output_payload(outputs),
        "losses": D._loss_payload(losses, total),
        "fixture_tensor_bindings": {
            "tokens": FROZEN.tensor_digest(tokens.detach().cpu()),
            "action_goal": FROZEN.tensor_digest(action_goal.detach().cpu()),
            "targets": {key: FROZEN.tensor_digest(value.detach().cpu())
                        for key, value in targets.items()},
            "batch": FROZEN.tensor_digest(batch.detach().cpu()),
        },
        "parameter_gradient_inventory": gradients,
        "gradient_verdict": verdict, "sdpa_audit": sdpa.audit(),
        "optimizer_constructed": True, "optimizer_steps": 0,
        "fresh_model_constructions": 1, "optimizer_constructions": 1,
        "whole_model_forwards": 1, "whole_model_backwards": 1,
        "gradient_clips": 0, "calibration_rows_opened": 0,
    }
    captures_receipt, captures = capture.export(model)
    receipt["captures"] = captures_receipt
    frozen = CONSUMER.validate_frozen_runtime_bytes(root)["artifacts"][
        "exact_reproduction.json"]
    exact_equal = D._pass_exact_payload(receipt) == D._pass_exact_payload(frozen)
    expected_pattern = set(C.NATIVE_NONFINITE_PARAMETER_NAMES)
    pattern = (set(verdict["nonfinite_parameter_set"])
               == set(C.NATIVE_NONFINITE_PARAMETER_NAMES)
               and len(verdict["nonfinite_parameter_set"])
               == len(C.NATIVE_NONFINITE_PARAMETER_NAMES))
    by_name = {row["fully_qualified_name"]: row for row in gradients}
    pattern = pattern and all(
        by_name[name]["shape"] == [512]
        and by_name[name]["finite_count"] == 256
        and by_name[name]["nan_count"] == 256
        and by_name[name]["positive_infinity_count"] == 0
        and by_name[name]["negative_infinity_count"] == 0
        for name in expected_pattern)
    negative = all(by_name[f"{C.NEGATIVE_CONTROL_PATH}.{suffix}"]["nan_count"] == 0
                   and by_name[f"{C.NEGATIVE_CONTROL_PATH}.{suffix}"][
                       "positive_infinity_count"] == 0
                   and by_name[f"{C.NEGATIVE_CONTROL_PATH}.{suffix}"][
                       "negative_infinity_count"] == 0
                   for suffix in ("weight", "bias"))
    efficient = (sdpa.audit()["exact_phase_kind_shape_dtype_ledger"]
                 and sdpa.audit()["backend_sequence"]
                 == ["EFFICIENT_ATTENTION"] * 7)
    receipt["exact_predecessor_receipt_equal"] = exact_equal
    receipt["native_nonfinite_pattern_exact"] = pattern
    receipt["negative_control_affine_gradients_finite"] = negative
    receipt["production_efficient_backend_exact"] = efficient
    receipt["model_state_unchanged"] = (
        FROZEN.state_dict_digest(FROZEN._cpu_state(model)) == initial_digest)
    receipt["optimizer_state_empty_unchanged"] = (
        optimizer.state_dict()["state"] == {})
    receipt["reproduced"] = (
        exact_equal and pattern and negative and efficient
        and receipt["model_state_unchanged"]
        and receipt["optimizer_state_empty_unchanged"])
    torch.cuda.synchronize(device)
    receipt["wall_time_seconds"] = time.monotonic() - started
    receipt["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    receipt = signed(receipt, REPRODUCTION_KEY)
    del model, optimizer, outputs, total, tokens, action_goal, targets
    torch.cuda.empty_cache()
    return receipt, captures


def local_case(captured: Mapping[str, Any], *, case: str,
               device: torch.device) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    target = torch.device("cpu") if case == "CPU_NATIVE" else device
    x = captured["input"].clone(memory_format=torch.preserve_format).to(
        target).requires_grad_(True)
    weight = captured["weight"].clone().to(target).requires_grad_(True)
    bias = captured["bias"].clone().to(target).requires_grad_(True)
    dy = captured["upstream"].clone(memory_format=torch.preserve_format).to(target)
    if case == "GPU_EXPLICIT_AFFINE":
        normalized = F.layer_norm(x, (C.FEATURE_WIDTH,), weight=None, bias=None,
                                  eps=float(captured["eps"]))
        output = normalized * weight + bias
        formula = C.IMPLEMENTATION_CONTRACT["externalised_formula"]
    else:
        output = F.layer_norm(x, (C.FEATURE_WIDTH,), weight, bias,
                              float(captured["eps"]))
        formula = C.IMPLEMENTATION_CONTRACT["native_formula"]
    output.backward(dy)
    tensors = {"forward": output.detach().cpu(),
               "input_gradient": x.grad.detach().cpu(),
               "weight_gradient": weight.grad.detach().cpu(),
               "bias_gradient": bias.grad.detach().cpu()}
    receipt = {
        "case": case, "device": str(target), "formula": formula,
        "dtype": str(x.dtype), "shape": list(x.shape), "stride": list(x.stride()),
        "eps": float(captured["eps"]),
        **{key: tensor_receipt(value) for key, value in tensors.items()},
    }
    return receipt, tensors


def run_local_cases(captures: Mapping[str, Any], contract_digest: str,
                    device: torch.device) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    rows = []
    all_paths_pass = True
    for name in C.LAYER_NORM_PATHS:
        captured = captures[name]
        cpu_receipt, cpu = local_case(captured, case="CPU_NATIVE", device=device)
        native_receipt, native = local_case(
            captured, case="GPU_NATIVE", device=device)
        explicit_receipt, explicit = local_case(
            captured, case="GPU_EXPLICIT_AFFINE", device=device)
        for receipt in (cpu_receipt, native_receipt, explicit_receipt):
            receipt["path"] = name
            receipt["captured_layout_equal"] = (
                receipt["dtype"] == "torch.float32"
                and receipt["shape"] == list(captured["input"].shape)
                and receipt["stride"] == list(captured["input"].stride()))
        for key, tolerance_key in (
                ("forward", "forward"), ("input_gradient", "input_gradient"),
                ("weight_gradient", "weight_gradient"),
                ("bias_gradient", "bias_gradient")):
            native_receipt[f"{key}_comparison_to_cpu"] = comparison(
                cpu[key], native[key], C.TOLERANCES[tolerance_key])
            explicit_receipt[f"{key}_comparison_to_cpu"] = comparison(
                cpu[key], explicit[key], C.TOLERANCES[tolerance_key])
            cpu_receipt[f"{key}_comparison_to_cpu"] = {
                "both_finite": finite(cpu[key]), "equivalent": finite(cpu[key]),
                "maximum_absolute_difference": 0.0,
                "maximum_relative_difference": 0.0,
                "common_finite_count": int(cpu[key].numel()),
                "maximum_tolerance_excess": (
                    -float(C.TOLERANCES[tolerance_key]["absolute"])),
                "absolute_tolerance": C.TOLERANCES[tolerance_key]["absolute"],
                "relative_tolerance": C.TOLERANCES[tolerance_key]["relative"],
            }
        native_matches_whole = {
            "weight": FROZEN.tensor_digest(native["weight_gradient"])
            == FROZEN.tensor_digest(captured["native_weight_gradient"]),
            "bias": FROZEN.tensor_digest(native["bias_gradient"])
            == FROZEN.tensor_digest(captured["native_bias_gradient"]),
        }
        native_receipt["exact_whole_model_native_affine_gradient_match"] = (
            native_matches_whole)
        native_pattern = all(
            native_receipt[key]["finite_count"] == 256
            and native_receipt[key]["nan_count"] == 256
            and native_receipt[key]["positive_infinity_count"] == 0
            and native_receipt[key]["negative_infinity_count"] == 0
            for key in ("weight_gradient", "bias_gradient"))
        cpu_finite = all(finite(cpu[key]) for key in (
            "forward", "input_gradient", "weight_gradient", "bias_gradient"))
        native_forward_input_finite = finite(native["forward"]) and finite(
            native["input_gradient"])
        explicit_finite = all(finite(explicit[key]) for key in (
            "forward", "input_gradient", "weight_gradient", "bias_gradient"))
        explicit_equivalent = all(explicit_receipt[
            f"{key}_comparison_to_cpu"]["equivalent"] for key in (
                "forward", "input_gradient", "weight_gradient", "bias_gradient"))
        path_pass = (cpu_finite and native_forward_input_finite and native_pattern
                     and explicit_finite
                     and explicit_equivalent
                     and all(row["captured_layout_equal"] for row in (
                         cpu_receipt, native_receipt, explicit_receipt)))
        rows.append({"path": name, "passed": path_pass,
                     "cases": {"CPU_NATIVE": cpu_receipt,
                               "GPU_NATIVE": native_receipt,
                               "GPU_EXPLICIT_AFFINE": explicit_receipt}})
        all_paths_pass = all_paths_pass and path_pass
        del cpu, native, explicit
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return signed({
        "schema": LOCAL_SCHEMA, "status": C.STATUS,
        "contract_digest": contract_digest, "path_count": len(rows),
        "case_count": len(rows) * 3, "paths": rows,
        "classification_rule": C.LOCAL_CLASSIFICATION_RULE,
        "all_paths_pass": all_paths_pass,
        "mechanism_classification": (
            C.LOCAL_SUCCESS_CLASSIFICATION if all_paths_pass else None),
        "captured_tensor_values_persisted": False,
        "calibration_rows_opened": 0,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
    }, LOCAL_KEY)


@contextmanager
def externalised_layernorms(model: nn.Module):
    modules = dict(model.named_modules())
    originals = {}
    for name in C.LAYER_NORM_PATHS:
        module = modules[name]
        originals[name] = module.forward

        def forward(this: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
            normalized = F.layer_norm(value, (C.FEATURE_WIDTH,), weight=None,
                                      bias=None, eps=float(this.eps))
            return normalized * this.weight + this.bias

        module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        for name, original in originals.items():
            modules[name].forward = original


def attention_and_query_evidence(model: nn.Module) -> dict[str, Any]:
    groups = (
        "pooler.blocks.0.attn", "pooler.blocks.1.attn",
        "pooler.blocks.2.attn", "pooler.cross_attention_block.xattn")
    group_rows = []
    named = dict(model.named_parameters())
    for group in groups:
        selected = [(name, parameter.grad) for name, parameter in named.items()
                    if name.startswith(group + ".")]
        finite_nonzero = bool(selected) and all(
            gradient is not None and finite(gradient) for _, gradient in selected)
        squared = sum(float(gradient.detach().to(torch.float64).square().sum())
                      for _, gradient in selected) if finite_nonzero else 0.0
        group_rows.append({"path": group, "finite_nonzero": finite_nonzero
                           and squared > 0.0,
                           "gradient_l2": (math.sqrt(squared)
                                           if finite_nonzero else None)})
    projection = model.token_projection.weight.grad
    query = model.pooler.query_tokens.grad
    query_rows = query.detach()[0] if query is not None else None
    queries_good = (query_rows is not None and finite(query_rows)
                    and all(float(row.to(torch.float64).norm()) > 0.0
                            for row in query_rows)
                    and all(not torch.equal(query_rows[a], query_rows[b])
                            for a in range(3) for b in range(a + 1, 3)))
    projection_finite = projection is not None and finite(projection)
    return {
        "token_projection_finite_nonzero": (
            projection_finite
            and float(projection.detach().to(torch.float64).norm()) > 0.0),
        "token_projection_gradient_digest": (
            FROZEN.tensor_digest(projection.detach().cpu())
            if projection is not None else None),
        "token_projection_gradient_l2": (
            float(projection.detach().to(torch.float64).norm())
            if projection_finite else None),
        "attention_modules": group_rows,
        "all_four_attention_modules_finite_nonzero": all(
            row["finite_nonzero"] for row in group_rows),
        "component_queries_finite_nonzero_pairwise_distinct": queries_good,
        "component_query_gradient_digest": (
            FROZEN.tensor_digest(query.detach().cpu()) if query is not None else None),
        "component_query_rows": ([{
            "index": index,
            "gradient_l2": (float(row.to(torch.float64).norm())
                             if finite(row) else None),
            "gradient_digest": FROZEN.tensor_digest(row.cpu()),
        } for index, row in enumerate(query_rows)]
            if query_rows is not None else []),
    }


def optimizer_tensors_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return finite(value)
    if isinstance(value, Mapping):
        return all(optimizer_tensors_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(optimizer_tensors_finite(item) for item in value)
    return True


def run_smoke(fit: Mapping[str, Any], reproduction: Mapping[str, Any],
              contract_digest: str, device: torch.device,
              root: Path) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model, initial_digest = fresh_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    action_goal, targets = ATTENTIVE._small_features(fit["rows"], device)
    batch_cpu = torch.arange(4, dtype=torch.int64)
    batch = batch_cpu.to(device)
    tokens = ATTENTIVE._token_batch(
        fit["rows"], fit["store"], batch_cpu.tolist(), device)
    optimizer.zero_grad(set_to_none=True)
    sdpa = D.SDPAInventory("production")
    backward_executed = 0
    with externalised_layernorms(model), sdpa:
        outputs = model(tokens, action_goal[batch])
        raw = D._raw_component_losses(outputs, targets, batch)
        loss = D._pass_loss(raw, "frozen_summed_loss")
        if finite(loss) and all(finite(value) for value in outputs):
            sdpa.backward_started = True
            loss.backward()
            backward_executed = 1
    gradient_inventory = D.named_parameter_inventory(model)
    gradient_verdict = D.gradient_verdict(gradient_inventory)
    all_gradients_finite = gradient_verdict["all_gradients_present_and_finite"]
    gradient_evidence = attention_and_query_evidence(model)
    modules = dict(model.named_modules())
    layernorm_affine_finite = {
        name: all(modules[name]._parameters[suffix].grad is not None
                  and finite(modules[name]._parameters[suffix].grad)
                  for suffix in ("weight", "bias"))
        for name in (*C.LAYER_NORM_PATHS, C.NEGATIVE_CONTROL_PATH)}
    output_comparisons = {
        component: comparison(
            torch.tensor(reproduction["outputs"][component]["values"],
                         dtype=torch.float32), value.detach().cpu(),
            C.TOLERANCES["forward"])
        for component, value in zip(D.COMPONENTS, outputs, strict=True)}
    reproduced_loss = torch.tensor(
        reproduction["losses"]["selected_total"]["value"], dtype=torch.float32)
    loss_comparison = comparison(
        reproduced_loss, loss.detach().cpu(), C.TOLERANCES["forward"])
    pre_step_pass = (
        finite(loss) and all(finite(value) for value in outputs)
        and all_gradients_finite
        and gradient_evidence["token_projection_finite_nonzero"]
        and gradient_evidence["all_four_attention_modules_finite_nonzero"]
        and gradient_evidence[
            "component_queries_finite_nonzero_pairwise_distinct"]
        and all(layernorm_affine_finite.values())
        and all(row["equivalent"] for row in output_comparisons.values())
        and loss_comparison["equivalent"]
        and D.sdpa_ledger_exact(sdpa.audit())
        and sdpa.audit()["backend_sequence"] == ["EFFICIENT_ATTENTION"] * 7)
    checkpoint_binding = None
    optimizer_steps = 0
    clip_norm = None
    clipped_gradients_finite = False
    post_step_finite = False
    if pre_step_pass:
        clip = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        RUNTIME_COUNTERS["gradient_clips"] += 1
        clip_norm = float(clip.detach().cpu())
        require(math.isfinite(clip_norm)
                and all(parameter.grad is not None and finite(parameter.grad)
                        for parameter in model.parameters()),
                "explicit-affine clipping became nonfinite")
        clipped_gradients_finite = True
        optimizer.step()
        RUNTIME_COUNTERS["optimizer_updates"] += 1
        optimizer_steps = 1
        state = FROZEN._cpu_state(model)
        optimizer_state = optimizer.state_dict()
        post_step_finite = (all(finite(value) for value in state.values())
                            and optimizer_tensors_finite(optimizer_state)
                            and FROZEN.state_dict_digest(state) != initial_digest)
        if post_step_finite:
            checkpoint = {
                "schema": CHECKPOINT_SCHEMA, "status": C.STATUS,
                "contract_digest": contract_digest,
                "implementation_digest": C.IMPLEMENTATION_DIGEST,
                "initial_state_digest": initial_digest,
                "model_state_dict": state,
                "model_state_digest": FROZEN.state_dict_digest(state),
                "optimizer_state_dict": optimizer_state,
                "optimizer_state_digest": FROZEN.structured_digest(optimizer_state),
                "completed_optimizer_updates": 1,
                "calibration_rows_opened": 0,
            }
            checkpoint_target = path(root, "checkpoint")
            require(not checkpoint_target.exists() and not checkpoint_target.is_symlink(),
                    "conditional smoke checkpoint exists")
            FROZEN.atomic_torch_save(checkpoint, checkpoint_target)
            checkpoint_target.chmod(0o444)
            loaded = torch.load(checkpoint_target, map_location="cpu",
                                weights_only=False)
            require(FROZEN.state_dict_digest(loaded["model_state_dict"])
                    == checkpoint["model_state_digest"]
                    and FROZEN.structured_digest(loaded["optimizer_state_dict"])
                    == checkpoint["optimizer_state_digest"],
                    "conditional smoke checkpoint reload changed")
            reload_model, reload_initial_digest = fresh_model(torch.device("cpu"))
            reload_optimizer = torch.optim.AdamW(
                reload_model.parameters(), lr=3e-4, weight_decay=0.01)
            with externalised_layernorms(reload_model):
                reload_model.load_state_dict(
                    loaded["model_state_dict"], strict=True)
                reload_optimizer.load_state_dict(
                    loaded["optimizer_state_dict"])
            strict_reload_passed = (
                reload_initial_digest == initial_digest
                and FROZEN.state_dict_digest(FROZEN._cpu_state(reload_model))
                == checkpoint["model_state_digest"]
                and FROZEN.structured_digest(reload_optimizer.state_dict())
                == checkpoint["optimizer_state_digest"])
            require(strict_reload_passed,
                    "strict conditional model/optimizer reload changed")
            checkpoint_binding = {
                "path": str(checkpoint_target.relative_to(ROOT)),
                "sha256": C.file_sha256(checkpoint_target),
                "byte_count": checkpoint_target.stat().st_size,
                "model_state_digest": checkpoint["model_state_digest"],
                "optimizer_state_digest": checkpoint["optimizer_state_digest"],
                "strict_model_optimizer_reload_passed": True,
            }
    passed = (pre_step_pass and optimizer_steps == 1 and post_step_finite
              and checkpoint_binding is not None
              and checkpoint_binding.get(
                  "strict_model_optimizer_reload_passed") is True)
    torch.cuda.synchronize(device)
    return signed({
        "schema": SMOKE_SCHEMA, "status": C.STATUS,
        "contract_digest": contract_digest,
        "implementation_name": C.IMPLEMENTATION_NAME,
        "implementation_digest": C.IMPLEMENTATION_DIGEST,
        "fixture_digest": fit["binding"]["fixture_digest"],
        "initial_state_digest": initial_digest,
        "output_comparisons_to_native": output_comparisons,
        "loss_comparison_to_native": loss_comparison,
        "all_outputs_and_loss_finite": (
            finite(loss) and all(finite(value) for value in outputs)),
        "all_parameter_gradients_finite": all_gradients_finite,
        "parameter_gradient_inventory": gradient_inventory,
        "complete_gradient_verdict": gradient_verdict,
        "gradient_evidence": gradient_evidence,
        "layernorm_affine_gradients_finite": layernorm_affine_finite,
        "sdpa_audit": sdpa.audit(),
        "gradient_clip_max_norm": 1.0, "preclip_total_norm": clip_norm,
        "all_clipped_gradients_finite": clipped_gradients_finite,
        "optimizer": {"name": "AdamW", "lr": 3e-4,
                      "weight_decay": 0.01},
        "completed_optimizer_updates": optimizer_steps,
        "fresh_model_constructions": 1 + (1 if checkpoint_binding else 0),
        "optimizer_constructions": 1 + (1 if checkpoint_binding else 0),
        "whole_model_forwards": 1, "whole_model_backwards": backward_executed,
        "post_step_model_and_optimizer_finite": post_step_finite,
        "checkpoint": checkpoint_binding, "passed": passed,
        "calibration_rows_opened": 0, "predictor_checkpoints_opened": 0,
        "scorer_training_started": False,
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
    }, SMOKE_KEY)


def binding(root: Path, name: str, value: Mapping[str, Any], key: str) -> dict[str, Any]:
    return artifact_binding(path(root, name), value, key)


def terminal_payload(*, contract: Mapping[str, Any], attempt: Mapping[str, Any],
                     reproduction: Mapping[str, Any], local: Mapping[str, Any] | None,
                     smoke: Mapping[str, Any] | None, root: Path) -> dict[str, Any]:
    if not reproduction["reproduced"]:
        kind, mechanism, eligible = "NONREPRODUCTION_STOP", None, False
    elif local is None or not local["all_paths_pass"]:
        kind, mechanism, eligible = "LOCAL_EQUIVALENCE_FAILURE_STOP", None, False
    elif smoke is None or not smoke["passed"]:
        kind = "CONDITIONAL_SMOKE_FAILURE_STOP"
        mechanism, eligible = C.LOCAL_SUCCESS_CLASSIFICATION, False
    else:
        kind = "SUCCESS_CLASSIFICATION"
        mechanism, eligible = C.LOCAL_SUCCESS_CLASSIFICATION, True
    bindings = {
        "attempt": binding(root, "attempt", attempt, ATTEMPT_KEY),
        "reproduction": binding(
            root, "reproduction", reproduction, REPRODUCTION_KEY),
    }
    if local is not None:
        bindings["local"] = binding(root, "local", local, LOCAL_KEY)
    if smoke is not None:
        bindings["smoke"] = binding(root, "smoke", smoke, SMOKE_KEY)
        if smoke["checkpoint"] is not None:
            bindings["checkpoint"] = smoke["checkpoint"]
    return signed({
        "schema": TERMINAL_SCHEMA, "status": C.STATUS, "complete": True,
        "contract_digest": contract[C.CONTRACT_SELF_KEY],
        "attempt_digest": attempt[ATTEMPT_KEY], "terminal_kind": kind,
        "mechanism_classification": mechanism,
        "implementation_successor_eligibility": (
            C.IMPLEMENTATION_NAME if eligible else None),
        "successor_requires_separately_committed_source_and_authority": True,
        "training_authorised_now": False, "repair_authorised_now": False,
        "readout_line_closed_on_non_success": not eligible,
        "artifact_bindings": bindings,
        "execution_counts": {
            "fresh_model_constructions": 1 + (
                smoke.get("fresh_model_constructions", 0)
                if smoke is not None else 0),
            "optimizer_constructions": 1 + (
                smoke.get("optimizer_constructions", 0)
                if smoke is not None else 0),
            "whole_model_forwards": 1 + (1 if smoke is not None else 0),
            "whole_model_backwards": 1 + (
                smoke.get("whole_model_backwards", 0)
                if smoke is not None else 0),
            "local_layernorm_forwards_backwards": (
                21 if local is not None else 0),
            "gradient_clips": (
                1 if smoke is not None
                and smoke.get("preclip_total_norm") is not None else 0),
            "optimizer_steps": (
                smoke.get("completed_optimizer_updates", 0)
                if smoke is not None else 0),
            "calibration_rows_opened": 0,
            "predictor_checkpoints_opened": 0,
        },
        "captured_tensor_values_persisted": False,
        "calibration_rows_opened": 0, "predictor_checkpoints_opened": 0,
        "scorer_training_started": False,
    }, TERMINAL_KEY)


def record_failure(root: Path, contract_digest: str | None,
                   attempt_digest: str | None, stage: str,
                   error: BaseException) -> dict[str, Any]:
    target = path(root, "failure")
    require(not target.exists() and not target.is_symlink(),
            "technical failure already exists")
    preserved = {}
    self_keys = {"contract": C.CONTRACT_SELF_KEY, "attempt": ATTEMPT_KEY,
                 "reproduction": REPRODUCTION_KEY, "local": LOCAL_KEY,
                 "smoke": SMOKE_KEY}
    for name, key in self_keys.items():
        target_path = path(root, name)
        if target_path.is_file() and not target_path.is_symlink():
            value = validate_signed(C.read_json(target_path, name), key, name)
            preserved[name] = artifact_binding(target_path, value, key)
    checkpoint_target = path(root, "checkpoint")
    if checkpoint_target.is_file() and not checkpoint_target.is_symlink():
        preserved["checkpoint"] = {
            "path": str(checkpoint_target.relative_to(ROOT)),
            "sha256": C.file_sha256(checkpoint_target),
            "byte_count": checkpoint_target.stat().st_size,
        }
    receipt = signed({
        "schema": FAILURE_SCHEMA, "status": C.STATUS, "complete": True,
        "contract_digest": contract_digest, "attempt_digest": attempt_digest,
        "stage": stage, "exception_type": type(error).__name__,
        "exception_message": str(error), "traceback": traceback.format_exc(),
        "preserved_artifacts": preserved,
        "completed_optimizer_updates": RUNTIME_COUNTERS["optimizer_updates"],
        "completed_gradient_clips": RUNTIME_COUNTERS["gradient_clips"],
        "retry_resume_or_replacement_authorised": False,
        "training_authorised": False, "repair_authorised": False,
        "calibration_rows_opened": 0, "predictor_checkpoints_opened": 0,
    }, FAILURE_KEY)
    publish_json_once(target, receipt, "technical failure")
    return receipt


def run_once(root: Path = ROOT) -> dict[str, Any]:
    contract = load_contract(root)
    runtime = C.runtime_root(root)
    require({item.name for item in runtime.iterdir()} == {ARTIFACTS["contract"]},
            "one-shot namespace was already consumed")
    stage = "device_preflight"
    attempt = None
    try:
        device, preflight = D.device_preflight()
        attempt = signed({
            "schema": ATTEMPT_SCHEMA, "status": C.STATUS,
            "attempt_number": 1, "maximum_attempts": 1,
            "contract_digest": contract[C.CONTRACT_SELF_KEY],
            "predecessor_terminal_digest": C.PREDECESSOR_TERMINAL_DIGEST,
            "fixture_digest": C.FROZEN_FIXTURE_DIGEST,
            "initial_state_digest": C.FROZEN_INITIAL_STATE_DIGEST,
            "technical_preflight": preflight,
            "execution_limits": C.EXECUTION_LIMITS,
            "scientific_training_authorised": False,
            "calibration_access_authorised": False,
            "predictor_access_authorised": False,
        }, ATTEMPT_KEY)
        publish_json_once(path(root, "attempt"), attempt, "attempt")
        stage = "fixture_and_reproduction"
        fit = fixture(root)
        reproduction, captures = run_reproduction(fit, device, root)
        publish_json_once(path(root, "reproduction"), reproduction, "reproduction")
        local = None
        smoke = None
        if reproduction["reproduced"]:
            stage = "twenty_one_local_cases"
            local = run_local_cases(
                captures, contract[C.CONTRACT_SELF_KEY], device)
            publish_json_once(path(root, "local"), local, "local cases")
            captures.clear()
            if local["all_paths_pass"]:
                stage = "conditional_whole_model_smoke"
                smoke = run_smoke(
                    fit, reproduction, contract[C.CONTRACT_SELF_KEY], device, root)
                publish_json_once(path(root, "smoke"), smoke, "conditional smoke")
        captures.clear()
        stage = "terminal"
        terminal = terminal_payload(
            contract=contract, attempt=attempt, reproduction=reproduction,
            local=local, smoke=smoke, root=root)
        publish_json_once(path(root, "terminal"), terminal, "terminal")
        return terminal
    except BaseException as exc:
        record_failure(root, contract.get(C.CONTRACT_SELF_KEY),
                       attempt.get(ATTEMPT_KEY) if attempt else None, stage, exc)
        raise


def read_artifact(root: Path, name: str, key: str, label: str) -> dict[str, Any]:
    return validate_signed(C.read_json(path(root, name), label), key, label)


def recompute_local_success(local: Mapping[str, Any]) -> bool:
    require(local.get("path_count") == 7 and local.get("case_count") == 21
            and [row.get("path") for row in local.get("paths", [])]
            == list(C.LAYER_NORM_PATHS), "local case cardinality changed")
    return all(row.get("passed") is True for row in local["paths"])


def _receipt_all_finite(value: Mapping[str, Any]) -> bool:
    count = math.prod(value.get("shape", []))
    return (value.get("finite_count") == count and value.get("nan_count") == 0
            and value.get("positive_infinity_count") == 0
            and value.get("negative_infinity_count") == 0)


def resource_receipt_valid(value: Mapping[str, Any]) -> bool:
    wall = value.get("wall_time_seconds")
    peak = value.get("peak_vram_bytes")
    return (isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0.0
            and isinstance(peak, int) and not isinstance(peak, bool) and peak > 0)


def validate_reproduction_semantics(reproduction: Mapping[str, Any],
                                    root: Path) -> bool:
    frozen = CONSUMER.validate_frozen_runtime_bytes(root)["artifacts"][
        "exact_reproduction.json"]
    exact_equal = D._pass_exact_payload(reproduction) == D._pass_exact_payload(frozen)
    verdict = reproduction.get("gradient_verdict", {})
    gradients = reproduction.get("parameter_gradient_inventory", [])
    by_name = {row.get("fully_qualified_name"): row for row in gradients}
    names = list(C.NATIVE_NONFINITE_PARAMETER_NAMES)
    pattern = (set(verdict.get("nonfinite_parameter_set", [])) == set(names)
               and len(verdict.get("nonfinite_parameter_set", [])) == len(names)
               and all(by_name.get(name, {}).get("shape") == [512]
                       and by_name[name].get("finite_count") == 256
                       and by_name[name].get("nan_count") == 256
                       and by_name[name].get("positive_infinity_count") == 0
                       and by_name[name].get("negative_infinity_count") == 0
                       for name in names))
    negative = all(
        by_name.get(f"{C.NEGATIVE_CONTROL_PATH}.{suffix}", {}).get("nan_count") == 0
        and by_name[f"{C.NEGATIVE_CONTROL_PATH}.{suffix}"].get(
            "positive_infinity_count") == 0
        and by_name[f"{C.NEGATIVE_CONTROL_PATH}.{suffix}"].get(
            "negative_infinity_count") == 0
        for suffix in ("weight", "bias"))
    sdpa = reproduction.get("sdpa_audit", {})
    efficient = (D.sdpa_ledger_exact(sdpa)
                 and sdpa.get("backend_sequence") == ["EFFICIENT_ATTENTION"] * 7)
    captures = reproduction.get("captures", {})
    capture_valid = (set(captures) == set(C.LAYER_NORM_PATHS)
                     and len(captures) == len(C.LAYER_NORM_PATHS))
    for index, name in enumerate(C.LAYER_NORM_PATHS):
        row = captures.get(name, {})
        calls = 1 if index == 0 else 2
        capture_valid = capture_valid and (
            set(row) == set(C.CAPTURE_CONTRACT["fields"])
            and row.get("path") == name and row.get("eps") == 1e-5
            and row.get("normalized_shape") == [512]
            and row.get("dtype") == "torch.float32"
            and row.get("shape", [])[-1:] == [512]
            and row.get("input_layout") == {
                "dtype": row.get("dtype"), "shape": row.get("shape"),
                "stride": row.get("stride")}
            and row.get("upstream_gradient_layout") == row.get("input_layout")
            and row.get("weight_layout") == {
                "dtype": "torch.float32", "shape": [512], "stride": [1]}
            and row.get("bias_layout") == {
                "dtype": "torch.float32", "shape": [512], "stride": [1]}
            and row.get("native_whole_model_weight_gradient", {}).get(
                "shape") == [512]
            and row.get("native_whole_model_bias_gradient", {}).get(
                "shape") == [512]
            and row.get("forward_call_count") == calls
            and row.get("paired_upstream_count") == 1
            and len(set(row.get("initial_recompute_input_digests", []))) == 1
            and len(set(row.get("initial_recompute_output_digests", []))) == 1
            and all(row.get(key) is True for key in (
                "input_finite", "upstream_gradient_finite",
                "weight_finite", "bias_finite")))
    recomputed = bool(
        reproduction.get("schema") == REPRODUCTION_SCHEMA
        and reproduction.get("fixture_binding", {}).get("fixture_digest")
        == C.FROZEN_FIXTURE_DIGEST
        and reproduction.get("initial_state_digest") == C.FROZEN_INITIAL_STATE_DIGEST
        and reproduction.get("registered_seed") == C.FROZEN_ATTENTIVE_SEED
        and reproduction.get("optimizer_constructed") is True
        and reproduction.get("optimizer_steps") == 0
        and reproduction.get("gradient_clips") == 0
        and reproduction.get("calibration_rows_opened") == 0
        and reproduction.get("model_state_unchanged") is True
        and reproduction.get("optimizer_state_empty_unchanged") is True
        and resource_receipt_valid(reproduction)
        and exact_equal and pattern and negative and efficient and capture_valid)
    require(reproduction.get("exact_predecessor_receipt_equal") == exact_equal
            and reproduction.get("native_nonfinite_pattern_exact") == pattern
            and reproduction.get("negative_control_affine_gradients_finite") == negative
            and reproduction.get("production_efficient_backend_exact") == efficient
            and reproduction.get("reproduced") == recomputed,
            "reproduction semantic replay changed")
    return recomputed


def validate_local_semantics(local: Mapping[str, Any],
                             reproduction: Mapping[str, Any]) -> bool:
    require(local.get("schema") == LOCAL_SCHEMA
            and local.get("status") == C.STATUS
            and local.get("contract_digest") == reproduction.get("contract_digest")
            and local.get("classification_rule") == C.LOCAL_CLASSIFICATION_RULE
            and local.get("path_count") == 7 and local.get("case_count") == 21
            and [row.get("path") for row in local.get("paths", [])]
            == list(C.LAYER_NORM_PATHS), "local case cardinality changed")
    all_paths = True
    captures = reproduction["captures"]
    for row in local["paths"]:
        name = row["path"]
        cases = row.get("cases", {})
        require(set(cases) == set(C.LOCAL_CASES),
                f"local case names changed at {name}")
        cpu, native, explicit = (cases[key] for key in (
            "CPU_NATIVE", "GPU_NATIVE", "GPU_EXPLICIT_AFFINE"))
        layout = captures[name]
        schema_ok = True
        for case_name, case in cases.items():
            expected_device = "cpu" if case_name == "CPU_NATIVE" else "cuda:0"
            expected_formula = C.IMPLEMENTATION_CONTRACT[
                "externalised_formula" if case_name == "GPU_EXPLICIT_AFFINE"
                else "native_formula"]
            schema_ok = schema_ok and (
                case.get("case") == case_name and case.get("path") == name
                and case.get("device") == expected_device
                and case.get("formula") == expected_formula
                and case.get("dtype") == layout["dtype"]
                and case.get("shape") == layout["shape"]
                and case.get("stride") == layout["stride"]
                and case.get("eps") == layout["eps"]
                and case.get("captured_layout_equal") is True)
            for tensor_name in (
                    "forward", "input_gradient", "weight_gradient", "bias_gradient"):
                tensor_row = case.get(tensor_name, {})
                schema_ok = schema_ok and set(tensor_row) == set(C.TENSOR_RESULT_FIELDS)
                comparison_row = case.get(f"{tensor_name}_comparison_to_cpu", {})
                schema_ok = schema_ok and set(comparison_row) == set(
                    C.COMPARISON_RESULT_FIELDS)
        cpu_finite = all(_receipt_all_finite(cpu[key]) for key in (
            "forward", "input_gradient", "weight_gradient", "bias_gradient"))
        native_forward_input = all(_receipt_all_finite(native[key]) for key in (
            "forward", "input_gradient"))
        native_pattern = all(
            native[key].get("finite_count") == 256
            and native[key].get("nan_count") == 256
            and native[key].get("positive_infinity_count") == 0
            and native[key].get("negative_infinity_count") == 0
            for key in ("weight_gradient", "bias_gradient"))
        native_finite_intersection = all(
            native[f"{key}_comparison_to_cpu"].get("common_finite_count") == 256
            and native[f"{key}_comparison_to_cpu"].get(
                "maximum_absolute_difference") is not None
            and native[f"{key}_comparison_to_cpu"].get(
                "maximum_relative_difference") is not None
            for key in ("weight_gradient", "bias_gradient"))
        derived_native_match = {
            suffix: (
                native[f"{suffix}_gradient"].get("tensor_digest")
                == captures[name][f"native_whole_model_{suffix}_gradient"].get(
                    "tensor_digest")
                and native[f"{suffix}_gradient"].get("finite_count")
                == captures[name][f"native_whole_model_{suffix}_gradient"].get(
                    "finite_count")
                and native[f"{suffix}_gradient"].get("nan_count")
                == captures[name][f"native_whole_model_{suffix}_gradient"].get(
                    "nan_count"))
            for suffix in ("weight", "bias")}
        require(native.get("exact_whole_model_native_affine_gradient_match")
                == derived_native_match,
                f"native whole-model pairing report changed at {name}")
        explicit_finite = all(_receipt_all_finite(explicit[key]) for key in (
            "forward", "input_gradient", "weight_gradient", "bias_gradient"))
        explicit_equal = all(
            explicit[f"{key}_comparison_to_cpu"].get("both_finite") is True
            and explicit[f"{key}_comparison_to_cpu"].get("equivalent") is True
            and explicit[f"{key}_comparison_to_cpu"].get(
                "maximum_tolerance_excess") <= 0.0
            and explicit[f"{key}_comparison_to_cpu"].get("absolute_tolerance")
            == C.TOLERANCES[key]["absolute"]
            and explicit[f"{key}_comparison_to_cpu"].get("relative_tolerance")
            == C.TOLERANCES[key]["relative"]
            for key in ("forward", "input_gradient", "weight_gradient",
                        "bias_gradient"))
        passed = bool(schema_ok and cpu_finite and native_forward_input
                      and native_pattern and native_finite_intersection
                      and explicit_finite
                      and explicit_equal)
        require(row.get("passed") == passed,
                f"local path predicate changed at {name}")
        all_paths = all_paths and passed
    require(local.get("all_paths_pass") == all_paths
            and local.get("mechanism_classification") == (
                C.LOCAL_SUCCESS_CLASSIFICATION if all_paths else None)
            and local.get("captured_tensor_values_persisted") is False
            and local.get("calibration_rows_opened") == 0,
            "local classification replay changed")
    require(resource_receipt_valid(local), "local resource receipt changed")
    return all_paths


def recompute_smoke_success(smoke: Mapping[str, Any]) -> bool:
    gradients = smoke.get("gradient_evidence", {})
    comparisons = smoke.get("output_comparisons_to_native", {})
    layernorms = smoke.get("layernorm_affine_gradients_finite", {})
    inventory = smoke.get("parameter_gradient_inventory", [])
    inventory_verdict = D.gradient_verdict(inventory) if isinstance(
        inventory, list) else {}
    inventory_schema = bool(
        isinstance(inventory, list)
        and len(inventory) == C.TRAINABLE_PARAMETER_TENSOR_COUNT
        and all(set(row) == set(C.PREDECESSOR.PARAMETER_INVENTORY_FIELDS)
                for row in inventory)
        and len({row["fully_qualified_name"] for row in inventory})
        == C.TRAINABLE_PARAMETER_TENSOR_COUNT
        and C.digest([[row["fully_qualified_name"], row["shape"],
                       row["parameter_dtype"]] for row in inventory])
        == C.FROZEN_PARAMETER_INVENTORY_DIGEST)
    by_name = ({row["fully_qualified_name"]: row for row in inventory}
               if inventory_schema else {})
    def finite_gradient_row(row: Mapping[str, Any] | None) -> bool:
        return bool(row and row.get("gradient_is_none") is False
                    and row.get("nan_count") == 0
                    and row.get("positive_infinity_count") == 0
                    and row.get("negative_infinity_count") == 0)
    token_row = by_name.get("token_projection.weight")
    token_derived = bool(
        finite_gradient_row(token_row)
        and token_row.get("finite_only_l2_norm", 0.0) > 0.0
        and gradients.get("token_projection_gradient_digest")
        == token_row.get("gradient_tensor_digest")
        and isinstance(gradients.get("token_projection_gradient_l2"),
                       (int, float))
        and math.isfinite(gradients["token_projection_gradient_l2"])
        and math.isclose(gradients.get("token_projection_gradient_l2", -1.0),
                         token_row.get("finite_only_l2_norm", -2.0),
                         rel_tol=1e-12, abs_tol=1e-12))
    attention_paths = ["pooler.blocks.0.attn", "pooler.blocks.1.attn",
                       "pooler.blocks.2.attn",
                       "pooler.cross_attention_block.xattn"]
    derived_attention = []
    for group in attention_paths:
        selected = [row for name, row in by_name.items()
                    if name.startswith(group + ".")]
        norm = math.sqrt(sum(float(row.get("finite_only_l2_norm", 0.0)) ** 2
                             for row in selected))
        derived_attention.append({
            "path": group,
            "finite_nonzero": bool(selected)
            and all(finite_gradient_row(row) for row in selected)
            and norm > 0.0,
            "gradient_l2": norm,
        })
    reported_attention = gradients.get("attention_modules", [])
    attention_derived = (
        len(reported_attention) == len(derived_attention)
        and all(reported.get("path") == derived.get("path")
                and reported.get("finite_nonzero") == derived.get("finite_nonzero")
                and isinstance(reported.get("gradient_l2"), (int, float))
                and math.isfinite(reported["gradient_l2"])
                and math.isclose(reported["gradient_l2"],
                                 derived["gradient_l2"], rel_tol=1e-12,
                                 abs_tol=1e-12)
                for reported, derived in zip(
                    reported_attention, derived_attention, strict=True)))
    query_row = by_name.get("pooler.query_tokens")
    query_rows = gradients.get("component_query_rows", [])
    query_derived = bool(
        finite_gradient_row(query_row)
        and gradients.get("component_query_gradient_digest")
        == query_row.get("gradient_tensor_digest")
        and len(query_rows) == 3
        and [row.get("index") for row in query_rows] == [0, 1, 2]
        and all(isinstance(row.get("gradient_l2"), (int, float))
                and math.isfinite(row["gradient_l2"])
                and row["gradient_l2"] > 0.0
                for row in query_rows)
        and len({row.get("gradient_digest") for row in query_rows}) == 3)
    derived_layernorms = {
        name: all(finite_gradient_row(by_name.get(f"{name}.{suffix}"))
                  for suffix in ("weight", "bias"))
        for name in (*C.LAYER_NORM_PATHS, C.NEGATIVE_CONTROL_PATH)}
    return bool(
        smoke.get("schema") == SMOKE_SCHEMA
        and smoke.get("status") == C.STATUS
        and isinstance(smoke.get("contract_digest"), str)
        and smoke.get("implementation_name") == C.IMPLEMENTATION_NAME
        and smoke.get("implementation_digest") == C.IMPLEMENTATION_DIGEST
        and smoke.get("fixture_digest") == C.FROZEN_FIXTURE_DIGEST
        and smoke.get("initial_state_digest") == C.FROZEN_INITIAL_STATE_DIGEST
        and smoke.get("all_outputs_and_loss_finite") is True
        and set(comparisons) == set(D.COMPONENTS)
        and all(row.get("both_finite") is True
                and row.get("equivalent") is True
                and row.get("maximum_tolerance_excess") <= 0.0
                and row.get("absolute_tolerance")
                == C.TOLERANCES["forward"]["absolute"]
                and row.get("relative_tolerance")
                == C.TOLERANCES["forward"]["relative"]
                for row in comparisons.values())
        and smoke.get("loss_comparison_to_native", {}).get("both_finite") is True
        and smoke.get("loss_comparison_to_native", {}).get("equivalent") is True
        and smoke.get("loss_comparison_to_native", {}).get(
            "maximum_tolerance_excess") <= 0.0
        and smoke.get("loss_comparison_to_native", {}).get("absolute_tolerance")
        == C.TOLERANCES["forward"]["absolute"]
        and smoke.get("loss_comparison_to_native", {}).get("relative_tolerance")
        == C.TOLERANCES["forward"]["relative"]
        and smoke.get("all_parameter_gradients_finite") is True
        and inventory_schema
        and smoke.get("complete_gradient_verdict") == inventory_verdict
        and inventory_verdict.get("all_gradients_present_and_finite") is True
        and gradients.get("token_projection_finite_nonzero") == token_derived
        and token_derived
        and gradients.get("all_four_attention_modules_finite_nonzero") is True
        and attention_derived
        and all(row["finite_nonzero"] for row in derived_attention)
        and gradients.get(
            "component_queries_finite_nonzero_pairwise_distinct") == query_derived
        and query_derived
        and set(layernorms) == set((*C.LAYER_NORM_PATHS, C.NEGATIVE_CONTROL_PATH))
        and layernorms == derived_layernorms
        and all(derived_layernorms.values())
        and smoke.get("gradient_clip_max_norm") == 1.0
        and isinstance(smoke.get("preclip_total_norm"), (int, float))
        and math.isfinite(smoke["preclip_total_norm"])
        and smoke["preclip_total_norm"] > 0.0
        and smoke.get("all_clipped_gradients_finite") is True
        and smoke.get("optimizer") == {
            "name": "AdamW", "lr": 3e-4, "weight_decay": 0.01}
        and smoke.get("completed_optimizer_updates") == 1
        and smoke.get("fresh_model_constructions") == 2
        and smoke.get("optimizer_constructions") == 2
        and smoke.get("whole_model_forwards") == 1
        and smoke.get("whole_model_backwards") == 1
        and smoke.get("post_step_model_and_optimizer_finite") is True
        and D.sdpa_ledger_exact(smoke.get("sdpa_audit", {}))
        and smoke.get("sdpa_audit", {}).get("backend_sequence")
        == ["EFFICIENT_ATTENTION"] * 7
        and isinstance(smoke.get("checkpoint"), Mapping)
        and smoke["checkpoint"].get(
            "strict_model_optimizer_reload_passed") is True
        and smoke.get("calibration_rows_opened") == 0
        and smoke.get("predictor_checkpoints_opened") == 0
        and smoke.get("scorer_training_started") is False
        and resource_receipt_valid(smoke))


def validate_terminal(root: Path = ROOT) -> dict[str, Any]:
    contract = load_contract(root)
    attempt = read_artifact(root, "attempt", ATTEMPT_KEY, "attempt")
    reproduction = read_artifact(
        root, "reproduction", REPRODUCTION_KEY, "reproduction")
    terminal = read_artifact(root, "terminal", TERMINAL_KEY, "terminal")
    local = (read_artifact(root, "local", LOCAL_KEY, "local cases")
             if path(root, "local").exists() else None)
    smoke = (read_artifact(root, "smoke", SMOKE_KEY, "smoke")
             if path(root, "smoke").exists() else None)
    require(attempt.get("schema") == ATTEMPT_SCHEMA
            and attempt.get("status") == C.STATUS
            and attempt.get("attempt_number") == 1
            and attempt.get("maximum_attempts") == 1
            and attempt.get("contract_digest") == contract[C.CONTRACT_SELF_KEY]
            and attempt.get("predecessor_terminal_digest")
            == C.PREDECESSOR_TERMINAL_DIGEST
            and attempt.get("fixture_digest") == C.FROZEN_FIXTURE_DIGEST
            and attempt.get("initial_state_digest") == C.FROZEN_INITIAL_STATE_DIGEST
            and attempt.get("execution_limits") == C.EXECUTION_LIMITS
            and attempt.get("scientific_training_authorised") is False
            and attempt.get("calibration_access_authorised") is False
            and attempt.get("predictor_access_authorised") is False
            and reproduction["contract_digest"] == contract[C.CONTRACT_SELF_KEY]
            and reproduction["calibration_rows_opened"] == 0,
            "attempt/reproduction binding changed")
    D._validate_technical_preflight(attempt["technical_preflight"])
    reproduced = validate_reproduction_semantics(reproduction, root)
    if local is not None:
        require(reproduced, "local cases exist without exact reproduction")
        local_success = validate_local_semantics(local, reproduction)
    else:
        local_success = False
    require(reproduced == (local is not None)
            and (not local_success or smoke is not None),
            "terminal stage progression changed")
    if smoke is not None:
        require(local_success, "conditional smoke exists without local success")
        require(smoke.get("contract_digest") == contract[C.CONTRACT_SELF_KEY],
                "conditional smoke contract binding changed")
        smoke_success = recompute_smoke_success(smoke)
        require(smoke.get("passed") == smoke_success,
                "conditional smoke predicate changed")
    else:
        smoke_success = False
    expected = terminal_payload(
        contract=contract, attempt=attempt, reproduction=reproduction,
        local=local, smoke=smoke, root=root)
    require(terminal == expected, "terminal classification/bindings changed")
    if smoke is not None and smoke["checkpoint"] is not None:
        checkpoint_target = path(root, "checkpoint")
        require(checkpoint_target.is_file() and not checkpoint_target.is_symlink()
                and smoke["checkpoint"].get("path")
                == str(checkpoint_target.relative_to(ROOT))
                and C.file_sha256(checkpoint_target)
                == smoke["checkpoint"]["sha256"],
                "conditional smoke checkpoint bytes changed")
        checkpoint = torch.load(checkpoint_target, map_location="cpu",
                                weights_only=False)
        require(checkpoint_target.stat().st_size
                == smoke["checkpoint"].get("byte_count")
                and checkpoint["schema"] == CHECKPOINT_SCHEMA
                and checkpoint.get("status") == C.STATUS
                and checkpoint.get("contract_digest")
                == contract[C.CONTRACT_SELF_KEY]
                and checkpoint.get("implementation_digest")
                == C.IMPLEMENTATION_DIGEST
                and checkpoint.get("initial_state_digest")
                == C.FROZEN_INITIAL_STATE_DIGEST
                and checkpoint.get("completed_optimizer_updates") == 1
                and checkpoint.get("calibration_rows_opened") == 0
                and FROZEN.state_dict_digest(checkpoint["model_state_dict"])
                == checkpoint["model_state_digest"]
                and FROZEN.structured_digest(checkpoint["optimizer_state_dict"])
                == checkpoint["optimizer_state_digest"],
                "conditional smoke checkpoint content changed")
        require(smoke["checkpoint"].get("model_state_digest")
                == checkpoint["model_state_digest"]
                and smoke["checkpoint"].get("optimizer_state_digest")
                == checkpoint["optimizer_state_digest"],
                "conditional smoke checkpoint receipt changed")
        model, _ = fresh_model(torch.device("cpu"))
        initial_state = FROZEN.state_dict_digest(FROZEN._cpu_state(model))
        with externalised_layernorms(model):
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=3e-4, weight_decay=0.01)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        require(FROZEN.state_dict_digest(FROZEN._cpu_state(model))
                == checkpoint["model_state_digest"]
                and checkpoint["model_state_digest"] != initial_state
                and FROZEN.structured_digest(optimizer.state_dict())
                == checkpoint["optimizer_state_digest"],
                "strict model/optimizer checkpoint reload changed")
    allowed = {ARTIFACTS["contract"], ARTIFACTS["attempt"],
               ARTIFACTS["reproduction"], ARTIFACTS["terminal"]}
    if local is not None:
        allowed.add(ARTIFACTS["local"])
    if smoke is not None:
        allowed.add(ARTIFACTS["smoke"])
        if smoke["checkpoint"] is not None:
            allowed.add(ARTIFACTS["checkpoint"])
    runtime = C.runtime_root(root)
    require({item.name for item in runtime.iterdir()} == allowed
            and all(item.is_file() and not item.is_symlink()
                    and item.stat().st_mode & 0o222 == 0
                    for item in runtime.iterdir()),
            "terminal runtime inventory changed")
    return terminal


def validate_failure(root: Path = ROOT) -> dict[str, Any]:
    contract = load_contract(root)
    failure = read_artifact(root, "failure", FAILURE_KEY, "technical failure")
    require(failure["schema"] == FAILURE_SCHEMA
            and failure.get("status") == C.STATUS
            and failure.get("contract_digest") == contract[C.CONTRACT_SELF_KEY]
            and failure["complete"] is True
            and failure["retry_resume_or_replacement_authorised"] is False
            and failure.get("training_authorised") is False
            and failure.get("repair_authorised") is False
            and failure["calibration_rows_opened"] == 0
            and failure["predictor_checkpoints_opened"] == 0,
            "technical failure changed")
    require(not path(root, "terminal").exists(),
            "technical failure coexists with terminal")
    stage = failure.get("stage")
    require(stage in {"device_preflight", "fixture_and_reproduction",
                      "twenty_one_local_cases",
                      "conditional_whole_model_smoke", "terminal"},
            "technical failure stage changed")
    updates = failure.get("completed_optimizer_updates")
    clips = failure.get("completed_gradient_clips")
    require(updates in (0, 1) and clips in (0, 1) and updates <= clips
            and (stage in {"conditional_whole_model_smoke", "terminal"}
                 or updates == clips == 0),
            "technical failure update/clip custody changed")
    preserved = failure.get("preserved_artifacts", {})
    require(isinstance(preserved, Mapping) and "contract" in preserved,
            "technical failure artifact bindings changed")
    known = {"contract": (C.CONTRACT_SELF_KEY, "contract"),
             "attempt": (ATTEMPT_KEY, "attempt"),
             "reproduction": (REPRODUCTION_KEY, "reproduction"),
             "local": (LOCAL_KEY, "local"), "smoke": (SMOKE_KEY, "smoke")}
    for name, receipt in preserved.items():
        target = path(root, name)
        require(name in (*known, "checkpoint") and target.is_file()
                and not target.is_symlink()
                and receipt.get("sha256") == C.file_sha256(target)
                and receipt.get("byte_count") == target.stat().st_size,
                f"technical failure binding changed at {name}")
        if name in known:
            key, label = known[name]
            value = read_artifact(root, name, key, label)
            require(receipt.get("self_digest") == value[key],
                    f"technical failure self binding changed at {name}")
    require(("attempt" in preserved) == (stage != "device_preflight")
            and ("reproduction" in preserved)
            == (stage in {"twenty_one_local_cases",
                          "conditional_whole_model_smoke", "terminal"}),
            "technical failure completed-stage prefix changed")
    if "attempt" in preserved:
        attempt = read_artifact(root, "attempt", ATTEMPT_KEY, "attempt")
        require(failure.get("attempt_digest") == attempt[ATTEMPT_KEY]
                and attempt.get("contract_digest") == contract[C.CONTRACT_SELF_KEY]
                and attempt.get("scientific_training_authorised") is False
                and attempt.get("calibration_access_authorised") is False
                and attempt.get("predictor_access_authorised") is False,
                "technical failure attempt changed")
    if "reproduction" in preserved:
        reproduction = read_artifact(
            root, "reproduction", REPRODUCTION_KEY, "reproduction")
        validate_reproduction_semantics(reproduction, root)
    else:
        reproduction = None
    if "local" in preserved:
        require(reproduction is not None,
                "local failure evidence lacks reproduction")
        validate_local_semantics(
            read_artifact(root, "local", LOCAL_KEY, "local cases"),
            reproduction)
    if "smoke" in preserved:
        smoke = read_artifact(root, "smoke", SMOKE_KEY, "smoke")
        require(smoke.get("passed") == recompute_smoke_success(smoke),
                "failure smoke predicate changed")
    if "checkpoint" in preserved:
        checkpoint = torch.load(path(root, "checkpoint"), map_location="cpu",
                                weights_only=False)
        require(checkpoint.get("schema") == CHECKPOINT_SCHEMA
                and checkpoint.get("status") == C.STATUS
                and checkpoint.get("contract_digest") == contract[C.CONTRACT_SELF_KEY]
                and checkpoint.get("implementation_digest") == C.IMPLEMENTATION_DIGEST
                and checkpoint.get("initial_state_digest")
                == C.FROZEN_INITIAL_STATE_DIGEST
                and checkpoint.get("completed_optimizer_updates") == 1
                and checkpoint.get("calibration_rows_opened") == 0
                and FROZEN.state_dict_digest(checkpoint["model_state_dict"])
                == checkpoint["model_state_digest"]
                and FROZEN.structured_digest(checkpoint["optimizer_state_dict"])
                == checkpoint["optimizer_state_digest"],
                "orphan conditional checkpoint changed")
        require(updates == 1 and clips == 1,
                "orphan checkpoint update custody changed")
    runtime = C.runtime_root(root)
    expected_files = {ARTIFACTS[name] for name in preserved} | {
        ARTIFACTS["failure"]}
    require({item.name for item in runtime.iterdir()} == expected_files
            and all(item.is_file() and not item.is_symlink()
                    and item.stat().st_mode & 0o222 == 0
                    for item in runtime.iterdir()),
            "technical failure runtime inventory changed")
    return failure


def validate_outcome(root: Path = ROOT) -> dict[str, Any]:
    has_terminal = path(root, "terminal").exists()
    has_failure = path(root, "failure").exists()
    require(has_terminal != has_failure, "exactly one terminal/failure required")
    return validate_terminal(root) if has_terminal else validate_failure(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("issue", "run", "validate"), required=True)
    args = parser.parse_args()
    if args.stage == "issue":
        result = issue_contract()
    elif args.stage == "run":
        result = run_once()
    else:
        result = validate_outcome()
    key = next((candidate for candidate in (
        C.CONTRACT_SELF_KEY, TERMINAL_KEY, FAILURE_KEY) if candidate in result), None)
    print(json.dumps({"stage": args.stage, "digest": result.get(key) if key else None,
                      "terminal_kind": result.get("terminal_kind"),
                      "mechanism_classification": result.get(
                          "mechanism_classification")}, sort_keys=True))


if __name__ == "__main__":
    main()
