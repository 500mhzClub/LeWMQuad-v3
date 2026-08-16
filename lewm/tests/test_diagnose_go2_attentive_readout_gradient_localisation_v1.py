"""Focused CPU/source tests for the bounded gradient-localisation runner."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.oracle import (
    go2_attentive_readout_gradient_localisation_v1_contract as C,
)
from scripts import diagnose_go2_attentive_readout_gradient_localisation_v1 as D


ROOT = Path(__file__).resolve().parents[2]


def test_tensor_stats_localise_first_nan_and_infinities() -> None:
    value = torch.tensor([[1.0, float("nan")],
                          [float("inf"), float("-inf")]])
    stats = D.tensor_numeric_stats(value)
    assert stats["finite_count"] == 1
    assert stats["nan_count"] == 1
    assert stats["positive_infinity_count"] == 1
    assert stats["negative_infinity_count"] == 1
    assert stats["maximum_absolute_finite_value"] == 1.0
    assert stats["finite_only_l2_norm"] == 1.0
    assert stats["first_nonfinite_flat_index"] == 1
    assert stats["first_nonfinite_multi_index"] == [0, 1]


def test_parameter_inventory_separates_none_from_nonfinite() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model[0].weight.grad = torch.ones_like(model[0].weight)
    model[0].bias.grad = torch.tensor([float("nan"), 0.0])
    rows = D.named_parameter_inventory(model)
    verdict = D.gradient_verdict(rows)
    assert verdict["nonfinite_parameter_set"] == ["0.bias"]
    assert verdict["offending_parameter_set"] == ["0.bias"]
    assert verdict["gradient_none_set"] == ["1.weight", "1.bias"]
    assert verdict["all_gradients_present_and_finite"] is False
    assert all(tuple(row) == C.PARAMETER_INVENTORY_FIELDS for row in rows)


def _pass_payload(seed: str) -> dict:
    return {
        "fixture_tensor_bindings": {"tokens": f"token-{seed}"},
        "outputs": {name: {"tensor_digest": f"{name}-{seed}"}
                    for name in D.COMPONENTS},
        "losses": {
            "components": {name: {"tensor_digest": f"loss-{name}-{seed}"}
                           for name in D.COMPONENTS},
            "selected_total": {"tensor_digest": f"total-{seed}"},
        },
        "parameter_gradient_inventory": [{
            "fully_qualified_name": "weight",
            "gradient_tensor_digest": f"grad-{seed}"}],
    }


def test_exact_harness_uses_tensor_receipt_digests_not_numeric_values() -> None:
    left = _pass_payload("a")
    same = _pass_payload("a")
    changed = _pass_payload("b")
    assert D.exact_pass_payload_equal(left, same)
    assert not D.exact_pass_payload_equal(left, changed)
    assert D._exact_mismatch(left, changed)["mismatches"]


def test_terminal_difference_payloads_are_deterministically_rederived() -> None:
    reproduction = _pass_payload("a")
    reproduction["complete_gradient_verdict"] = {
        "nonfinite_parameter_set": [], "gradient_none_set": ["head.weight"]}
    reproduction["pre_backward_finiteness"] = {"total_loss_finite": True}
    assert D._nonreproduction_difference(reproduction) == {
        "expected": "at least one nonfinite parameter gradient",
        "observed_nonfinite_parameter_set": [],
        "observed_gradient_none_set": ["head.weight"],
        "pre_backward_finiteness": {"total_loss_finite": True},
    }
    hook = _pass_payload("a")
    hook["sdpa_audit"] = {
        "exact_phase_kind_shape_dtype_ledger": False, "invocations": []}
    difference = D._hook_harness_difference(reproduction, hook)
    assert difference["reason"] == "hook pass production SDPA ledger changed"
    assert difference["tensor_receipt_difference"] is None


def test_frozen_loss_uses_one_final_division() -> None:
    raw = {name: torch.tensor(value, dtype=torch.float32)
           for name, value in zip(D.COMPONENTS, (3.25, 1.5, 0.75), strict=True)}
    observed = D._pass_loss(raw, "frozen_summed_loss")
    expected = (raw["progress"] + raw["safety"] + raw["completion"]) / 64
    assert torch.equal(observed, expected)
    assert torch.equal(D._pass_loss(raw, "safety_only"), raw["safety"] / 64)


def _matrix(*, c_finite: bool, c_math: bool,
            d_finite: bool, d_audit: bool, a_nonmath: bool = True) -> dict:
    def row(finite: bool) -> dict:
        return {
            "complete_gradient_verdict": {
                "all_gradients_present_and_finite": finite},
            "forward_equivalence_to_A": {"equivalent": True},
        }
    return {
        "A": {"sdpa_audit": {"has_non_math_dispatch": a_nonmath}},
        "B": {},
        "C": {**row(c_finite), "sdpa_audit": {
            "all_seven_dispatches_math": c_math}},
        "D": {**row(d_finite), "official_manual_audit": {
            "passed": d_audit}},
    }


def test_classification_is_predeclared_and_backend_has_precedence() -> None:
    both = _matrix(c_finite=True, c_math=True, d_finite=True, d_audit=True)
    assert D.classify_mechanism(both, exact_ab=True) == (
        "BACKEND_NUMERICAL_DEFECT_CONTRACT_PRESERVING")
    implementation = _matrix(
        c_finite=False, c_math=True, d_finite=True, d_audit=True)
    assert D.classify_mechanism(implementation, exact_ab=True) == (
        "IMPLEMENTATION_DEFECT_CONTRACT_PRESERVING")
    neither = _matrix(
        c_finite=False, c_math=False, d_finite=False, d_audit=False)
    assert D.classify_mechanism(neither, exact_ab=True) == (
        "ARCHITECTURE_OR_OBJECTIVE_CHANGE_REQUIRED")
    assert D.classify_mechanism(both, exact_ab=False) is None


def _ledger(backend: str = "MATH") -> dict:
    rows = []
    specs = [
        ("initial_forward", "self", 3072),
        ("initial_forward", "self", 3072),
        ("initial_forward", "self", 3072),
        ("initial_forward", "cross", 3),
        ("backward_checkpoint_recompute", "self", 3072),
        ("backward_checkpoint_recompute", "self", 3072),
        ("backward_checkpoint_recompute", "self", 3072),
    ]
    for index, (phase, kind, query_tokens) in enumerate(specs):
        rows.append({
            "invocation_index": index, "phase": phase, "kind": kind,
            "query_shape": [4, 16, query_tokens, 32],
            "key_shape": [4, 16, 3072, 32],
            "value_shape": [4, 16, 3072, 32],
            "dtype": "torch.float32",
            "selected_backend_inside_effective_context": backend,
            "dropout_p": 0.0, "is_causal": False,
        })
    return {"invocations": rows}


def test_sdpa_ledger_requires_all_seven_ordered_calls_and_shapes() -> None:
    audit = _ledger()
    assert D.sdpa_ledger_exact(audit)
    assert D._recompute_sdpa_summary(audit)["all_seven_dispatches_math"]
    audit["invocations"][3]["query_shape"] = [4, 16, 4, 32]
    assert not D.sdpa_ledger_exact(audit)


def test_forced_math_wrapper_encloses_the_actual_cpu_sdpa_call() -> None:
    original = F.scaled_dot_product_attention
    q = torch.randn(1, 1, 2, 4, requires_grad=True)
    with D.SDPAInventory("forced_math") as inventory:
        output = F.scaled_dot_product_attention(q, q, q)
        inventory.backward_started = True
        output.sum().backward()
    assert F.scaled_dot_product_attention is original
    assert inventory.invocations[0][
        "selected_backend_inside_effective_context"] == "MATH"


def test_historical_core_has_no_diagnostic_receipt_before_backward() -> None:
    source = inspect.getsource(D.run_pass)
    zero = source.index("optimizer.zero_grad(set_to_none=True)")
    forward = source.index("outputs = model(tokens, action_goal[batch])")
    raw_loss = source.index("raw_losses = _raw_component_losses")
    selected = source.index("total = _pass_loss")
    finite = source.index("torch.isfinite(total)")
    backward = source.index("total.backward()")
    receipt = source.index("tensor_bindings =")
    assert zero < forward < raw_loss < selected < finite < backward < receipt


def test_source_has_no_step_clip_repair_training_or_predictor_route() -> None:
    source = (ROOT / "scripts/diagnose_go2_attentive_readout_gradient_localisation_v1.py").read_text()
    assert ".step(" not in source
    assert "clip_grad" not in source
    assert "predictor checkpoint" not in source.lower()
    assert "calibration_evidence" not in source
    assert "train_once(" not in source
    assert "repair_authorised_now\": True" not in source


def test_contract_source_closure_is_exactly_four_additive_files() -> None:
    assert Path(__file__).relative_to(ROOT).as_posix() in C.NEW_SOURCE_PATHS
    assert len(C.NEW_SOURCE_PATHS) == 4
    assert C.EXECUTION_COUNTS["optimizer_constructions"] == 10
    assert C.EXECUTION_COUNTS["optimizer_steps"] == 0
