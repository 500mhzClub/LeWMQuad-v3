#!/usr/bin/env python3
"""Run the bounded no-step attentive-gradient localisation diagnostic.

The diagnostic preserves the failed production smoke, reconstructs the same
four-row calculation from the registered seed, and performs forward/backward
localisation only.  It constructs the frozen empty AdamW for historical
preparation but performs no optimiser step, state allocation, checkpoint,
training, scorer repair, calibration, or predictor route.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import (  # noqa: E402
    go2_attentive_readout_gradient_localisation_v1_contract as CONTRACT,
)
from scripts import train_go2_utility_scorer_v1_2 as FROZEN  # noqa: E402
from scripts import (  # noqa: E402
    train_go2_utility_scorer_v1_3_attentive_readout_v1 as ATTENTIVE,
)
from scripts import (  # noqa: E402
    train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1 as AMENDED,
)


STATUS = CONTRACT.STATUS
ATTEMPT_SCHEMA = "go2_attentive_readout_gradient_localisation_v1_attempt_v1"
PASS_SCHEMA = "go2_attentive_readout_gradient_localisation_v1_pass_v1"
GROUP_SCHEMA = "go2_attentive_readout_gradient_localisation_v1_group_v1"
FIXTURE_SCHEMA = "go2_attentive_readout_gradient_localisation_v1_fixture_v1"
TERMINAL_SCHEMA = "go2_attentive_readout_gradient_localisation_v1_terminal_v1"
FAILURE_SCHEMA = "go2_attentive_readout_gradient_localisation_v1_technical_failure_v1"

ATTEMPT_SELF_KEY = "gradient_localisation_attempt_digest"
PASS_SELF_KEY = "gradient_localisation_pass_digest"
GROUP_SELF_KEY = "gradient_localisation_group_digest"
FIXTURE_SELF_KEY = "gradient_localisation_fixture_digest"
TERMINAL_SELF_KEY = "gradient_localisation_terminal_digest"
FAILURE_SELF_KEY = "gradient_localisation_technical_failure_digest"

ARTIFACT_NAMES = {
    "attempt": "attempt.json",
    "fixture": "fit_only_fixture.json",
    "reproduction": "exact_reproduction.json",
    "hook": "hook_inventory.json",
    "isolation": "loss_isolation.json",
    "matrix": "backend_matrix.json",
    "terminal": "terminal.json",
    "failure": "technical_failure.json",
}
EXPECTED_SDPA_CALLS = 7
COMPONENTS = ("progress", "safety", "completion")
EXECUTION_COUNTERS = {
    "fresh_model_constructions": 0, "optimizer_constructions": 0,
    "forward_attempts": 0, "completed_forwards": 0,
    "backward_attempts": 0, "completed_backwards": 0,
    "optimizer_steps": 0, "gradient_clips": 0,
    "fixture_validation_row_record_opens": 0,
    "fixture_validation_latent_shard_opens": 0,
    "unique_fit_row_record_files": 0,
    "unique_fit_latent_shard_files": 0,
    "pass_latent_shard_loads": 0,
    "batch_presentations": 0, "examples_presented": 0,
}
COMPLETED_PASS_RECEIPTS: dict[str, dict[str, Any]] = {}


class GradientLocalisationError(RuntimeError):
    """The frozen diagnostic calculation or one-shot lineage changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GradientLocalisationError(message)


def _path(root: Path, name: str) -> Path:
    return CONTRACT.runtime_root(root) / ARTIFACT_NAMES[name]


def signed(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    require(key not in result, f"duplicate self key {key}")
    result[key] = CONTRACT.digest(result)
    return result


def validate_signed(value: Mapping[str, Any], key: str,
                    label: str) -> dict[str, Any]:
    try:
        return CONTRACT.validate_signed(value, key, label)
    except CONTRACT.GradientLocalisationContractError as exc:
        raise GradientLocalisationError(str(exc)) from exc


def publish_json_once(path: Path, value: Mapping[str, Any], label: str) -> None:
    require(not path.exists() and not path.is_symlink(), f"{label} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.exists() and not temporary.is_symlink(),
            f"{label} temporary path exists")
    with temporary.open("xb") as sink:
        sink.write(encoded)
        sink.flush()
        os.fsync(sink.fileno())
    temporary.replace(path)
    path.chmod(0o444)


def read_signed(path: Path, key: str, label: str) -> dict[str, Any]:
    try:
        value = CONTRACT.read_json(path, label)
    except CONTRACT.GradientLocalisationContractError as exc:
        raise GradientLocalisationError(str(exc)) from exc
    return validate_signed(value, key, label)


def artifact_binding(path: Path, key: str, label: str) -> dict[str, Any]:
    value = read_signed(path, key, label)
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
        else str(path),
        "self_digest": value[key],
        "sha256": CONTRACT.file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _dtype_name(value: torch.Tensor) -> str:
    return str(value.dtype)


def _first_multi_index(flat_index: int, shape: Sequence[int]) -> list[int]:
    if not shape:
        return []
    remaining = int(flat_index)
    result = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        size = int(shape[axis])
        result[axis] = remaining % size
        remaining //= size
    return result


def tensor_numeric_stats(value: torch.Tensor, *, include_digest: bool = False,
                         ) -> dict[str, Any]:
    tensor = value.detach()
    flat = tensor.reshape(-1)
    finite = torch.isfinite(flat)
    nan = torch.isnan(flat)
    positive_infinity = torch.isposinf(flat)
    negative_infinity = torch.isneginf(flat)
    finite_count = int(finite.sum().item())
    nan_count = int(nan.sum().item())
    positive_count = int(positive_infinity.sum().item())
    negative_count = int(negative_infinity.sum().item())
    finite_values = flat[finite].to(dtype=torch.float64)
    maximum = (float(finite_values.abs().max().item())
               if finite_count else None)
    l2 = (float(torch.linalg.vector_norm(finite_values).item())
          if finite_count else 0.0)
    bad = torch.nonzero(~finite, as_tuple=False).reshape(-1)
    first_flat = int(bad[0].item()) if int(bad.numel()) else None
    result = {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor),
        "element_count": int(tensor.numel()),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "positive_infinity_count": positive_count,
        "negative_infinity_count": negative_count,
        "all_finite": not (nan_count or positive_count or negative_count),
        "maximum_absolute_finite_value": maximum,
        "finite_only_l2_norm": l2,
        "first_nonfinite_flat_index": first_flat,
        "first_nonfinite_multi_index": (
            _first_multi_index(first_flat, tensor.shape)
            if first_flat is not None else None),
    }
    if include_digest:
        result["tensor_digest"] = FROZEN.tensor_digest(tensor.cpu())
    return result


def _tensor_tree(value: Any, prefix: str = "value",
                 ) -> list[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        result: list[tuple[str, torch.Tensor]] = []
        for key in sorted(value, key=str):
            result.extend(_tensor_tree(value[key], f"{prefix}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_tensor_tree(item, f"{prefix}[{index}]"))
        return result
    return []


def tensor_tree_stats(value: Any, prefix: str = "value") -> dict[str, Any]:
    tensors = _tensor_tree(value, prefix)
    rows = []
    for path, tensor in tensors:
        row = tensor_numeric_stats(tensor)
        row["tensor_path"] = path
        rows.append(row)
    offenders = [row for row in rows if not row["all_finite"]]
    finite_maxima = [row["maximum_absolute_finite_value"] for row in rows
                     if row["maximum_absolute_finite_value"] is not None]
    return {
        "tensor_count": len(rows),
        "all_finite": not offenders,
        "maximum_absolute_finite_value": max(finite_maxima, default=None),
        "first_nonfinite_tensor_path": (
            offenders[0]["tensor_path"] if offenders else None),
        "first_nonfinite_multi_index": (
            offenders[0]["first_nonfinite_multi_index"] if offenders else None),
        "tensors": rows,
    }


def named_parameter_inventory(model: nn.Module) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        module_path = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules[module_path]
        gradient = parameter.grad
        if gradient is None:
            gradient_fields = {
                "gradient_dtype": None, "gradient_is_none": True,
                "finite_count": 0, "nan_count": 0,
                "positive_infinity_count": 0, "negative_infinity_count": 0,
                "maximum_absolute_finite_value": None,
                "finite_only_l2_norm": 0.0,
                "first_nonfinite_flat_index": None,
                "first_nonfinite_multi_index": None,
                "gradient_tensor_digest": None,
            }
        else:
            stats = tensor_numeric_stats(gradient, include_digest=True)
            gradient_fields = {
                "gradient_dtype": stats["dtype"], "gradient_is_none": False,
                "finite_count": stats["finite_count"],
                "nan_count": stats["nan_count"],
                "positive_infinity_count": stats["positive_infinity_count"],
                "negative_infinity_count": stats["negative_infinity_count"],
                "maximum_absolute_finite_value":
                    stats["maximum_absolute_finite_value"],
                "finite_only_l2_norm": stats["finite_only_l2_norm"],
                "first_nonfinite_flat_index":
                    stats["first_nonfinite_flat_index"],
                "first_nonfinite_multi_index":
                    stats["first_nonfinite_multi_index"],
                "gradient_tensor_digest": stats["tensor_digest"],
            }
        row = {
            "fully_qualified_name": name,
            "module_path": module_path,
            "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
            "shape": list(parameter.shape),
            "parameter_dtype": _dtype_name(parameter),
            **gradient_fields,
        }
        require(tuple(row) == CONTRACT.PARAMETER_INVENTORY_FIELDS,
                f"parameter inventory schema changed for {name}")
        rows.append(row)
    return rows


def parameter_shape_dtype_inventory(model: nn.Module) -> list[list[Any]]:
    return [[name, list(parameter.shape), str(parameter.dtype)]
            for name, parameter in model.named_parameters()]


def gradient_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing = [str(row["fully_qualified_name"]) for row in rows
               if row["gradient_is_none"]]
    nonfinite = [str(row["fully_qualified_name"]) for row in rows
                 if row["nan_count"] or row["positive_infinity_count"]
                 or row["negative_infinity_count"]]
    return {
        "all_gradients_present_and_finite": not missing and not nonfinite,
        "gradient_none_set": missing,
        "offending_parameter_set": nonfinite,
        "nonfinite_parameter_set": nonfinite,
        "first_nonfinite_parameter_gradient": (
            nonfinite[0] if nonfinite else None),
        "all_nonfinite_parameter_gradients": nonfinite,
    }


def exact_gradient_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(row["fully_qualified_name"]): row["gradient_tensor_digest"]
            for row in rows}


def forward_equivalence(reference: Sequence[float], candidate: Sequence[float],
                        *, exact: bool = False) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    require(left.shape == right.shape, "forward comparison shape changed")
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    difference = np.abs(right - left)
    denominator = np.maximum(np.abs(left),
                             CONTRACT.FORWARD_EQUIVALENCE[
                                 "relative_denominator_floor"])
    maximum_absolute = float(difference.max(initial=0.0))
    maximum_relative = float((difference / denominator).max(initial=0.0))
    if exact:
        # This helper is intentionally numeric-reporting only.  Scientific
        # exactness is enforced by exact_pass_payload_equal() over canonical
        # tensor receipt digests, retaining dtype, shape, and raw bytes.
        equivalent = finite and bool(np.array_equal(left, right))
        rule = "numeric_exact_report_only_not_harness_gate"
    else:
        atol = CONTRACT.FORWARD_EQUIVALENCE[
            "a_c_d_absolute_tolerance"]
        rtol = CONTRACT.FORWARD_EQUIVALENCE[
            "a_c_d_relative_tolerance"]
        equivalent = finite and bool(np.all(
            difference <= atol + rtol * np.abs(left)))
        rule = "abs(diff)<=1e-5+1.3e-6*abs(A)"
    return {
        "all_values_finite": finite, "equivalent": equivalent,
        "rule": rule, "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
    }


def exact_pass_payload_equal(left: Mapping[str, Any],
                             right: Mapping[str, Any]) -> bool:
    """Compare canonical tensor-receipt digests, never converted values."""
    return _pass_exact_payload(left) == _pass_exact_payload(right)


def classify_mechanism(matrix: Mapping[str, Any], *,
                       exact_ab: bool,
                       reproduction_succeeded: bool = True) -> str | None:
    if not reproduction_succeeded or not exact_ab:
        return None
    a, c, d = (matrix[key] for key in ("A", "C", "D"))
    backend = (
        a["sdpa_audit"]["has_non_math_dispatch"]
        and c["complete_gradient_verdict"][
            "all_gradients_present_and_finite"]
        and c["forward_equivalence_to_A"]["equivalent"]
        and c["sdpa_audit"]["all_seven_dispatches_math"]
    )
    if backend:
        return CONTRACT.MECHANISM_CLASSIFICATIONS[0]
    implementation = (
        d["complete_gradient_verdict"][
            "all_gradients_present_and_finite"]
        and d["forward_equivalence_to_A"]["equivalent"]
        and d["official_manual_audit"]["passed"]
    )
    if implementation:
        return CONTRACT.MECHANISM_CLASSIFICATIONS[1]
    return CONTRACT.MECHANISM_CLASSIFICATIONS[2]


class ModuleHookInventory:
    """Temporary forward hooks with tensor-gradient boundary hooks."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.records: list[dict[str, Any]] = []
        self.backward_events: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.backward_started = False
        self._install()

    def _install(self) -> None:
        modules = dict(self.model.named_modules())
        for target in CONTRACT.HOOK_TARGETS:
            path = target["path"]
            if path in ("horizon_embeddings", "pooler.query_tokens"):
                continue
            require(path in modules, f"frozen hook target is absent: {path}")
            module = modules[path]
            actual_type = f"{type(module).__module__}.{type(module).__qualname__}"
            require(actual_type == target["module_type"],
                    f"frozen hook target type changed: {path}: {actual_type}")
            self.handles.append(module.register_forward_hook(
                self._forward_hook(path, target["role"])))
        query = self.model.pooler.query_tokens
        require(isinstance(query, nn.Parameter), "component queries changed")
        self.handles.append(query.register_hook(
            self._gradient_hook("pooler.query_tokens", "parameter", None)))

    def _gradient_hook(self, path: str, boundary: str,
                       record_index: int | None):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            stats = tensor_numeric_stats(gradient)
            event = {
                "event_index": len(self.backward_events),
                "module_path": path, "boundary": boundary,
                "forward_record_index": record_index,
                "all_finite": stats["all_finite"],
                "maximum_absolute_finite_value":
                    stats["maximum_absolute_finite_value"],
                "first_nonfinite_multi_index":
                    stats["first_nonfinite_multi_index"],
                "dtype": stats["dtype"], "shape": stats["shape"],
            }
            self.backward_events.append(event)
            if record_index is not None:
                self.records[record_index][f"gradient_{boundary}"].append(event)
            return gradient
        return hook

    def _forward_hook(self, path: str, role: str):
        def hook(_module: nn.Module, inputs: Any, output: Any) -> None:
            index = len(self.records)
            record = {
                "forward_record_index": index,
                "module_path": path, "role": role,
                "phase": ("backward_checkpoint_recompute"
                          if self.backward_started else "initial_forward"),
                "input": tensor_tree_stats(inputs, "input"),
                "output": tensor_tree_stats(output, "output"),
                "gradient_input": [], "gradient_output": [],
            }
            self.records.append(record)
            for _, tensor in _tensor_tree(inputs, "input"):
                if tensor.requires_grad:
                    tensor.register_hook(self._gradient_hook(
                        path, "input", index))
            for _, tensor in _tensor_tree(output, "output"):
                if tensor.requires_grad:
                    tensor.register_hook(self._gradient_hook(
                        path, "output", index))
        return hook

    def mark_backward(self) -> None:
        self.backward_started = True

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def receipt(self, model: nn.Module) -> dict[str, Any]:
        candidates = []
        for record in self.records:
            outputs = record["gradient_output"]
            inputs = record["gradient_input"]
            if (outputs and inputs and all(row["all_finite"] for row in outputs)
                    and any(not row["all_finite"] for row in inputs)):
                first_input = min(row["event_index"] for row in inputs
                                  if not row["all_finite"])
                candidates.append((first_input, record["module_path"],
                                   record["forward_record_index"]))
        candidates.sort()
        all_forward_finite = all(
            row["input"]["all_finite"] and row["output"]["all_finite"]
            for row in self.records)
        query = tensor_numeric_stats(model.pooler.query_tokens)
        horizon = tensor_numeric_stats(model.horizon_embeddings)
        return {
            "temporary_hooks_removed": not self.handles,
            "forward_records": self.records,
            "backward_event_order": self.backward_events,
            "all_forward_module_inputs_and_outputs_finite": all_forward_finite,
            "component_query_parameter_pre_backward": query,
            "horizon_embedding_buffer_pre_backward": horizon,
            "first_reverse_module_with_finite_downstream_and_nonfinite_upstream": (
                {"event_index": candidates[0][0],
                 "module_path": candidates[0][1],
                 "forward_record_index": candidates[0][2]}
                if candidates else None),
        }


class SDPAInventory:
    """Scoped per-invocation SDPA selector and q/k/v/out inventory."""

    def __init__(self, policy: str, *, capture_boundaries: bool = False) -> None:
        self.policy = policy
        self.capture_boundaries = capture_boundaries
        self.invocations: list[dict[str, Any]] = []
        self.backward_started = False
        self._original = None

    @staticmethod
    def _backend_name(choice: Any) -> str:
        try:
            return torch.nn.attention.SDPBackend(int(choice)).name
        except (TypeError, ValueError):
            return f"UNKNOWN_{choice}"

    @staticmethod
    def _arguments(args: Sequence[Any], kwargs: Mapping[str, Any],
                   ) -> dict[str, Any]:
        return {
            "attn_mask": kwargs.get("attn_mask", args[3] if len(args) > 3 else None),
            "dropout_p": kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0),
            "is_causal": kwargs.get("is_causal", args[5] if len(args) > 5 else False),
            "scale": kwargs.get("scale", None),
            "enable_gqa": kwargs.get("enable_gqa", False),
        }

    def _choice(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                values: Mapping[str, Any]) -> str:
        choice = torch._fused_sdp_choice(
            q, k, v, values["attn_mask"], values["dropout_p"],
            values["is_causal"], scale=values["scale"],
            enable_gqa=values["enable_gqa"])
        return self._backend_name(choice)

    def _capture_gradient(self, row: dict[str, Any], key: str):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            row[f"{key}_gradient"] = tensor_numeric_stats(gradient)
            return gradient
        return hook

    def _wrapped(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 *args: Any, **kwargs: Any) -> torch.Tensor:
        require(self._original is not None, "SDPA wrapper is not installed")
        values = self._arguments(args, kwargs)
        context = (torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.MATH])
                   if self.policy == "forced_math" else _null_context())
        with context:
            backend = self._choice(q, k, v, values)
            row: dict[str, Any] = {
                "invocation_index": len(self.invocations),
                "phase": ("backward_checkpoint_recompute"
                          if self.backward_started else "initial_forward"),
                "kind": "cross" if int(q.shape[-2]) == 3 else "self",
                "query_shape": list(q.shape), "key_shape": list(k.shape),
                "value_shape": list(v.shape), "dtype": str(q.dtype),
                "selected_backend_inside_effective_context": backend,
                "dropout_p": float(values["dropout_p"]),
                "is_causal": bool(values["is_causal"]),
            }
            if self.capture_boundaries:
                row["query"] = tensor_numeric_stats(q)
                row["key"] = tensor_numeric_stats(k)
                row["value"] = tensor_numeric_stats(v)
                for key, tensor in (("query", q), ("key", k), ("value", v)):
                    if tensor.requires_grad:
                        tensor.register_hook(self._capture_gradient(row, key))
            self.invocations.append(row)
            output = self._original(q, k, v, *args, **kwargs)
            if self.capture_boundaries:
                row["output"] = tensor_numeric_stats(output)
                if output.requires_grad:
                    output.register_hook(self._capture_gradient(row, "output"))
            return output

    def __enter__(self) -> "SDPAInventory":
        require(self.policy in ("production", "forced_math"),
                "invalid SDPA policy")
        self._original = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._wrapped
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        require(self._original is not None, "SDPA wrapper was not installed")
        F.scaled_dot_product_attention = self._original
        self._original = None

    def audit(self) -> dict[str, Any]:
        backends = [row["selected_backend_inside_effective_context"]
                    for row in self.invocations]
        result = {
            "invocation_count": len(backends),
            "expected_invocation_count": EXPECTED_SDPA_CALLS,
            "backend_sequence": backends,
            "all_seven_dispatches_math": (
                len(backends) == EXPECTED_SDPA_CALLS
                and all(value == "MATH" for value in backends)),
            "has_non_math_dispatch": any(value != "MATH" for value in backends),
            "every_selector_sampled_inside_effective_inner_context": True,
            "invocations": self.invocations,
        }
        result["exact_phase_kind_shape_dtype_ledger"] = sdpa_ledger_exact(result)
        return result


def sdpa_ledger_exact(audit: Mapping[str, Any]) -> bool:
    rows = audit.get("invocations", [])
    if len(rows) != EXPECTED_SDPA_CALLS:
        return False
    expected = [
        ("initial_forward", "self", [4, 16, 3072, 32],
         [4, 16, 3072, 32], [4, 16, 3072, 32]),
        ("initial_forward", "self", [4, 16, 3072, 32],
         [4, 16, 3072, 32], [4, 16, 3072, 32]),
        ("initial_forward", "self", [4, 16, 3072, 32],
         [4, 16, 3072, 32], [4, 16, 3072, 32]),
        ("initial_forward", "cross", [4, 16, 3, 32],
         [4, 16, 3072, 32], [4, 16, 3072, 32]),
        ("backward_checkpoint_recompute", "self", [4, 16, 3072, 32],
         [4, 16, 3072, 32], [4, 16, 3072, 32]),
        ("backward_checkpoint_recompute", "self", [4, 16, 3072, 32],
         [4, 16, 3072, 32], [4, 16, 3072, 32]),
        ("backward_checkpoint_recompute", "self", [4, 16, 3072, 32],
         [4, 16, 3072, 32], [4, 16, 3072, 32]),
    ]
    observed = [
        (row.get("phase"), row.get("kind"), row.get("query_shape"),
         row.get("key_shape"), row.get("value_shape"))
        for row in rows]
    return (observed == expected
            and all(row.get("dtype") == "torch.float32" for row in rows)
            and all(row.get("dropout_p") == 0.0
                    and row.get("is_causal") is False
                    and row.get("selected_backend_inside_effective_context")
                    in {"MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION",
                        "CUDNN_ATTENTION", "OVERRIDEABLE"}
                    for row in rows)
            and [row.get("invocation_index") for row in rows]
            == list(range(EXPECTED_SDPA_CALLS)))


@contextmanager
def _null_context():
    yield


@contextmanager
def official_explicit_attention(model: nn.Module, enabled: bool):
    attention_modules = []
    for path in (
            "pooler.blocks.0.attn", "pooler.blocks.1.attn",
            "pooler.blocks.2.attn", "pooler.cross_attention_block.xattn"):
        module = dict(model.named_modules())[path]
        require(getattr(module, "use_sdpa", None) is True,
                f"official attention flag changed at {path}")
        attention_modules.append((path, module))
    if enabled:
        for _, module in attention_modules:
            module.use_sdpa = False
    try:
        yield attention_modules
    finally:
        if enabled:
            for _, module in attention_modules:
                module.use_sdpa = True


def official_manual_audit(model: nn.Module,
                          attention_modules: Sequence[tuple[str, nn.Module]],
                          sdpa_invocations: int) -> dict[str, Any]:
    modules = dict(model.named_modules())
    layer_norms = [(path, module) for path, module in modules.items()
                   if path.startswith("pooler.") and isinstance(module, nn.LayerNorm)]
    attention_rows = []
    for path, module in attention_modules:
        parameters = list(module.parameters())
        attention_rows.append({
            "path": path, "use_sdpa_during_pass": module.use_sdpa is False,
            "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
            "scale": float(module.scale),
            "all_parameters_float32": all(p.dtype == torch.float32
                                           for p in parameters),
        })
    norm_rows = [{
        "path": path, "eps": float(module.eps),
        "all_parameters_float32": all(p.dtype == torch.float32
                                       for p in module.parameters()),
    } for path, module in layer_norms]
    passed = (
        len(attention_rows) == 4 and len(norm_rows) == 8
        and sdpa_invocations == 0
        and all(row["use_sdpa_during_pass"]
                and row["all_parameters_float32"] for row in attention_rows)
        and all(row["eps"] == 1e-5 and row["all_parameters_float32"]
                for row in norm_rows)
    )
    return {
        "passed": passed,
        "official_source_binding_digest":
            CONTRACT.OFFICIAL_POOLER_BINDING_DIGEST,
        "official_non_sdpa_formula": (
            "scaled q@k.T, softmax, dropout(0), weighted reduction; "
            "LayerNorm eps=1e-5"),
        "outer_and_reduction_dtype": "torch.float32",
        "attention_modules": attention_rows, "layer_norms": norm_rows,
        "sdpa_invocation_count": sdpa_invocations,
        "state_restored_after_context": False,
    }


def production_ambient() -> dict[str, Any]:
    preferred = torch.backends.cuda.preferred_rocm_fa_library()
    preferred_name = getattr(preferred, "name", str(preferred).split(".")[-1])
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdpa_memory_efficient_enabled": bool(
            torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdpa_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "sdpa_cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        "sdpa_priority_order": list(torch._C._get_sdp_priority_order()),
        "rocm_preferred_flash_attention_library": preferred_name,
        "deterministic_algorithms_after_fresh_model_state":
            bool(torch.are_deterministic_algorithms_enabled()),
    }


def device_preflight() -> tuple[torch.device, dict[str, Any]]:
    expected = CONTRACT.EXECUTION_ENVIRONMENT
    executable = Path(sys.executable).absolute()
    expected_executable = (ROOT / expected["python"]).absolute()
    require(executable == expected_executable,
            "diagnostic interpreter is not the authorised ROCm environment")
    expected_prefix = expected_executable.parents[1]
    require(Path(sys.prefix).absolute() == expected_prefix
            and Path(sys.base_prefix).resolve() == Path("/usr").resolve(),
            "diagnostic virtual-environment prefix changed")
    require(torch.__version__ == expected["torch_version"]
            and torch.version.hip == expected["torch_hip_version"],
            "diagnostic PyTorch/ROCm build changed")
    torch_path = Path(torch.__file__).resolve()
    expected_distribution = (
        ROOT / expected["torch_distribution_environment"]).resolve()
    require(torch_path.is_relative_to(expected_distribution),
            "diagnostic torch was not loaded from the authorised distribution")
    require(torch.cuda.is_available()
            and torch.cuda.device_count() == expected["visible_hip_device_count"],
            "diagnostic HIP device visibility changed")
    device = torch.device(expected["device"])
    name = torch.cuda.get_device_name(device)
    capability = list(torch.cuda.get_device_capability(device))
    properties = torch.cuda.get_device_properties(device)
    architecture = str(getattr(properties, "gcnArchName", ""))
    require(name == expected["device_name"]
            and capability == expected["device_capability"]
            and architecture.startswith(expected["device_architecture"]),
            "diagnostic R9700 identity changed")
    observed_ambient = production_ambient()
    expected_ambient = expected["production_ambient"]
    # configure_determinism runs with every model construction, so this one
    # field is checked after construction rather than at process entry.
    entry_ambient = dict(observed_ambient)
    entry_ambient.pop("deterministic_algorithms_after_fresh_model_state")
    expected_entry = dict(expected_ambient)
    expected_entry.pop("deterministic_algorithms_after_fresh_model_state")
    require(entry_ambient == expected_entry,
            "production FP32/SDPA ambient changed")
    require(observed_ambient[
                "deterministic_algorithms_after_fresh_model_state"]
            == expected["deterministic_algorithms_at_process_entry"],
            "diagnostic deterministic-algorithm state changed at process entry")
    return device, {
        "python": str(executable), "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix, "torch_path": str(torch_path),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip, "device": str(device),
        "device_name": name, "device_capability": capability,
        "device_architecture": architecture,
        "visible_hip_device_count": torch.cuda.device_count(),
        "entry_ambient": observed_ambient,
    }


def load_installed_contract(root: Path = ROOT) -> dict[str, Any]:
    path = CONTRACT.contract_path(root)
    try:
        value = CONTRACT.validate_contract(
            CONTRACT.read_json(path, "installed gradient-localisation contract"))
        source = CONTRACT.source_closure(root)
        lineage = CONTRACT.failed_smoke_lineage(root)
    except CONTRACT.GradientLocalisationContractError as exc:
        raise GradientLocalisationError(str(exc)) from exc
    require(value == CONTRACT.build_contract(source, lineage),
            "installed diagnostic contract does not bind live clean source")
    return value


def _fixture(root: Path) -> dict[str, Any]:
    rows, store, binding = AMENDED._fit_only_smoke_fixture(root)
    require(binding["fixture_digest"] == CONTRACT.FROZEN_FIXTURE_DIGEST
            and len(rows) == 4 and binding["calibration_latent_shards_opened"] == 0
            and binding["calibration_label_rows_opened"] == 0,
            "four-row fit-only fixture changed")
    EXECUTION_COUNTERS["fixture_validation_row_record_opens"] += 4
    EXECUTION_COUNTERS["fixture_validation_latent_shard_opens"] += 4
    EXECUTION_COUNTERS["unique_fit_row_record_files"] = 4
    EXECUTION_COUNTERS["unique_fit_latent_shard_files"] = 4
    return {
        "rows": rows, "store": store,
        "binding": binding,
    }


def _raw_component_losses(outputs: Sequence[torch.Tensor],
                          targets: Mapping[str, torch.Tensor],
                          indices: torch.Tensor,
                          ) -> dict[str, torch.Tensor]:
    require(len(outputs) == 3 and all(tuple(value.shape) == (4,)
                                      for value in outputs),
            "component output shape changed")
    return {
        "progress": F.mse_loss(
            outputs[0], targets["progress"][indices], reduction="sum"),
        "safety": F.binary_cross_entropy_with_logits(
            outputs[1], targets["safety"][indices], reduction="sum"),
        "completion": F.binary_cross_entropy_with_logits(
            outputs[2], targets["completion"][indices], reduction="sum"),
    }


def _pass_loss(raw_losses: Mapping[str, torch.Tensor],
               objective: str) -> torch.Tensor:
    denominator = CONTRACT.LOSS_CONTRACT["effective_batch_denominator"]
    if objective == "progress_only":
        return raw_losses["progress"] / denominator
    if objective == "safety_only":
        return raw_losses["safety"] / denominator
    if objective == "completion_only":
        return raw_losses["completion"] / denominator
    require(objective == "frozen_summed_loss", "unknown diagnostic objective")
    # Exact failed-smoke association: two FP32 additions, then one division.
    return (raw_losses["progress"] + raw_losses["safety"]
            + raw_losses["completion"]) / denominator


def _model_parameter_finiteness(model: nn.Module) -> dict[str, Any]:
    offenders = []
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            offenders.append(name)
    return {"all_finite": not offenders, "offenders": offenders,
            "parameter_tensor_count": sum(1 for _ in model.parameters())}


def _output_payload(outputs: Sequence[torch.Tensor]) -> dict[str, Any]:
    return {
        component: {
            "values": [float(value) for value in tensor.detach().cpu().tolist()],
            "tensor_digest": FROZEN.tensor_digest(tensor.detach().cpu()),
            "stats": tensor_numeric_stats(tensor),
        }
        for component, tensor in zip(COMPONENTS, outputs, strict=True)
    }


def _loss_payload(losses: Mapping[str, torch.Tensor],
                  total: torch.Tensor) -> dict[str, Any]:
    return {
        "components": {key: {
            "value": float(value.detach().cpu()),
            "tensor_digest": FROZEN.tensor_digest(value.detach().cpu()),
            "stats": tensor_numeric_stats(value),
        } for key, value in losses.items()},
        "selected_total": {
            "value": float(total.detach().cpu()),
            "tensor_digest": FROZEN.tensor_digest(total.detach().cpu()),
            "stats": tensor_numeric_stats(total),
        },
    }


def _pass_exact_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fixture_tensor_bindings": value["fixture_tensor_bindings"],
        "component_output_digests": {
            key: row["tensor_digest"]
            for key, row in value["outputs"].items()},
        "component_loss_digests": {
            key: row["tensor_digest"]
            for key, row in value["losses"]["components"].items()},
        "total_loss_digest": value["losses"]["selected_total"]["tensor_digest"],
        "named_gradient_digests": exact_gradient_map(
            value["parameter_gradient_inventory"]),
    }


def _fresh_model(device: torch.device) -> tuple[nn.Module, str]:
    model, state, state_digest = AMENDED._fresh_model_state()
    EXECUTION_COUNTERS["fresh_model_constructions"] += 1
    require(state_digest == CONTRACT.FROZEN_INITIAL_STATE_DIGEST,
            "registered attentive initial state changed")
    inventory = parameter_shape_dtype_inventory(model)
    require(CONTRACT.digest(inventory)
            == CONTRACT.FROZEN_PARAMETER_INVENTORY_DIGEST
            and len(inventory) == CONTRACT.TRAINABLE_PARAMETER_TENSOR_COUNT
            and sum(parameter.numel() for parameter in model.parameters())
            == CONTRACT.TRAINABLE_PARAMETER_COUNT,
            "registered attentive parameter inventory changed")
    require(model.training and model.pooler.use_activation_checkpointing is True,
            "production train/checkpointing mode changed")
    require(production_ambient()
            == CONTRACT.EXECUTION_ENVIRONMENT["production_ambient"],
            "production deterministic/FP32 ambient changed after construction")
    model.to(device)
    return model, FROZEN.state_dict_digest(state)


def run_pass(*, name: str, objective: str, fixture: Mapping[str, Any],
             device: torch.device, backend: str = "production",
             hooks: bool = False, record_sdpa: bool = False,
             explicit_official: bool = False) -> dict[str, Any]:
    require(backend in ("production", "forced_math"), "invalid pass backend")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model, initial_state_digest = _fresh_model(device)
    model.train()
    budget = ATTENTIVE.frozen_budget()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(budget["lr"]),
        weight_decay=float(budget["weight_decay"]))
    EXECUTION_COUNTERS["optimizer_constructions"] += 1
    optimizer_state_before = optimizer.state_dict()
    require(optimizer_state_before["state"] == {},
            "diagnostic AdamW unexpectedly has initial state")
    optimizer_digest_before = FROZEN.structured_digest(optimizer_state_before)
    action_goal, targets = ATTENTIVE._small_features(fixture["rows"], device)
    batch_cpu = torch.arange(4, dtype=torch.int64)
    batch = batch_cpu.to(device)
    tokens = ATTENTIVE._token_batch(
        fixture["rows"], fixture["store"], batch_cpu.tolist(), device)
    EXECUTION_COUNTERS["pass_latent_shard_loads"] += 4
    EXECUTION_COUNTERS["batch_presentations"] += 1
    EXECUTION_COUNTERS["examples_presented"] += 4
    require(not torch.is_autocast_enabled("cuda"),
            "diagnostic unexpectedly entered autocast")
    module_hooks = ModuleHookInventory(model) if hooks else None
    sdpa = (SDPAInventory(backend, capture_boundaries=hooks)
            if record_sdpa else None)
    attention_context = official_explicit_attention(model, explicit_official)
    sdpa_context = sdpa if sdpa is not None else _null_context()
    try:
        with attention_context as attention_modules, sdpa_context:
            # Historical core: no diagnostic allocation/synchronisation is
            # inserted between zero_grad, forward, the exact frozen loss,
            # its single finiteness check, and backward.
            optimizer.zero_grad(set_to_none=True)
            EXECUTION_COUNTERS["forward_attempts"] += 1
            outputs = model(tokens, action_goal[batch])
            EXECUTION_COUNTERS["completed_forwards"] += 1
            raw_losses = _raw_component_losses(outputs, targets, batch)
            total = _pass_loss(raw_losses, objective)
            require(bool(torch.isfinite(total).item()),
                    "diagnostic loss is non-finite before backward")
            if module_hooks is not None:
                module_hooks.mark_backward()
            if sdpa is not None:
                sdpa.backward_started = True
            EXECUTION_COUNTERS["backward_attempts"] += 1
            total.backward()
            EXECUTION_COUNTERS["completed_backwards"] += 1

            # All general-purpose receipts are deliberately post-backward.
            losses = {key: value / 64 for key, value in raw_losses.items()}
            parameter_pre = _model_parameter_finiteness(model)
            input_pre = tensor_tree_stats(
                {"tokens": tokens, "action_goal": action_goal}, "inputs")
            target_pre = tensor_tree_stats(targets, "targets")
            output_pre = tensor_tree_stats(outputs, "component_outputs")
            activation_pre = (
                all(row["input"]["all_finite"] and row["output"]["all_finite"]
                    for row in module_hooks.records)
                if module_hooks is not None else output_pre["all_finite"])
            pre_backward = {
                "all_model_parameters_finite": parameter_pre["all_finite"],
                "all_inputs_finite": input_pre["all_finite"],
                "all_activations_finite": activation_pre,
                "all_targets_finite": target_pre["all_finite"],
                "all_component_losses_finite": all(
                    bool(torch.isfinite(value).item()) for value in losses.values()),
                "total_loss_finite": bool(torch.isfinite(total).item()),
                "parameter_offenders": parameter_pre["offenders"],
                "input_offenders": ([] if input_pre["all_finite"] else
                                    [input_pre["first_nonfinite_tensor_path"]]),
                "activation_offenders": ([] if activation_pre else
                                           ["hook_inventory"]),
                "target_offenders": ([] if target_pre["all_finite"] else
                                     [target_pre["first_nonfinite_tensor_path"]]),
                "component_loss_offenders": [
                    key for key, value in losses.items()
                    if not bool(torch.isfinite(value).item())],
            }
            require(tuple(pre_backward) == CONTRACT.PRE_BACKWARD_FINITE_FIELDS,
                    "pre-backward finiteness schema changed")
            require(all(pre_backward[key] for key in (
                "all_model_parameters_finite", "all_inputs_finite",
                "all_activations_finite", "all_targets_finite",
                "all_component_losses_finite", "total_loss_finite")),
                "diagnostic pass is nonfinite before backward")
            outputs_payload = _output_payload(outputs)
            losses_payload = _loss_payload(losses, total)
            tensor_bindings = {
                "tokens": FROZEN.tensor_digest(tokens.detach().cpu()),
                "action_goal": FROZEN.tensor_digest(action_goal.detach().cpu()),
                "targets": {key: FROZEN.tensor_digest(value.detach().cpu())
                            for key, value in targets.items()},
                "batch": FROZEN.tensor_digest(batch.detach().cpu()),
            }
            gradients = named_parameter_inventory(model)
            verdict = gradient_verdict(gradients)
            manual_audit = (official_manual_audit(
                model, attention_modules,
                len(sdpa.invocations) if sdpa is not None else 0)
                if explicit_official else None)
        if module_hooks is not None:
            module_hooks.close()
            hook_receipt = module_hooks.receipt(model)
            if sdpa is not None:
                hook_receipt["sdpa_internal_boundaries"] = sdpa.invocations
        else:
            hook_receipt = None
        sdpa_audit = (sdpa.audit() if sdpa is not None else {
            "invocation_count": None, "not_instrumented": True})
        if manual_audit is not None:
            manual_audit["state_restored_after_context"] = all(
                getattr(module, "use_sdpa", None) is True
                for _, module in attention_modules)
            manual_audit["passed"] = (
                manual_audit["passed"]
                and manual_audit["state_restored_after_context"])
        state_after = FROZEN.state_dict_digest(FROZEN._cpu_state(model))
        require(state_after == initial_state_digest,
                "forward/backward diagnostic mutated model state")
        optimizer_state_after = optimizer.state_dict()
        optimizer_digest_after = FROZEN.structured_digest(optimizer_state_after)
        require(optimizer_state_after["state"] == {}
                and optimizer_digest_after == optimizer_digest_before,
                "diagnostic AdamW state changed without a step")
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        receipt = {
            "schema": PASS_SCHEMA, "status": STATUS, "pass_name": name,
            "objective": objective, "backend_policy": backend,
            "explicit_official_non_sdpa": explicit_official,
            "fixture_digest": fixture["binding"]["fixture_digest"],
            "fixture_tensor_bindings": tensor_bindings,
            "registered_seed": CONTRACT.FROZEN_ATTENTIVE_SEED,
            "initial_state_digest": initial_state_digest,
            "final_state_digest": state_after,
            "state_unchanged": True, "model_mode": "train",
            "activation_checkpointing": True,
            "autocast": False, "parameter_dtype": "torch.float32",
            "latent_dtype": str(tokens.dtype),
            "loss_effective_batch_denominator": 64,
            "pre_backward_finiteness": pre_backward,
            "finiteness_receipt_timing": (
                "historical core ran forward-loss-single-finite-check-backward; "
                "general receipts were collected afterward from retained "
                "tensors; internal hook boundaries were captured in their "
                "forward calls only for the hook pass"),
            "pre_backward_activation_coverage": (
                "all_frozen_internal_hook_boundaries"
                if hooks else "component_outputs_only"),
            "internal_activation_finiteness_proven_by_this_pass": hooks,
            "outputs": outputs_payload, "losses": losses_payload,
            "parameter_gradient_inventory": gradients,
            "complete_gradient_verdict": verdict,
            "hook_inventory": hook_receipt, "sdpa_audit": sdpa_audit,
            "official_manual_audit": manual_audit,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "wall_time_seconds": elapsed,
            "optimizer": {
                "class": "torch.optim.AdamW", "settings": budget,
                "zero_grad_set_to_none_before_forward": True,
                "state_entry_count_before": 0, "state_entry_count_after": 0,
                "state_digest_before": optimizer_digest_before,
                "state_digest_after": optimizer_digest_after,
                "state_unchanged": True,
            },
            "optimizer_constructions": 1, "optimizer_steps": 0,
            "gradient_clips": 0,
        }
        return signed(receipt, PASS_SELF_KEY)
    finally:
        if module_hooks is not None:
            module_hooks.close()
        del optimizer, model, tokens, action_goal, targets, batch
        torch.cuda.empty_cache()


def _combined_numeric_vector(value: Mapping[str, Any]) -> list[float]:
    result = []
    for component in COMPONENTS:
        result.extend(value["outputs"][component]["values"])
    for component in COMPONENTS:
        result.append(value["losses"]["components"][component]["value"])
    result.append(value["losses"]["selected_total"]["value"])
    return result


def _exact_mismatch(reference: Mapping[str, Any],
                    candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    left = _pass_exact_payload(reference)
    right = _pass_exact_payload(candidate)
    if left == right:
        return None
    mismatches = []
    for section in sorted(left):
        if left[section] != right.get(section):
            if isinstance(left[section], Mapping):
                names = sorted(set(left[section]) | set(right.get(section, {})))
                changed = [name for name in names
                           if left[section].get(name)
                           != right.get(section, {}).get(name)]
            else:
                changed = [section]
            mismatches.append({"section": section, "changed_fields": changed})
    return {"reference_payload_digest": CONTRACT.digest(left),
            "candidate_payload_digest": CONTRACT.digest(right),
            "mismatches": mismatches}


def _nonreproduction_difference(
        reproduction: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected": "at least one nonfinite parameter gradient",
        "observed_nonfinite_parameter_set": reproduction[
            "complete_gradient_verdict"]["nonfinite_parameter_set"],
        "observed_gradient_none_set": reproduction[
            "complete_gradient_verdict"]["gradient_none_set"],
        "pre_backward_finiteness": reproduction[
            "pre_backward_finiteness"],
    }


def _hook_harness_difference(reference: Mapping[str, Any],
                             hook: Mapping[str, Any]) -> dict[str, Any] | None:
    mismatch = _exact_mismatch(reference, hook)
    if not hook["sdpa_audit"]["exact_phase_kind_shape_dtype_ledger"]:
        return {
            "reason": "hook pass production SDPA ledger changed",
            "sdpa_audit": hook["sdpa_audit"],
            "tensor_receipt_difference": mismatch,
        }
    return mismatch


def _terminal_payload(*, contract: Mapping[str, Any], attempt: Mapping[str, Any],
                      terminal_kind: str, mechanism: str | None,
                      artifacts: Mapping[str, Any],
                      first_reverse: Any,
                      first_parameter: str | None,
                      all_parameters: Sequence[str],
                      exact_difference: Mapping[str, Any] | None = None,
                      technical_preflight: Mapping[str, Any] | None = None,
                      loss_component_attribution: Mapping[str, Any] | None = None,
                      production_backend_reproduction_evidence:
                          Mapping[str, Any] | None = None,
                      completed_passes: Sequence[str] = (),
                      ) -> dict[str, Any]:
    require((mechanism in CONTRACT.MECHANISM_CLASSIFICATIONS)
            == (terminal_kind == "COMPLETED_MECHANISM_CLASSIFICATION"),
            "terminal mechanism cardinality changed")
    repair_support = mechanism in CONTRACT.MECHANISM_CLASSIFICATIONS[:2]
    return signed({
        "schema": TERMINAL_SCHEMA, "status": STATUS,
        "terminal_kind": terminal_kind, "complete": True,
        "mechanism_classification": mechanism,
        "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
        "attempt_digest": attempt[ATTEMPT_SELF_KEY],
        "artifact_bindings": dict(artifacts),
        "first_reverse_module_with_finite_downstream_and_nonfinite_upstream":
            first_reverse,
        "first_nonfinite_parameter_gradient": first_parameter,
        "all_nonfinite_parameter_gradients": list(all_parameters),
        "exact_harness_or_reproduction_difference": exact_difference,
        "technical_preflight": (dict(technical_preflight)
                                if technical_preflight is not None else None),
        "loss_component_attribution": (
            dict(loss_component_attribution)
            if loss_component_attribution is not None else None),
        "production_backend_reproduction_evidence": (
            dict(production_backend_reproduction_evidence)
            if production_backend_reproduction_evidence is not None else None),
        "completed_passes": list(completed_passes),
        "later_repair_gate": {
            "classification_can_support_separate_repair_decision":
                repair_support,
            "repair_authorised_now": False,
            "training_authorised_now": False,
            "automatic_repair_or_training": False,
        },
        "execution_counts": dict(EXECUTION_COUNTERS),
        "predictor_checkpoints_opened": 0,
        "calibration_rows_opened": 0,
        "scorer_training_started": False,
    }, TERMINAL_SELF_KEY)


def _publish_terminal(root: Path, **kwargs: Any) -> dict[str, Any]:
    terminal = _terminal_payload(**kwargs)
    publish_json_once(_path(root, "terminal"), terminal,
                      "gradient-localisation terminal")
    return validate_terminal(root)


def _matrix_entry(receipt: Mapping[str, Any],
                  reference: Mapping[str, Any], *, exact: bool = False,
                  ) -> dict[str, Any]:
    numerical = forward_equivalence(
        _combined_numeric_vector(reference), _combined_numeric_vector(receipt),
        exact=False)
    return {
        **dict(receipt),
        "forward_equivalence_to_A": numerical,
        "exact_tensor_receipt_equality_to_A": (
            exact_pass_payload_equal(reference, receipt) if exact else None),
        "forward_maximum_absolute_difference_from_A":
            numerical["maximum_absolute_difference"],
        "forward_maximum_relative_difference_from_A":
            numerical["maximum_relative_difference"],
        "component_losses": {
            key: row["value"]
            for key, row in receipt["losses"]["components"].items()},
        "offending_parameter_set": receipt["complete_gradient_verdict"][
            "offending_parameter_set"],
        "actual_sdpa_backend_per_invocation": receipt["sdpa_audit"].get(
            "backend_sequence", []),
    }


def loss_component_attribution(
        passes: Sequence[Mapping[str, Any]],
        original_nonfinite: Sequence[str]) -> dict[str, Any]:
    require([value["objective"] for value in passes]
            == list(CONTRACT.LOSS_ISOLATION_PASSES),
            "loss-attribution pass order changed")
    original = set(original_nonfinite)
    result = {}
    for value in passes:
        observed = value["complete_gradient_verdict"][
            "nonfinite_parameter_set"]
        observed_set = set(observed)
        result[value["objective"]] = {
            "nonfinite_parameter_set": list(observed),
            "gradient_none_set": value["complete_gradient_verdict"][
                "gradient_none_set"],
            "at_least_one_original_offender_reproduced":
                bool(original & observed_set),
            "all_original_offenders_reproduced": original <= observed_set,
            "exact_original_offender_set_reproduced":
                original == observed_set,
        }
    return result


def _record_failure(root: Path, *, contract: Mapping[str, Any],
                    attempt: Mapping[str, Any], stage: str, error: BaseException,
                    completed_passes: Sequence[str],
                    technical_preflight: Mapping[str, Any] | None) -> dict[str, Any]:
    path = _path(root, "failure")
    if path.exists() or path.is_symlink():
        return read_signed(path, FAILURE_SELF_KEY,
                           "gradient-localisation technical failure")
    traceback_value = traceback.format_exc()
    compatibility = (isinstance(error, torch.OutOfMemoryError)
                     and stage in ("backend_matrix_C", "backend_matrix_D"))
    value = signed({
        "schema": FAILURE_SCHEMA,
        "status": ("EXACT_SHAPE_BACKEND_COMPATIBILITY_TECHNICAL_STOP"
                   if compatibility else
                   "INVALID_GRADIENT_LOCALISATION_TECHNICAL_STOP"),
        "complete": True, "stage": stage,
        "exception_type": type(error).__name__,
        "exception_message": str(error), "traceback": traceback_value,
        "traceback_sha256": hashlib.sha256(
            traceback_value.encode("utf-8")).hexdigest(),
        "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
        "attempt_digest": attempt[ATTEMPT_SELF_KEY],
        "completed_passes": list(completed_passes),
        "completed_pass_receipts": {
            name: COMPLETED_PASS_RECEIPTS[name]
            for name in completed_passes},
        "technical_preflight": (dict(technical_preflight)
                                if technical_preflight is not None else None),
        "mechanism_classification": None,
        "contracted_four_row_shape_not_reduced": True,
        "four_row_pass_attempted": EXECUTION_COUNTERS["forward_attempts"] > 0,
        "batch_size_reduction_attempted": False,
        "execution_counts": dict(EXECUTION_COUNTERS),
        "repair_authorised": False,
        "training_authorised": False, "automatic_retry_authorised": False,
        "predictor_checkpoints_opened": 0, "calibration_rows_opened": 0,
    }, FAILURE_SELF_KEY)
    publish_json_once(path, value, "gradient-localisation technical failure")
    return value


def _ensure_runtime_unconsumed(root: Path) -> None:
    allowed = {"contract.json"}
    runtime = CONTRACT.runtime_root(root)
    require(runtime.is_dir() and not runtime.is_symlink(),
            "diagnostic contract namespace is absent")
    observed = {path.name for path in runtime.iterdir()}
    require(observed <= allowed,
            "gradient-localisation runtime was already consumed")


def run_once(root: Path = ROOT) -> dict[str, Any]:
    terminal_path = _path(root, "terminal")
    failure_path = _path(root, "failure")
    if terminal_path.exists() or terminal_path.is_symlink():
        return validate_terminal(root)
    if failure_path.exists() or failure_path.is_symlink():
        failure = validate_failure(root)
        raise GradientLocalisationError(
            f"diagnostic ended in technical failure: {failure['stage']}")
    contract = load_installed_contract(root)
    _ensure_runtime_unconsumed(root)
    require(not any(EXECUTION_COUNTERS.values()),
            "diagnostic execution counters were already consumed")
    require(not COMPLETED_PASS_RECEIPTS,
            "diagnostic completed-pass cache was already consumed")
    official = ATTENTIVE.validate_official_pooler_source()
    require(official["binding_digest"]
            == CONTRACT.OFFICIAL_POOLER_BINDING_DIGEST,
            "official pooler calculation binding changed")
    attempt = signed({
        "schema": ATTEMPT_SCHEMA, "status": STATUS,
        "attempt_number": 1, "maximum_attempts": 1,
        "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
        "source_closure_digest": contract["source_closure"][
            CONTRACT.SOURCE_CLOSURE_SELF_KEY],
        "failed_smoke_digest": CONTRACT.SMOKE_FAILURE_DIGEST,
        "official_pooler_binding_digest":
            CONTRACT.OFFICIAL_POOLER_BINDING_DIGEST,
        "fixture_digest": CONTRACT.FROZEN_FIXTURE_DIGEST,
        "initial_state_digest": CONTRACT.FROZEN_INITIAL_STATE_DIGEST,
        "registered_seed": CONTRACT.FROZEN_ATTENTIVE_SEED,
        "planned_execution_counts": CONTRACT.EXECUTION_COUNTS,
        "optimizer_constructions_planned": 10, "optimizer_steps": 0,
        "gradient_clips": 0, "repair_authorised": False,
        "training_authorised": False,
    }, ATTEMPT_SELF_KEY)
    publish_json_once(_path(root, "attempt"), attempt,
                      "gradient-localisation attempt")
    stage = "device_preflight"
    completed: list[str] = []
    preflight: dict[str, Any] | None = None
    try:
        device, preflight = device_preflight()
        stage = "fit_fixture_preflight"
        fixture = _fixture(root)
        fixture_receipt = signed({
            "schema": FIXTURE_SCHEMA, "status": STATUS,
            "contract_digest": contract[CONTRACT.CONTRACT_SELF_KEY],
            "attempt_digest": attempt[ATTEMPT_SELF_KEY],
            "fit_only_fixture_binding": fixture["binding"],
            "fixture_digest": fixture["binding"]["fixture_digest"],
            "row_count": 4, "calibration_label_rows_opened": 0,
            "calibration_latent_shards_opened": 0,
        }, FIXTURE_SELF_KEY)
        publish_json_once(_path(root, "fixture"), fixture_receipt,
                          "fit-only fixture binding")

        stage = "exact_reproduction"
        reproduction = run_pass(
            name="exact_reproduction", objective="frozen_summed_loss",
            fixture=fixture, device=device)
        COMPLETED_PASS_RECEIPTS["exact_reproduction"] = reproduction
        publish_json_once(_path(root, "reproduction"), reproduction,
                          "exact reproduction")
        completed.append("exact_reproduction")
        reproduction_nonfinite = reproduction[
            "complete_gradient_verdict"]["nonfinite_parameter_set"]
        if not reproduction_nonfinite:
            artifacts = {
                "attempt": artifact_binding(_path(root, "attempt"),
                                            ATTEMPT_SELF_KEY, "attempt"),
                "fixture": artifact_binding(_path(root, "fixture"),
                                            FIXTURE_SELF_KEY, "fixture"),
                "reproduction": artifact_binding(
                    _path(root, "reproduction"), PASS_SELF_KEY,
                    "exact reproduction"),
            }
            difference = _nonreproduction_difference(reproduction)
            return _publish_terminal(
                root, contract=contract, attempt=attempt,
                terminal_kind="NONREPRODUCTION_TECHNICAL_STOP",
                mechanism=None, artifacts=artifacts, first_reverse=None,
                first_parameter=None, all_parameters=[],
                exact_difference=difference, technical_preflight=preflight,
                completed_passes=completed)

        stage = "hook_inventory"
        hook_receipt = run_pass(
            name="hook_inventory", objective="frozen_summed_loss",
            fixture=fixture, device=device, hooks=True, record_sdpa=True)
        COMPLETED_PASS_RECEIPTS["hook_inventory"] = hook_receipt
        publish_json_once(_path(root, "hook"), hook_receipt,
                          "hook inventory")
        completed.append("hook_inventory")
        mismatch = _hook_harness_difference(reproduction, hook_receipt)
        if mismatch is not None:
            artifacts = {
                name: artifact_binding(_path(root, name), key, name)
                for name, key in (("attempt", ATTEMPT_SELF_KEY),
                                  ("fixture", FIXTURE_SELF_KEY),
                                  ("reproduction", PASS_SELF_KEY),
                                  ("hook", PASS_SELF_KEY))}
            return _publish_terminal(
                root, contract=contract, attempt=attempt,
                terminal_kind="INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP",
                mechanism=None, artifacts=artifacts,
                first_reverse=hook_receipt["hook_inventory"][
                    "first_reverse_module_with_finite_downstream_and_nonfinite_upstream"],
                first_parameter=reproduction_nonfinite[0],
                all_parameters=reproduction_nonfinite,
                exact_difference=mismatch, technical_preflight=preflight,
                completed_passes=completed)

        stage = "loss_isolation"
        isolation_passes = []
        for objective in CONTRACT.LOSS_ISOLATION_PASSES:
            pass_receipt = run_pass(
                name=f"loss_isolation_{objective}", objective=objective,
                fixture=fixture, device=device)
            isolation_passes.append(pass_receipt)
            COMPLETED_PASS_RECEIPTS[f"loss_isolation_{objective}"] = pass_receipt
            completed.append(f"loss_isolation_{objective}")
        isolation = signed({
            "schema": GROUP_SCHEMA, "status": STATUS,
            "group": "loss_isolation", "passes": isolation_passes,
            "fresh_model_constructions": 4,
            "optimizer_constructions": 4, "optimizer_steps": 0,
            "gradient_clips": 0,
        }, GROUP_SELF_KEY)
        publish_json_once(_path(root, "isolation"), isolation,
                          "loss-isolation group")
        summed = isolation_passes[-1]
        attribution = loss_component_attribution(
            isolation_passes, reproduction_nonfinite)
        mismatch = _exact_mismatch(reproduction, summed)
        if mismatch is not None:
            artifacts = {
                "attempt": artifact_binding(_path(root, "attempt"),
                                            ATTEMPT_SELF_KEY, "attempt"),
                "fixture": artifact_binding(_path(root, "fixture"),
                                            FIXTURE_SELF_KEY, "fixture"),
                "reproduction": artifact_binding(
                    _path(root, "reproduction"), PASS_SELF_KEY,
                    "exact reproduction"),
                "hook": artifact_binding(_path(root, "hook"), PASS_SELF_KEY,
                                         "hook inventory"),
                "isolation": artifact_binding(
                    _path(root, "isolation"), GROUP_SELF_KEY,
                    "loss isolation"),
            }
            return _publish_terminal(
                root, contract=contract, attempt=attempt,
                terminal_kind="INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP",
                mechanism=None, artifacts=artifacts,
                first_reverse=hook_receipt["hook_inventory"][
                    "first_reverse_module_with_finite_downstream_and_nonfinite_upstream"],
                first_parameter=reproduction_nonfinite[0],
                all_parameters=reproduction_nonfinite,
                exact_difference=mismatch, technical_preflight=preflight,
                loss_component_attribution=attribution,
                completed_passes=completed)

        matrix_passes: dict[str, dict[str, Any]] = {}
        for key, backend, explicit in (
                ("A", "production", False), ("B", "production", False),
                ("C", "forced_math", False), ("D", "production", True)):
            stage = f"backend_matrix_{key}"
            receipt = run_pass(
                name=f"backend_matrix_{key}", objective="frozen_summed_loss",
                fixture=fixture, device=device, backend=backend,
                record_sdpa=True, explicit_official=explicit)
            COMPLETED_PASS_RECEIPTS[f"backend_matrix_{key}"] = receipt
            matrix_passes[key] = _matrix_entry(
                receipt, reproduction, exact=key in ("A", "B"))
            completed.append(f"backend_matrix_{key}")

        ab_exact = (exact_pass_payload_equal(matrix_passes["A"],
                                             matrix_passes["B"])
                    and matrix_passes["A"]["sdpa_audit"]["backend_sequence"]
                    == matrix_passes["B"]["sdpa_audit"]["backend_sequence"])
        group_exact = all(exact_pass_payload_equal(reproduction, value)
                          for value in (hook_receipt, summed,
                                        matrix_passes["A"],
                                        matrix_passes["B"]))
        ledger_exact = (
            hook_receipt["sdpa_audit"][
                "exact_phase_kind_shape_dtype_ledger"]
            and all(matrix_passes[key]["sdpa_audit"][
                "exact_phase_kind_shape_dtype_ledger"]
                    for key in ("A", "B", "C"))
            and matrix_passes["D"]["sdpa_audit"]["invocation_count"] == 0
            and matrix_passes["D"]["official_manual_audit"]["passed"])
        mismatch = None
        if not group_exact or not ab_exact or not ledger_exact:
            mismatch = {
                "exact_equality_group_passed": group_exact,
                "A_B_exact_tensor_receipts_and_backend_ledger": ab_exact,
                "all_expected_sdpa_and_manual_ledgers_exact": ledger_exact,
                "pairwise": {
                    name: _exact_mismatch(reproduction, value)
                    for name, value in (("hook", hook_receipt),
                                        ("isolation_summed", summed),
                                        ("A", matrix_passes["A"]),
                                        ("B", matrix_passes["B"]))},
            }
        matrix = signed({
            "schema": GROUP_SCHEMA, "status": STATUS,
            "group": "backend_matrix", "passes": matrix_passes,
            "A_B_exact_tensor_receipt_equality": ab_exact,
            "all_exact_harness_reference_passes_equal": group_exact,
            "all_expected_sdpa_and_manual_ledgers_exact": ledger_exact,
            "exact_harness_difference": mismatch,
            "fresh_model_constructions": 4,
            "optimizer_constructions": 4, "optimizer_steps": 0,
            "gradient_clips": 0,
        }, GROUP_SELF_KEY)
        publish_json_once(_path(root, "matrix"), matrix,
                          "backend matrix")
        backend_reproduction = {
            "reproduction_exact_equal_to_hook":
                exact_pass_payload_equal(reproduction, hook_receipt),
            "reproduction_exact_equal_to_matrix_A":
                exact_pass_payload_equal(reproduction, matrix_passes["A"]),
            "hook_production_sdpa_ledger": hook_receipt["sdpa_audit"],
            "matrix_A_production_sdpa_ledger":
                matrix_passes["A"]["sdpa_audit"],
        }
        if mismatch is not None:
            artifacts = {
                "attempt": artifact_binding(_path(root, "attempt"),
                                            ATTEMPT_SELF_KEY, "attempt"),
                "fixture": artifact_binding(_path(root, "fixture"),
                                            FIXTURE_SELF_KEY, "fixture"),
                "reproduction": artifact_binding(
                    _path(root, "reproduction"), PASS_SELF_KEY,
                    "exact reproduction"),
                "hook": artifact_binding(_path(root, "hook"), PASS_SELF_KEY,
                                         "hook inventory"),
                "isolation": artifact_binding(
                    _path(root, "isolation"), GROUP_SELF_KEY,
                    "loss isolation"),
                "matrix": artifact_binding(_path(root, "matrix"),
                                           GROUP_SELF_KEY, "backend matrix"),
            }
            return _publish_terminal(
                root, contract=contract, attempt=attempt,
                terminal_kind="INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP",
                mechanism=None, artifacts=artifacts,
                first_reverse=hook_receipt["hook_inventory"][
                    "first_reverse_module_with_finite_downstream_and_nonfinite_upstream"],
                first_parameter=reproduction_nonfinite[0],
                all_parameters=reproduction_nonfinite,
                exact_difference=mismatch, technical_preflight=preflight,
                loss_component_attribution=attribution,
                production_backend_reproduction_evidence=backend_reproduction,
                completed_passes=completed)

        mechanism = classify_mechanism(matrix_passes, exact_ab=ab_exact)
        require(mechanism in CONTRACT.MECHANISM_CLASSIFICATIONS,
                "completed matrix did not yield exactly one mechanism")
        artifacts = {
            "attempt": artifact_binding(_path(root, "attempt"),
                                        ATTEMPT_SELF_KEY, "attempt"),
            "fixture": artifact_binding(_path(root, "fixture"),
                                        FIXTURE_SELF_KEY, "fixture"),
            "reproduction": artifact_binding(
                _path(root, "reproduction"), PASS_SELF_KEY,
                "exact reproduction"),
            "hook": artifact_binding(_path(root, "hook"), PASS_SELF_KEY,
                                     "hook inventory"),
            "isolation": artifact_binding(_path(root, "isolation"),
                                           GROUP_SELF_KEY, "loss isolation"),
            "matrix": artifact_binding(_path(root, "matrix"),
                                       GROUP_SELF_KEY, "backend matrix"),
        }
        terminal = _publish_terminal(
            root, contract=contract, attempt=attempt,
            terminal_kind="COMPLETED_MECHANISM_CLASSIFICATION",
            mechanism=mechanism, artifacts=artifacts,
            first_reverse=hook_receipt["hook_inventory"][
                "first_reverse_module_with_finite_downstream_and_nonfinite_upstream"],
            first_parameter=reproduction_nonfinite[0],
            all_parameters=reproduction_nonfinite, exact_difference=None,
            technical_preflight=preflight,
            loss_component_attribution=attribution,
            production_backend_reproduction_evidence=backend_reproduction,
            completed_passes=completed)
        return terminal
    except BaseException as exc:
        if not terminal_path.exists() and not terminal_path.is_symlink():
            _record_failure(root, contract=contract, attempt=attempt,
                            stage=stage, error=exc,
                            completed_passes=completed,
                            technical_preflight=preflight)
        raise


MATRIX_ADDITIONS = {
    "forward_equivalence_to_A", "exact_tensor_receipt_equality_to_A",
    "forward_maximum_absolute_difference_from_A",
    "forward_maximum_relative_difference_from_A",
    "component_losses", "offending_parameter_set",
    "actual_sdpa_backend_per_invocation",
}


def _recompute_sdpa_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    rows = value.get("invocations", [])
    backends = [row.get("selected_backend_inside_effective_context")
                for row in rows]
    return {
        "invocation_count": len(rows),
        "expected_invocation_count": EXPECTED_SDPA_CALLS,
        "backend_sequence": backends,
        "all_seven_dispatches_math": (
            len(rows) == EXPECTED_SDPA_CALLS
            and all(backend == "MATH" for backend in backends)),
        "has_non_math_dispatch": any(backend != "MATH"
                                     for backend in backends),
        "every_selector_sampled_inside_effective_inner_context": True,
        "invocations": rows,
        "exact_phase_kind_shape_dtype_ledger": sdpa_ledger_exact(value),
    }


def _recompute_manual_audit(value: Mapping[str, Any]) -> bool:
    attention = value.get("attention_modules", [])
    norms = value.get("layer_norms", [])
    expected_attention = {
        "pooler.blocks.0.attn": "src.models.utils.modules.Attention",
        "pooler.blocks.1.attn": "src.models.utils.modules.Attention",
        "pooler.blocks.2.attn": "src.models.utils.modules.Attention",
        "pooler.cross_attention_block.xattn":
            "src.models.utils.modules.CrossAttention",
    }
    expected_norms = {
        "pooler.blocks.0.norm1", "pooler.blocks.0.norm2",
        "pooler.blocks.1.norm1", "pooler.blocks.1.norm2",
        "pooler.blocks.2.norm1", "pooler.blocks.2.norm2",
        "pooler.cross_attention_block.norm1",
        "pooler.cross_attention_block.norm2",
    }
    return (
        value.get("official_source_binding_digest")
        == CONTRACT.OFFICIAL_POOLER_BINDING_DIGEST
        and value.get("outer_and_reduction_dtype") == "torch.float32"
        and value.get("official_non_sdpa_formula") == (
            "scaled q@k.T, softmax, dropout(0), weighted reduction; "
            "LayerNorm eps=1e-5")
        and value.get("sdpa_invocation_count") == 0
        and value.get("state_restored_after_context") is True
        and {row.get("path"): row.get("module_type") for row in attention}
        == expected_attention
        and {row.get("path") for row in norms} == expected_norms
        and all(row.get("use_sdpa_during_pass") is True
                and row.get("all_parameters_float32") is True
                and row.get("scale") == 32 ** -0.5
                for row in attention)
        and all(row.get("eps") == 1e-5
                and row.get("all_parameters_float32") is True
                for row in norms)
    )


PASS_SPECS = {
    "exact_reproduction": ("frozen_summed_loss", "production", False),
    "hook_inventory": ("frozen_summed_loss", "production", False),
    "loss_isolation_progress_only": ("progress_only", "production", False),
    "loss_isolation_safety_only": ("safety_only", "production", False),
    "loss_isolation_completion_only": (
        "completion_only", "production", False),
    "loss_isolation_frozen_summed_loss": (
        "frozen_summed_loss", "production", False),
    "backend_matrix_A": ("frozen_summed_loss", "production", False),
    "backend_matrix_B": ("frozen_summed_loss", "production", False),
    "backend_matrix_C": ("frozen_summed_loss", "forced_math", False),
    "backend_matrix_D": ("frozen_summed_loss", "production", True),
}
PASS_ORDER = tuple(PASS_SPECS)


def _validate_pass(value: Mapping[str, Any], label: str, *,
                   matrix_entry: bool = False,
                   expected_name: str | None = None) -> dict[str, Any]:
    result = dict(value)
    if matrix_entry:
        base = {key: item for key, item in result.items()
                if key not in MATRIX_ADDITIONS}
        validate_signed(base, PASS_SELF_KEY, label)
    else:
        validate_signed(result, PASS_SELF_KEY, label)
    pass_name = result.get("pass_name")
    require(pass_name in PASS_SPECS
            and (expected_name is None or pass_name == expected_name)
            and (result.get("objective"), result.get("backend_policy"),
                 result.get("explicit_official_non_sdpa"))
            == PASS_SPECS[pass_name], f"{label} pass identity changed")
    require(result.get("schema") == PASS_SCHEMA
            and result.get("status") == STATUS
            and result.get("fixture_digest") == CONTRACT.FROZEN_FIXTURE_DIGEST
            and result.get("registered_seed") == CONTRACT.FROZEN_ATTENTIVE_SEED
            and result.get("initial_state_digest")
            == CONTRACT.FROZEN_INITIAL_STATE_DIGEST
            and result.get("final_state_digest")
            == CONTRACT.FROZEN_INITIAL_STATE_DIGEST
            and result.get("state_unchanged") is True
            and result.get("model_mode") == "train"
            and result.get("activation_checkpointing") is True
            and result.get("autocast") is False
            and result.get("parameter_dtype") == "torch.float32"
            and result.get("latent_dtype") == "torch.float32"
            and result.get("loss_effective_batch_denominator") == 64
            and result.get("optimizer_constructions") == 1
            and result.get("optimizer_steps") == 0
            and result.get("gradient_clips") == 0,
            f"{label} frozen pass invariants changed")
    optimizer = result.get("optimizer", {})
    require(optimizer.get("class") == "torch.optim.AdamW"
            and optimizer.get("settings") == ATTENTIVE.frozen_budget()
            and optimizer.get("zero_grad_set_to_none_before_forward") is True
            and optimizer.get("state_entry_count_before") == 0
            and optimizer.get("state_entry_count_after") == 0
            and optimizer.get("state_digest_before")
            == optimizer.get("state_digest_after")
            and optimizer.get("state_unchanged") is True,
            f"{label} frozen AdamW no-step evidence changed")
    pre = result.get("pre_backward_finiteness", {})
    require(set(pre) == set(CONTRACT.PRE_BACKWARD_FINITE_FIELDS)
            and all(pre.get(key) is True for key in (
                "all_model_parameters_finite", "all_inputs_finite",
                "all_activations_finite", "all_targets_finite",
                "all_component_losses_finite", "total_loss_finite")),
            f"{label} pre-backward finiteness changed")
    rows = result.get("parameter_gradient_inventory")
    require(isinstance(rows, list)
            and len(rows) == CONTRACT.TRAINABLE_PARAMETER_TENSOR_COUNT
            and all(set(row) == set(CONTRACT.PARAMETER_INVENTORY_FIELDS)
                    for row in rows)
            and result.get("complete_gradient_verdict")
            == gradient_verdict(rows),
            f"{label} gradient inventory changed")
    shape_dtype = [[row["fully_qualified_name"], row["shape"],
                    row["parameter_dtype"]] for row in rows]
    require(CONTRACT.digest(shape_dtype)
            == CONTRACT.FROZEN_PARAMETER_INVENTORY_DIGEST
            and len({row["fully_qualified_name"] for row in rows}) == len(rows),
            f"{label} frozen parameter names/shapes/dtypes changed")
    outputs = result.get("outputs", {})
    require(set(outputs) == set(COMPONENTS), f"{label} output keys changed")
    for component in COMPONENTS:
        row = outputs[component]
        tensor = torch.tensor(row.get("values"), dtype=torch.float32)
        require(tuple(tensor.shape) == (4,)
                and bool(torch.isfinite(tensor).all())
                and row.get("tensor_digest") == FROZEN.tensor_digest(tensor)
                and row.get("stats") == tensor_numeric_stats(tensor),
                f"{label} {component} output receipt changed")
    loss_payload = result.get("losses", {})
    require(set(loss_payload.get("components", {})) == set(COMPONENTS),
            f"{label} component-loss keys changed")
    for component in COMPONENTS:
        row = loss_payload["components"][component]
        tensor = torch.tensor(row.get("value"), dtype=torch.float32)
        require(bool(torch.isfinite(tensor))
                and row.get("tensor_digest") == FROZEN.tensor_digest(tensor)
                and row.get("stats") == tensor_numeric_stats(tensor),
                f"{label} {component} loss receipt changed")
    selected = loss_payload.get("selected_total", {})
    selected_tensor = torch.tensor(selected.get("value"), dtype=torch.float32)
    require(bool(torch.isfinite(selected_tensor))
            and selected.get("tensor_digest")
            == FROZEN.tensor_digest(selected_tensor)
            and selected.get("stats") == tensor_numeric_stats(selected_tensor),
            f"{label} selected loss receipt changed")
    instrumented = pass_name in (
        "hook_inventory", "backend_matrix_A", "backend_matrix_B",
        "backend_matrix_C", "backend_matrix_D")
    sdpa = result.get("sdpa_audit", {})
    if instrumented:
        recomputed_sdpa = _recompute_sdpa_summary(sdpa)
        require(sdpa == recomputed_sdpa,
                f"{label} SDPA summary is not derived from invocation rows")
    else:
        require(sdpa == {"invocation_count": None, "not_instrumented": True},
                f"{label} unexpectedly claims an SDPA ledger")
    if pass_name == "backend_matrix_D":
        manual = result.get("official_manual_audit")
        require(isinstance(manual, Mapping)
                and manual.get("passed") == _recompute_manual_audit(manual),
                f"{label} official manual audit changed")
    else:
        require(result.get("official_manual_audit") is None,
                f"{label} unexpectedly claims a manual audit")
    if pass_name == "hook_inventory":
        require(result.get("pre_backward_activation_coverage")
                == "all_frozen_internal_hook_boundaries"
                and result.get(
                    "internal_activation_finiteness_proven_by_this_pass") is True,
                f"{label} internal activation coverage changed")
    else:
        require(result.get("pre_backward_activation_coverage")
                == "component_outputs_only"
                and result.get(
                    "internal_activation_finiteness_proven_by_this_pass") is False,
                f"{label} activation coverage is overstated")
    return result


def _binding_matches(root: Path, binding: Mapping[str, Any], path: Path,
                     key: str, label: str) -> bool:
    expected = artifact_binding(path, key, label)
    return dict(binding) == expected


def _validate_fixture_receipt(root: Path, contract: Mapping[str, Any],
                              attempt: Mapping[str, Any]) -> dict[str, Any]:
    value = read_signed(_path(root, "fixture"), FIXTURE_SELF_KEY,
                        "fit-only fixture binding")
    require(AMENDED.AMENDMENT.digest(AMENDED.SMOKE_FIT_FIXTURE)
            == AMENDED.SMOKE_FIT_FIXTURE_DIGEST,
            "frozen fit-only fixture source changed")
    binding = {
        "fixture_digest": AMENDED.SMOKE_FIT_FIXTURE_DIGEST,
        "row_count": 4,
        "row_record_files_opened": 4,
        "fit_latent_shards_opened": 4,
        "calibration_rows_materialized": 0,
        "calibration_label_rows_opened": 0,
        "calibration_latent_shards_opened": 0,
        "global_training_view_digest":
            AMENDED.CONTRACT.FROZEN_TRAINING_VIEW_DIGEST,
        "global_latent_index_digest":
            AMENDED.CONTRACT.FROZEN_LATENT_INDEX_DIGEST,
        "registered_data_order_contract_digest": AMENDED.AMENDMENT.digest(
            AMENDED.CONTRACT.DATA_ORDER_CONTRACT),
        "files": [{
            "training_view_row_digest": row["training_view_row_digest"],
            "branch_identity_digest": row["branch_identity_digest"],
            "row_record_path": row["input"]["path"],
            "row_record_sha256": row["input"]["sha256"],
            "row_record_self_digest": row["input"]["self_digest"],
            "latent_path": row["latent"]["path"],
            "latent_sha256": row["latent"]["sha256"],
        } for row in AMENDED.SMOKE_FIT_FIXTURE],
    }
    require(value.get("schema") == FIXTURE_SCHEMA
            and value.get("status") == STATUS
            and value.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and value.get("attempt_digest") == attempt[ATTEMPT_SELF_KEY]
            and value.get("fixture_digest") == CONTRACT.FROZEN_FIXTURE_DIGEST
            and value.get("fit_only_fixture_binding") == binding
            and value.get("row_count") == 4
            and value.get("calibration_label_rows_opened") == 0
            and value.get("calibration_latent_shards_opened") == 0,
            "fit-only fixture receipt changed")
    return value


def _validate_runtime_inventory(root: Path, expected_names: set[str]) -> None:
    runtime = CONTRACT.runtime_root(root)
    require(runtime.is_dir() and not runtime.is_symlink(),
            "diagnostic runtime root changed")
    children = list(runtime.iterdir())
    require({path.name for path in children} == expected_names
            and all(path.is_file() and not path.is_symlink()
                    and path.stat().st_mode & 0o222 == 0 for path in children),
            "diagnostic runtime namespace inventory or immutability changed")


def _validate_terminal_inventory(root: Path,
                                 bindings: Mapping[str, Any]) -> None:
    expected = {"contract.json", "terminal.json"}
    expected.update(Path(str(binding["path"])).name
                    for binding in bindings.values())
    _validate_runtime_inventory(root, expected)


def _validate_technical_preflight(value: Mapping[str, Any]) -> None:
    expected = CONTRACT.EXECUTION_ENVIRONMENT
    expected_python = str((ROOT / expected["python"]).absolute())
    expected_prefix = str(Path(expected_python).parents[1])
    torch_path = Path(str(value.get("torch_path", "")))
    torch_distribution = (
        ROOT / expected["torch_distribution_environment"]).resolve()
    entry = dict(expected["production_ambient"])
    entry["deterministic_algorithms_after_fresh_model_state"] = expected[
        "deterministic_algorithms_at_process_entry"]
    require(value.get("python") == expected_python
            and value.get("python_prefix") == expected_prefix
            and Path(str(value.get("python_base_prefix", ""))).resolve()
            == Path("/usr").resolve()
            and torch_path.is_file() and not torch_path.is_symlink()
            and torch_path.resolve().is_relative_to(torch_distribution)
            and value.get("torch_version") == expected["torch_version"]
            and value.get("torch_hip_version") == expected["torch_hip_version"]
            and value.get("device") == expected["device"]
            and value.get("device_name") == expected["device_name"]
            and value.get("device_capability") == expected["device_capability"]
            and str(value.get("device_architecture", "")).startswith(
                expected["device_architecture"])
            and value.get("visible_hip_device_count")
            == expected["visible_hip_device_count"]
            and value.get("entry_ambient") == entry,
            "terminal technical preflight changed")


def validate_failure(root: Path = ROOT) -> dict[str, Any]:
    contract = load_installed_contract(root)
    require(not _path(root, "terminal").exists()
            and not _path(root, "terminal").is_symlink(),
            "technical failure coexists with a terminal")
    attempt = read_signed(_path(root, "attempt"), ATTEMPT_SELF_KEY,
                          "gradient-localisation attempt")
    failure = read_signed(_path(root, "failure"), FAILURE_SELF_KEY,
                          "gradient-localisation technical failure")
    traceback_value = failure.get("traceback")
    require(attempt.get("schema") == ATTEMPT_SCHEMA
            and attempt.get("attempt_number") == 1
            and attempt.get("maximum_attempts") == 1
            and attempt.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and attempt.get("source_closure_digest")
            == contract["source_closure"][CONTRACT.SOURCE_CLOSURE_SELF_KEY]
            and attempt.get("failed_smoke_digest")
            == CONTRACT.SMOKE_FAILURE_DIGEST
            and attempt.get("official_pooler_binding_digest")
            == CONTRACT.OFFICIAL_POOLER_BINDING_DIGEST
            and attempt.get("fixture_digest") == CONTRACT.FROZEN_FIXTURE_DIGEST
            and attempt.get("initial_state_digest")
            == CONTRACT.FROZEN_INITIAL_STATE_DIGEST
            and attempt.get("registered_seed") == CONTRACT.FROZEN_ATTENTIVE_SEED
            and attempt.get("planned_execution_counts")
            == CONTRACT.EXECUTION_COUNTS
            and attempt.get("optimizer_constructions_planned") == 10
            and attempt.get("optimizer_steps") == 0
            and attempt.get("gradient_clips") == 0
            and attempt.get("repair_authorised") is False
            and attempt.get("training_authorised") is False,
            "technical-failure attempt binding changed")
    require(failure.get("schema") == FAILURE_SCHEMA
            and failure.get("status") in (
                "EXACT_SHAPE_BACKEND_COMPATIBILITY_TECHNICAL_STOP",
                "INVALID_GRADIENT_LOCALISATION_TECHNICAL_STOP")
            and failure.get("complete") is True
            and failure.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and failure.get("attempt_digest") == attempt[ATTEMPT_SELF_KEY]
            and isinstance(traceback_value, str)
            and failure.get("traceback_sha256") == hashlib.sha256(
                traceback_value.encode("utf-8")).hexdigest()
            and failure.get("mechanism_classification") is None
            and failure.get("contracted_four_row_shape_not_reduced") is True
            and failure.get("four_row_pass_attempted")
            == (failure.get("execution_counts", {}).get("forward_attempts", 0)
                > 0)
            and failure.get("batch_size_reduction_attempted") is False
            and failure.get("repair_authorised") is False
            and failure.get("training_authorised") is False
            and failure.get("automatic_retry_authorised") is False
            and failure.get("predictor_checkpoints_opened") == 0
            and failure.get("calibration_rows_opened") == 0,
            "gradient-localisation technical failure changed")
    completed = failure.get("completed_passes")
    receipts = failure.get("completed_pass_receipts")
    require(isinstance(completed, list)
            and completed == list(PASS_ORDER[:len(completed)])
            and isinstance(receipts, Mapping)
            and set(receipts) == set(completed),
            "technical-failure completed-pass prefix changed")
    stage = failure.get("stage")
    allowed_lengths = {
        "device_preflight": {0}, "fit_fixture_preflight": {0},
        "exact_reproduction": {0}, "hook_inventory": {1},
        "loss_isolation": {2, 3, 4, 5, 6},
        "backend_matrix_A": {6}, "backend_matrix_B": {7},
        "backend_matrix_C": {8}, "backend_matrix_D": {9, 10},
    }
    require(stage in allowed_lengths and len(completed) in allowed_lengths[stage],
            "technical-failure stage/completed-pass prefix changed")
    for name in completed:
        _validate_pass(receipts[name], f"failure-embedded {name}",
                       expected_name=name)
    preflight = failure.get("technical_preflight")
    if preflight is not None:
        require(isinstance(preflight, Mapping),
                "technical-failure preflight type changed")
        _validate_technical_preflight(preflight)
    else:
        require(failure.get("stage") == "device_preflight",
                "technical failure omitted a completed device preflight")
    counts = failure.get("execution_counts")
    require(isinstance(counts, Mapping)
            and counts.get("optimizer_steps") == 0
            and counts.get("gradient_clips") == 0
            and len(completed) <= counts.get("fresh_model_constructions", -1)
            <= len(completed) + 1
            and len(completed) <= counts.get("optimizer_constructions", -1)
            <= len(completed) + 1
            and len(completed) <= counts.get("forward_attempts", -1)
            <= len(completed) + 1
            and len(completed) <= counts.get("completed_forwards", -1)
            <= counts.get("forward_attempts")
            and len(completed) <= counts.get("backward_attempts", -1)
            <= len(completed) + 1
            and len(completed) <= counts.get("completed_backwards", -1)
            <= counts.get("backward_attempts"),
            "technical-failure execution counters changed")
    if failure["status"] == "EXACT_SHAPE_BACKEND_COMPATIBILITY_TECHNICAL_STOP":
        require(failure.get("stage") in ("backend_matrix_C", "backend_matrix_D")
                and failure.get("exception_type") == "OutOfMemoryError",
                "compatibility-failure stage/type changed")
    for artifact_name, pass_name in (
            ("reproduction", "exact_reproduction"),
            ("hook", "hook_inventory")):
        path = _path(root, artifact_name)
        if path.exists() or path.is_symlink():
            on_disk = read_signed(path, PASS_SELF_KEY, artifact_name)
            require(receipts.get(pass_name) == on_disk,
                    f"technical-failure {artifact_name} receipt diverged")
    require((len(completed) < 1) == (not _path(root, "reproduction").exists())
            and (len(completed) < 2) == (not _path(root, "hook").exists()),
            "technical-failure required pass artifacts changed")
    if stage not in ("device_preflight", "fit_fixture_preflight"):
        require(_path(root, "fixture").is_file()
                and not _path(root, "fixture").is_symlink(),
                "technical-failure required fixture receipt is absent")
    isolation_path = _path(root, "isolation")
    if isolation_path.exists() or isolation_path.is_symlink():
        group = read_signed(isolation_path, GROUP_SELF_KEY, "loss isolation")
        require(group.get("schema") == GROUP_SCHEMA
                and group.get("status") == STATUS
                and group.get("group") == "loss_isolation"
                and group.get("fresh_model_constructions") == 4
                and group.get("optimizer_constructions") == 4
                and group.get("optimizer_steps") == 0
                and group.get("gradient_clips") == 0
                and group.get("passes") == [
            receipts[f"loss_isolation_{objective}"]
            for objective in CONTRACT.LOSS_ISOLATION_PASSES],
            "technical-failure isolation group diverged")
    matrix_path = _path(root, "matrix")
    if matrix_path.exists() or matrix_path.is_symlink():
        group = read_signed(matrix_path, GROUP_SELF_KEY, "backend matrix")
        require(group.get("schema") == GROUP_SCHEMA
                and group.get("status") == STATUS
                and group.get("group") == "backend_matrix"
                and group.get("fresh_model_constructions") == 4
                and group.get("optimizer_constructions") == 4
                and group.get("optimizer_steps") == 0
                and group.get("gradient_clips") == 0
                and set(group.get("passes", {})) == {"A", "B", "C", "D"},
                "technical-failure matrix group schema changed")
        for key in ("A", "B", "C", "D"):
            entry = _validate_pass(
                group["passes"][key], f"failure matrix {key}",
                matrix_entry=True, expected_name=f"backend_matrix_{key}")
            base = {field: item for field, item in group["passes"][key].items()
                    if field not in MATRIX_ADDITIONS}
            require(receipts[f"backend_matrix_{key}"] == base,
                    f"technical-failure matrix {key} receipt diverged")
            reproduction = receipts["exact_reproduction"]
            numerical = forward_equivalence(
                _combined_numeric_vector(reproduction),
                _combined_numeric_vector(entry))
            require(entry["forward_equivalence_to_A"] == numerical
                    and entry["forward_maximum_absolute_difference_from_A"]
                    == numerical["maximum_absolute_difference"]
                    and entry["forward_maximum_relative_difference_from_A"]
                    == numerical["maximum_relative_difference"]
                    and entry["exact_tensor_receipt_equality_to_A"]
                    == (exact_pass_payload_equal(reproduction, entry)
                        if key in ("A", "B") else None)
                    and entry["component_losses"] == {
                        name: row["value"] for name, row in
                        entry["losses"]["components"].items()}
                    and entry["offending_parameter_set"]
                    == entry["complete_gradient_verdict"][
                        "offending_parameter_set"]
                    and entry["actual_sdpa_backend_per_invocation"]
                    == entry["sdpa_audit"].get("backend_sequence", []),
                    f"technical-failure matrix {key} derivation changed")
        matrix_passes = group["passes"]
        ab_exact = (exact_pass_payload_equal(matrix_passes["A"],
                                             matrix_passes["B"])
                    and matrix_passes["A"]["sdpa_audit"]["backend_sequence"]
                    == matrix_passes["B"]["sdpa_audit"]["backend_sequence"])
        group_exact = all(exact_pass_payload_equal(
            receipts["exact_reproduction"], value) for value in (
                receipts["hook_inventory"],
                receipts["loss_isolation_frozen_summed_loss"],
                matrix_passes["A"], matrix_passes["B"]))
        ledger_exact = (
            matrix_passes["A"]["sdpa_audit"][
                "exact_phase_kind_shape_dtype_ledger"]
            and matrix_passes["B"]["sdpa_audit"][
                "exact_phase_kind_shape_dtype_ledger"]
            and matrix_passes["C"]["sdpa_audit"][
                "exact_phase_kind_shape_dtype_ledger"]
            and matrix_passes["D"]["sdpa_audit"]["invocation_count"] == 0
            and _recompute_manual_audit(
                matrix_passes["D"]["official_manual_audit"]))
        require(group["A_B_exact_tensor_receipt_equality"] == ab_exact
                and group["all_exact_harness_reference_passes_equal"]
                == group_exact
                and group["all_expected_sdpa_and_manual_ledgers_exact"]
                == ledger_exact,
                "technical-failure matrix group derivation changed")
    expected = {"contract.json", "attempt.json", "technical_failure.json"}
    for name in ("fixture", "reproduction", "hook", "isolation", "matrix"):
        path = _path(root, name)
        if path.exists() or path.is_symlink():
            expected.add(path.name)
    if _path(root, "fixture").exists() or _path(root, "fixture").is_symlink():
        _validate_fixture_receipt(root, contract, attempt)
    _validate_runtime_inventory(root, expected)
    return failure


def validate_outcome(root: Path = ROOT) -> dict[str, Any]:
    terminal = _path(root, "terminal")
    failure = _path(root, "failure")
    terminal_exists = terminal.exists() or terminal.is_symlink()
    failure_exists = failure.exists() or failure.is_symlink()
    require(terminal_exists != failure_exists,
            "diagnostic must have exactly one terminal or technical failure")
    return validate_terminal(root) if terminal_exists else validate_failure(root)


def validate_terminal(root: Path = ROOT) -> dict[str, Any]:
    contract = load_installed_contract(root)
    require(not _path(root, "failure").exists()
            and not _path(root, "failure").is_symlink(),
            "terminal coexists with a technical failure")
    attempt = read_signed(_path(root, "attempt"), ATTEMPT_SELF_KEY,
                          "gradient-localisation attempt")
    require(attempt.get("schema") == ATTEMPT_SCHEMA
            and attempt.get("attempt_number") == 1
            and attempt.get("maximum_attempts") == 1
            and attempt.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and attempt.get("source_closure_digest")
            == contract["source_closure"][CONTRACT.SOURCE_CLOSURE_SELF_KEY]
            and attempt.get("failed_smoke_digest")
            == CONTRACT.SMOKE_FAILURE_DIGEST
            and attempt.get("official_pooler_binding_digest")
            == CONTRACT.OFFICIAL_POOLER_BINDING_DIGEST
            and attempt.get("fixture_digest") == CONTRACT.FROZEN_FIXTURE_DIGEST
            and attempt.get("initial_state_digest")
            == CONTRACT.FROZEN_INITIAL_STATE_DIGEST
            and attempt.get("registered_seed") == CONTRACT.FROZEN_ATTENTIVE_SEED
            and attempt.get("planned_execution_counts")
            == CONTRACT.EXECUTION_COUNTS
            and attempt.get("optimizer_constructions_planned") == 10
            and attempt.get("optimizer_steps") == 0
            and attempt.get("gradient_clips") == 0
            and attempt.get("repair_authorised") is False
            and attempt.get("training_authorised") is False,
            "gradient-localisation attempt changed")
    terminal = read_signed(_path(root, "terminal"), TERMINAL_SELF_KEY,
                           "gradient-localisation terminal")
    require(terminal.get("schema") == TERMINAL_SCHEMA
            and terminal.get("status") == STATUS
            and terminal.get("complete") is True
            and terminal.get("contract_digest")
            == contract[CONTRACT.CONTRACT_SELF_KEY]
            and terminal.get("attempt_digest") == attempt[ATTEMPT_SELF_KEY]
            and isinstance(terminal.get("execution_counts"), Mapping)
            and terminal["execution_counts"].get("optimizer_steps") == 0
            and terminal["execution_counts"].get("gradient_clips") == 0
            and terminal.get("predictor_checkpoints_opened") == 0
            and terminal.get("calibration_rows_opened") == 0
            and terminal.get("scorer_training_started") is False,
            "gradient-localisation terminal bindings changed")
    kind = terminal.get("terminal_kind")
    mechanism = terminal.get("mechanism_classification")
    require(kind in (
        "NONREPRODUCTION_TECHNICAL_STOP",
        "INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP",
        "COMPLETED_MECHANISM_CLASSIFICATION")
        and ((kind == "COMPLETED_MECHANISM_CLASSIFICATION")
             == (mechanism in CONTRACT.MECHANISM_CLASSIFICATIONS)),
        "terminal mechanism cardinality changed")
    repair_support = mechanism in CONTRACT.MECHANISM_CLASSIFICATIONS[:2]
    require(terminal.get("later_repair_gate") == {
        "classification_can_support_separate_repair_decision": repair_support,
        "repair_authorised_now": False,
        "training_authorised_now": False,
        "automatic_repair_or_training": False,
    }, "terminal repair gate changed")
    completed_passes = terminal.get("completed_passes")
    counts = terminal["execution_counts"]
    preflight = terminal.get("technical_preflight")
    require(isinstance(preflight, Mapping),
            "terminal technical preflight is absent")
    _validate_technical_preflight(preflight)
    require(isinstance(completed_passes, list)
            and completed_passes == list(PASS_ORDER[:len(completed_passes)])
            and counts.get("fresh_model_constructions")
            == len(completed_passes)
            and counts.get("optimizer_constructions")
            == len(completed_passes)
            and counts.get("forward_attempts") == len(completed_passes)
            and counts.get("completed_forwards") == len(completed_passes)
            and counts.get("backward_attempts") == len(completed_passes)
            and counts.get("completed_backwards") == len(completed_passes)
            and counts.get("fixture_validation_row_record_opens") == 4
            and counts.get("fixture_validation_latent_shard_opens") == 4
            and counts.get("unique_fit_row_record_files") == 4
            and counts.get("unique_fit_latent_shard_files") == 4
            and counts.get("pass_latent_shard_loads")
            == 4 * len(completed_passes)
            and counts.get("batch_presentations") == len(completed_passes)
            and counts.get("examples_presented") == 4 * len(completed_passes),
            "terminal completed-pass/counter lineage changed")
    expected_terminal_shape = {
        "NONREPRODUCTION_TECHNICAL_STOP": (
            1, {"attempt", "fixture", "reproduction"}),
        "COMPLETED_MECHANISM_CLASSIFICATION": (
            10, {"attempt", "fixture", "reproduction", "hook",
                 "isolation", "matrix"}),
    }
    if kind == "INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP":
        harness_shapes = {
            2: {"attempt", "fixture", "reproduction", "hook"},
            6: {"attempt", "fixture", "reproduction", "hook", "isolation"},
            10: {"attempt", "fixture", "reproduction", "hook",
                 "isolation", "matrix"},
        }
        require(len(completed_passes) in harness_shapes,
                "harness terminal completed-pass cardinality changed")
        expected_binding_keys = harness_shapes[len(completed_passes)]
    else:
        expected_count, expected_binding_keys = expected_terminal_shape[kind]
        require(len(completed_passes) == expected_count,
                "terminal completed-pass cardinality changed")
    bindings = terminal.get("artifact_bindings")
    fixture_receipt = _validate_fixture_receipt(root, contract, attempt)
    require(isinstance(bindings, Mapping)
            and set(bindings) == expected_binding_keys
            and _binding_matches(root, bindings["attempt"],
                                 _path(root, "attempt"), ATTEMPT_SELF_KEY,
                                 "attempt")
            and _binding_matches(root, bindings["reproduction"],
                                 _path(root, "reproduction"), PASS_SELF_KEY,
                                 "exact reproduction")
            and _binding_matches(root, bindings["fixture"],
                                 _path(root, "fixture"), FIXTURE_SELF_KEY,
                                 "fixture"),
            "terminal base artifact binding changed")
    reproduction = _validate_pass(read_signed(
        _path(root, "reproduction"), PASS_SELF_KEY, "exact reproduction"),
        "exact reproduction", expected_name="exact_reproduction")
    nonfinite = reproduction["complete_gradient_verdict"][
        "nonfinite_parameter_set"]
    if kind == "NONREPRODUCTION_TECHNICAL_STOP":
        require(not nonfinite and mechanism is None
                and terminal.get("first_nonfinite_parameter_gradient") is None
                and terminal.get("all_nonfinite_parameter_gradients") == []
                and terminal.get(
                    "first_reverse_module_with_finite_downstream_and_nonfinite_upstream")
                is None
                and terminal.get("loss_component_attribution") is None
                and terminal.get(
                    "production_backend_reproduction_evidence") is None
                and terminal.get("exact_harness_or_reproduction_difference")
                == _nonreproduction_difference(reproduction),
                "nonreproduction terminal changed")
        _validate_terminal_inventory(root, bindings)
        return terminal

    hook = _validate_pass(read_signed(
        _path(root, "hook"), PASS_SELF_KEY, "hook inventory"),
        "hook inventory", expected_name="hook_inventory")
    require(_binding_matches(root, bindings["hook"], _path(root, "hook"),
                             PASS_SELF_KEY, "hook inventory"),
            "terminal hook binding changed")
    hook_inventory = hook.get("hook_inventory")
    require(isinstance(hook_inventory, Mapping)
            and hook_inventory.get("temporary_hooks_removed") is True
            and hook_inventory.get(
                "all_forward_module_inputs_and_outputs_finite") is True,
            "hook inventory is incomplete")
    require(terminal.get(
                "first_reverse_module_with_finite_downstream_and_nonfinite_upstream")
            == hook_inventory.get(
                "first_reverse_module_with_finite_downstream_and_nonfinite_upstream")
            and terminal.get("first_nonfinite_parameter_gradient")
            == (nonfinite[0] if nonfinite else None)
            and terminal.get("all_nonfinite_parameter_gradients") == nonfinite,
            "terminal hook/parameter offender attribution changed")
    hook_ledger_exact = hook["sdpa_audit"][
        "exact_phase_kind_shape_dtype_ledger"]

    if kind == "INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP" \
            and "isolation" not in bindings:
        recomputed_hook_difference = _hook_harness_difference(
            reproduction, hook)
        require(recomputed_hook_difference is not None
                and mechanism is None
                and terminal.get("loss_component_attribution") is None
                and terminal.get(
                    "production_backend_reproduction_evidence") is None
                and terminal.get("exact_harness_or_reproduction_difference")
                == recomputed_hook_difference,
                "early harness stop does not reproduce its mismatch")
        _validate_terminal_inventory(root, bindings)
        return terminal

    isolation = read_signed(_path(root, "isolation"), GROUP_SELF_KEY,
                            "loss isolation")
    require(isolation.get("schema") == GROUP_SCHEMA
            and isolation.get("group") == "loss_isolation"
            and len(isolation.get("passes", [])) == 4
            and isolation.get("fresh_model_constructions") == 4
            and isolation.get("optimizer_constructions") == 4
            and isolation.get("optimizer_steps") == 0
            and isolation.get("gradient_clips") == 0
            and _binding_matches(root, bindings["isolation"],
                                 _path(root, "isolation"), GROUP_SELF_KEY,
                                 "loss isolation"),
            "loss-isolation group changed")
    isolation_passes = [
        _validate_pass(value, f"loss isolation {objective}",
                       expected_name=f"loss_isolation_{objective}")
        for objective, value in zip(CONTRACT.LOSS_ISOLATION_PASSES,
                                    isolation["passes"], strict=True)]
    require([value["objective"] for value in isolation_passes]
            == list(CONTRACT.LOSS_ISOLATION_PASSES),
            "loss-isolation objective order changed")
    require(all(value["fixture_tensor_bindings"]
                == reproduction["fixture_tensor_bindings"]
                for value in [hook, *isolation_passes]),
            "hook/isolation fixture tensors changed across passes")
    recomputed_attribution = loss_component_attribution(
        isolation_passes, nonfinite)
    require(terminal.get("loss_component_attribution")
            == recomputed_attribution,
            "terminal loss-component attribution changed")
    summed = isolation_passes[-1]
    if kind == "INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP" \
            and "matrix" not in bindings:
        recomputed_isolation_difference = _exact_mismatch(
            reproduction, summed)
        require(recomputed_isolation_difference is not None
                and mechanism is None
                and terminal.get(
                    "production_backend_reproduction_evidence") is None
                and terminal.get("exact_harness_or_reproduction_difference")
                == recomputed_isolation_difference,
                "isolation harness stop does not reproduce its mismatch")
        _validate_terminal_inventory(root, bindings)
        return terminal

    matrix = read_signed(_path(root, "matrix"), GROUP_SELF_KEY,
                         "backend matrix")
    require(matrix.get("schema") == GROUP_SCHEMA
            and matrix.get("group") == "backend_matrix"
            and set(matrix.get("passes", {})) == {"A", "B", "C", "D"}
            and matrix.get("fresh_model_constructions") == 4
            and matrix.get("optimizer_constructions") == 4
            and matrix.get("optimizer_steps") == 0
            and matrix.get("gradient_clips") == 0
            and _binding_matches(root, bindings["matrix"],
                                 _path(root, "matrix"), GROUP_SELF_KEY,
                                 "backend matrix"),
            "backend matrix changed")
    passes = {key: _validate_pass(
                  value, f"backend matrix {key}", matrix_entry=True,
                  expected_name=f"backend_matrix_{key}")
              for key, value in matrix["passes"].items()}
    require(all(value["fixture_tensor_bindings"]
                == reproduction["fixture_tensor_bindings"]
                for value in passes.values()),
            "backend-matrix fixture tensors changed across passes")
    for key, value in passes.items():
        numerical = forward_equivalence(
            _combined_numeric_vector(reproduction),
            _combined_numeric_vector(value), exact=False)
        require(value.get("forward_equivalence_to_A") == numerical
                and value.get("forward_maximum_absolute_difference_from_A")
                == numerical["maximum_absolute_difference"]
                and value.get("forward_maximum_relative_difference_from_A")
                == numerical["maximum_relative_difference"]
                and value.get("exact_tensor_receipt_equality_to_A")
                == (exact_pass_payload_equal(reproduction, value)
                    if key in ("A", "B") else None)
                and value.get("component_losses") == {
                    name: row["value"] for name, row in
                    value["losses"]["components"].items()}
                and value.get("offending_parameter_set")
                == value["complete_gradient_verdict"][
                    "offending_parameter_set"]
                and value.get("actual_sdpa_backend_per_invocation")
                == value["sdpa_audit"].get("backend_sequence", []),
                f"backend matrix {key} derived comparison changed")
    ab_exact = (exact_pass_payload_equal(passes["A"], passes["B"])
                and passes["A"]["sdpa_audit"]["backend_sequence"]
                == passes["B"]["sdpa_audit"]["backend_sequence"])
    group_exact = all(exact_pass_payload_equal(reproduction, value)
                      for value in (hook, summed, passes["A"], passes["B"]))
    ledger_exact = (
        hook_ledger_exact
        and all(passes[key]["sdpa_audit"][
            "exact_phase_kind_shape_dtype_ledger"]
                for key in ("A", "B", "C"))
        and passes["D"]["sdpa_audit"]["invocation_count"] == 0
        and _recompute_manual_audit(passes["D"]["official_manual_audit"]))
    recomputed_backend_reproduction = {
        "reproduction_exact_equal_to_hook":
            exact_pass_payload_equal(reproduction, hook),
        "reproduction_exact_equal_to_matrix_A":
            exact_pass_payload_equal(reproduction, passes["A"]),
        "hook_production_sdpa_ledger": hook["sdpa_audit"],
        "matrix_A_production_sdpa_ledger": passes["A"]["sdpa_audit"],
    }
    require(terminal.get("production_backend_reproduction_evidence")
            == recomputed_backend_reproduction,
            "terminal production-backend reproduction evidence changed")
    require(matrix.get("A_B_exact_tensor_receipt_equality") == ab_exact
            and matrix.get("all_exact_harness_reference_passes_equal")
            == group_exact
            and matrix.get("all_expected_sdpa_and_manual_ledgers_exact")
            == ledger_exact,
            "backend-matrix exact receipt replay changed")
    if kind == "INVALID_DIAGNOSTIC_HARNESS_TECHNICAL_STOP":
        require((not ab_exact or not group_exact or not ledger_exact)
                and mechanism is None
                and matrix.get("exact_harness_difference")
                and terminal.get("exact_harness_or_reproduction_difference")
                == matrix.get("exact_harness_difference"),
                "matrix harness stop does not reproduce its mismatch")
        _validate_terminal_inventory(root, bindings)
        return terminal
    require(nonfinite and len(completed_passes) == 10
            and counts == CONTRACT.EXECUTION_COUNTS
            and passes["A"]["sdpa_audit"]["invocation_count"]
            == EXPECTED_SDPA_CALLS
            and passes["B"]["sdpa_audit"]["invocation_count"]
            == EXPECTED_SDPA_CALLS
            and passes["C"]["sdpa_audit"]["invocation_count"]
            == EXPECTED_SDPA_CALLS
            and passes["D"]["sdpa_audit"]["invocation_count"] == 0,
            "completed matrix SDPA ledger changed")
    recomputed = classify_mechanism(passes, exact_ab=ab_exact)
    require(recomputed == mechanism
            and terminal.get("exact_harness_or_reproduction_difference") is None
            and terminal.get("first_nonfinite_parameter_gradient")
            == nonfinite[0]
            and terminal.get("all_nonfinite_parameter_gradients") == nonfinite
            and terminal.get(
                "first_reverse_module_with_finite_downstream_and_nonfinite_upstream")
            == hook_inventory.get(
                "first_reverse_module_with_finite_downstream_and_nonfinite_upstream"),
            "terminal deterministic classification changed")
    _validate_terminal_inventory(root, bindings)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("issue-contract", "run", "validate"))
    arguments = parser.parse_args()
    if arguments.stage == "issue-contract":
        value = CONTRACT.issue_contract(ROOT)
    elif arguments.stage == "run":
        value = run_once(ROOT)
    else:
        value = validate_outcome(ROOT)
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
