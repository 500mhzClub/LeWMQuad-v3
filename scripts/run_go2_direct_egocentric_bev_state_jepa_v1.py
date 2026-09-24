#!/usr/bin/env python3
"""Run the one-shot Direct Egocentric BEV-State JEPA V1 probe.

Importing this module is source-only.  Torch, PIL, NumPy, generated inputs,
RGB, raster labels, and the N320 checkpoint are not imported or opened until
the frozen authority has passed, a new output root has been reserved, and the
post-reservation hardware preflight has passed.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V1_PREFLIGHT_JSON"
)
MATCHED_RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_shared_jepa_v5_matched_training_v1.py"
)
SCHEDULE_ADAPTER_RELATIVE_PATH = (
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v2_schedule.py"
)


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_direct_egocentric_bev_state_jepa_v1_runner_contract",
    ROOT / "lewm/benchmarks/go2_direct_egocentric_bev_state_jepa_v1.py",
)

# This reviewed base is imported for custody-neutral file, inventory, raw-input,
# N320, state-hash, and RNG helpers only.  Its scientific contract is retained
# separately for the frozen target-mapping builders; Direct V1 receipts and
# authority are implemented below against ``contract``.
_BASE = _source_only_module(
    "_lewm_direct_egocentric_bev_state_jepa_v1_frozen_base_runner",
    ROOT / "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py",
)
_LEGACY_CONTRACT = _BASE.contract

_read_regular = _BASE._read_regular
_write_exclusive = _BASE._write_exclusive
_register_output_content_sha256 = _BASE._register_output_content_sha256
_register_output_semantic_metadata = _BASE._register_output_semantic_metadata
_reset_output_binding_registry = _BASE._reset_output_binding_registry
_terminal_inventory = _BASE._terminal_inventory
_seal_terminal_with_repair = _BASE._seal_terminal_with_repair
_construct_raw_inputs_with_progress = _BASE._construct_raw_inputs_with_progress
_load_n320_with_progress = _BASE._load_n320_with_progress
_normalize_endpoint_paths = _BASE._normalize_endpoint_paths
_run_with_rng_preserved = _BASE._run_with_rng_preserved
_check_gpu_time = _BASE._check_gpu_time
_state_sha = _BASE._state_sha


class ScientificGateFailure(RuntimeError):
    """Internal marker used only after publishing an exact gate failure."""


def _publish_json(
    path: Path,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive(path, raw)
    _register_output_content_sha256(path, str(value["content_sha256"]))
    return value, raw


def _binding(
    relative: str,
    value: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    return contract.artifact_binding(
        relative,
        raw,
        content_sha256=str(value["content_sha256"]),
    )


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu() if hasattr(value, "detach") else value)
    if not math.isfinite(result):
        raise FloatingPointError("Direct BEV scalar became nonfinite")
    return result


def _zero_forbidden_semantic_counters() -> dict[str, int]:
    """Counters narrower than the public access schema, all permanently zero."""

    return {
        "camera_origin_array_open_count": 0,
        "camera_basis_array_open_count": 0,
        "ray_direction_array_open_count": 0,
        "ray_label_array_open_count": 0,
        "depth_array_open_count": 0,
        "ground_array_open_count": 0,
        "general_raw_frame_loader_call_count": 0,
        "other_supervision_array_open_count": 0,
    }


class DirectBevNarrowLoader:
    """Load only bound RGB and ``raster_labels.u1`` rows.

    The general raw-frame API is deliberately absent from this implementation.
    Every request, row-cache outcome, underlying array-cache outcome, and
    physical read attempt is ledgered as it happens so an exception still has
    a complete operational receipt.
    """

    _IMAGE_KINDS = ("current", "next", "fixed_negative", "endpoint")
    _SCOPES = ("training", "observation", "endpoint_observation")

    def __init__(
        self,
        runtime: Any,
        inputs: Any,
        *,
        progress: dict[str, Any] | None = None,
        maximum_image_cache: int = 10_000,
        maximum_label_cache: int = 10_000,
    ) -> None:
        self.runtime = runtime
        self.inputs = inputs
        self.progress = progress
        self.maximum_image_cache = int(maximum_image_cache)
        self.maximum_label_cache = int(maximum_label_cache)
        self.image_cache: OrderedDict[str, Any] = OrderedDict()
        self.label_cache: OrderedDict[str, Any] = OrderedDict()
        self._physical_image_path_kind: dict[str, str] = {}
        self._physical_raster_paths: set[str] = set()
        self._counters: dict[str, Any] = {
            "rgb_request_count": {name: 0 for name in self._IMAGE_KINDS},
            "rgb_cache_hit_count": {name: 0 for name in self._IMAGE_KINDS},
            "rgb_cache_miss_count": {name: 0 for name in self._IMAGE_KINDS},
            "rgb_physical_read_attempt_count": {
                name: 0 for name in self._IMAGE_KINDS
            },
            "rgb_physical_read_success_count": {
                name: 0 for name in self._IMAGE_KINDS
            },
            "raster_row_request_count": {name: 0 for name in self._SCOPES},
            "raster_row_cache_hit_count": {name: 0 for name in self._SCOPES},
            "raster_row_cache_miss_count": {name: 0 for name in self._SCOPES},
            "raster_row_array_call_count": 0,
            "raster_underlying_array_cache_hit_count": 0,
            "raster_underlying_array_cache_miss_count": 0,
            "raster_physical_array_open_attempt_count": 0,
            "raster_physical_array_open_success_count": 0,
            "raster_shard_request_count": 0,
            "allowed_supervision_array_open_attempt_count": {
                "raster_labels.u1": 0
            },
            "allowed_supervision_array_open_success_count": {
                "raster_labels.u1": 0
            },
            "forbidden_semantic_counters": (
                _zero_forbidden_semantic_counters()
            ),
        }
        self._sync_progress()

    def _sync_progress(self) -> None:
        if self.progress is not None:
            self.progress["direct_bev_loader_access"] = self.receipt()

    @staticmethod
    def _increment(mapping: dict[str, int], name: str) -> None:
        mapping[name] += 1

    def receipt(self) -> dict[str, Any]:
        result = copy.deepcopy(self._counters)
        result.update({
            "image_cache_entry_count": len(self.image_cache),
            "label_row_cache_entry_count": len(self.label_cache),
            "physical_image_path_count": len(self._physical_image_path_kind),
            "physical_raster_array_path_count": len(self._physical_raster_paths),
            "raw_inputs_frame_attribute_invocation_count": 0,
            "only_allowed_supervision_array": "raster_labels.u1",
        })
        return result

    def _endpoint(self, endpoint_identity: str, *, role: str) -> dict[str, Any]:
        endpoint = self.inputs.endpoints.get(endpoint_identity)
        if type(endpoint) is not dict or endpoint.get("dataset_role") != role:
            raise PermissionError("Direct BEV endpoint crossed its dataset role")
        return endpoint

    def image(
        self,
        endpoint_identity: str,
        *,
        role: str,
        stage: str,
        kind: str,
    ) -> Any:
        if kind not in self._IMAGE_KINDS:
            raise ValueError("Direct BEV RGB request kind changed")
        endpoint = self._endpoint(endpoint_identity, role=role)
        self._increment(self._counters["rgb_request_count"], kind)
        cached = self.image_cache.get(endpoint_identity)
        if cached is not None:
            self._increment(self._counters["rgb_cache_hit_count"], kind)
            self.image_cache.move_to_end(endpoint_identity)
            self._sync_progress()
            return cached

        self._increment(self._counters["rgb_cache_miss_count"], kind)
        self._increment(
            self._counters["rgb_physical_read_attempt_count"], kind
        )
        self._sync_progress()
        raw = self.inputs.read_rgb(
            str(endpoint["image_path_metadata_only"]),
            str(endpoint["image_sha256_commitment_only"]),
            role=role,
            arm="rgb_direct_egocentric_bev_state_jepa_v1",
            stage=stage,
        )
        self._increment(
            self._counters["rgb_physical_read_success_count"], kind
        )
        relative = str(endpoint["image_path_metadata_only"])
        self._physical_image_path_kind.setdefault(relative, kind)
        with self.runtime.Image.open(io.BytesIO(raw)) as decoded:
            image = decoded.convert("RGB").resize(
                (112, 112),
                self.runtime.Image.Resampling.BILINEAR,
            )
            array = (
                self.runtime.np.asarray(
                    image, dtype=self.runtime.np.float32
                )
                / 255.0
            )
        tensor = (
            self.runtime.torch.from_numpy(array.copy())
            .permute(2, 0, 1)
            .contiguous()
        )
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        normalized = (tensor - mean) / std
        if normalized.dtype != self.runtime.torch.float32:
            raise TypeError("Direct BEV normalized RGB is not float32")
        self.image_cache[endpoint_identity] = normalized
        self.image_cache.move_to_end(endpoint_identity)
        while len(self.image_cache) > self.maximum_image_cache:
            self.image_cache.popitem(last=False)
        self._sync_progress()
        return normalized

    def raster_label(
        self,
        endpoint_identity: str,
        *,
        role: str,
        stage: str,
        scope: str,
        filename: str = "raster_labels.u1",
    ) -> Any:
        if filename != "raster_labels.u1":
            self._counters["forbidden_semantic_counters"][
                "other_supervision_array_open_count"
            ] += 1
            self._sync_progress()
            raise PermissionError("only raster_labels.u1 is authorized")
        if scope not in self._SCOPES:
            raise ValueError("Direct BEV raster request scope changed")
        endpoint = self._endpoint(endpoint_identity, role=role)
        self._increment(self._counters["raster_row_request_count"], scope)
        cached = self.label_cache.get(endpoint_identity)
        if cached is not None:
            self._increment(
                self._counters["raster_row_cache_hit_count"], scope
            )
            self.label_cache.move_to_end(endpoint_identity)
            self._sync_progress()
            return cached

        self._increment(self._counters["raster_row_cache_miss_count"], scope)
        self._counters["raster_shard_request_count"] += 1
        self._sync_progress()
        shard = self.inputs._shard(
            endpoint,
            arm="rgb_direct_egocentric_bev_state_jepa_v1",
            stage=stage,
        )
        relative = (
            Path(str(endpoint["scene_shard"])).parent / filename
        ).as_posix()
        underlying_hit = self.inputs.array_cache.get(relative) is not None
        if underlying_hit:
            self._counters[
                "raster_underlying_array_cache_hit_count"
            ] += 1
        else:
            self._counters[
                "raster_underlying_array_cache_miss_count"
            ] += 1
            self._counters["raster_physical_array_open_attempt_count"] += 1
            self._counters["allowed_supervision_array_open_attempt_count"][
                filename
            ] += 1
        self._counters["raster_row_array_call_count"] += 1
        self._sync_progress()
        row = self.inputs._row_array(
            endpoint,
            shard,
            "raster_labels.u1",
            arm="rgb_direct_egocentric_bev_state_jepa_v1",
            stage=stage,
        )
        if not underlying_hit:
            self._counters["raster_physical_array_open_success_count"] += 1
            self._counters["allowed_supervision_array_open_success_count"][
                filename
            ] += 1
            self._physical_raster_paths.add(
                f"{contract.RAW_ROOT_RELATIVE_PATH}/{relative}"
            )
        if (
            tuple(row.shape) != (64, 64)
            or row.dtype != self.runtime.torch.uint8
            or not bool(((row >= 0) & (row <= 2)).all())
        ):
            raise PermissionError("raster_labels.u1 row shape, dtype, or values changed")
        row = row.contiguous()
        self.label_cache[endpoint_identity] = row
        self.label_cache.move_to_end(endpoint_identity)
        while len(self.label_cache) > self.maximum_label_cache:
            self.label_cache.popitem(last=False)
        self._sync_progress()
        return row

    def batch(
        self,
        pairs: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        device: Any,
        *,
        role: str,
        stage: str,
        mapped_negative_indices: Sequence[int],
        scope: str,
    ) -> dict[str, Any]:
        if len(mapped_negative_indices) != len(pairs):
            raise PermissionError("mapped-negative inventory length changed")
        selected = [pairs[int(index)] for index in indices]
        mapped = [
            pairs[int(mapped_negative_indices[int(index)])]
            for index in indices
        ]
        if any(row.get("dataset_role") != role for row in selected):
            raise PermissionError("Direct BEV batch crossed dataset roles")
        if any(
            candidate.get("dataset_role") != role
            or candidate.get("scene_id") != primary.get("scene_id")
            or candidate.get("next_endpoint_sha256")
            == primary.get("next_endpoint_sha256")
            for primary, candidate in zip(selected, mapped, strict=True)
        ):
            raise PermissionError("fixed mapped-negative identity changed")

        torch = self.runtime.torch
        current = torch.stack([
            self.image(
                str(row["current_endpoint_sha256"]),
                role=role,
                stage=stage,
                kind="current",
            )
            for row in selected
        ]).to(device)
        next_rgb = torch.stack([
            self.image(
                str(row["next_endpoint_sha256"]),
                role=role,
                stage=stage,
                kind="next",
            )
            for row in selected
        ]).to(device)
        fixed_negative = torch.stack([
            self.image(
                str(row["next_endpoint_sha256"]),
                role=role,
                stage=stage,
                kind="fixed_negative",
            )
            for row in mapped
        ]).to(device)
        label_scope = (
            "training" if scope == "training" else "observation"
        )
        current_labels = torch.stack([
            self.raster_label(
                str(row["current_endpoint_sha256"]),
                role=role,
                stage=stage,
                scope=label_scope,
            )
            for row in selected
        ]).to(device=device, dtype=torch.long)
        next_labels = torch.stack([
            self.raster_label(
                str(row["next_endpoint_sha256"]),
                role=role,
                stage=stage,
                scope=label_scope,
            )
            for row in selected
        ]).to(device=device, dtype=torch.long)
        action_indices = torch.tensor(
            [contract.ACTION_VOCABULARY.index(str(row["primitive"]))
             for row in selected],
            dtype=torch.long,
            device=device,
        )
        actions = torch.zeros(
            (len(selected), len(contract.ACTION_VOCABULARY)),
            dtype=torch.float32,
            device=device,
        )
        actions[torch.arange(len(selected), device=device), action_indices] = 1.0
        non_hold = action_indices != contract.HOLD_ACTION_INDEX
        return {
            "current_rgb": current,
            "next_rgb": next_rgb,
            "fixed_negative_rgb": fixed_negative,
            "current_labels": current_labels,
            "next_labels": next_labels,
            "action_one_hot": actions,
            "action_indices": action_indices,
            "non_hold_mask": non_hold,
            "rows": selected,
        }

    def endpoint_batch(
        self,
        endpoint_identities: Sequence[str],
        device: Any,
        *,
        role: str,
        stage: str,
    ) -> tuple[Any, Any]:
        torch = self.runtime.torch
        images = torch.stack([
            self.image(
                identity,
                role=role,
                stage=stage,
                kind="endpoint",
            )
            for identity in endpoint_identities
        ]).to(device)
        labels = torch.stack([
            self.raster_label(
                identity,
                role=role,
                stage=stage,
                scope="endpoint_observation",
            )
            for identity in endpoint_identities
        ]).to(device=device, dtype=torch.long)
        return images, labels

    def model_facing_access_counts(self) -> dict[str, int]:
        """Exact public loader counters before the terminal integrity rehash."""

        rgb_requests = self._counters["rgb_request_count"]
        rgb_hits = self._counters["rgb_cache_hit_count"]
        rgb_misses = self._counters["rgb_cache_miss_count"]
        row_requests = sum(self._counters["raster_row_request_count"].values())
        row_cache_hits = sum(
            self._counters["raster_row_cache_hit_count"].values()
        )
        underlying_hits = self._counters[
            "raster_underlying_array_cache_hit_count"
        ]
        underlying_misses = self._counters[
            "raster_underlying_array_cache_miss_count"
        ]
        result = {
            "current_rgb_row_request_count": int(rgb_requests["current"]),
            "next_rgb_row_request_count": int(rgb_requests["next"]),
            "fixed_negative_rgb_row_request_count": int(
                rgb_requests["fixed_negative"]
            ),
            "endpoint_rgb_row_request_count": int(rgb_requests["endpoint"]),
            "rgb_cache_hit_count": int(sum(rgb_hits.values())),
            "rgb_cache_miss_count": int(sum(rgb_misses.values())),
            "rgb_physical_file_open_count": int(sum(
                self._counters["rgb_physical_read_success_count"].values()
            )),
            "raster_label_row_request_count": int(row_requests),
            "raster_label_row_cache_hit_count": int(row_cache_hits),
            "raster_label_row_cache_miss_count": int(
                sum(self._counters["raster_row_cache_miss_count"].values())
            ),
            "raster_label_underlying_array_cache_hit_count": int(
                underlying_hits
            ),
            "raster_label_underlying_array_cache_miss_count": int(
                underlying_misses
            ),
            "raster_label_physical_array_open_count": int(
                self._counters["raster_physical_array_open_success_count"]
            ),
        }
        if (
            sum(result[name] for name in (
                "current_rgb_row_request_count",
                "next_rgb_row_request_count",
                "fixed_negative_rgb_row_request_count",
                "endpoint_rgb_row_request_count",
            ))
            != result["rgb_cache_hit_count"] + result["rgb_cache_miss_count"]
            or result["rgb_cache_miss_count"]
            != result["rgb_physical_file_open_count"]
            or result["raster_label_row_request_count"]
            != result["raster_label_row_cache_hit_count"]
            + result["raster_label_row_cache_miss_count"]
            or result["raster_label_row_cache_miss_count"]
            != result["raster_label_underlying_array_cache_hit_count"]
            + result["raster_label_underlying_array_cache_miss_count"]
            or result["raster_label_underlying_array_cache_miss_count"]
            != result["raster_label_physical_array_open_count"]
        ):
            raise RuntimeError("Direct BEV loader access accounting changed")
        return result

    def terminal_rehash_classification(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Classify the one terminal reread of each consumed payload."""

        result: dict[str, Any] = {
            "rgb_file_open_count_by_first_request_kind": {
                name: 0 for name in self._IMAGE_KINDS
            },
            "raster_labels_u1_file_open_count": 0,
            "metadata_shard_open_count": 0,
        }
        for record in records:
            relative = str(record["path"])
            kind = self._physical_image_path_kind.get(relative)
            if kind is not None:
                result["rgb_file_open_count_by_first_request_kind"][kind] += 1
            elif relative in self._physical_raster_paths:
                result["raster_labels_u1_file_open_count"] += 1
            elif record.get("kind") == "raw_supervision" and relative.endswith(
                ".json"
            ):
                result["metadata_shard_open_count"] += 1
        return result


def _confusion_metrics(
    confusion: Sequence[Sequence[int]],
    *,
    nll_sum: float,
    cell_count: int,
) -> dict[str, Any]:
    """Exact target-row/predicted-column three-class raster reduction."""

    matrix = [[int(value) for value in row] for row in confusion]
    if (
        len(matrix) != 3
        or any(len(row) != 3 for row in matrix)
        or any(value < 0 for row in matrix for value in row)
        or type(cell_count) is not int
        or cell_count <= 0
        or sum(sum(row) for row in matrix) != cell_count
        or not math.isfinite(float(nll_sum))
    ):
        raise ValueError("invalid three-class raster accumulator")
    recalls: list[float | None] = []
    for index, row in enumerate(matrix):
        total = sum(row)
        recalls.append(None if total == 0 else row[index] / total)
    present = [value for value in recalls if value is not None]
    if not present:
        raise ValueError("raster accumulator has no present class")
    return {
        "confusion_target_row_predicted_column": matrix,
        "unknown_recall": recalls[0],
        "free_recall": recalls[1],
        "occupied_recall": recalls[2],
        "balanced_accuracy": sum(present) / len(present),
        "nll": float(nll_sum) / cell_count,
        "cell_count": cell_count,
        "present_class_count": len(present),
    }


def _macro_balanced_accuracy(
    actual: Sequence[int],
    predicted: Sequence[int],
    *,
    class_count: int = 9,
) -> tuple[float, list[float]]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("action metric populations differ or are empty")
    recalls: list[float] = []
    for action in range(class_count):
        mask = [index for index, value in enumerate(actual) if value == action]
        if not mask:
            raise ValueError("an action class is absent from the selection role")
        recalls.append(
            sum(predicted[index] == action for index in mask) / len(mask)
        )
    return sum(recalls) / class_count, recalls


def _selection_endpoint_population(
    inputs: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    endpoint_families: dict[str, str] = {}
    for row in selection_pairs:
        if row.get("dataset_role") != "checkpoint_selection":
            raise PermissionError("selection endpoint population crossed roles")
        for field in ("current_endpoint_sha256", "next_endpoint_sha256"):
            identity = str(row[field])
            endpoint = inputs.endpoints.get(identity)
            if (
                type(endpoint) is not dict
                or endpoint.get("dataset_role") != "checkpoint_selection"
                or endpoint.get("family") != row.get("family")
                or endpoint.get("scene_id") != row.get("scene_id")
            ):
                raise PermissionError("selection endpoint metadata changed")
            previous = endpoint_families.setdefault(
                identity, str(endpoint["family"])
            )
            if previous != endpoint["family"]:
                raise PermissionError("endpoint family identity changed")
    aggregate = sorted(endpoint_families)
    rough = [
        identity for identity in aggregate
        if endpoint_families[identity] == contract.ROUGH_RASTER_FAMILY
    ]
    if (
        len(aggregate) != contract.AGGREGATE_RASTER_ENDPOINT_COUNT
        or contract.canonical_json_sha256(aggregate)
        != contract.AGGREGATE_RASTER_ORDERED_ENDPOINT_IDENTITY_SHA256
        or len(rough) != contract.ROUGH_RASTER_ENDPOINT_COUNT
    ):
        raise PermissionError("registered endpoint observation population changed")
    return aggregate, rough


def _parameter_partition(model: Any) -> dict[str, Any]:
    """Validate the exact reviewed trainable/target parameter inventory."""

    groups: dict[str, list[tuple[str, Any]]] = {
        "encoder": [],
        "decoder_state": [],
        "predictor": [],
        "detached_target_encoder_decoder_state": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            group = "encoder"
        elif name.startswith(("bev_decoder.", "state_head.")):
            group = "decoder_state"
        elif name.startswith("predictor."):
            group = "predictor"
        elif name.startswith((
            "target_encoder.",
            "target_bev_decoder.",
            "target_state_head.",
        )):
            group = "detached_target_encoder_decoder_state"
        else:
            raise RuntimeError(f"unregistered Direct BEV parameter: {name}")
        groups[group].append((name, parameter))

    receipt: dict[str, Any] = {}
    all_ids: list[int] = []
    for group, rows in groups.items():
        names = [name for name, _ in rows]
        parameters = [parameter for _, parameter in rows]
        observed = {
            "parameter_count": sum(item.numel() for item in parameters),
            "tensor_count": len(parameters),
            "ordered_parameter_name_sha256": (
                contract.canonical_json_sha256(names)
            ),
        }
        if observed != contract.MODEL_PARAMETER_INVENTORY[group]:
            raise RuntimeError(f"Direct BEV {group} parameter inventory changed")
        target = group == "detached_target_encoder_decoder_state"
        if any(parameter.requires_grad == target for parameter in parameters):
            raise RuntimeError(f"Direct BEV {group} requires_grad policy changed")
        ids = [id(parameter) for parameter in parameters]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Direct BEV {group} repeats a parameter")
        all_ids.extend(ids)
        receipt[group] = observed
    if (
        len(all_ids) != len(set(all_ids))
        or sum(row["parameter_count"] for row in receipt.values())
        != contract.MODEL_PARAMETER_INVENTORY["total"]["parameter_count"]
        or sum(row["tensor_count"] for row in receipt.values())
        != contract.MODEL_PARAMETER_INVENTORY["total"]["tensor_count"]
    ):
        raise RuntimeError("Direct BEV complete parameter partition changed")
    receipt["total"] = dict(contract.MODEL_PARAMETER_INVENTORY["total"])
    return {"groups": groups, "receipt": receipt}


def _initialize_model(
    runtime: Any,
    model_api: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Migrate only N320 encoder weights, then construct all fresh modules."""

    torch = runtime.torch
    n320_encoder = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in fit.encoder.state_dict().items()
    }
    n320_sha256 = _state_sha(runtime, n320_encoder)
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = [item.clone() for item in torch.cuda.get_rng_state_all()]
    config = model_api.DirectEgocentricBevStateJepaV1Config()
    model = model_api.DirectEgocentricBevStateJepaV1(
        n320_encoder_state_dict=n320_encoder,
        config=config,
    )
    if (
        not torch.equal(torch.random.get_rng_state(), cpu_rng_before)
        or any(
            not torch.equal(before, after)
            for before, after in zip(
                cuda_rng_before,
                torch.cuda.get_rng_state_all(),
                strict=True,
            )
        )
    ):
        raise RuntimeError("isolated Direct BEV construction changed global RNG")
    if _state_sha(runtime, model.encoder) != n320_sha256:
        raise RuntimeError("N320 online encoder migration changed")
    if _state_sha(runtime, model.target_encoder) != n320_sha256:
        raise RuntimeError("N320 target encoder hard sync changed")
    for online, target in zip(
        model._online_modules(), model._target_modules(), strict=True
    ):
        online_state = online.state_dict()
        target_state = target.state_dict()
        if online_state.keys() != target_state.keys() or any(
            not torch.equal(online_state[name], target_state[name])
            for name in online_state
        ):
            raise RuntimeError("Direct BEV target hard sync changed")
    if int(model.ema_update_count.detach().cpu().item()) != 0:
        raise RuntimeError("Direct BEV EMA count did not start at zero")
    if (
        int(torch.count_nonzero(model.predictor.net[-1].weight).item()) != 0
        or int(torch.count_nonzero(model.predictor.net[-1].bias).item()) != 0
    ):
        raise RuntimeError("Direct BEV final residual layer is not exact zero")
    model = model.to(device)
    model.train()
    partition = _parameter_partition(model)
    receipt = {
        "schema": f"{contract.SCHEMA_PREFIX}_initialization_v1",
        "fresh_parameter_seed": contract.BASE_INITIALIZATION_SEED,
        "n320_fit_seed": contract.N320_FIT_SEED,
        "n320_encoder_state_sha256": n320_sha256,
        "n320_encoder_only_migration": True,
        "fresh_modules_in_draw_order": [
            "bev_decoder",
            "state_head",
            "predictor",
        ],
        "prior_runtime_parameter_reuse_count": 0,
        "target_hard_sync_count_before_update_zero": 1,
        "target_ema_update_count_before_update_zero": 0,
        "target_ema_momentum": contract.TARGET_EMA_MOMENTUM,
        "predictor_target_copy_count": 0,
        "predictor_final_residual_layer_exact_zero": True,
        "complete_initial_state_sha256": _state_sha(runtime, model),
        "parameter_partition": partition["receipt"],
    }
    return model, partition, receipt


def _build_optimizer(runtime: Any, partition: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    torch = runtime.torch
    groups = partition["groups"]
    encoder_parameters = [item for _, item in groups["encoder"]]
    decoder_state_parameters = [
        item for _, item in groups["decoder_state"]
    ]
    predictor_parameters = [item for _, item in groups["predictor"]]
    target_parameters = [
        item for _, item in groups["detached_target_encoder_decoder_state"]
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 1e-4},
            {
                "params": [*decoder_state_parameters, *predictor_parameters],
                "lr": 3e-4,
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for parameter in (
            *encoder_parameters,
            *decoder_state_parameters,
            *predictor_parameters,
        )
    }
    if (
        optimizer_ids != expected_ids
        or optimizer_ids.intersection(map(id, target_parameters))
        or [group["lr"] for group in optimizer.param_groups] != [1e-4, 3e-4]
    ):
        raise RuntimeError("Direct BEV optimizer parameter partition changed")
    return optimizer, {
        "name": "AdamW",
        "precision": "float32",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-4,
        "encoder_learning_rate": 1e-4,
        "decoder_state_predictor_learning_rate": 3e-4,
        "encoder_decoder_state_joint_clip_norm": 1.0,
        "predictor_separate_clip_norm": 1.0,
        "target_parameters_excluded": True,
        "optimizer_group_count": 2,
    }


def _snapshot_model(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    update: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if update not in contract.CHECKPOINT_UPDATES:
        raise ValueError("snapshot update is not preregistered")
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    state_sha256 = _state_sha(runtime, state)
    semantic = {
        "schema": f"{contract.SCHEMA_PREFIX}_checkpoint_v1",
        "update": update,
        "state_sha256": state_sha256,
        "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256[update],
        "metadata": dict(metadata),
        "development_only": True,
        "resume_authorized": False,
        "runtime_ready": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    content_sha256 = contract.canonical_json_sha256(semantic)
    buffer = io.BytesIO()
    runtime.torch.save(
        {
            **semantic,
            "content_sha256": content_sha256,
            "model_state_dict": state,
        },
        buffer,
    )
    raw = buffer.getvalue()
    relative = f"checkpoints/update_{update}.pt"
    _write_exclusive(output_root / relative, raw)
    _register_output_semantic_metadata(
        output_root / relative,
        content_sha256=content_sha256,
        state_sha256=state_sha256,
        # The reviewed custody registry exposes only its Phase-A/Phase-B
        # namespace.  This perception-only probe is the Phase-A arm; the
        # direct-BEV identity remains bound by the checkpoint schema/content.
        phase="phase_a",
        update=update,
        schedule_prefix_sha256=contract.SCHEDULE_PREFIX_SHA256[update],
    )
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "content_sha256": content_sha256,
        "state_sha256": state_sha256,
        "update": update,
        "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256[update],
        "write_only": True,
        "read_count_after_write": 0,
    }


def _objective(model: Any, batch: Mapping[str, Any]) -> Any:
    return model.training_objective(
        current_rgb=batch["current_rgb"],
        next_rgb=batch["next_rgb"],
        fixed_negative_rgb=batch["fixed_negative_rgb"],
        action_one_hot=batch["action_one_hot"],
        non_hold_mask=batch["non_hold_mask"],
        current_labels=batch["current_labels"],
        next_labels=batch["next_labels"],
    )


def _gradient_integrity_probe(
    runtime: Any,
    model: Any,
    partition: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove update-zero gradient and exact six-call O/T isolation in-place."""

    torch = runtime.torch
    parameters = list(model.parameters())
    previous_grads = [
        None if item.grad is None else item.grad.detach().clone()
        for item in parameters
    ]
    call_counts = {
        "online_state_stack": 0,
        "predictor": 0,
        "target_state_stack": 0,
    }
    modules = {
        # The encoder API calls ``forward_tokens`` directly, so a Module
        # forward hook on VisionEncoder itself would silently observe zero.
        # The sole state head is traversed exactly once per complete O/T stack
        # call and is therefore the fail-closed stack-call witness.
        "online_state_stack": model.state_head,
        "predictor": model.predictor,
        "target_state_stack": model.target_state_head,
    }
    handles = []
    for name, module in modules.items():
        def count_call(_module: Any, _args: Any, _output: Any, *, key: str = name) -> None:
            call_counts[key] += 1
        handles.append(module.register_forward_hook(count_call))

    current = batch["current_rgb"].detach().clone().requires_grad_(True)
    next_rgb = batch["next_rgb"].detach().clone().requires_grad_(True)
    fixed = batch["fixed_negative_rgb"].detach().clone().requires_grad_(True)
    probe_batch = dict(batch)
    probe_batch.update({
        "current_rgb": current,
        "next_rgb": next_rgb,
        "fixed_negative_rgb": fixed,
    })
    for parameter in parameters:
        parameter.grad = None
    try:
        result = _objective(model, probe_batch)
        total_next_gradient = torch.autograd.grad(
            result.total,
            next_rgb,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grounding_next_gradient = torch.autograd.grad(
            0.5 * result.G_next / math.log(2.0),
            next_rgb,
            retain_graph=True,
            allow_unused=True,
        )[0]
        fixed_gradient = torch.autograd.grad(
            result.total,
            fixed,
            retain_graph=True,
            allow_unused=True,
        )[0]
        result.total.backward()
        with torch.no_grad():
            wrong_rgb_observation_state = model.online_state(fixed.detach())
        target_parameters = [
            item
            for _, item in partition["groups"][
                "detached_target_encoder_decoder_state"
            ]
        ]
        group_gradient_norms: dict[str, float] = {}
        for group in ("encoder", "decoder_state", "predictor"):
            gradients = [
                parameter.grad
                for _, parameter in partition["groups"][group]
                if parameter.grad is not None
            ]
            finite = all(bool(torch.isfinite(item).all()) for item in gradients)
            total = sum(float(item.detach().abs().sum().cpu()) for item in gradients)
            if not gradients or not finite or not math.isfinite(total):
                raise RuntimeError(f"Direct BEV {group} gradient integrity changed")
            group_gradient_norms[group] = total
        intended_nonzero = all(value > 0.0 for value in group_gradient_norms.values())
        target_gradient_free = all(item.grad is None for item in target_parameters)
        isolation = (
            total_next_gradient is not None
            and grounding_next_gradient is not None
            and torch.allclose(
                total_next_gradient,
                grounding_next_gradient,
                rtol=1e-5,
                atol=1e-7,
            )
            and fixed_gradient is None
            and call_counts == {
                "online_state_stack": 3,
                "predictor": 1,
                "target_state_stack": 3,
            }
            and not result.target_next_logits.requires_grad
            and not result.target_current_logits.requires_grad
            and not result.target_mapped_negative_logits.requires_grad
            and not wrong_rgb_observation_state.requires_grad
        )
        return {
            "target_parameters_gradient_free": target_gradient_free,
            "intended_online_path_gradient_nonzero": intended_nonzero,
            "six_call_graph_isolation_exact": isolation,
            "training_objective_call_counts": call_counts,
            "online_group_absolute_gradient_sums": group_gradient_norms,
            "next_online_gradient_equals_G_next_only": (
                total_next_gradient is not None
                and grounding_next_gradient is not None
                and torch.allclose(
                    total_next_gradient,
                    grounding_next_gradient,
                    rtol=1e-5,
                    atol=1e-7,
                )
            ),
            "fixed_negative_rgb_optimizer_gradient_absent": (
                fixed_gradient is None
            ),
            "observation_only_fixed_negative_output_requires_grad": (
                wrong_rgb_observation_state.requires_grad
            ),
        }
    finally:
        for handle in handles:
            handle.remove()
        for parameter, previous in zip(parameters, previous_grads, strict=True):
            parameter.grad = previous


def _empty_scene_accumulator() -> dict[str, dict[str, float]]:
    return {
        str(binding["scene_id"]): {
            "row_count": 0.0,
            "correct_rgb_loss_sum": 0.0,
            "wrong_rgb_loss_sum": 0.0,
            "hardest_wrong_margin_sum": 0.0,
            "non_hold_row_count": 0.0,
            "eligible_target_count": 0.0,
            "target_margin_sum": 0.0,
        }
        for binding in contract.SELECTION_FAMILY_BINDINGS.values()
    }


def _evaluate_observation_impl(
    runtime: Any,
    model_api: Any,
    model: Any,
    partition: Mapping[str, Any],
    loader: DirectBevNarrowLoader,
    selection_pairs: Sequence[Mapping[str, Any]],
    selection_mapping: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    update_zero: Mapping[str, Any] | None,
    prior_gates_passed: bool,
) -> dict[str, Any]:
    torch = runtime.torch
    if len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]:
        raise PermissionError("checkpoint-selection pair population changed")
    aggregate_endpoints, rough_endpoints = _selection_endpoint_population(
        loader.inputs, selection_pairs
    )
    mapping_indices = selection_mapping["negative_indices"]
    eligible = selection_mapping["same_action_eligible"]
    if (
        len(mapping_indices) != len(selection_pairs)
        or sum(bool(value) for value in eligible) != 494
    ):
        raise PermissionError("selection target eligibility changed")

    state_before = _state_sha(runtime, model)
    was_training = bool(model.training)
    model.eval()
    row_count = 0
    g_sum = 0.0
    j_sum = 0.0
    c_sum = 0.0
    action_nll_sum = 0.0
    target_nll_sum = 0.0
    target_strict_wins = 0
    actual_actions: list[int] = []
    predicted_actions: list[int] = []
    scenes = _empty_scene_accumulator()
    all_registered_finite = True
    all_nine_equal = True
    exact_persistence = True
    state_min = math.inf
    state_max = -math.inf
    candidate_count_histogram = {"10": 0, "11": 0}
    gradient_integrity: dict[str, Any] | None = None

    try:
        for start in range(0, len(selection_pairs), contract.MICROBATCH_SIZE):
            indices = list(range(start, min(
                start + contract.MICROBATCH_SIZE, len(selection_pairs)
            )))
            batch = loader.batch(
                selection_pairs,
                indices,
                device,
                role="checkpoint_selection",
                stage=f"observation_update_{update}",
                mapped_negative_indices=mapping_indices,
                scope="observation",
            )
            if update == 0 and gradient_integrity is None:
                gradient_integrity = _gradient_integrity_probe(
                    runtime, model, partition, batch
                )
            with torch.no_grad():
                result = _objective(model, batch)
                # Sixth conceptual call: O(fixed_negative_rgb), observation
                # only.  The correct O(next_rgb) logits are reused from the
                # exact five-call objective rather than recomputed.
                wrong_state = model.online_state(batch["fixed_negative_rgb"])
                correct_rgb_losses = model_api._hard_hierarchical_loss_per_row(
                    result.next_online_state_logits, batch["next_labels"]
                )
                wrong_rgb_losses = model_api._hard_hierarchical_loss_per_row(
                    wrong_state, batch["next_labels"]
                )

                size = len(indices)
                row_count += size
                g_sum += _scalar(result.G) * size
                j_sum += _scalar(result.J) * size
                c_sum += _scalar(result.C) * size
                action_nll_sum += _scalar(result.action_nll_per_row.sum())
                actual = batch["action_indices"].detach().cpu().tolist()
                predicted = result.action_logits.argmax(dim=1).detach().cpu().tolist()
                actual_actions.extend(int(value) for value in actual)
                predicted_actions.extend(int(value) for value in predicted)

                action_mask = torch.ones_like(
                    result.action_energies, dtype=torch.bool
                )
                rows = torch.arange(size, device=device)
                action_mask[rows, batch["action_indices"]] = False
                hardest_wrong = result.action_energies.masked_fill(
                    ~action_mask, torch.inf
                ).min(dim=1).values
                hardest_margin = hardest_wrong - result.executed_energy

                binary_logits = torch.stack((
                    -result.executed_energy / result.candidate_energy_scale,
                    -result.mapped_negative_energy / result.candidate_energy_scale,
                ), dim=1)
                binary_nll = torch.nn.functional.cross_entropy(
                    binary_logits,
                    torch.zeros(size, dtype=torch.long, device=device),
                    reduction="none",
                )
                target_margin = (
                    result.mapped_negative_energy - result.executed_energy
                )
                for offset, source_index in enumerate(indices):
                    scene = str(selection_pairs[source_index]["scene_id"])
                    if scene not in scenes:
                        raise PermissionError("unregistered selection scene")
                    accumulator = scenes[scene]
                    accumulator["row_count"] += 1
                    accumulator["correct_rgb_loss_sum"] += _scalar(
                        correct_rgb_losses[offset]
                    )
                    accumulator["wrong_rgb_loss_sum"] += _scalar(
                        wrong_rgb_losses[offset]
                    )
                    accumulator["hardest_wrong_margin_sum"] += _scalar(
                        hardest_margin[offset]
                    )
                    accumulator["non_hold_row_count"] += int(
                        bool(batch["non_hold_mask"][offset])
                    )
                    if bool(eligible[source_index]):
                        accumulator["eligible_target_count"] += 1
                        accumulator["target_margin_sum"] += _scalar(
                            target_margin[offset]
                        )
                        target_nll_sum += _scalar(binary_nll[offset])
                        target_strict_wins += int(
                            bool(
                                result.executed_energy[offset]
                                < result.mapped_negative_energy[offset]
                            )
                        )
                current = result.current_state_logits
                predictions = result.all_action_prediction_logits
                expanded = current[:, None].expand_as(predictions)
                all_nine_equal = all_nine_equal and torch.equal(
                    predictions,
                    predictions[:, :1].expand_as(predictions),
                )
                exact_persistence = exact_persistence and torch.equal(
                    predictions, expanded
                )
                state_min = min(state_min, _scalar(current.min()))
                state_max = max(state_max, _scalar(current.max()))
                counts = result.candidate_counts.detach().cpu().tolist()
                for count in counts:
                    key = str(int(count))
                    if key not in candidate_count_histogram:
                        raise RuntimeError("conditional-NCE candidate count changed")
                    candidate_count_histogram[key] += 1
                finite_tensors = (
                    result.total,
                    result.G,
                    result.J,
                    result.C,
                    result.current_state_logits,
                    result.next_online_state_logits,
                    result.executed_prediction_logits,
                    result.all_action_prediction_logits,
                    result.target_next_logits,
                    result.target_current_logits,
                    result.target_mapped_negative_logits,
                    result.action_energies,
                    result.action_logits,
                    result.action_nll_per_row,
                    result.candidate_energy_scale,
                    result.conditional_nce_per_row,
                    correct_rgb_losses,
                    wrong_rgb_losses,
                )
                all_registered_finite = all_registered_finite and all(
                    bool(torch.isfinite(value).all()) for value in finite_tensors
                )

        if row_count != 495:
            raise RuntimeError("pairwise observation did not cover all 495 rows")
        if candidate_count_histogram != {"10": 60, "11": 435}:
            raise RuntimeError("selection conditional-NCE population changed")
        action_ba, action_recalls = _macro_balanced_accuracy(
            actual_actions, predicted_actions
        )
        scene_receipts: dict[str, Any] = {}
        correct_rgb_scene_wins = 0
        hardest_wrong_positive_scenes = 0
        target_positive_scenes = 0
        eligible_count = 0
        for family, binding in contract.SELECTION_FAMILY_BINDINGS.items():
            scene_id = str(binding["scene_id"])
            accumulator = scenes[scene_id]
            count = int(accumulator["row_count"])
            if count != int(binding["row_count"]):
                raise RuntimeError("selection scene row population changed")
            non_hold_count = int(accumulator["non_hold_row_count"])
            if non_hold_count != int(binding["non_hold_row_count"]):
                raise RuntimeError("selection scene non-hold population changed")
            target_count = int(accumulator["eligible_target_count"])
            if target_count != int(binding["same_action_row_count"]):
                raise RuntimeError(
                    "selection scene same-action population changed"
                )
            eligible_count += target_count
            correct_mean = accumulator["correct_rgb_loss_sum"] / count
            wrong_mean = accumulator["wrong_rgb_loss_sum"] / count
            hardest_mean = accumulator["hardest_wrong_margin_sum"] / count
            target_mean = (
                accumulator["target_margin_sum"] / target_count
                if target_count else None
            )
            correct_win = correct_mean < wrong_mean
            hardest_positive = hardest_mean > 0.0
            target_positive = target_mean is not None and target_mean > 0.0
            correct_rgb_scene_wins += int(correct_win)
            hardest_wrong_positive_scenes += int(hardest_positive)
            target_positive_scenes += int(target_positive)
            scene_receipts[family] = {
                "scene_id": scene_id,
                "row_count": count,
                "non_hold_row_count": non_hold_count,
                "same_action_eligible_row_count": target_count,
                "correct_rgb_mean_loss": correct_mean,
                "mapped_negative_rgb_mean_loss": wrong_mean,
                "correct_rgb_strict_win": correct_win,
                "hardest_wrong_minus_executed_mean_energy": hardest_mean,
                "hardest_wrong_positive": hardest_positive,
                "mapped_target_minus_correct_mean_energy": target_mean,
                "target_positive": target_positive,
            }
        if eligible_count != 494:
            raise RuntimeError("same-action target metric population changed")

        aggregate_confusion = torch.zeros(9, dtype=torch.long)
        rough_confusion = torch.zeros(9, dtype=torch.long)
        aggregate_nll_sum = 0.0
        rough_nll_sum = 0.0
        aggregate_cells = 0
        rough_cells = 0
        rough_set = set(rough_endpoints)
        family_endpoint_counts: dict[str, int] = {
            family: 0 for family in contract.SCENE_FAMILIES
        }
        for identity in aggregate_endpoints:
            family = str(loader.inputs.endpoints[identity]["family"])
            family_endpoint_counts[family] += 1
        with torch.no_grad():
            for start in range(0, len(aggregate_endpoints), contract.MICROBATCH_SIZE):
                identities = aggregate_endpoints[
                    start : start + contract.MICROBATCH_SIZE
                ]
                images, labels = loader.endpoint_batch(
                    identities,
                    device,
                    role="checkpoint_selection",
                    stage=f"raster_observation_update_{update}",
                )
                logits = model.online_state(images)
                probabilities = torch.softmax(logits, dim=1)
                prediction = probabilities.argmax(dim=1)
                codes = (labels * 3 + prediction).reshape(-1)
                batch_confusion = torch.bincount(codes, minlength=9).to("cpu")
                aggregate_confusion += batch_confusion
                target_probability = probabilities.gather(
                    1, labels[:, None]
                ).squeeze(1).clamp_min(torch.finfo(torch.float32).eps)
                per_cell_nll = -target_probability.log()
                aggregate_nll_sum += float(per_cell_nll.double().sum().cpu())
                aggregate_cells += int(labels.numel())
                rough_rows = [
                    offset for offset, identity in enumerate(identities)
                    if identity in rough_set
                ]
                if rough_rows:
                    rough_index = torch.tensor(
                        rough_rows, dtype=torch.long, device=device
                    )
                    rough_labels = labels.index_select(0, rough_index)
                    rough_prediction = prediction.index_select(0, rough_index)
                    rough_codes = (rough_labels * 3 + rough_prediction).reshape(-1)
                    rough_confusion += torch.bincount(
                        rough_codes, minlength=9
                    ).to("cpu")
                    rough_nll_sum += float(
                        per_cell_nll.index_select(0, rough_index)
                        .double().sum().cpu()
                    )
                    rough_cells += int(rough_labels.numel())
                all_registered_finite = all_registered_finite and bool(
                    torch.isfinite(logits).all()
                    and torch.isfinite(probabilities).all()
                    and torch.isfinite(per_cell_nll).all()
                )

        aggregate_raster = _confusion_metrics(
            aggregate_confusion.reshape(3, 3).tolist(),
            nll_sum=aggregate_nll_sum,
            cell_count=aggregate_cells,
        )
        rough_raster = _confusion_metrics(
            rough_confusion.reshape(3, 3).tolist(),
            nll_sum=rough_nll_sum,
            cell_count=rough_cells,
        )
        metrics: dict[str, Any] = {
            "G": g_sum / row_count,
            "J": j_sum / row_count,
            "C": c_sum / row_count,
            "action_nll": action_nll_sum / row_count,
            "action_macro_balanced_accuracy": action_ba,
            "action_per_class_recall": action_recalls,
            "hardest_wrong_positive_scene_count": (
                hardest_wrong_positive_scenes
            ),
            "same_action_target_nll": target_nll_sum / eligible_count,
            "same_action_target_strict_win_rate": (
                target_strict_wins / eligible_count
            ),
            "target_positive_scene_count": target_positive_scenes,
            "correct_rgb_scene_win_count": correct_rgb_scene_wins,
            "aggregate_raster_balanced_accuracy": (
                aggregate_raster["balanced_accuracy"]
            ),
            "aggregate_free_recall": aggregate_raster["free_recall"],
            "aggregate_occupied_recall": aggregate_raster["occupied_recall"],
            "aggregate_raster_nll": aggregate_raster["nll"],
            "rough_raster_balanced_accuracy": (
                rough_raster["balanced_accuracy"]
            ),
            "rough_raster_occupied_recall": rough_raster["occupied_recall"],
            "all_registered_values_finite": all_registered_finite,
            "state_nonconstant": state_max > state_min,
        }
        if update == 0:
            if gradient_integrity is None:
                raise RuntimeError("update-zero gradient probe did not run")
            metrics.update({
                "three_logit_bottleneck_exact": (
                    model.state_head.out_channels == 3
                    and model.predictor.net[0].in_channels == 6
                ),
                "no_hidden_or_auxiliary_bypass": (
                    set(partition["groups"])
                    == {
                        "encoder",
                        "decoder_state",
                        "predictor",
                        "detached_target_encoder_decoder_state",
                    }
                ),
                "prediction_is_exact_persistence": exact_persistence,
                "all_nine_action_predictions_bitwise_equal": all_nine_equal,
                "target_parameters_gradient_free": gradient_integrity[
                    "target_parameters_gradient_free"
                ],
                "intended_online_path_gradient_nonzero": gradient_integrity[
                    "intended_online_path_gradient_nonzero"
                ],
                "six_call_graph_isolation_exact": gradient_integrity[
                    "six_call_graph_isolation_exact"
                ],
            })
        gate = contract.evaluate_gate(
            update,
            metrics,
            update_zero=update_zero,
            prior_gates_passed=prior_gates_passed,
        )
        return {
            "schema": f"{contract.SCHEMA_PREFIX}_observation_v1",
            "update": update,
            "metrics": metrics,
            "gate": gate,
            "populations": {
                "G_J_C_action_wrong_rgb_pairs": row_count,
                "same_action_target_pairs": eligible_count,
                "same_action_target_fallback_pairs_excluded": 1,
                "aggregate_raster_unique_endpoints": len(aggregate_endpoints),
                "aggregate_raster_ordered_endpoint_identity_sha256": (
                    contract.canonical_json_sha256(aggregate_endpoints)
                ),
                "rough_raster_unique_endpoints": len(rough_endpoints),
                "rough_raster_family": contract.ROUGH_RASTER_FAMILY,
                "family_endpoint_counts": family_endpoint_counts,
            },
            "candidate_count_histogram": candidate_count_histogram,
            "scene_metrics": scene_receipts,
            "aggregate_raster": aggregate_raster,
            "rough_raster": rough_raster,
            "gradient_integrity": gradient_integrity,
            "call_graph": {
                "training_objective_online_calls": 2,
                "training_objective_detached_target_calls": 3,
                "observation_only_fixed_negative_online_calls_per_pair": 1,
                "wrong_rgb_correct_logits_reused_from_isolated_O_next": True,
                "raster_labels_enter_encoder_or_transition_count": 0,
            },
            "endpoint_label_identity": {
                "rgb_endpoint_identity_equals_raster_endpoint_identity": True,
                "raster_filename": "raster_labels.u1",
                "aggregate_ordered_endpoint_identity_sha256": (
                    contract.canonical_json_sha256(aggregate_endpoints)
                ),
                "aggregate_endpoint_count": len(aggregate_endpoints),
                "rough_endpoint_count": len(rough_endpoints),
            },
            "loader_access_after_observation": loader.receipt(),
        }
    finally:
        model.train(was_training)
        if _state_sha(runtime, model) != state_before:
            raise RuntimeError("registered observation changed model state")


def _evaluate_observation(
    runtime: Any,
    model_api: Any,
    model: Any,
    partition: Mapping[str, Any],
    loader: DirectBevNarrowLoader,
    selection_pairs: Sequence[Mapping[str, Any]],
    selection_mapping: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    update_zero: Mapping[str, Any] | None,
    prior_gates_passed: bool,
) -> dict[str, Any]:
    return _run_with_rng_preserved(
        runtime,
        lambda: _evaluate_observation_impl(
            runtime,
            model_api,
            model,
            partition,
            loader,
            selection_pairs,
            selection_mapping,
            device,
            update=update,
            update_zero=update_zero,
            prior_gates_passed=prior_gates_passed,
        ),
    )


def _run_with_strict_determinism(runtime: Any, operation: Any) -> tuple[Any, dict[str, Any]]:
    torch = runtime.torch
    previous_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    previous_warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    previous_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        return operation(), {
            "strict_deterministic_algorithms": True,
            "warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms, warn_only=previous_warn_only
        )
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


def _write_training_trace(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    value, raw = _publish_json(
        output_root / "training_trace.json",
        {
            "schema": f"{contract.SCHEMA_PREFIX}_training_trace_v1",
            "row_count": len(rows),
            "rows": [dict(row) for row in rows],
            "write_only": True,
            "read_count_after_write": 0,
            "resume_authorized": False,
        },
    )
    return _binding("training_trace.json", value, raw)


def _train_probe(
    runtime: Any,
    model_api: Any,
    fit: Any,
    loader: DirectBevNarrowLoader,
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    train_mapping: Mapping[str, Any],
    selection_mapping: Mapping[str, Any],
    schedule: Sequence[int],
    device: Any,
    output_root: Path,
    *,
    gpu_started: float,
    progress: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = runtime.torch
    if (
        len(train_pairs) != contract.TRAIN_ROLE_COUNTS["pairs"]
        or len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
        or len(schedule) != contract.MAXIMUM_PRESENTATIONS
    ):
        raise PermissionError("Direct BEV train/selection/schedule population changed")
    model, partition, initialization = _initialize_model(
        runtime, model_api, fit, device
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("Direct BEV model parameters must remain float32")
    optimizer, optimizer_receipt = _build_optimizer(runtime, partition)
    observations: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    updates = 0
    presentations = 0
    backward_calls = 0
    objective_evaluations = 0
    pair_loads = 0

    progress["stage"] = "observation_update_0"
    _check_gpu_time(
        gpu_started,
        maximum_minutes=contract.GPU_ACTIVE_TIME_CAP_MINUTES,
        stage="Direct BEV before update-zero observation",
    )
    observation_zero = _evaluate_observation(
        runtime,
        model_api,
        model,
        partition,
        loader,
        selection_pairs,
        selection_mapping,
        device,
        update=0,
        update_zero=None,
        prior_gates_passed=True,
    )
    observations.append(observation_zero)
    _check_gpu_time(
        gpu_started,
        maximum_minutes=contract.GPU_ACTIVE_TIME_CAP_MINUTES,
        stage="Direct BEV update-zero observation",
    )
    progress["registered_observations"] = 1
    progress["last_gate_control"] = observation_zero["gate"]["control"]
    update_zero_metrics = observation_zero["metrics"]
    terminal_gate = observation_zero["gate"]

    if terminal_gate["passed"]:
        model.train()
        for update in range(1, contract.MAXIMUM_UPDATES + 1):
            progress["stage"] = f"training_update_{update}"
            _check_gpu_time(
                gpu_started,
                maximum_minutes=contract.GPU_ACTIVE_TIME_CAP_MINUTES,
                stage=f"Direct BEV update {update}",
            )
            start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
            stop = update * contract.EFFECTIVE_BATCH_SIZE
            update_indices = [int(value) for value in schedule[start:stop]]
            if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
                raise RuntimeError("Direct BEV schedule exhausted before hard cap")
            optimizer.zero_grad(set_to_none=True)
            component_sums = {name: 0.0 for name in ("total", "G", "J", "C")}
            for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
                micro_start = microbatch * contract.MICROBATCH_SIZE
                indices = update_indices[
                    micro_start : micro_start + contract.MICROBATCH_SIZE
                ]
                if len(indices) != contract.MICROBATCH_SIZE:
                    raise RuntimeError("Direct BEV microbatch schedule changed")
                batch = loader.batch(
                    train_pairs,
                    indices,
                    device,
                    role="train",
                    stage=f"training_update_{update}_microbatch_{microbatch}",
                    mapped_negative_indices=train_mapping["negative_indices"],
                    scope="training",
                )
                pair_loads += len(indices)
                progress["pair_loads"] = pair_loads
                result = _objective(model, batch)
                objective_evaluations += 1
                progress["objective_evaluations"] = objective_evaluations
                if not bool(torch.isfinite(result.total)):
                    raise FloatingPointError("Direct BEV training objective is nonfinite")
                (result.total / contract.MICROBATCHES_PER_UPDATE).backward()
                backward_calls += 1
                progress["backward_calls"] = backward_calls
                component_sums["total"] += _scalar(result.total)
                component_sums["G"] += _scalar(result.G)
                component_sums["J"] += _scalar(result.J)
                component_sums["C"] += _scalar(result.C)

            joint_parameters = [
                parameter
                for group in ("encoder", "decoder_state")
                for _, parameter in partition["groups"][group]
            ]
            predictor_parameters = [
                parameter for _, parameter in partition["groups"]["predictor"]
            ]
            joint_preclip_norm = torch.nn.utils.clip_grad_norm_(
                joint_parameters,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            predictor_preclip_norm = torch.nn.utils.clip_grad_norm_(
                predictor_parameters,
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            progress["last_training_gradients_finite"] = bool(
                math.isfinite(_scalar(joint_preclip_norm))
                and math.isfinite(_scalar(predictor_preclip_norm))
            )
            progress["optimizer_updates"] = update
            before_ema = int(model.ema_update_count.detach().cpu().item())
            model.update_target_ema_after_optimizer_step()
            after_ema = int(model.ema_update_count.detach().cpu().item())
            if before_ema != update - 1 or after_ema != update:
                raise RuntimeError("target EMA did not update exactly once")
            updates = update
            presentations += contract.EFFECTIVE_BATCH_SIZE
            progress["updates"] = updates
            progress["presentations"] = presentations
            progress["ema_updates"] = after_ema
            if (
                updates > contract.MAXIMUM_UPDATES
                or presentations > contract.MAXIMUM_PRESENTATIONS
                or pair_loads != presentations
            ):
                raise RuntimeError("Direct BEV hard update/presentation cap changed")
            trace_rows.append({
                "update": update,
                "presentations": presentations,
                "schedule_slice_sha256": contract.canonical_json_sha256(
                    update_indices
                ),
                "mean_total": (
                    component_sums["total"]
                    / contract.MICROBATCHES_PER_UPDATE
                ),
                "mean_G": component_sums["G"] / contract.MICROBATCHES_PER_UPDATE,
                "mean_J": component_sums["J"] / contract.MICROBATCHES_PER_UPDATE,
                "mean_C": component_sums["C"] / contract.MICROBATCHES_PER_UPDATE,
                "encoder_decoder_state_preclip_norm": _scalar(joint_preclip_norm),
                "predictor_preclip_norm": _scalar(predictor_preclip_norm),
                "encoder_decoder_state_clip_max_norm": 1.0,
                "predictor_clip_max_norm": 1.0,
                "ema_update_count": after_ema,
            })
            if update in contract.OBSERVATION_UPDATES:
                progress["stage"] = f"observation_update_{update}"
                observation = _evaluate_observation(
                    runtime,
                    model_api,
                    model,
                    partition,
                    loader,
                    selection_pairs,
                    selection_mapping,
                    device,
                    update=update,
                    update_zero=update_zero_metrics,
                    prior_gates_passed=all(
                        bool(item["gate"]["passed"]) for item in observations
                    ),
                )
                observations.append(observation)
                terminal_gate = observation["gate"]
                progress["registered_observations"] = len(observations)
                progress["last_gate_control"] = terminal_gate["control"]
                snapshots.append(_snapshot_model(
                    runtime,
                    model,
                    output_root,
                    update=update,
                    metadata={
                        "gate": terminal_gate,
                        "metrics": observation["metrics"],
                        "optimizer_updates": updates,
                        "presentations": presentations,
                        "ema_updates": after_ema,
                    },
                ))
                _check_gpu_time(
                    gpu_started,
                    maximum_minutes=contract.GPU_ACTIVE_TIME_CAP_MINUTES,
                    stage=f"Direct BEV observation and snapshot {update}",
                )
                if not terminal_gate["passed"]:
                    break
            _check_gpu_time(
                gpu_started,
                maximum_minutes=contract.GPU_ACTIVE_TIME_CAP_MINUTES,
                stage=f"Direct BEV completed update {update}",
            )

    trace_binding = _write_training_trace(output_root, trace_rows)
    if terminal_gate["passed"] and terminal_gate["control"] != contract.CONTROL_PASS:
        raise RuntimeError("Direct BEV training stopped before a terminal gate")
    if not terminal_gate["passed"] and terminal_gate["control"] not in contract.FAILURE_CONTROLS:
        raise RuntimeError("Direct BEV failure control is unregistered")
    if terminal_gate["control"] == contract.CONTROL_PASS and (
        updates != contract.MAXIMUM_UPDATES
        or presentations != contract.MAXIMUM_PRESENTATIONS
        or int(model.ema_update_count.detach().cpu().item()) != updates
    ):
        raise RuntimeError("Direct BEV pass escaped the hard cap")
    progress["stage"] = "training_complete"
    progress["training_passed"] = bool(
        terminal_gate["control"] == contract.CONTROL_PASS
    )
    return model, {
        "schema": f"{contract.SCHEMA_PREFIX}_bounded_probe_v1",
        "status": terminal_gate["control"],
        "terminal_gate": terminal_gate,
        "updates": updates,
        "presentations": presentations,
        "optimizer_updates": updates,
        "ema_updates": int(model.ema_update_count.detach().cpu().item()),
        "pair_loads": pair_loads,
        "objective_evaluations": objective_evaluations,
        "backward_calls": backward_calls,
        "registered_observation_count": len(observations),
        "observations": observations,
        "snapshots": snapshots,
        "training_trace": trace_binding,
        "initialization": initialization,
        "optimizer": optimizer_receipt,
        "terminal_model_state_sha256": _state_sha(runtime, model),
        "hard_caps": {
            "maximum_updates": contract.MAXIMUM_UPDATES,
            "maximum_presentations": contract.MAXIMUM_PRESENTATIONS,
            "gpu_active_minutes_maximum": (
                contract.GPU_ACTIVE_TIME_CAP_MINUTES
            ),
        },
        "retry_resume_or_replacement_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }


def _load_authority_pre_reservation(
    review_sha256: str,
    authorization_sha256: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, str]]:
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise PermissionError("Torch imported before Direct BEV reservation")
    sources = contract.current_source_bindings(ROOT)
    source_manifest_raw = _read_regular(
        ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH,
        expected_sha256=sources[contract.SOURCE_MANIFEST_RELATIVE_PATH],
    )
    source_manifest = contract.validate_source_manifest(source_manifest_raw)
    source_manifest_binding = contract.artifact_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH,
        source_manifest_raw,
        content_sha256=str(source_manifest["content_sha256"]),
    )
    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_sha256,
    )
    review = contract.validate_review(
        contract.parse_canonical_json(review_raw, name="Direct BEV source review"),
        expected_sources=sources,
        source_manifest_binding=source_manifest_binding,
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=authorization_sha256,
    )
    authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            authorization_raw, name="Direct BEV execution authorization"
        ),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    return review, review_raw, authorization, authorization_raw, sources


def _source_authority_receipt(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "source_binding_count": len(sources),
        "source_bindings_sha256": contract.canonical_json_sha256(sources),
        "source_review": contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=str(review["content_sha256"]),
        ),
        "execution_authorization": contract.artifact_binding(
            contract.AUTHORIZATION_RELATIVE_PATH,
            authorization_raw,
            content_sha256=str(authorization["content_sha256"]),
        ),
        "generated_runtime_input_open_count": 0,
        "torch_imported": False,
    }


def _reserve(
    output_root: Path,
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
) -> tuple[dict[str, Any], bytes]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("the sole Direct BEV V1 attempt is already consumed")
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_binding = contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=str(authorization["content_sha256"]),
    )
    science = contract.science_contract()
    attempt_identity = contract.canonical_json_sha256({
        "schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1",
        "review": review_binding,
        "authorization": authorization_binding,
        "science_contract_sha256": contract.canonical_json_sha256(science),
    })
    core = {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "RESERVED_0700_BEFORE_TORCH_RUNTIME_RGB_RASTER_OR_CHECKPOINT",
        "attempt_index": contract.ATTEMPT_INDEX,
        "maximum_attempts": contract.MAXIMUM_ATTEMPTS,
        "attempt_identity": attempt_identity,
        "independent_source_review": review_binding,
        "execution_authorization": authorization_binding,
        "reviewed_sources": dict(sources),
        "science_contract": science,
        "output_root_absent_before_reservation": True,
        "output_root_mode": "0700",
        "torch_imported_before_reservation": False,
        "runtime_input_opened_before_reservation": False,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_root, mode=0o700)
    try:
        _reset_output_binding_registry(output_root)
        return _publish_json(output_root / "reservation.json", core)
    except BaseException as error:
        partial = _terminal_inventory(output_root)
        failure, failure_raw = _publish_json(
            output_root / "failure.json",
            {
                "schema": contract.FAILURE_SCHEMA,
                "status": contract.RESERVATION_PUBLICATION_FAILURE_STATUS,
                "attempt_identity": attempt_identity,
                "reservation_publication_succeeded": False,
                "error": _error_evidence(error),
                "exact_partial_inventory_before_failure_receipt": partial,
                "missing_normal_receipts": list(contract.NORMAL_RECEIPT_PATHS),
                "missing_normal_receipts_synthesized": False,
                "retry_resume_repair_or_replacement_authorized": False,
                "authority": dict(contract.DOWNSTREAM_DENIALS),
            },
        )
        inventory = _terminal_inventory(output_root)
        _publish_json(
            output_root / "completed.json",
            {
                "schema": contract.COMPLETION_SCHEMA,
                "status": contract.RESERVATION_PUBLICATION_FAILURE_STATUS,
                "attempt_identity": attempt_identity,
                "failure": _binding("failure.json", failure, failure_raw),
                "exact_precompletion_files": inventory["files"],
                "exact_precompletion_file_bindings": inventory["file_bindings"],
                "partial_evidence_bindings": inventory["partial_evidence_bindings"],
                "exact_terminal_files": sorted([
                    *inventory["files"], "completed.json"
                ]),
                "exact_terminal_directories_including_root": (
                    inventory["directories_including_root"]
                ),
                "retry_authorized": False,
                "authority": dict(contract.DOWNSTREAM_DENIALS),
            },
        )
        _seal_terminal_with_repair(output_root)
        raise


def _run_preflight_after_reservation(
    *,
    launcher_source_sha256: str,
    expected_source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    previous_contract = _BASE.contract
    previous_key = _BASE.PREFLIGHT_ENVIRONMENT_KEY
    _BASE.contract = contract
    _BASE.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
    try:
        return _BASE._run_preflight_after_reservation(
            launcher_source_sha256=launcher_source_sha256,
            expected_source_authority=expected_source_authority,
        )
    finally:
        _BASE.contract = previous_contract
        _BASE.PREFLIGHT_ENVIRONMENT_KEY = previous_key


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, Any, Any, Any]:
    required = (
        MATCHED_RUNNER_RELATIVE_PATH,
        SCHEDULE_ADAPTER_RELATIVE_PATH,
        contract.MODEL_RELATIVE_PATH,
    )
    if any(
        relative not in sources or not contract.is_sha256(sources[relative])
        for relative in required
    ):
        raise PermissionError("reviewed Direct BEV runtime source is incomplete")
    # Complete closure rehash immediately before the first tensor-capable import.
    for relative, expected_sha256 in sources.items():
        _read_regular(ROOT / relative, expected_sha256=expected_sha256)
    matched = _BASE._load_source_module(
        "_lewm_direct_bev_v1_matched_runtime",
        ROOT / MATCHED_RUNNER_RELATIVE_PATH,
    )
    runtime = matched._load_runtime()
    schedule_adapter = _BASE._load_source_module(
        "_lewm_direct_bev_v1_schedule_adapter",
        ROOT / SCHEDULE_ADAPTER_RELATIVE_PATH,
    )
    model_source = ROOT / contract.MODEL_RELATIVE_PATH
    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        model_api = _source_only_module(
            "lewm.models.direct_egocentric_bev_state_jepa_v1",
            model_source,
        )
    finally:
        sys.path[:] = original_path
    observed = Path(model_api.__file__)
    if (
        observed.is_symlink()
        or model_source.is_symlink()
        or observed.resolve() != model_source.resolve()
    ):
        raise PermissionError("imported Direct BEV model source changed")
    for relative in required:
        _read_regular(ROOT / relative, expected_sha256=sources[relative])
    return matched, runtime, schedule_adapter, model_api


def _read_bound(path: Path, binding: Mapping[str, Any]) -> bytes:
    validated = contract.validate_binding(
        dict(binding), path=path.relative_to(ROOT).as_posix()
    )
    return _read_regular(
        path,
        expected_sha256=validated["file_sha256"],
        expected_byte_count=validated["byte_count"],
    )


def _load_schedule(
    schedule_adapter: Any,
    authorization: Mapping[str, Any],
    train_pairs: Sequence[Mapping[str, Any]],
    *,
    progress: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    identity = authorization["runtime_inputs"]["schedule"]
    binding = identity["source"]
    progress["schedule_open_attempted"] = True
    raw = _read_bound(ROOT / binding["path"], binding)
    progress["schedule_open_succeeded"] = True
    state = schedule_adapter.validate_bound_schedule_phase_a(
        raw=raw, binding=binding
    )
    indices, observed_binding, adapter_record = (
        schedule_adapter.finalize_train_identity(
            state=state,
            ordered_train_pair_ids=[
                str(row["content_sha256"]) for row in train_pairs
            ],
        )
    )
    indices = list(indices)
    if (
        observed_binding != binding
        or len(indices) != contract.MAXIMUM_PRESENTATIONS
        or contract.canonical_json_sha256(indices[:1_600])
        != contract.SCHEDULE_PREFIX_SHA256[100]
        or contract.canonical_json_sha256(indices[:6_400])
        != contract.SCHEDULE_PREFIX_SHA256[400]
        or contract.canonical_json_sha256(indices)
        != contract.SCHEDULE_PREFIX_SHA256[1_000]
        or identity != contract.build_schedule_identity()
    ):
        raise PermissionError("frozen Direct BEV presentation schedule changed")
    progress["schedule_validated"] = True
    return indices, {
        "binding": dict(binding),
        "adapter_record": adapter_record,
        "identity": dict(identity),
        "used_presentation_count": len(indices),
        "schedule_regeneration_count": 0,
    }


def _validate_target_mappings(
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train = _LEGACY_CONTRACT.validate_same_action_target_mapping(
        train_pairs, role="train"
    )
    selection = _LEGACY_CONTRACT.validate_same_action_target_mapping(
        selection_pairs, role="checkpoint_selection"
    )
    permutation = _LEGACY_CONTRACT.validate_selection_action_permutation(
        selection_pairs
    )
    if (
        train["binding"] != contract.TARGET_MAPPING_BINDINGS["train"]
        or selection["binding"]
        != contract.TARGET_MAPPING_BINDINGS["checkpoint_selection"]
        or permutation["binding"] != contract.SELECTION_ACTION_PERMUTATION_BINDING
    ):
        raise PermissionError("Direct BEV frozen target mapping changed")
    return train, selection, permutation


def _terminal_authority_rehash(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    observed_sources = contract.current_source_bindings(ROOT)
    if observed_sources != dict(sources):
        raise PermissionError("reviewed Direct BEV source changed during execution")
    manifest_raw = _read_regular(
        ROOT / contract.SOURCE_MANIFEST_RELATIVE_PATH,
        expected_sha256=sources[contract.SOURCE_MANIFEST_RELATIVE_PATH],
    )
    manifest = contract.validate_source_manifest(manifest_raw)
    manifest_binding = contract.artifact_binding(
        contract.SOURCE_MANIFEST_RELATIVE_PATH,
        manifest_raw,
        content_sha256=str(manifest["content_sha256"]),
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    observed_review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_binding["file_sha256"],
        expected_byte_count=review_binding["byte_count"],
    )
    observed_review = contract.validate_review(
        contract.parse_canonical_json(
            observed_review_raw, name="terminal Direct BEV source review"
        ),
        expected_sources=sources,
        source_manifest_binding=manifest_binding,
    )
    authorization_binding = contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=str(authorization["content_sha256"]),
    )
    observed_authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=authorization_binding["file_sha256"],
        expected_byte_count=authorization_binding["byte_count"],
    )
    observed_authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            observed_authorization_raw,
            name="terminal Direct BEV execution authorization",
        ),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    if (
        observed_review_raw != review_raw
        or observed_authorization_raw != authorization_raw
        or observed_review != dict(review)
        or observed_authorization != dict(authorization)
    ):
        raise PermissionError("Direct BEV authority changed during execution")
    return {
        "source_bindings_sha256": contract.canonical_json_sha256(sources),
        "source_review": {
            **review_binding,
            "exact_pre_reservation_bytes_match": True,
        },
        "execution_authorization": {
            **authorization_binding,
            "exact_pre_reservation_bytes_match": True,
        },
    }


def _terminal_fixed_runtime_rehash(
    authorization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runtime_inputs = authorization["runtime_inputs"]
    bindings = (
        runtime_inputs["n320"]["gate"],
        runtime_inputs["n320"]["checkpoint"],
        runtime_inputs["schedule"]["source"],
    )
    result = []
    for binding in bindings:
        raw = _read_bound(ROOT / binding["path"], binding)
        result.append({
            **dict(binding),
            "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
            "observed_byte_count": len(raw),
        })
    return result


def _access_counters(loader: DirectBevNarrowLoader) -> dict[str, int]:
    detailed = loader.receipt()
    forbidden = detailed["forbidden_semantic_counters"]
    if (
        any(int(value) != 0 for value in forbidden.values())
        or int(detailed["raw_inputs_frame_attribute_invocation_count"]) != 0
    ):
        raise PermissionError("a forbidden Direct BEV loader access occurred")
    value = {
        "raw_manifest_open_count": 1,
        "raw_audit_open_count": 1,
        "pair_index_open_count": 1,
        "endpoint_index_open_count": 1,
        **loader.model_facing_access_counts(),
        "n320_gate_open_count": 1,
        "n320_checkpoint_open_count": 1,
        "schedule_open_count": 1,
        **{name: 0 for name in contract.FORBIDDEN_ACCESS_ZERO_COUNTER_FIELDS},
    }
    return contract.validate_access_counters(value)


def _error_evidence(error: BaseException) -> dict[str, str]:
    text = "".join(traceback.format_exception_only(type(error), error)).strip()
    return {
        "module": type(error).__module__,
        "type": type(error).__name__,
        "message": text[:2_000],
        "message_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    *,
    error: BaseException,
    progress: Mapping[str, Any],
) -> None:
    if (output_root / "completed.json").exists():
        _seal_terminal_with_repair(output_root)
        return
    public_progress = {
        name: value for name, value in progress.items()
        if not name.startswith("_")
    }
    loader = progress.get("_loader")
    loader_access = (
        loader.receipt()
        if isinstance(loader, DirectBevNarrowLoader)
        else public_progress.get("direct_bev_loader_access")
    )
    inputs = progress.get("_inputs")
    consumed_ledger = []
    if inputs is not None and type(getattr(inputs, "consumed", None)) is dict:
        consumed_ledger = [
            copy.deepcopy(inputs.consumed[name])
            for name in sorted(inputs.consumed)
        ]
    partial = _terminal_inventory(output_root)
    binding_by_path = {
        str(item["path"]): item for item in partial["file_bindings"]
    }
    present = [
        path for path in contract.NORMAL_RECEIPT_PATHS
        if path in binding_by_path
    ]
    missing = [
        path for path in contract.NORMAL_RECEIPT_PATHS if path not in present
    ]
    failure, failure_raw = _publish_json(
        output_root / "failure.json",
        {
            "schema": contract.FAILURE_SCHEMA,
            "status": contract.OPERATIONAL_FAILURE_STATUS,
            "reservation": _binding("reservation.json", reservation, reservation_raw),
            "progress": public_progress,
            "error": _error_evidence(error),
            "exact_partial_inventory_before_failure_receipt": partial,
            "normal_receipts_present": present,
            "normal_receipt_bindings_present": [
                binding_by_path[path] for path in present
            ],
            "missing_normal_receipts": missing,
            "missing_normal_receipts_synthesized": False,
            "loader_access_at_failure": loader_access,
            "raw_constructor_read_ledger": copy.deepcopy(
                progress.get("_raw_constructor_reads", {})
            ),
            "consumed_input_ledger_without_payload_reopen": consumed_ledger,
            "consumed_input_rehash_attempted_after_failure": False,
            "updates": int(public_progress.get("updates", 0)),
            "presentations": int(public_progress.get("presentations", 0)),
            "optimizer_updates": int(public_progress.get("optimizer_updates", 0)),
            "ema_updates": int(public_progress.get("ema_updates", 0)),
            "retry_resume_repair_or_replacement_authorized": False,
            "checkpoint_qualified": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    inventory = _terminal_inventory(output_root)
    _publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": contract.OPERATIONAL_FAILURE_STATUS,
            "attempt_identity": reservation["attempt_identity"],
            "failure": _binding("failure.json", failure, failure_raw),
            "exact_precompletion_files": inventory["files"],
            "exact_precompletion_file_bindings": inventory["file_bindings"],
            "partial_evidence_bindings": inventory["partial_evidence_bindings"],
            "exact_terminal_files": sorted([
                *inventory["files"], "completed.json"
            ]),
            "exact_terminal_directories_including_root": (
                inventory["directories_including_root"]
            ),
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    _seal_terminal_with_repair(output_root)


def _execute_after_reservation(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    output_root: Path,
    progress: dict[str, Any],
) -> int:
    progress["stage"] = "post_reservation_source_rehash"
    if contract.current_source_bindings(ROOT) != dict(sources):
        raise PermissionError("reviewed source changed across reservation")
    progress["stage"] = "post_reservation_hardware_preflight"
    preflight = _run_preflight_after_reservation(
        launcher_source_sha256=sources[contract.LAUNCHER_RELATIVE_PATH],
        expected_source_authority=_source_authority_receipt(
            review=review,
            review_raw=review_raw,
            authorization=authorization,
            authorization_raw=authorization_raw,
            sources=sources,
        ),
    )
    progress["preflight_validated"] = True
    progress["stage"] = "post_preflight_authority_rehash"
    _terminal_authority_rehash(
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
    )

    progress["stage"] = "deferred_tensor_runtime_import"
    matched, runtime, schedule_adapter, model_api = (
        _load_post_reservation_stack(sources)
    )
    progress["_runtime"] = runtime
    runtime_inputs = authorization["runtime_inputs"]
    adapted_authorization = {
        "raw": runtime_inputs["raw"],
        "camera": runtime_inputs["n320"],
    }
    progress["stage"] = "raw_authority_and_index_validation"
    inputs = _construct_raw_inputs_with_progress(
        matched,
        runtime,
        adapted_authorization,
        progress,
    )
    _normalize_endpoint_paths(inputs)
    train_pairs = inputs.role_pairs("train")
    selection_pairs = inputs.role_pairs("checkpoint_selection")
    if (
        len(train_pairs) != contract.TRAIN_ROLE_COUNTS["pairs"]
        or len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise PermissionError("Direct BEV development role population changed")
    train_mapping, selection_mapping, action_permutation = (
        _validate_target_mappings(train_pairs, selection_pairs)
    )
    progress["target_mapping_bindings"] = {
        "train": dict(train_mapping["binding"]),
        "checkpoint_selection": dict(selection_mapping["binding"]),
        "selection_action_permutation": dict(action_permutation["binding"]),
    }
    schedule, schedule_receipt = _load_schedule(
        schedule_adapter,
        authorization,
        train_pairs,
        progress=progress,
    )

    progress["stage"] = "reserved_runtime_device_validation"
    gpu_started = time.monotonic()
    trainer = matched.Trainer(runtime, inputs, output_root, reservation)
    device, hardware = trainer.device()
    if (
        hardware["visible_device_count"] != 1
        or "r9700" not in hardware["name"].casefold().replace(" ", "")
        or hardware["name"] != preflight["visible_device_name"]
        or hardware["total_memory_bytes"] != preflight["total_memory_bytes"]
    ):
        raise PermissionError("Direct BEV runtime GPU differs from preflight")
    progress["gpu_active_started"] = True

    progress["stage"] = "n320_initialization_checkpoint_load"
    fit, n320_gate, n320_checkpoint_binding = _load_n320_with_progress(
        matched,
        runtime,
        adapted_authorization,
        progress,
    )
    progress["n320_checkpoint_loaded"] = True
    loader = DirectBevNarrowLoader(runtime, inputs, progress=progress)
    progress["_loader"] = loader
    progress["stage"] = "direct_bev_bounded_probe"
    (model, probe), determinism = _run_with_strict_determinism(
        runtime,
        lambda: _train_probe(
            runtime,
            model_api,
            fit,
            loader,
            train_pairs,
            selection_pairs,
            train_mapping,
            selection_mapping,
            schedule,
            device,
            output_root,
            gpu_started=gpu_started,
            progress=progress,
        ),
    )
    probe["determinism"] = determinism
    status = str(probe["status"])
    passed = status == contract.CONTROL_PASS
    progress["updates"] = int(probe["updates"])
    progress["presentations"] = int(probe["presentations"])
    progress["training_passed"] = passed
    del fit
    _check_gpu_time(
        gpu_started,
        maximum_minutes=contract.GPU_ACTIVE_TIME_CAP_MINUTES,
        stage="Direct BEV terminal model release",
    )
    model.to("cpu")
    del model
    runtime.torch.cuda.empty_cache()
    gpu_active_elapsed_seconds = _check_gpu_time(
        gpu_started,
        maximum_minutes=contract.GPU_ACTIVE_TIME_CAP_MINUTES,
        stage="Direct BEV completed terminal GPU release",
    )

    progress["stage"] = "metrics_publication"
    metrics, metrics_raw = _publish_json(
        output_root / "metrics.json",
        {
            "schema": contract.METRICS_SCHEMA,
            "status": status,
            "registered_observation_count": probe["registered_observation_count"],
            "observations": [
                {
                    "update": item["update"],
                    "metrics": item["metrics"],
                    "gate": item["gate"],
                    "populations": item["populations"],
                    "candidate_count_histogram": item[
                        "candidate_count_histogram"
                    ],
                    "scene_metrics": item["scene_metrics"],
                    "aggregate_raster": item["aggregate_raster"],
                    "rough_raster": item["rough_raster"],
                    "gradient_integrity": item["gradient_integrity"],
                    "call_graph": item["call_graph"],
                    "endpoint_label_identity": item[
                        "endpoint_label_identity"
                    ],
                }
                for item in probe["observations"]
            ],
            "terminal_gate": probe["terminal_gate"],
            "all_gates_conjunctive": True,
            "stopped_at_first_failure": not passed,
            "checkpoint_qualified": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    progress["stage"] = "artifact_publication"
    artifact, artifact_raw = _publish_json(
        output_root / "artifact.json",
        {
            "schema": contract.ARTIFACT_SCHEMA,
            "status": status,
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "metrics": _binding("metrics.json", metrics, metrics_raw),
            "probe": probe,
            "schedule": schedule_receipt,
            "target_mapping_bindings": progress["target_mapping_bindings"],
            "hardware_preflight": preflight,
            "hardware": hardware,
            "n320": {
                "gate_content_sha256": n320_gate["content_sha256"],
                "checkpoint": n320_checkpoint_binding,
                "encoder_only_migration": True,
            },
            "last_training_gradients_finite": bool(
                progress.get("last_training_gradients_finite", probe["updates"] == 0)
            ),
            "checkpoint_qualified": False,
            "retry_resume_repair_or_replacement_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )

    progress["stage"] = "terminal_consumed_input_rehash"
    consumed = inputs.rehash_consumed()
    consumed_roles = {
        role
        for record in consumed["records"]
        for role in record["roles"]
    }
    permitted_roles = {
        "authority", "index", "train", "checkpoint_selection"
    }
    required_model_roles = {"checkpoint_selection"}
    if int(probe["presentations"]) > 0:
        required_model_roles.add("train")
    if (
        not consumed_roles.issubset(permitted_roles)
        or not required_model_roles.issubset(consumed_roles)
    ):
        raise PermissionError("Direct BEV consumed an unauthorized data role")
    array_records = [
        record for record in consumed["records"]
        if Path(str(record["path"])).suffix in {".u1", ".f4"}
    ]
    if any(Path(str(record["path"])).name != "raster_labels.u1"
           for record in array_records):
        raise PermissionError("Direct BEV consumed a forbidden supervision array")
    if not array_records:
        raise PermissionError("Direct BEV did not consume bound raster labels")
    terminal_rehash_classification = loader.terminal_rehash_classification(
        consumed["records"]
    )
    fixed_runtime_rehash = _terminal_fixed_runtime_rehash(authorization)
    authority_rehash = _terminal_authority_rehash(
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
    )
    access_counters = _access_counters(loader)
    if terminal_rehash_classification["raster_labels_u1_file_open_count"] != (
        access_counters["raster_label_physical_array_open_count"]
    ):
        raise PermissionError("terminal raster-label rehash accounting changed")
    if sum(
        terminal_rehash_classification[
            "rgb_file_open_count_by_first_request_kind"
        ].values()
    ) != access_counters["rgb_physical_file_open_count"]:
        raise PermissionError("terminal RGB rehash accounting changed")
    progress["stage"] = "access_publication"
    access, access_raw = _publish_json(
        output_root / "access.json",
        {
            "schema": contract.ACCESS_SCHEMA,
            "status": "ALL_AUTHORIZED_DEVELOPMENT_INPUTS_REHASHED",
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "roles_opened": sorted(consumed_roles),
            "model_facing_roles_opened": sorted(required_model_roles),
            "access_counters": access_counters,
            "loader": loader.receipt(),
            "loader_counter_mapping": dict(
                contract.RUNNER_ACCESS_COUNTER_MAPPING
            ),
            "terminal_rehash_classification": terminal_rehash_classification,
            "consumed": consumed,
            "fixed_runtime_input_rehash": fixed_runtime_rehash,
            "source_authority_rehash": authority_rehash,
            "allowed_supervision_arrays_opened": ["raster_labels.u1"],
            "forbidden_supervision_arrays_opened": [],
            "general_raw_frame_loader_call_count": 0,
            "checkpoint_or_training_trace_read_after_write_count": 0,
            "all_consumed_inputs_rehashed": True,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )

    if not passed:
        contract.validate_failure_status_chain({
            "metrics": status,
            "artifact": status,
            "result": status,
            "completion": status,
        })
    progress["stage"] = "result_publication"
    result, result_raw = _publish_json(
        output_root / "result.json",
        {
            "schema": contract.RESULT_SCHEMA,
            "status": status,
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "metrics": _binding("metrics.json", metrics, metrics_raw),
            "artifact": _binding("artifact.json", artifact, artifact_raw),
            "access": _binding("access.json", access, access_raw),
            "terminal_gate": probe["terminal_gate"],
            "operation_counts": {
                "optimizer_updates": probe["optimizer_updates"],
                "pair_presentations": probe["presentations"],
                "ema_updates": probe["ema_updates"],
                "objective_evaluations": probe["objective_evaluations"],
                "backward_calls": probe["backward_calls"],
                "observer_reruns": 0,
            },
            "gpu_active_elapsed_seconds": gpu_active_elapsed_seconds,
            "checkpoint_qualified": False,
            "pass_authorizes": (
                "perception_gate_requalification_preregistration_only"
                if passed else "nothing"
            ),
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    loader.image_cache.clear()
    loader.label_cache.clear()
    runtime.torch.cuda.empty_cache()

    inventory = _terminal_inventory(output_root)
    progress["stage"] = "completion_publication"
    _publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": status,
            "attempt_identity": reservation["attempt_identity"],
            "result": _binding("result.json", result, result_raw),
            "exact_precompletion_files": inventory["files"],
            "exact_precompletion_file_bindings": inventory["file_bindings"],
            "partial_evidence_bindings": inventory["partial_evidence_bindings"],
            "exact_terminal_files": sorted([
                *inventory["files"], "completed.json"
            ]),
            "exact_terminal_directories_including_root": (
                inventory["directories_including_root"]
            ),
            "all_inputs_rehashed": True,
            "all_terminal_files_sealed_read_only": True,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    progress["completion_published"] = True
    progress["stage"] = "terminal_sealing"
    _seal_terminal_with_repair(output_root)
    return 0 if passed else 2


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
) -> int:
    review, review_raw, authorization, authorization_raw, sources = (
        _load_authority_pre_reservation(
            review_file_sha256,
            authorization_file_sha256,
        )
    )
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation, reservation_raw = _reserve(
        output_root,
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
    )
    progress: dict[str, Any] = {
        "stage": "reserved",
        "preflight_validated": False,
        "gpu_active_started": False,
        "schedule_open_attempted": False,
        "schedule_open_succeeded": False,
        "schedule_validated": False,
        "raw_inputs_constructed": False,
        "n320_load_entered": False,
        "n320_gate_open_attempted": False,
        "n320_gate_open_succeeded": False,
        "n320_checkpoint_open_attempted": False,
        "n320_checkpoint_open_succeeded": False,
        "n320_checkpoint_loaded": False,
        "updates": 0,
        "optimizer_updates": 0,
        "ema_updates": 0,
        "presentations": 0,
        "pair_loads": 0,
        "objective_evaluations": 0,
        "backward_calls": 0,
        "registered_observations": 0,
        "training_passed": False,
        "completion_published": False,
    }
    try:
        return _execute_after_reservation(
            review=review,
            review_raw=review_raw,
            authorization=authorization,
            authorization_raw=authorization_raw,
            sources=sources,
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=output_root,
            progress=progress,
        )
    except BaseException as error:
        try:
            _terminal_failure(
                output_root,
                reservation,
                reservation_raw,
                error=error,
                progress=progress,
            )
        except BaseException as receipt_error:
            raise RuntimeError(
                "Direct BEV experiment failed and terminal receipt publication failed"
            ) from receipt_error
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("Direct BEV execution requires --run")
    for name in ("review_sha256", "authorization_sha256"):
        if not contract.is_sha256(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be an exact SHA-256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
