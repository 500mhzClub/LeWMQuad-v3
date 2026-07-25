#!/usr/bin/env python3
"""Run the V5 State-Dependent Latent-Flow JEPA probe.

Importing this module is source-only.  Torch, PIL, NumPy, generated inputs,
RGB payloads, and checkpoints are first reachable after exact source
authority has been validated and the fresh attempt root has been reserved.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, is_dataclass, replace
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import stat
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import warnings


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_lewm_go2_rgb_jepa_encoder_pretraining_v5_latent_flow_contract",
    _CONTRACT_PATH,
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise ImportError("cannot load RGB JEPA encoder-pretraining contract")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(contract)

PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_PATCH_WHITENED_ACTION_RESIDUAL_JEPA_"
    "V5_STATE_DEPENDENT_LATENT_FLOW_PREFLIGHT_JSON"
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
CONFLICTING_ACCELERATOR_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
)
class ScientificGateFailure(RuntimeError):
    """The experiment completed valid science but did not pass a fixed gate."""


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_byte_count: int | None = None,
) -> bytes:
    if path.is_symlink():
        raise PermissionError(f"symlink input forbidden: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"input is not regular: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    if not (
        _fingerprint(before)
        == _fingerprint(opened)
        == _fingerprint(after_open)
        == _fingerprint(after)
    ):
        raise RuntimeError(f"input changed while read: {path}")
    raw = b"".join(chunks)
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise PermissionError(f"input hash changed: {path}")
    if expected_byte_count is not None and len(raw) != expected_byte_count:
        raise PermissionError(f"input byte count changed: {path}")
    return raw


def _write_exclusive(path: Path, raw: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_json(
    path: Path,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive(path, raw)
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


def _state_sha(runtime: Any, state_or_model: Any) -> str:
    state = (
        state_or_model.state_dict()
        if hasattr(state_or_model, "state_dict")
        else state_or_model
    )
    normalized = {
        name: value.detach().to(device="cpu").contiguous()
        for name, value in sorted(state.items())
    }
    return runtime.model_module.tensor_state_dict_sha256(normalized)


def _subset_sha(runtime: Any, model: Any, prefixes: Sequence[str]) -> str:
    state = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(tuple(prefixes))
    }
    if not state:
        raise RuntimeError(f"state subset is empty: {tuple(prefixes)}")
    return _state_sha(runtime, state)


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu() if hasattr(value, "detach") else value)
    if not math.isfinite(result):
        raise FloatingPointError("experiment scalar became nonfinite")
    return result


def _receipt_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        result = asdict(value)
    elif hasattr(value, "to_dict"):
        result = value.to_dict()
    elif type(value) is dict:
        result = dict(value)
    else:
        raise TypeError("initialization receipt is not structured")
    if type(result) is not dict:
        raise TypeError("initialization receipt did not normalize to a dict")
    return result


def _load_source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_authority_pre_reservation(
    review_sha256: str,
    authorization_sha256: str,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, str],
]:
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise PermissionError("Torch imported before attempt reservation")
    sources = contract.current_source_bindings(ROOT)
    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_sha256,
    )
    review = contract.validate_review(
        contract.parse_canonical_json(review_raw, name="source review"),
        expected_sources=sources,
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
            authorization_raw, name="execution authorization"
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
        raise RuntimeError("the sole RGB JEPA pretraining attempt is consumed")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_root, mode=0o700)
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
    attempt_identity = contract.canonical_json_sha256({
        "schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1",
        "review": review_binding,
        "authorization": authorization_binding,
        "science_contract_sha256":
            contract.canonical_json_sha256(contract.science_contract()),
    })
    core = {
        "schema": f"{contract.SCHEMA_PREFIX}_reservation_v1",
        "status":
            "RESERVED_0700_BEFORE_TORCH_RUNTIME_INPUT_RGB_OR_CHECKPOINT",
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": attempt_identity,
        "independent_source_review": review_binding,
        "execution_authorization": authorization_binding,
        "reviewed_sources": dict(sources),
        "science_contract": contract.science_contract(),
        "output_root_absent_before_reservation": True,
        "output_root_mode": "0700",
        "torch_imported_before_reservation": False,
        "runtime_input_opened_before_reservation": False,
        "preflight_validation_deferred_until_after_reservation": True,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    try:
        return _publish_json(output_root / "reservation.json", core)
    except BaseException:
        try:
            _publish_json(output_root / "failure.json", {
                "schema": f"{contract.SCHEMA_PREFIX}_failure_v1",
                "status": "TERMINAL_RESERVATION_PUBLICATION_FAILURE",
                "attempt_identity": attempt_identity,
                "retry_authorized": False,
                "authority": dict(contract.DOWNSTREAM_DENIALS),
            })
        finally:
            _seal_terminal(output_root)
        raise


def _validate_preflight_after_reservation(
    *,
    expected_sha256: str,
    launcher_source_sha256: str,
    expected_source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("runner requires python -I -B")
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise PermissionError("Torch imported before post-reservation preflight")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("runner requires HIP_VISIBLE_DEVICES=0")
    if any(name in os.environ for name in CONFLICTING_ACCELERATOR_ENVIRONMENT):
        raise PermissionError("conflicting accelerator selector is present")
    if any(os.environ.get(name) != "1" for name in THREAD_ENVIRONMENT):
        raise PermissionError("all native thread selectors must equal one")
    encoded = os.environ.get(PREFLIGHT_ENVIRONMENT_KEY)
    if type(encoded) is not str:
        raise PermissionError("isolated no-tensor preflight receipt is absent")
    raw = encoded.encode("ascii") + b"\n"
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("preflight receipt hash changed")
    value = contract.parse_canonical_json(raw, name="hardware preflight receipt")
    fields = {
        "schema",
        "status",
        "launcher_process_id",
        "source_authority",
        "preflight_child_process_id",
        "visible_device_count",
        "visible_device_index",
        "visible_device_name",
        "total_memory_bytes",
        "torch_version",
        "hip_version",
        "tensor_allocation_count",
        "payload_open_count",
        "torch_device_api_call_count",
        "launcher_source_sha256",
        "immediate_exec_required",
        "intervening_gpu_query_count",
        "content_sha256",
    }
    name = value.get("visible_device_name")
    if (
        set(value) != fields
        or value["schema"] != f"{contract.SCHEMA_PREFIX}_hardware_preflight_v1"
        or value["status"] != "PASS_EXACTLY_ONE_VISIBLE_DISCRETE_R9700"
        or value["launcher_process_id"] != os.getpid()
        or value["source_authority"] != dict(expected_source_authority)
        or type(value["preflight_child_process_id"]) is not int
        or value["preflight_child_process_id"] <= 0
        or value["visible_device_count"] != 1
        or value["visible_device_index"] != 0
        or type(name) is not str
        or "r9700" not in name.casefold().replace(" ", "")
        or type(value["total_memory_bytes"]) is not int
        or value["total_memory_bytes"] < 32_000_000_000
        or value["tensor_allocation_count"] != 0
        or value["payload_open_count"] != 0
        or value["torch_device_api_call_count"] != 3
        or value["launcher_source_sha256"] != launcher_source_sha256
        or value["immediate_exec_required"] is not True
        or value["intervening_gpu_query_count"] != 0
    ):
        raise PermissionError("hardware preflight receipt changed")
    return value


def _seal_terminal(output_root: Path) -> dict[str, Any]:
    files: list[str] = []
    directories: list[Path] = []
    for current, names, filenames in os.walk(output_root, topdown=False):
        current_path = Path(current)
        directories.append(current_path)
        for name in filenames:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise PermissionError("terminal output contains a nonregular file")
            os.chmod(path, 0o444, follow_symlinks=False)
            files.append(path.relative_to(output_root).as_posix())
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise PermissionError("terminal output contains a nondirectory")
    for path in directories:
        os.chmod(path, 0o555, follow_symlinks=False)
    return {
        "file_count": len(files),
        "files": sorted(files),
        "directory_count_including_root": len(directories),
        "all_files_mode": "0444",
        "all_directories_mode": "0555",
    }


def _phase_a_ops() -> SimpleNamespace:
    """Load the exact frozen Phase-A objective helpers at the point of use."""
    from lewm.models.phase2d_spatial_lewm import (  # type: ignore
        normalize_spatial_tokens,
    )
    from lewm.models.patch_whitened_action_residual_jepa import (  # type: ignore
        ActionConditionedLatentFlow,
        action_indexed_energy_nll,
        initialize_action_gate_rows,
        patch_whitening_terms,
        predict_action_conditioned_flow_warps,
    )

    return SimpleNamespace(
        ActionConditionedLatentFlow=ActionConditionedLatentFlow,
        action_indexed_energy_nll=action_indexed_energy_nll,
        initialize_action_gate_rows=initialize_action_gate_rows,
        normalize_spatial_tokens=normalize_spatial_tokens,
        patch_whitening_terms=patch_whitening_terms,
        predict_action_conditioned_flow_warps=
            predict_action_conditioned_flow_warps,
    )


def _phase_a_current_only_loss(
    model: Any,
    current_rgb: Any,
    next_rgb: Any,
    action: Any,
    *,
    ops: Any | None = None,
) -> dict[str, Any]:
    """Exact residual Phase-A objective; the online path sees current RGB only."""
    ops = _phase_a_ops() if ops is None else ops
    if (
        current_rgb.ndim != 4
        or next_rgb.shape != current_rgb.shape
        or action.ndim != 2
        or action.shape != (current_rgb.shape[0], 9)
    ):
        raise ValueError("Phase-A current-only batch shape changed")

    online_tokens = model.encoder.forward_tokens(current_rgb)
    online_patches = online_tokens[:, 1:]
    online_state = model.online_geometry(online_patches)
    online_projected = ops.normalize_spatial_tokens(
        model.online_target_projector(online_state)
    )

    import torch  # deferred: this function is called only after reservation

    # Both EMA branches and the current-state skip are exact stop-gradients.
    with torch.no_grad():
        target_current_tokens = model.target_encoder.forward_tokens(current_rgb)
        target_next_tokens = model.target_encoder.forward_tokens(next_rgb)
        target_current_raw = model.target_geometry_module(
            target_current_tokens[:, 1:]
        )
        target_next_raw = model.target_geometry_module(
            target_next_tokens[:, 1:]
        )
        target_current_pre = model.target_projector(target_current_raw)
        target_next_pre = model.target_projector(target_next_raw)
        target_current = ops.normalize_spatial_tokens(target_current_pre)
        target_next = ops.normalize_spatial_tokens(target_next_pre)
    predictions = ops.predict_action_conditioned_flow_warps(
        model.predictor,
        model.prediction_projector,
        online_state,
        action,
        target_current,
    )
    action_losses = ops.action_indexed_energy_nll(
        predictions,
        target_next,
    )
    raw_whitening = ops.patch_whitening_terms(online_state.float())
    projected_whitening = ops.patch_whitening_terms(
        online_projected.float()
    )
    total = (
        action_losses.jepa
        + contract.ACTION_INDEXED_ENERGY_NLL_WEIGHT
        * action_losses.identification
        + contract.WHITENING_VARIANCE_WEIGHT
        * (raw_whitening.variance + projected_whitening.variance)
        + contract.WHITENING_COVARIANCE_WEIGHT
        * (raw_whitening.covariance + projected_whitening.covariance)
    )
    return {
        "loss": total,
        "jepa_loss": action_losses.jepa,
        "action_identification_loss": action_losses.identification,
        "raw_whitening_variance_loss": raw_whitening.variance,
        "raw_whitening_covariance_loss": raw_whitening.covariance,
        "projected_whitening_variance_loss":
            projected_whitening.variance,
        "projected_whitening_covariance_loss":
            projected_whitening.covariance,
        "prediction": predictions.executed,
        "all_action_predictions": predictions.all_predictions,
        "control_predictions": predictions.controls,
        "control_indices": predictions.control_indices,
        "all_flows_cell": predictions.all_flows_cell,
        "online_state": online_state,
        "raw_target_next": target_next_raw,
        "projected_target_next": target_next,
        "projected_target_current": target_current,
    }


def _verify_update_zero_action_symmetry_batch(
    torch: Any,
    all_predictions: Any,
) -> tuple[int, int]:
    """Compare all 36 candidate pairs for one update-zero microbatch."""
    if (
        all_predictions.ndim != 4
        or all_predictions.shape[0] < 1
        or all_predictions.shape[1] != 9
    ):
        raise PermissionError(
            "update-zero all-action prediction population changed"
        )
    comparison_count = 0
    for left in range(9):
        for right in range(left + 1, 9):
            comparison_count += 1
            if not torch.equal(
                all_predictions[:, left],
                all_predictions[:, right],
            ):
                raise PermissionError(
                    "update-zero all-action predictions are not bitwise equal"
                )
    return int(all_predictions.shape[0]), comparison_count


def _update_zero_action_symmetry_receipt(
    *,
    row_count: int,
    comparison_count: int | None,
) -> dict[str, Any]:
    """Bind the complete update-zero action-symmetry population."""
    expected_rows = contract.SELECTION_ROLE_COUNTS["pairs"]
    if row_count != expected_rows:
        raise PermissionError(
            "update-zero all-action prediction row count changed"
        )
    if comparison_count != 36:
        raise RuntimeError(
            "update-zero unordered action-pair comparison count changed"
        )
    return {
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": comparison_count,
        "all_action_prediction_row_count": row_count,
    }


def _scene_derangement(
    pairs: Sequence[Mapping[str, Any]],
    *,
    endpoint_key: str,
) -> tuple[int, ...]:
    """Return the first valid later row in each scene's canonical cycle."""
    if endpoint_key not in {
        "current_endpoint_sha256",
        "next_endpoint_sha256",
    }:
        raise ValueError("derangement endpoint key changed")
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(pairs):
        scene = str(row["scene_id"])
        groups.setdefault(scene, []).append(index)
    result = [-1] * len(pairs)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: str(pairs[index]["content_sha256"]))
        size = len(ordered)
        for position, index in enumerate(ordered):
            identity = str(pairs[index][endpoint_key])
            for offset in range(1, size):
                candidate = ordered[(position + offset) % size]
                if str(pairs[candidate][endpoint_key]) != identity:
                    result[index] = candidate
                    break
            if result[index] < 0:
                raise PermissionError(
                    "scene-local endpoint derangement cannot be constructed"
                )
    if any(
        str(pairs[index][endpoint_key])
        == str(pairs[candidate][endpoint_key])
        for index, candidate in enumerate(result)
    ):
        raise PermissionError("derangement retained an endpoint identity")
    return tuple(result)


def _effective_rank(torch: Any, tokens: Any) -> float:
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    samples = centered.reshape(-1, centered.shape[-1]).float()
    covariance = samples.T @ samples / max(1, samples.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool((total > 0).item()):
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum()
    return float(torch.exp(entropy))


class RGBOnlyLoader:
    """Decode only bound RGB endpoints; never call RawInputs.frame()."""

    def __init__(self, runtime: Any, inputs: Any, *, maximum_cache: int = 10_000):
        self.runtime = runtime
        self.inputs = inputs
        self.maximum_cache = int(maximum_cache)
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.supervision_array_open_count = 0
        self.general_frame_loader_call_count = 0

    def image(
        self,
        endpoint_identity: str,
        *,
        role: str,
        stage: str,
    ) -> Any:
        endpoint = self.inputs.endpoints.get(endpoint_identity)
        if type(endpoint) is not dict or endpoint.get("dataset_role") != role:
            raise PermissionError("RGB-only endpoint crossed its role")
        cached = self.cache.get(endpoint_identity)
        if cached is not None:
            self.cache.move_to_end(endpoint_identity)
            return cached
        raw = self.inputs.read_rgb(
            str(endpoint["image_path_metadata_only"]),
            str(endpoint["image_sha256_commitment_only"]),
            role=role,
            arm="rgb_jepa_encoder_pretraining_phase_a",
            stage=stage,
        )
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
        self.cache[endpoint_identity] = normalized
        self.cache.move_to_end(endpoint_identity)
        while len(self.cache) > self.maximum_cache:
            self.cache.popitem(last=False)
        return normalized

    def batch(
        self,
        pairs: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        device: Any,
        *,
        role: str,
        stage: str,
        include_non_hold: bool = True,
    ) -> tuple[Any, ...]:
        selected = [pairs[index] for index in indices]
        if any(row.get("dataset_role") != role for row in selected):
            raise PermissionError("RGB-only batch crossed dataset roles")
        current = self.runtime.torch.stack([
            self.image(
                str(row["current_endpoint_sha256"]),
                role=role,
                stage=stage,
            )
            for row in selected
        ]).to(device)
        next_rgb = self.runtime.torch.stack([
            self.image(
                str(row["next_endpoint_sha256"]),
                role=role,
                stage=stage,
            )
            for row in selected
        ]).to(device)
        actions = self.runtime.torch.zeros(
            (len(selected), 9),
            dtype=self.runtime.torch.float32,
            device=device,
        )
        action_indices = [
            contract.ACTION_VOCABULARY.index(str(row["primitive"]))
            for row in selected
        ]
        actions[
            self.runtime.torch.arange(len(selected), device=device),
            self.runtime.torch.tensor(action_indices, device=device),
        ] = 1.0
        if not include_non_hold:
            return current, next_rgb, actions
        non_hold = self.runtime.torch.tensor(
            [row["primitive"] != "hold" for row in selected],
            dtype=self.runtime.torch.bool,
            device=device,
        )
        return current, next_rgb, actions, non_hold


def _phase_a_parameter_partition(model: Any) -> dict[str, Any]:
    encoder: list[tuple[str, Any]] = []
    other: list[tuple[str, Any]] = []
    frozen: list[tuple[str, Any]] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith(contract.PHASE_A_ENCODER_PARAMETER_PREFIXES):
            encoder.append((name, parameter))
        elif name.startswith(contract.PHASE_A_AUXILIARY_PARAMETER_PREFIXES):
            other.append((name, parameter))
        elif name.startswith(contract.PHASE_A_FROZEN_PARAMETER_PREFIXES):
            frozen.append((name, parameter))
        elif parameter.numel() > 0:
            unexpected.append(name)
    if unexpected:
        raise PermissionError(
            f"Phase-A parameter partition changed: {unexpected[:4]}"
        )
    if not encoder or not other or not frozen:
        raise PermissionError("Phase-A parameter partition is empty")
    if any(parameter.requires_grad for _, parameter in frozen):
        raise PermissionError("Phase-A frozen parameter became trainable")
    if any(not parameter.requires_grad for _, parameter in (*encoder, *other)):
        raise PermissionError("Phase-A online parameter became frozen")
    return {
        "encoder": [parameter for _, parameter in encoder],
        "other": [parameter for _, parameter in other],
        "frozen": [parameter for _, parameter in frozen],
        "receipt": {
            "encoder_parameter_count":
                sum(parameter.numel() for _, parameter in encoder),
            "encoder_tensor_count": len(encoder),
            "other_parameter_count":
                sum(parameter.numel() for _, parameter in other),
            "other_tensor_count": len(other),
            "frozen_parameter_count":
                sum(parameter.numel() for _, parameter in frozen),
            "frozen_tensor_count": len(frozen),
            "appearance_projector_frozen": all(
                not parameter.requires_grad
                for parameter in model.appearance_projector.parameters()
            ),
            "appearance_projector_excluded_from_optimizer_and_clip": True,
            "encoder_names_sha256":
                contract.canonical_json_sha256([name for name, _ in encoder]),
            "other_names_sha256":
                contract.canonical_json_sha256([name for name, _ in other]),
            "frozen_names_sha256":
                contract.canonical_json_sha256([name for name, _ in frozen]),
        },
    }


def _runtime_with_tail_depth_loss_adapter(runtime: Any, tail_depth: Any) -> Any:
    return replace(
        runtime,
        loss_adapter=SimpleNamespace(
            observable_camera_ray_v4_loss_v4=(
                tail_depth.observable_camera_ray_v4_tail_depth_loss_v4
            )
        ),
    )


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, Any, Any, Any, Any]:
    """First Torch/PIL/NumPy-capable import point."""
    required = (
        contract.MATCHED_V1_RUNNER_RELATIVE_PATH,
        contract.SCHEDULE_ADAPTER_RELATIVE_PATH,
        contract.PHASE2D_MODEL_RELATIVE_PATH,
        contract.OBJECTIVE_MODEL_RELATIVE_PATH,
        contract.MULTIRES_MODEL_RELATIVE_PATH,
        contract.TAIL_DEPTH_LOSS_RELATIVE_PATH,
    )
    if any(
        path not in sources or not contract.is_sha256(sources[path])
        for path in required
    ):
        raise PermissionError("reviewed runtime source closure is incomplete")
    matched_path = ROOT / contract.MATCHED_V1_RUNNER_RELATIVE_PATH
    _read_regular(
        matched_path,
        expected_sha256=sources[contract.MATCHED_V1_RUNNER_RELATIVE_PATH],
    )
    matched = _load_source_module(
        "_lewm_jepa_encoder_v5_latent_flow_matched_loader",
        matched_path,
    )
    runtime = matched._load_runtime()
    _read_regular(
        ROOT / contract.SCHEDULE_ADAPTER_RELATIVE_PATH,
        expected_sha256=sources[contract.SCHEDULE_ADAPTER_RELATIVE_PATH],
    )
    schedule_adapter = _load_source_module(
        "_lewm_jepa_encoder_v5_latent_flow_schedule_adapter",
        ROOT / contract.SCHEDULE_ADAPTER_RELATIVE_PATH,
    )

    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        _read_regular(
            ROOT / contract.PHASE2D_MODEL_RELATIVE_PATH,
            expected_sha256=sources[contract.PHASE2D_MODEL_RELATIVE_PATH],
        )
        from lewm.models import phase2d_spatial_lewm as phase2d  # type: ignore
        _read_regular(
            ROOT / contract.OBJECTIVE_MODEL_RELATIVE_PATH,
            expected_sha256=sources[contract.OBJECTIVE_MODEL_RELATIVE_PATH],
        )
        from lewm.models import (  # type: ignore
            patch_whitened_action_residual_jepa as objective,
        )
        _read_regular(
            ROOT / contract.MULTIRES_MODEL_RELATIVE_PATH,
            expected_sha256=sources[contract.MULTIRES_MODEL_RELATIVE_PATH],
        )
        from lewm.models import (  # type: ignore
            shared_observable_camera_ray_jepa_v5_multires_v1 as multires,
        )
        _read_regular(
            ROOT / contract.TAIL_DEPTH_LOSS_RELATIVE_PATH,
            expected_sha256=sources[contract.TAIL_DEPTH_LOSS_RELATIVE_PATH],
        )
        from lewm.models import (  # type: ignore
            shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth
            as tail_depth,
        )
    finally:
        sys.path[:] = original_path
    for module, relative in (
        (phase2d, contract.PHASE2D_MODEL_RELATIVE_PATH),
        (objective, contract.OBJECTIVE_MODEL_RELATIVE_PATH),
        (multires, contract.MULTIRES_MODEL_RELATIVE_PATH),
        (tail_depth, contract.TAIL_DEPTH_LOSS_RELATIVE_PATH),
    ):
        observed = Path(module.__file__)
        expected = ROOT / relative
        if (
            observed.is_symlink()
            or expected.is_symlink()
            or observed.resolve() != expected.resolve()
        ):
            raise PermissionError(f"imported runtime source changed: {relative}")
        _read_regular(expected, expected_sha256=sources[relative])
    runtime = _runtime_with_tail_depth_loss_adapter(runtime, tail_depth)
    return matched, runtime, schedule_adapter, phase2d, multires


def _read_bound(path: Path, binding: Mapping[str, Any]) -> bytes:
    validated = contract.validate_binding(
        dict(binding),
        path=path.relative_to(ROOT).as_posix(),
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
) -> tuple[list[int], dict[str, Any]]:
    binding = authorization["runtime_inputs"]["schedule"]
    raw = _read_bound(ROOT / binding["path"], binding)
    state = schedule_adapter.validate_bound_schedule_phase_a(
        raw=raw,
        binding=binding,
    )
    indices, observed_binding, adapter_record = (
        schedule_adapter.finalize_train_identity(
            state=state,
            ordered_train_pair_ids=[
                str(row["content_sha256"]) for row in train_pairs
            ],
        )
    )
    if (
        observed_binding != binding
        or len(indices) != contract.MAXIMUM_PRESENTATIONS
        or contract.canonical_json_sha256(indices[:1_600])
        != contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[100]
        or contract.canonical_json_sha256(indices[:6_400])
        != contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[400]
        or contract.canonical_json_sha256(indices)
        != contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[1_000]
    ):
        raise PermissionError("frozen presentation schedule changed")
    return indices, {
        "binding": dict(binding),
        "adapter_record": adapter_record,
        "phase_a_identity":
            contract.build_schedule_identity("phase_a"),
        "phase_b_identity":
            contract.build_schedule_identity("phase_b"),
    }


def _normalize_endpoint_paths(inputs: Any) -> None:
    for endpoint in inputs.endpoints.values():
        image_path = Path(str(endpoint["image_path_metadata_only"]))
        if image_path.is_absolute():
            try:
                image_path = image_path.relative_to(ROOT)
            except ValueError as error:
                raise PermissionError(
                    "development RGB path escaped repository root"
                ) from error
        relative = image_path.as_posix()
        contract.safe_relative_path(relative, name="development RGB path")
        lowered = {part.casefold() for part in Path(relative).parts}
        if (
            "sealed" in lowered
            or any(part.startswith("sealed_") for part in lowered)
            or "heldout" in lowered
        ):
            raise PermissionError("protected RGB role entered endpoint index")
        endpoint["image_path_metadata_only"] = relative


def _run_with_rng_preserved(runtime: Any, operation: Any) -> Any:
    """Run an observation without perturbing later stochastic training."""
    torch = runtime.torch
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [item.clone() for item in torch.cuda.get_rng_state_all()]
    try:
        return operation()
    finally:
        torch.random.set_rng_state(cpu_before)
        torch.cuda.set_rng_state_all(cuda_before)
        if (
            not torch.equal(torch.random.get_rng_state(), cpu_before)
            or any(
                not torch.equal(before, after)
                for before, after in zip(
                    cuda_before,
                    torch.cuda.get_rng_state_all(),
                    strict=True,
                )
            )
        ):
            raise RuntimeError("observation RNG state was not restored exactly")


def _check_gpu_time(
    started: float,
    *,
    maximum_minutes: int,
    stage: str,
) -> float:
    elapsed = time.monotonic() - started
    if elapsed > maximum_minutes * 60.0:
        raise TimeoutError(
            f"{stage} exceeded the fixed {maximum_minutes}-minute GPU-active cap"
        )
    return elapsed


def _run_phase_b_with_reviewed_determinism(
    runtime: Any,
    operation: Any,
) -> tuple[Any, dict[str, Any]]:
    """Install the two reviewed matched-V1 ROCm/NumPy compatibility seams."""
    expected_prefix = (
        contract.PHASE_A_GRID_SAMPLE_DETERMINISM_WARNING_PREFIX
    )
    original_from_numpy = runtime.torch.from_numpy
    scalar_adaptation_count = 0

    def from_numpy_with_scalar(value: Any) -> Any:
        nonlocal scalar_adaptation_count
        if isinstance(value, runtime.np.generic):
            scalar_adaptation_count += 1
            return runtime.torch.as_tensor(value)
        return original_from_numpy(value)

    runtime.torch.from_numpy = from_numpy_with_scalar
    try:
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("once")
            runtime.torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                result = operation()
            finally:
                runtime.torch.use_deterministic_algorithms(
                    True, warn_only=False
                )
    finally:
        runtime.torch.from_numpy = original_from_numpy
    if not observed or any(
        item.category is not UserWarning
        or not str(item.message).startswith(expected_prefix)
        for item in observed
    ):
        raise RuntimeError(
            "Phase-B training emitted an unexpected determinism warning set"
        )
    if scalar_adaptation_count <= 0:
        raise RuntimeError(
            "Phase-B scalar from_numpy compatibility seam was not exercised"
        )
    messages = [str(item.message) for item in observed]
    return result, {
        "strict_deterministic_algorithms_restored": True,
        "warn_only_scope": "phase_b_training_and_inline_evaluation",
        "expected_grid_sampler_warning_count": len(messages),
        "warning_messages_sha256":
            contract.canonical_json_sha256(messages),
        "unexpected_warning_count": 0,
        "numpy_scalar_from_numpy_adaptation_count":
            scalar_adaptation_count,
        "torch_from_numpy_restored": True,
    }


def _run_phase_a_with_reviewed_determinism(
    runtime: Any,
    operation: Any,
) -> tuple[Any, dict[str, Any]]:
    """Permit only the preregistered latent-warp ROCm backward warning."""

    expected_prefix = (
        contract.PHASE_A_GRID_SAMPLE_DETERMINISM_WARNING_PREFIX
    )
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("once")
        runtime.torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            result = operation()
        finally:
            runtime.torch.use_deterministic_algorithms(
                True, warn_only=False
            )
    if not observed or any(
        item.category is not UserWarning
        or not str(item.message).startswith(expected_prefix)
        for item in observed
    ):
        raise RuntimeError(
            "Phase-A training emitted an unexpected determinism warning set"
        )
    messages = [str(item.message) for item in observed]
    return result, {
        "strict_deterministic_algorithms_restored": True,
        "warn_only_scope": "phase_a_training_and_checkpoint_selection",
        "expected_grid_sampler_warning_count": len(messages),
        "warning_messages_sha256":
            contract.canonical_json_sha256(messages),
        "unexpected_warning_count": 0,
    }


def _snapshot_model(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    phase: str,
    update: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    state_sha256 = _state_sha(runtime, state)
    semantic = {
        "schema": f"{contract.SCHEMA_PREFIX}_{phase}_checkpoint_v1",
        "phase": phase,
        "update": update,
        "state_sha256": state_sha256,
        "schedule_prefix_sha256":
            contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[update],
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
    relative = f"{phase}/checkpoints/update_{update}.pt"
    _write_exclusive(output_root / relative, raw)
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
        "state_sha256": state_sha256,
    }


def _phase_a_model(
    runtime: Any,
    phase2d: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch = runtime.torch
    ops = _phase_a_ops()
    torch.manual_seed(contract.BASE_INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(contract.BASE_INITIALIZATION_SEED)
    model = phase2d.Phase2DSpatialLeWorldModel(
        **contract.phase_a_model_config()
    )
    cpu_rng_before_gate = torch.random.get_rng_state().clone()
    cuda_rng_before_gate = [
        value.clone() for value in torch.cuda.get_rng_state_all()
    ]
    gate_initialization = ops.initialize_action_gate_rows(model.predictor)
    expected_gate_initialization = {
        "seed": contract.BASE_INITIALIZATION_SEED,
        "block_count": len(model.predictor.blocks),
        "latent_dim": 192,
        "attention_gate_rows": [384, 576],
        "mlp_gate_rows": [960, 1_152],
        "weight_std": contract.ACTION_GATE_WEIGHT_STD,
        "bias": contract.ACTION_GATE_BIAS,
        "changed_weight_scalar_count":
            len(model.predictor.blocks) * 2 * 192 * 192,
        "changed_bias_scalar_count":
            len(model.predictor.blocks) * 2 * 192,
    }
    if gate_initialization != expected_gate_initialization:
        raise RuntimeError("action-gate initialization receipt changed")
    if (
        not torch.equal(torch.random.get_rng_state(), cpu_rng_before_gate)
        or any(
            not torch.equal(before, after)
            for before, after in zip(
                cuda_rng_before_gate,
                torch.cuda.get_rng_state_all(),
                strict=True,
            )
        )
    ):
        raise RuntimeError("isolated action-gate initialization changed global RNG")
    gate_predictor_sha = _state_sha(runtime, model.predictor)
    cpu_rng_before_flow_install = torch.random.get_rng_state().clone()
    cuda_rng_before_flow_install = [
        value.clone() for value in torch.cuda.get_rng_state_all()
    ]
    shared_projector = model.prediction_projector
    shared_projector_sha = _state_sha(runtime, shared_projector)
    model.prediction_projector = ops.ActionConditionedLatentFlow(
        shared_projector
    )
    action_one_hot = torch.eye(9, dtype=torch.float32)[:, None, :]
    action_embeddings = model.predictor.action_embed(action_one_hot)[:, 0, :]
    relative_action_embeddings = (
        action_embeddings
        - action_embeddings[contract.HOLD_ACTION_INDEX]
    )
    pairwise_embedding_equality = (
        action_embeddings[:, None] == action_embeddings[None, :]
    ).all(dim=-1)
    expected_embedding_equality = torch.eye(9, dtype=torch.bool)
    non_hold_embedding_rows = torch.arange(9) != contract.HOLD_ACTION_INDEX
    action_embeddings_valid = (
        tuple(action_embeddings.shape) == (9, 192)
        and bool(torch.isfinite(action_embeddings).all().item())
        and torch.equal(
            pairwise_embedding_equality,
            expected_embedding_equality,
        )
        and int(
            torch.count_nonzero(
                relative_action_embeddings[non_hold_embedding_rows],
                dim=1,
            ).eq(0).sum().item()
        ) == 0
        and int(
            torch.count_nonzero(
                relative_action_embeddings[contract.HOLD_ACTION_INDEX]
            ).item()
        ) == 0
    )
    if (
        not torch.equal(
            torch.random.get_rng_state(),
            cpu_rng_before_flow_install,
        )
        or any(
            not torch.equal(before, after)
            for before, after in zip(
                cuda_rng_before_flow_install,
                torch.cuda.get_rng_state_all(),
                strict=True,
            )
        )
    ):
        raise RuntimeError(
            "latent-flow installation changed global RNG"
        )
    flow_weight = getattr(
        model.prediction_projector,
        "flow_weight",
        None,
    )
    direct_parameters = dict(
        model.prediction_projector.named_parameters(recurse=False)
    )
    flow_scalar_count = 2 * 192
    if (
        getattr(model.prediction_projector, "shared_projector", None)
        is not shared_projector
        or _state_sha(runtime, shared_projector) != shared_projector_sha
        or not isinstance(flow_weight, torch.nn.Parameter)
        or tuple(flow_weight.shape) != (2, 192)
        or flow_weight.dtype != torch.float32
        or set(direct_parameters) != {"flow_weight"}
        or direct_parameters["flow_weight"] is not flow_weight
        or int(flow_weight.numel()) != flow_scalar_count
        or int(torch.count_nonzero(flow_weight).item()) != 0
        or hasattr(model.prediction_projector, "flow_bias")
        or not action_embeddings_valid
    ):
        raise RuntimeError(
            "state-dependent latent-flow initialization changed"
        )
    flow_initialization = {
        "action_count": 9,
        "latent_dim": 192,
        "flow_dim": 2,
        "weight_shape": [2, 192],
        "weight_scalar_count": flow_scalar_count,
        "exact_zero_weight_scalar_count": flow_scalar_count,
        "nonzero_weight_scalar_count": 0,
        "bias_parameter_count": 0,
        "bias": False,
        "action_embedding_shape": [9, 192],
        "action_embeddings_pairwise_distinct": True,
        "all_eight_non_hold_relative_embeddings_nonzero": True,
        "hold_relative_embedding_exactly_zero": True,
        "maximum_absolute_displacement_patch_cells": 1.0,
        "normalized_patch_step": 2.0 / 15.0,
        "grid_shape": [16, 16],
        "grid_component_order": ["x_column", "y_row"],
        "grid_sample_mode": "bilinear",
        "grid_sample_padding_mode": "border",
        "grid_sample_align_corners": True,
        "wrapped_existing_shared_projector": True,
        "shared_projector_state_sha256_before_and_after":
            shared_projector_sha,
        "zero_initialized_without_rng_draw": True,
        "global_rng_state_preserved": True,
    }
    n320_encoder = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in fit.encoder.state_dict().items()
    }
    n320_sha = _state_sha(runtime, n320_encoder)
    model.encoder.load_state_dict(n320_encoder, strict=True)
    model.target_encoder.load_state_dict(n320_encoder, strict=True)
    if (
        _state_sha(runtime, model.encoder) != n320_sha
        or _state_sha(runtime, model.target_encoder) != n320_sha
    ):
        raise RuntimeError("Phase-A N320 encoder copy changed")
    model.appearance_projector.requires_grad_(False)
    model.appearance_projector.eval()
    model = model.to(device)
    model.train()
    model.appearance_projector.eval()
    partition = _phase_a_parameter_partition(model)
    receipt = {
        "schema": f"{contract.SCHEMA_PREFIX}_phase_a_initialization_v1",
        "seed": contract.BASE_INITIALIZATION_SEED,
        "model_config": contract.phase_a_model_config(),
        "n320_online_encoder_state_sha256": n320_sha,
        "n320_ema_encoder_state_sha256": n320_sha,
        "online_and_ema_encoder_exactly_equal": True,
        "predictor_and_projectors_fixed_seed_initialized": True,
        "action_gate_initialization": gate_initialization,
        "action_gate_initialization_preserved_global_rng": True,
        "predictor_state_sha256_after_action_gate_initialization":
            gate_predictor_sha,
        "state_dependent_latent_flow_initialization":
            flow_initialization,
        "appearance_projector_frozen_and_eval": (
            not model.appearance_projector.training
            and all(
                not parameter.requires_grad
                for parameter in model.appearance_projector.parameters()
            )
        ),
        "n320_evidence_head_copy_count": 0,
        "rejected_checkpoint_open_count": 0,
        "complete_initial_state_sha256": _state_sha(runtime, model),
        "partition": partition["receipt"],
    }
    return model, partition, receipt


def _phase_a_diagnostics(
    runtime: Any,
    model: Any,
    loader: RGBOnlyLoader,
    pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    update: int,
) -> dict[str, Any]:
    """Compute the exact 495-row controls while preserving all RNG streams."""
    torch = runtime.torch
    ops = _phase_a_ops()
    before_state = _state_sha(runtime, model)
    was_training = bool(model.training)

    def observe() -> dict[str, Any]:
        model.eval()
        states: list[Any] = []
        ema_current_skips: list[Any] = []
        actions: list[Any] = []
        predictions: list[Any] = []
        flows_cell: list[Any] = []
        update_zero_action_row_count = 0
        update_zero_action_pair_count: int | None = None
        control_energies: list[Any] = []
        control_indices: list[Any] = []
        raw_targets: list[Any] = []
        projected_targets: list[Any] = []
        non_hold_rows: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                indices = list(
                    range(start, min(start + contract.MICROBATCH_SIZE, len(pairs)))
                )
                current, next_rgb, action, non_hold = loader.batch(
                    pairs,
                    indices,
                    device,
                    role="checkpoint_selection",
                    stage=f"phase_a_diagnostic_update_{update}",
                )
                online_tokens = model.encoder.forward_tokens(current)
                state = model.online_geometry(online_tokens[:, 1:])
                current_target_tokens = (
                    model.target_encoder.forward_tokens(current)
                )
                current_target_raw = model.target_geometry_module(
                    current_target_tokens[:, 1:]
                )
                current_skip = ops.normalize_spatial_tokens(
                    model.target_projector(current_target_raw)
                )
                next_target_tokens = model.target_encoder.forward_tokens(
                    next_rgb
                )
                raw_target = model.target_geometry_module(
                    next_target_tokens[:, 1:]
                )
                projected_target = ops.normalize_spatial_tokens(
                    model.target_projector(raw_target)
                )
                residual_predictions = (
                    ops.predict_action_conditioned_flow_warps(
                        model.predictor,
                        model.prediction_projector,
                        state,
                        action,
                        current_skip,
                    )
                )
                if update == 0:
                    (
                        verified_rows,
                        verified_pairs,
                    ) = _verify_update_zero_action_symmetry_batch(
                        torch,
                        residual_predictions.all_predictions,
                    )
                    update_zero_action_row_count += verified_rows
                    if update_zero_action_pair_count is None:
                        update_zero_action_pair_count = verified_pairs
                    elif update_zero_action_pair_count != verified_pairs:
                        raise RuntimeError(
                            "update-zero action-pair count changed by batch"
                        )
                states.append(state.detach().cpu())
                ema_current_skips.append(current_skip.detach().cpu())
                actions.append(action.detach().cpu())
                predictions.append(
                    residual_predictions.executed.detach().cpu()
                )
                flows_cell.append(
                    residual_predictions.all_flows_cell.detach().cpu()
                )
                control_energies.append(
                    (
                        residual_predictions.controls
                        - projected_target[:, None]
                    ).square().mean(dim=(2, 3)).detach().cpu()
                )
                control_indices.append(
                    residual_predictions.control_indices.detach().cpu()
                )
                raw_targets.append(raw_target.detach().cpu())
                projected_targets.append(projected_target.detach().cpu())
                non_hold_rows.append(non_hold.detach().cpu())

        state = torch.cat(states)
        ema_current_skip = torch.cat(ema_current_skips)
        action = torch.cat(actions)
        prediction = torch.cat(predictions)
        all_flows_cell = torch.cat(flows_cell).float()
        control_mse = torch.cat(control_energies).float()
        candidate_indices = torch.cat(control_indices)
        raw_target = torch.cat(raw_targets).float()
        target = torch.cat(projected_targets).float()
        non_hold = torch.cat(non_hold_rows).bool()
        requested_indices = action.argmax(dim=1)
        if (
            len(pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
            or state.shape[0] != len(pairs)
            or tuple(raw_target.shape[1:]) != (256, 192)
            or tuple(all_flows_cell.shape) != (len(pairs), 9, 256, 2)
            or tuple(control_mse.shape) != (len(pairs), 8)
            or tuple(candidate_indices.shape) != (len(pairs), 8)
            or int(non_hold.sum())
            != contract.SELECTION_NON_HOLD_PAIR_COUNT
            or not torch.equal(
                non_hold,
                requested_indices != contract.HOLD_ACTION_INDEX,
            )
        ):
            raise PermissionError("Phase-A selection population changed")
        action_indexed_symmetry = (
            _update_zero_action_symmetry_receipt(
                row_count=update_zero_action_row_count,
                comparison_count=update_zero_action_pair_count,
            )
            if update == 0
            else None
        )
        per_action_any_nonzero = {
            action_name: bool(
                torch.count_nonzero(all_flows_cell[:, action_index]).item()
            )
            for action_index, action_name in enumerate(
                contract.ACTION_VOCABULARY
            )
        }
        latent_flow = {
            "all_values_finite": bool(
                torch.isfinite(all_flows_cell).all().item()
            ),
            "all_components_within_closed_one_patch_bound": bool(
                (all_flows_cell.abs() <= 1.0).all().item()
            ),
            "hold_flow_exactly_zero": (
                int(
                    torch.count_nonzero(
                        all_flows_cell[:, contract.HOLD_ACTION_INDEX]
                    ).item()
                )
                == 0
            ),
            "maximum_absolute_flow_cell":
                float(all_flows_cell.abs().max()),
            "non_hold_action_nonzero_count": sum(
                int(active)
                for action_name, active in per_action_any_nonzero.items()
                if action_name != "hold"
            ),
            "per_action_any_nonzero": per_action_any_nonzero,
        }

        current_mapping = torch.tensor(
            _scene_derangement(
                pairs, endpoint_key="current_endpoint_sha256"
            ),
            dtype=torch.long,
        )
        next_mapping = torch.tensor(
            _scene_derangement(
                pairs, endpoint_key="next_endpoint_sha256"
            ),
            dtype=torch.long,
        )
        shuffled_current_predictions: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(pairs), contract.MICROBATCH_SIZE):
                stop = min(start + contract.MICROBATCH_SIZE, len(pairs))
                shuffled_state = state[current_mapping[start:stop]].to(device)
                shuffled_skip = ema_current_skip[
                    current_mapping[start:stop]
                ].to(device)
                original_action = action[start:stop].to(device)
                shuffled_current_predictions.append(
                    ops.predict_action_conditioned_flow_warps(
                        model.predictor,
                        model.prediction_projector,
                        shuffled_state,
                        original_action,
                        shuffled_skip,
                    ).executed.cpu()
                )
        shuffled_current = torch.cat(shuffled_current_predictions)
        shuffled_next = target[next_mapping]
        mean_target = target.mean(dim=0, keepdim=True).expand_as(target)

        def row_mse(left: Any, right: Any) -> Any:
            return (left.float() - right.float()).square().mean(dim=(1, 2))

        true_mse = row_mse(prediction, target)
        cyclic_indices = (requested_indices + 1) % len(
            contract.ACTION_VOCABULARY
        )
        cyclic_matches = candidate_indices == cyclic_indices[:, None]
        if not bool((cyclic_matches.sum(dim=1) == 1).all()):
            raise PermissionError(
                "Phase-A cyclic wrong-action population changed"
            )
        cyclic_positions = cyclic_matches.to(torch.int64).argmax(dim=1)
        rows = torch.arange(len(pairs), dtype=torch.long)
        cyclic_wrong_mse = control_mse[rows, cyclic_positions]
        hardest_wrong_mse = control_mse.min(dim=1).values
        hold_matches = (
            candidate_indices[non_hold] == contract.HOLD_ACTION_INDEX
        )
        if not bool((hold_matches.sum(dim=1) == 1).all()):
            raise PermissionError(
                "Phase-A real-hold control population changed"
            )
        hold_positions = hold_matches.to(torch.int64).argmax(dim=1)
        non_hold_rows_index = rows[non_hold]
        hold_mse = control_mse[
            non_hold_rows_index,
            hold_positions,
        ]
        shuffled_next_mse = row_mse(prediction, shuffled_next)
        mean_target_mse = row_mse(prediction, mean_target)
        shuffled_current_mse = row_mse(shuffled_current, target)
        if not bool(non_hold.any()):
            raise PermissionError("Phase-A selection has no non-hold rows")

        q_raw = raw_target - raw_target.mean(dim=0, keepdim=True)
        raw_variance = raw_target.var(dim=0, unbiased=False).mean()
        spatial_diversity = q_raw.var(dim=1, unbiased=False).mean()
        per_family: dict[str, dict[str, Any]] = {}
        for family in contract.SCENE_FAMILIES:
            family_mask = torch.tensor(
                [row["family"] == family for row in pairs],
                dtype=torch.bool,
            )
            family_non_hold = family_mask & non_hold
            if not bool(family_mask.any()) or not bool(family_non_hold.any()):
                raise PermissionError(
                    f"Phase-A control population is empty: {family}"
                )
            per_family[family] = {
                "cyclic_wrong_action_minus_true_mse": float(
                    (
                        cyclic_wrong_mse[family_mask]
                        - true_mse[family_mask]
                    ).mean()
                ),
                "hardest_wrong_action_minus_true_mse": float(
                    (
                        hardest_wrong_mse[family_mask]
                        - true_mse[family_mask]
                    ).mean()
                ),
                "hold_action_minus_non_hold_true_mse": float(
                    (
                        hold_mse[family_non_hold[non_hold]]
                        - true_mse[family_non_hold]
                    ).mean()
                ),
                "hold_action_rows_match_non_hold_rows": True,
            }

        finite_tensors = (
            state,
            ema_current_skip,
            prediction,
            control_mse,
            raw_target,
            target,
            shuffled_current,
            true_mse,
            cyclic_wrong_mse,
            hardest_wrong_mse,
            hold_mse,
            all_flows_cell,
        )
        metric = {
            "all_values_finite": bool(
                torch.stack([
                    torch.isfinite(value).all() for value in finite_tensors
                ]).all()
            ),
            "ema_target_gradient_free": (
                all(
                    parameter.grad is None and not parameter.requires_grad
                    for module in (
                        model.target_encoder,
                        model.target_geometry_module,
                        model.target_projector,
                    )
                    for parameter in module.parameters()
                )
            ),
            "pair_count": len(pairs),
            "scene_family_count": len(contract.SCENE_FAMILIES),
            "cyclic_wrong_action_pair_count": len(pairs),
            "all_wrong_action_candidate_count":
                int(control_mse.numel()),
            "non_hold_pair_count": int(non_hold.sum()),
            "hold_action_pair_count": int(hold_mse.numel()),
            "hold_action_rows_match_non_hold_rows": True,
            "centered_raw_patch_effective_rank":
                _effective_rank(torch, raw_target),
            "centered_projected_target_effective_rank":
                _effective_rank(torch, target),
            "raw_cross_sample_variance": float(raw_variance),
            "content_residual_spatial_diversity":
                float(spatial_diversity),
            "true_pair_mse": float(true_mse.mean()),
            "shuffled_next_mse": float(shuffled_next_mse.mean()),
            "mean_target_mse": float(mean_target_mse.mean()),
            "cyclic_wrong_action_mse":
                float(cyclic_wrong_mse.mean()),
            "hardest_wrong_action_mse":
                float(hardest_wrong_mse.mean()),
            "non_hold_true_pair_mse": float(true_mse[non_hold].mean()),
            "hold_action_mse": float(hold_mse.mean()),
            "shuffled_current_mse": float(shuffled_current_mse.mean()),
            "per_family": per_family,
            "latent_flow": latent_flow,
        }
        if set(metric) != contract.PHASE_A_METRIC_FIELDS:
            raise RuntimeError("Phase-A diagnostic fields changed")
        return {
            "metric": metric,
            "action_indexed_symmetry": action_indexed_symmetry,
        }

    try:
        observed = _run_with_rng_preserved(runtime, observe)
    finally:
        if was_training:
            model.train()
            model.appearance_projector.eval()
    if _state_sha(runtime, model) != before_state:
        raise RuntimeError("Phase-A diagnostics mutated model state")
    return {
        "update": update,
        "role": "checkpoint_selection",
        "metric": observed["metric"],
        "action_indexed_symmetry": observed["action_indexed_symmetry"],
        "model_state_sha256_before_and_after": before_state,
        "rng_state_preserved": True,
        "state_mutation_count": 0,
    }


def _phase_a_train(
    runtime: Any,
    phase2d: Any,
    fit: Any,
    loader: RGBOnlyLoader,
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    schedule: Sequence[int],
    device: Any,
    output_root: Path,
    *,
    gpu_started: float,
    progress: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    torch = runtime.torch
    model, partition, initialization = _phase_a_model(
        runtime, phase2d, fit, device
    )
    trainable = [*partition["encoder"], *partition["other"]]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(partition["encoder"]),
                "lr": 1e-4,
                "group_name": "encoder",
            },
            {
                "params": list(partition["other"]),
                "lr": 3e-4,
                "group_name": "auxiliary",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
    )
    torch.cuda.manual_seed_all(contract.BASE_INITIALIZATION_SEED)
    diagnostics = [
        _phase_a_diagnostics(
            runtime,
            model,
            loader,
            selection_pairs,
            device,
            update=0,
        )
    ]
    update0_health = {
        name: diagnostics[0]["metric"][name]
        for name in (
            "raw_cross_sample_variance",
            "content_residual_spatial_diversity",
        )
    }
    update0_health.update(diagnostics[0]["action_indexed_symmetry"])
    update0_health["latent_flow"] = diagnostics[0]["metric"]["latent_flow"]
    if set(update0_health) != contract.PHASE_A_UPDATE0_FIELDS:
        raise RuntimeError("Phase-A update-zero receipt fields changed")
    trace: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    continuation_gates: list[dict[str, Any]] = []
    early_failure: dict[str, Any] | None = None
    ema_update_count = 0
    for update in range(1, contract.PHASE_A_MAXIMUM_UPDATE + 1):
        _check_gpu_time(
            gpu_started,
            maximum_minutes=contract.PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES,
            stage="Phase A",
        )
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, Any] = {}
        start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
        update_indices = list(
            schedule[start : start + contract.EFFECTIVE_BATCH_SIZE]
        )
        if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
            raise PermissionError("Phase-A schedule ended early")
        for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
            low = microbatch * contract.MICROBATCH_SIZE
            indices = update_indices[low : low + contract.MICROBATCH_SIZE]
            current, next_rgb, action = loader.batch(
                train_pairs,
                indices,
                device,
                role="train",
                stage="phase_a_gradient",
                include_non_hold=False,
            )
            loss = _phase_a_current_only_loss(
                model,
                current,
                next_rgb,
                action,
            )
            if not bool(torch.isfinite(loss["loss"]).item()):
                raise FloatingPointError("Phase-A objective became nonfinite")
            (loss["loss"] / contract.MICROBATCHES_PER_UPDATE).backward()
            for name in (
                "loss",
                "jepa_loss",
                "action_identification_loss",
                "raw_whitening_variance_loss",
                "raw_whitening_covariance_loss",
                "projected_whitening_variance_loss",
                "projected_whitening_covariance_loss",
            ):
                contribution = (
                    loss[name].detach()
                    / contract.MICROBATCHES_PER_UPDATE
                )
                sums[name] = sums.get(name, contribution.new_zeros(())) + contribution
        if any(parameter.grad is not None for parameter in partition["frozen"]):
            raise RuntimeError("Phase-A frozen parameter acquired a gradient")
        gradient_before = torch.nn.utils.clip_grad_norm_(
            trainable, max_norm=1.0
        )
        if not bool(torch.isfinite(gradient_before).item()):
            raise FloatingPointError("Phase-A gradient became nonfinite")
        gradient_after = _scalar(torch.stack([
            parameter.grad.detach().float().square().sum()
            for parameter in trainable
            if parameter.grad is not None
        ]).sum().sqrt())
        if gradient_after > 1.00001:
            raise RuntimeError("Phase-A global clip norm changed")
        optimizer.step()
        model.update_target_encoder()
        ema_update_count += 1
        progress["phase_a_updates"] = update
        progress["phase_a_presentations"] = (
            update * contract.EFFECTIVE_BATCH_SIZE
        )
        trace.append({
            "schema": f"{contract.SCHEMA_PREFIX}_phase_a_trace_row_v1",
            "update": update,
            "presentation_indices_sha256":
                contract.canonical_json_sha256(update_indices),
            "encoder_learning_rate": 1e-4,
            "auxiliary_learning_rate": 3e-4,
            "microbatch_count": contract.MICROBATCHES_PER_UPDATE,
            "pair_presentations":
                update * contract.EFFECTIVE_BATCH_SIZE,
            "backward_count": update * contract.MICROBATCHES_PER_UPDATE,
            "optimizer_step_count": update,
            "ema_update_count": ema_update_count,
            "global_clip_count": update,
            "gradient_norm_before_clip": _scalar(gradient_before),
            "gradient_norm_after_clip": gradient_after,
            "losses": {
                name: value
                for name, value in zip(
                    sums,
                    torch.stack(tuple(sums.values())).cpu().tolist(),
                    strict=True,
                )
            },
        })
        if update in contract.CHECKPOINT_UPDATES:
            diagnostic = _phase_a_diagnostics(
                runtime,
                model,
                loader,
                selection_pairs,
                device,
                update=update,
            )
            diagnostics.append(diagnostic)
            if update in {100, 400}:
                continuation = contract.evaluate_phase_a_continuation(
                    update,
                    diagnostic["metric"],
                    update0_health,
                    {
                        "rng_state_preserved":
                            diagnostic["rng_state_preserved"],
                        "state_mutation_count":
                            diagnostic["state_mutation_count"],
                    },
                )
                continuation_gates.append(continuation)
                if not continuation["passed"]:
                    early_failure = continuation
            snapshot = _snapshot_model(
                runtime,
                model,
                output_root,
                phase="phase_a",
                update=update,
                metadata={
                    "initialization": initialization,
                    "partition": partition["receipt"],
                },
            )
            snapshots.append(snapshot)
        _check_gpu_time(
            gpu_started,
            maximum_minutes=contract.PHASE_A_GPU_ACTIVE_TIME_CAP_MINUTES,
            stage="Phase A",
        )
        if early_failure is not None:
            break
    if ema_update_count != len(trace):
        raise RuntimeError("Phase-A EMA update count changed")
    if early_failure is not None:
        terminal_gate = early_failure
        phase_status = str(early_failure["control"])
    else:
        if ema_update_count != contract.PHASE_A_MAXIMUM_UPDATE:
            raise RuntimeError("Phase-A terminal update count changed")
        terminal_observation = diagnostics[-1]
        terminal_gate = contract.evaluate_phase_a(
            terminal_observation["metric"],
            update0_health,
            {
                "rng_state_preserved":
                    terminal_observation["rng_state_preserved"],
                "state_mutation_count":
                    terminal_observation["state_mutation_count"],
            },
        )
        phase_status = (
            "PASS_PHASE_A"
            if terminal_gate["passed"]
            else "FAIL_PHASE_A_TERMINAL"
        )
    trace_raw = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in trace
    )
    _write_exclusive(output_root / "phase_a/training_trace.jsonl", trace_raw)
    metrics, metrics_raw = _publish_json(
        output_root / "phase_a/metrics.json",
        {
            "schema": contract.PHASE_A_METRICS_SCHEMA,
            "status": phase_status,
            "observations": diagnostics,
            "update0_health": update0_health,
            "continuation_gates": continuation_gates,
            "terminal_gate": terminal_gate,
            "selection_evaluation_updates": [
                observation["update"] for observation in diagnostics
            ],
            "observer_rerun_count": 0,
            "rng_state_preserved_at_every_observation": True,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    artifact = {
        "schema": contract.PHASE_A_ARTIFACT_SCHEMA,
        "status": (
            "QUALIFIED_FOR_CONDITIONAL_PHASE_B"
            if terminal_gate["passed"]
            else phase_status
        ),
        "initialization": initialization,
        "partition": partition["receipt"],
        "snapshots": snapshots,
        "metrics": _binding("phase_a/metrics.json", metrics, metrics_raw),
        "training_trace": {
            "path": "phase_a/training_trace.jsonl",
            "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
            "byte_count": len(trace_raw),
            "row_count": len(trace),
            "content_sha256": contract.canonical_json_sha256(trace),
        },
        "updates": len(trace),
        "presentations": len(trace) * contract.EFFECTIVE_BATCH_SIZE,
        "ema_update_count": ema_update_count,
        "terminal_online_encoder_state_sha256":
            _state_sha(runtime, model.encoder),
        "target_state_gradient_count": 0,
        "frozen_state_gradient_count": 0,
        "appearance_projector_gradient_count": 0,
        "camera_supervision_array_open_count":
            loader.supervision_array_open_count,
        "general_raw_v13_frame_loader_call_count":
            loader.general_frame_loader_call_count,
        "gate": terminal_gate,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    return model, artifact


def _changed_state_keys(
    torch: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    if set(before) != set(after):
        raise RuntimeError("model state key set changed")
    return sorted(
        name
        for name in before
        if not bool(torch.equal(before[name], after[name]))
    )


def _set_phase_b_mode(model: Any, *, training: bool) -> None:
    """Set evidence-head mode while keeping every frozen module in eval."""
    model.train(training)
    model.evidence_head.train(training)
    for name in (
        "encoder",
        "bev_decoder",
        "predictor",
        "occupancy_head",
        "target_encoder",
        "target_bev_decoder",
    ):
        getattr(model, name).eval()
    if any(
        getattr(model, name).training
        for name in (
            "encoder",
            "bev_decoder",
            "predictor",
            "occupancy_head",
            "target_encoder",
            "target_bev_decoder",
        )
    ):
        raise RuntimeError("a frozen Phase-B module entered training mode")


def _phase_b_model(
    runtime: Any,
    multires: Any,
    fit: Any,
    phase_a_encoder_state: Mapping[str, Any],
    device: Any,
) -> tuple[Any, list[Any], list[Any], dict[str, Any]]:
    torch = runtime.torch
    model, raw_migration = (
        multires.SharedObservableCameraRayJepaV5MultiresV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=contract.RUNTIME_FILE_SHA256[
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
            n320_checkpoint_content_sha256=contract.RUNTIME_CONTENT_SHA256[
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
        )
    )
    migration = _receipt_dict(raw_migration)
    initialized_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    initial_target_bev_sha = _state_sha(runtime, model.target_bev_decoder)
    initial_online_bev_sha = _state_sha(runtime, model.bev_decoder)
    if initial_target_bev_sha != initial_online_bev_sha:
        raise RuntimeError(
            "Phase-B target BEV decoder was not identical at initialization"
        )
    phase_a_encoder_sha = _state_sha(runtime, phase_a_encoder_state)
    model.encoder.load_state_dict(phase_a_encoder_state, strict=True)
    after_online_copy = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    online_changes = _changed_state_keys(
        torch, initialized_state, after_online_copy
    )
    if any(not name.startswith("encoder.") for name in online_changes):
        raise RuntimeError("Phase-B online encoder copy escaped its scope")

    # Deliberately do not call hard_sync_ema_target_from_online(): that helper
    # also copies target_bev_decoder. V5 permits exactly this encoder-only sync.
    model.target_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
    model.target_encoder.requires_grad_(False)
    model.target_encoder.eval()
    after_target_sync = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    target_changes = _changed_state_keys(
        torch, after_online_copy, after_target_sync
    )
    if any(not name.startswith("target_encoder.") for name in target_changes):
        raise RuntimeError("Phase-B target-encoder hard sync escaped its scope")
    if (
        _state_sha(runtime, model.encoder) != phase_a_encoder_sha
        or _state_sha(runtime, model.target_encoder) != phase_a_encoder_sha
        or _state_sha(runtime, model.target_bev_decoder)
        != initial_target_bev_sha
        or _state_sha(runtime, model.bev_decoder) != initial_online_bev_sha
    ):
        raise RuntimeError("Phase-B transfer state identity changed")

    model.requires_grad_(False)
    trainable: list[tuple[str, Any]] = []
    frozen: list[tuple[str, Any]] = []
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith(contract.PHASE_B_TRAINABLE_PARAMETER_PREFIXES):
            parameter.requires_grad_(True)
            trainable.append((name, parameter))
        elif name.startswith(contract.PHASE_B_FROZEN_PARAMETER_PREFIXES):
            parameter.requires_grad_(False)
            frozen.append((name, parameter))
        else:
            unexpected.append(name)
    if unexpected or not trainable or not frozen:
        raise PermissionError(
            f"Phase-B parameter partition changed: {unexpected[:4]}"
        )
    model = model.to(device)
    _set_phase_b_mode(model, training=True)
    receipt = {
        "schema": f"{contract.SCHEMA_PREFIX}_phase_b_initialization_v1",
        "n320_migration": migration,
        "phase_a_online_encoder_state_sha256": phase_a_encoder_sha,
        "copied_online_state_prefixes": ["encoder."],
        "online_copy_changed_state_keys": online_changes,
        "post_copy_hard_sync": {
            "method": "manual_strict_target_encoder_load_state_dict",
            "count": 1,
            "copied_state_prefixes": ["target_encoder."],
            "changed_state_keys": target_changes,
            "target_bev_decoder_copy_count": 0,
            "target_bev_decoder_initialization_identity_verified": True,
            "target_bev_decoder_state_sha256": initial_target_bev_sha,
        },
        "shared_v5_bev_decoder_or_predictor_copy_count": 0,
        "jepa_predictor_projector_or_ema_transfer_count": 0,
        "trainable_prefixes":
            list(contract.PHASE_B_TRAINABLE_PARAMETER_PREFIXES),
        "frozen_prefixes": list(contract.PHASE_B_FROZEN_PARAMETER_PREFIXES),
        "trainable_parameter_count":
            sum(parameter.numel() for _, parameter in trainable),
        "trainable_tensor_count": len(trainable),
        "frozen_parameter_count":
            sum(parameter.numel() for _, parameter in frozen),
        "frozen_tensor_count": len(frozen),
        "trainable_names_sha256":
            contract.canonical_json_sha256([name for name, _ in trainable]),
        "frozen_names_sha256":
            contract.canonical_json_sha256([name for name, _ in frozen]),
    }
    return (
        model,
        [parameter for _, parameter in trainable],
        [parameter for _, parameter in frozen],
        receipt,
    )


def _phase_b_batch(
    trainer: Any,
    pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: Any,
    *,
    role: str,
    stage: str,
) -> dict[str, Any]:
    selected = [pairs[index] for index in indices]
    if any(row.get("dataset_role") != role for row in selected):
        raise PermissionError("Phase-B batch crossed dataset roles")
    current = [
        trainer.inputs.frame(
            str(row["current_endpoint_sha256"]),
            role=role,
            arm="rgb_jepa_encoder_pretraining_phase_b",
            stage=stage,
        )
        for row in selected
    ]
    next_frames = [
        trainer.inputs.frame(
            str(row["next_endpoint_sha256"]),
            role=role,
            arm="rgb_jepa_encoder_pretraining_phase_b",
            stage=stage,
        )
        for row in selected
    ]

    def stack(frames: Sequence[Mapping[str, Any]], name: str) -> Any:
        return trainer.r.torch.stack([row[name] for row in frames]).to(device)

    return {
        "forward": {
            "current_image": stack(current, "image"),
            "next_image": stack(next_frames, "image"),
            "current_camera_origin_body_m":
                stack(current, "camera_origin").float(),
            "current_camera_basis_body_fru":
                stack(current, "camera_basis").float(),
            "current_ground_plane_z_body_m":
                stack(current, "ground").float(),
            "next_camera_origin_body_m":
                stack(next_frames, "camera_origin").float(),
            "next_camera_basis_body_fru":
                stack(next_frames, "camera_basis").float(),
            "next_ground_plane_z_body_m":
                stack(next_frames, "ground").float(),
        },
        "current_supervision": trainer.supervision(current, device),
        "next_supervision": trainer.supervision(next_frames, device),
    }


def _phase_b_pair(runtime: Any, model: Any, batch: Mapping[str, Any]) -> Any:
    forward = batch["forward"]
    current = model.forward_frame(
        forward["current_image"],
        forward["current_camera_origin_body_m"],
        forward["current_camera_basis_body_fru"],
        forward["current_ground_plane_z_body_m"],
    )
    next_frame = model.forward_frame(
        forward["next_image"],
        forward["next_camera_origin_body_m"],
        forward["next_camera_basis_body_fru"],
        forward["next_ground_plane_z_body_m"],
    )
    overlap = runtime.torch.ones_like(
        current.bev[:, :1], dtype=runtime.torch.bool
    )
    return runtime.model_module.SharedTrainingPairV5(
        current=current,
        next=next_frame,
        predicted_next_bev=next_frame.bev,
        stop_gradient_target_next_bev=next_frame.bev.detach(),
        commanded_warped_current_bev=current.bev,
        commanded_overlap_mask=overlap,
        realized_warped_current_bev=current.bev,
        realized_overlap_mask=overlap,
        jepa=None,
    )


def _phase_b_loss_components(loss: Any) -> dict[str, Any]:
    result = {"camera_total": loss.total}
    for side in ("current", "next"):
        frame = getattr(loss, side)
        result.update({
            f"{side}_hierarchical_first_hit_nll":
                frame.hierarchical_first_hit_nll,
            f"{side}_tail_depth_p95_cvar":
                frame.tail_depth_p95_cvar,
            f"{side}_ground_clear_distance_state_balanced_bce":
                frame.ground_clear_distance_state_balanced_bce,
            f"{side}_derived_raster_hierarchical_bce":
                frame.derived_raster_hierarchical_bce.total,
            f"{side}_derived_raster_cell_nll":
                frame.derived_raster_cell_nll,
        })
    return result


def _phase_b_diagnostics(
    runtime: Any,
    trainer: Any,
    model: Any,
    frozen_sha256: str,
    selection_pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    update: int,
) -> dict[str, Any]:
    before = _state_sha(runtime, model)

    def observe() -> tuple[dict[str, Any], float]:
        _set_phase_b_mode(model, training=False)
        return trainer.physical_metrics(
            model,
            selection_pairs,
            device,
            arm="rgb_jepa_encoder_pretraining_phase_b",
            stage=f"phase_b_diagnostic_update_{update}",
        )

    try:
        physical, camera_loss = _run_with_rng_preserved(runtime, observe)
    finally:
        _set_phase_b_mode(model, training=True)
    after = _state_sha(runtime, model)
    if (
        before != after
        or _subset_sha(
            runtime, model, contract.PHASE_B_FROZEN_PARAMETER_PREFIXES
        )
        != frozen_sha256
    ):
        raise RuntimeError("Phase-B diagnostics mutated model state")
    full_evaluation = contract.evaluate_physical_scopes(physical)
    summary = {
        "complete_physical_scope_count":
            full_evaluation["complete_physical_scope_count"],
        "margin_count": full_evaluation["margin_count"],
        "passed_margin_count": full_evaluation["passed_margin_count"],
        "total_shortfall": full_evaluation["total_shortfall"],
        "rough_motion": full_evaluation["rough_motion"],
    }
    gate = contract.evaluate_phase_b(summary)
    return {
        "update": update,
        "role": "checkpoint_selection",
        "physical_scopes": physical,
        "aggregate_complete_v4_tail_depth_loss": float(camera_loss),
        "evaluation": full_evaluation,
        "terminal_conjunction": gate,
        "informational_only": update in (100, 400),
        "model_state_sha256_before_and_after": before,
        "frozen_state_sha256_before_and_after": frozen_sha256,
        "rng_state_preserved": True,
        "state_mutation_count": 0,
    }


def _phase_b_train(
    runtime: Any,
    multires: Any,
    fit: Any,
    phase_a_encoder_state: Mapping[str, Any],
    trainer: Any,
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    schedule: Sequence[int],
    device: Any,
    output_root: Path,
    *,
    gpu_started: float,
    progress: dict[str, Any],
) -> dict[str, Any]:
    torch = runtime.torch
    model, head, frozen, initialization = _phase_b_model(
        runtime,
        multires,
        fit,
        phase_a_encoder_state,
        device,
    )
    frozen_sha256 = _subset_sha(
        runtime, model, contract.PHASE_B_FROZEN_PARAMETER_PREFIXES
    )
    optimizer = torch.optim.AdamW(
        [{"params": list(head), "lr": contract.learning_rates(1)[0]}],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for update in range(1, contract.PHASE_B_MAXIMUM_UPDATE + 1):
        _check_gpu_time(
            gpu_started,
            maximum_minutes=
                contract.CUMULATIVE_GPU_ACTIVE_TIME_CAP_MINUTES,
            stage="cumulative Phase A plus Phase B",
        )
        learning_rate = contract.learning_rates(update)[0]
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, Any] = {}
        start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
        update_indices = list(
            schedule[start : start + contract.EFFECTIVE_BATCH_SIZE]
        )
        if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
            raise PermissionError("Phase-B schedule ended early")
        for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
            low = microbatch * contract.MICROBATCH_SIZE
            indices = update_indices[low : low + contract.MICROBATCH_SIZE]
            batch = _phase_b_batch(
                trainer,
                train_pairs,
                indices,
                device,
                role="train",
                stage="phase_b_camera_gradient",
            )
            pair = _phase_b_pair(runtime, model, batch)
            camera = runtime.loss_adapter.observable_camera_ray_v4_loss_v4(
                model,
                pair,
                batch["current_supervision"],
                batch["next_supervision"],
            )
            if not bool(torch.isfinite(camera.total).item()):
                raise FloatingPointError("Phase-B objective became nonfinite")
            (camera.total / contract.MICROBATCHES_PER_UPDATE).backward()
            for name, value in _phase_b_loss_components(camera).items():
                contribution = (
                    value.detach() / contract.MICROBATCHES_PER_UPDATE
                )
                sums[name] = sums.get(name, contribution.new_zeros(())) + contribution
        if any(parameter.grad is not None for parameter in frozen):
            raise RuntimeError("a frozen Phase-B parameter acquired a gradient")
        gradient_before = torch.nn.utils.clip_grad_norm_(head, max_norm=1.0)
        if not bool(torch.isfinite(gradient_before).item()):
            raise FloatingPointError("Phase-B gradient became nonfinite")
        gradient_after = _scalar(torch.stack([
            parameter.grad.detach().float().square().sum()
            for parameter in head
            if parameter.grad is not None
        ]).sum().sqrt())
        if gradient_after > 1.00001:
            raise RuntimeError("Phase-B head clip norm changed")
        optimizer.step()
        progress["phase_b_updates"] = update
        progress["phase_b_presentations"] = (
            update * contract.EFFECTIVE_BATCH_SIZE
        )
        trace.append({
            "schema": f"{contract.SCHEMA_PREFIX}_phase_b_trace_row_v1",
            "update": update,
            "presentation_indices_sha256":
                contract.canonical_json_sha256(update_indices),
            "evidence_head_learning_rate": learning_rate,
            "microbatch_count": contract.MICROBATCHES_PER_UPDATE,
            "pair_presentations":
                update * contract.EFFECTIVE_BATCH_SIZE,
            "camera_objective_count":
                update * contract.MICROBATCHES_PER_UPDATE,
            "backward_count":
                update * contract.MICROBATCHES_PER_UPDATE,
            "optimizer_step_count": update,
            "head_clip_count": update,
            "gradient_norm_before_clip": _scalar(gradient_before),
            "gradient_norm_after_clip": gradient_after,
            "losses": {
                name: value
                for name, value in zip(
                    sums,
                    torch.stack(tuple(sums.values())).cpu().tolist(),
                    strict=True,
                )
            },
            "jepa_objective_count": 0,
            "ema_update_count": 0,
        })
        if update in contract.CHECKPOINT_UPDATES:
            if (
                _subset_sha(
                    runtime,
                    model,
                    contract.PHASE_B_FROZEN_PARAMETER_PREFIXES,
                )
                != frozen_sha256
            ):
                raise RuntimeError("Phase-B frozen state changed")
            snapshots.append(
                _snapshot_model(
                    runtime,
                    model,
                    output_root,
                    phase="phase_b",
                    update=update,
                    metadata={
                        "initialization": initialization,
                        "frozen_state_sha256": frozen_sha256,
                    },
                )
            )
            metrics.append(
                _phase_b_diagnostics(
                    runtime,
                    trainer,
                    model,
                    frozen_sha256,
                    selection_pairs,
                    device,
                    update=update,
                )
            )
        _check_gpu_time(
            gpu_started,
            maximum_minutes=
                contract.CUMULATIVE_GPU_ACTIVE_TIME_CAP_MINUTES,
            stage="cumulative Phase A plus Phase B",
        )
    terminal_gate = metrics[-1]["terminal_conjunction"]
    if (
        _subset_sha(
            runtime, model, contract.PHASE_B_FROZEN_PARAMETER_PREFIXES
        )
        != frozen_sha256
    ):
        raise RuntimeError("Phase-B terminal frozen state changed")
    trace_raw = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in trace
    )
    _write_exclusive(output_root / "phase_b/training_trace.jsonl", trace_raw)
    metrics_value, metrics_raw = _publish_json(
        output_root / "phase_b/metrics.json",
        {
            "schema": contract.PHASE_B_METRICS_SCHEMA,
            "status": (
                "PASS_PHASE_B"
                if terminal_gate["passed"]
                else "FAIL_PHASE_B_TERMINAL"
            ),
            "observations": metrics,
            "selection_evaluation_updates": list(contract.CHECKPOINT_UPDATES),
            "observer_rerun_count": 0,
            "rng_state_preserved_at_every_observation": True,
            "terminal_gate": terminal_gate,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    result = {
        "schema": f"{contract.SCHEMA_PREFIX}_phase_b_artifact_v1",
        "status": (
            "PASS_FROZEN_ENCODER_PHYSICAL_PROBE"
            if terminal_gate["passed"]
            else "FAIL_FROZEN_ENCODER_PHYSICAL_PROBE_TERMINAL"
        ),
        "initialization": initialization,
        "frozen_state_sha256": frozen_sha256,
        "final_evidence_head_state_sha256":
            _state_sha(runtime, model.evidence_head),
        "snapshots": snapshots,
        "metrics":
            _binding("phase_b/metrics.json", metrics_value, metrics_raw),
        "training_trace": {
            "path": "phase_b/training_trace.jsonl",
            "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
            "byte_count": len(trace_raw),
            "row_count": len(trace),
            "content_sha256": contract.canonical_json_sha256(trace),
        },
        "updates": len(trace),
        "presentations": len(trace) * contract.EFFECTIVE_BATCH_SIZE,
        "jepa_objective_count": 0,
        "ema_update_count": 0,
        "gate": terminal_gate,
        "checkpoint_qualified": False,
        "perception_only_nonpromotable": True,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    model.to("cpu")
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def _terminal_inventory(output_root: Path) -> dict[str, Any]:
    files: list[str] = []
    directories: list[str] = ["."]
    for current, names, filenames in os.walk(output_root):
        current_path = Path(current)
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise PermissionError("terminal output contains a nondirectory")
            directories.append(path.relative_to(output_root).as_posix())
        for name in filenames:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise PermissionError("terminal output contains a nonregular file")
            files.append(path.relative_to(output_root).as_posix())
    return {
        "files": sorted(files),
        "directories_including_root": sorted(directories),
    }


def _error_evidence(error: BaseException) -> dict[str, str]:
    message = str(error)
    return {
        "type": type(error).__name__,
        "message": message,
        "message_sha256": hashlib.sha256(
            message.encode("utf-8")
        ).hexdigest(),
    }


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    *,
    error: BaseException,
    progress: Mapping[str, Any],
) -> None:
    """Publish one complete best-effort terminal failure chain and seal it."""
    if (output_root / "completed.json").exists():
        return
    failure, failure_raw = _publish_json(
        output_root / "failure.json",
        {
            "schema": contract.FAILURE_SCHEMA,
            "status": "TERMINAL_INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY",
            "reservation":
                _binding("reservation.json", reservation, reservation_raw),
            "progress": dict(progress),
            "error": _error_evidence(error),
            "phase_b_entered":
                bool(progress.get("phase_b_entered", False)),
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
            "status": "TERMINAL_FAILURE",
            "attempt_identity": reservation["attempt_identity"],
            "failure": _binding("failure.json", failure, failure_raw),
            "exact_precompletion_files": inventory["files"],
            "exact_terminal_files":
                sorted([*inventory["files"], "completed.json"]),
            "exact_terminal_directories_including_root":
                inventory["directories_including_root"],
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    _seal_terminal(output_root)


def _terminal_runtime_rehash(
    authorization: Mapping[str, Any],
) -> list[dict[str, Any]]:
    runtime_inputs = authorization["runtime_inputs"]
    bindings = (
        runtime_inputs["raw"]["manifest"],
        runtime_inputs["raw"]["audit"],
        runtime_inputs["camera"]["gate"],
        runtime_inputs["camera"]["checkpoint"],
        runtime_inputs["schedule"],
    )
    records = []
    for binding in bindings:
        raw = _read_bound(ROOT / binding["path"], binding)
        records.append({
            **dict(binding),
            "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
            "observed_byte_count": len(raw),
        })
    return records


def _execute_after_reservation(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    preflight_sha256: str,
    output_root: Path,
    progress: dict[str, Any],
) -> int:
    progress["stage"] = "post_reservation_preflight_validation"
    preflight = _validate_preflight_after_reservation(
        expected_sha256=preflight_sha256,
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

    progress["stage"] = "post_reservation_source_authority_rehash"
    if contract.current_source_bindings(ROOT) != dict(sources):
        raise PermissionError("reviewed source changed across reservation")
    observed_review = contract.validate_review(
        contract.parse_canonical_json(review_raw, name="source review rehash"),
        expected_sources=sources,
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    observed_authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            authorization_raw, name="execution authorization rehash"
        ),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    if (
        observed_review != dict(review)
        or observed_authorization != dict(authorization)
    ):
        raise PermissionError("authority changed across reservation")

    progress["stage"] = "deferred_runtime_import"
    matched, runtime, schedule_adapter, phase2d, multires = (
        _load_post_reservation_stack(sources)
    )
    runtime_authority = authorization["runtime_inputs"]
    adapted_authorization = {
        "raw": runtime_authority["raw"],
        "camera": runtime_authority["camera"],
    }
    progress["stage"] = "raw_authority_and_index_validation"
    inputs = matched.RawInputs(runtime, adapted_authorization)
    _normalize_endpoint_paths(inputs)
    trainer = matched.Trainer(runtime, inputs, output_root, reservation)
    train_pairs = inputs.role_pairs("train")
    selection_pairs = inputs.role_pairs("checkpoint_selection")
    if (
        len(train_pairs) != contract.TRAIN_ROLE_COUNTS["pairs"]
        or len(selection_pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
    ):
        raise PermissionError("development role population changed")
    schedule, schedule_receipt = _load_schedule(
        schedule_adapter,
        authorization,
        train_pairs,
    )

    progress["stage"] = "reserved_runtime_device_validation"
    gpu_started = time.monotonic()
    device, hardware = trainer.device()
    if (
        hardware["visible_device_count"] != 1
        or "r9700" not in hardware["name"].casefold().replace(" ", "")
        or hardware["name"] != preflight["visible_device_name"]
        or hardware["total_memory_bytes"] != preflight["total_memory_bytes"]
    ):
        raise PermissionError("runtime GPU differs from isolated preflight")
    progress["gpu_active_started"] = True

    progress["stage"] = "n320_initialization_checkpoint_load"
    fit, gate, camera_binding = matched._camera_model_after_reservation(
        runtime, adapted_authorization
    )
    progress["n320_checkpoint_loaded"] = True
    progress["stage"] = "phase_a"
    loader = RGBOnlyLoader(runtime, inputs)
    (
        (phase_a_model, phase_a),
        phase_a_determinism_receipt,
    ) = _run_phase_a_with_reviewed_determinism(
        runtime,
        lambda: _phase_a_train(
            runtime,
            phase2d,
            fit,
            loader,
            train_pairs,
            selection_pairs,
            schedule,
            device,
            output_root,
            gpu_started=gpu_started,
            progress=progress,
        ),
    )
    phase_a["determinism_compatibility"] = (
        phase_a_determinism_receipt
    )
    progress["phase_a_updates"] = phase_a["updates"]
    progress["phase_a_presentations"] = phase_a["presentations"]
    progress["phase_a_passed"] = bool(phase_a["gate"]["passed"])
    phase_a_value, phase_a_raw = _publish_json(
        output_root / "phase_a/artifact.json",
        phase_a,
    )

    phase_b: dict[str, Any] | None = None
    phase_b_binding: dict[str, Any] | None = None
    if phase_a["gate"]["passed"]:
        progress["phase_b_entered"] = True
        phase_a_encoder_state = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in phase_a_model.encoder.state_dict().items()
        }
        if (
            _state_sha(runtime, phase_a_encoder_state)
            != phase_a["terminal_online_encoder_state_sha256"]
        ):
            raise RuntimeError("Phase-A terminal encoder transfer changed")
        phase_a_model.to("cpu")
        del phase_a_model
        loader.cache.clear()
        runtime.torch.cuda.empty_cache()
        progress["stage"] = "phase_b"
        phase_b, determinism_receipt = (
            _run_phase_b_with_reviewed_determinism(
                runtime,
                lambda: _phase_b_train(
                    runtime,
                    multires,
                    fit,
                    phase_a_encoder_state,
                    trainer,
                    train_pairs,
                    selection_pairs,
                    schedule,
                    device,
                    output_root,
                    gpu_started=gpu_started,
                    progress=progress,
                ),
            )
        )
        phase_b["determinism_compatibility"] = determinism_receipt
        progress["phase_b_updates"] = phase_b["updates"]
        progress["phase_b_presentations"] = phase_b["presentations"]
        progress["phase_b_passed"] = bool(phase_b["gate"]["passed"])
        phase_b_value, phase_b_raw = _publish_json(
            output_root / "phase_b/artifact.json",
            phase_b,
        )
        phase_b_binding = _binding(
            "phase_b/artifact.json", phase_b_value, phase_b_raw
        )
    else:
        progress["phase_b_entered"] = False
        phase_a_model.to("cpu")
        del phase_a_model
        loader.cache.clear()
        runtime.torch.cuda.empty_cache()
    del fit

    progress["stage"] = "terminal_input_rehash"
    consumed = inputs.rehash_consumed()
    consumed_roles = {
        role
        for record in consumed["records"]
        for role in record["roles"]
    }
    permitted_roles = {
        "authority",
        "index",
        "train",
        "checkpoint_selection",
    }
    if (
        "probability_calibration" in consumed_roles
        or not consumed_roles.issubset(permitted_roles)
        or not {"train", "checkpoint_selection"}.issubset(consumed_roles)
        or contract.current_source_bindings(ROOT) != dict(sources)
    ):
        raise PermissionError("experiment consumed an unauthorized role or source")
    runtime_rehash = _terminal_runtime_rehash(authorization)
    access, access_raw = _publish_json(
        output_root / "access.json",
        {
            "schema": contract.ACCESS_SCHEMA,
            "status": "ALL_CONSUMED_DEVELOPMENT_INPUTS_REHASHED",
            "reservation":
                _binding("reservation.json", reservation, reservation_raw),
            "roles_opened": ["train", "checkpoint_selection"],
            "phase_a": {
                "dedicated_rgb_only_loader": True,
                "general_raw_v13_frame_loader_call_count":
                    loader.general_frame_loader_call_count,
                "camera_supervision_array_open_count":
                    loader.supervision_array_open_count,
            },
            "phase_b_entered": bool(progress["phase_b_entered"]),
            "consumed": consumed,
            "fixed_runtime_input_rehash": runtime_rehash,
            "schedule": schedule_receipt,
            "n320": {
                "gate_content_sha256": gate["content_sha256"],
                "checkpoint": camera_binding,
                "initialization_only": True,
            },
            "reviewed_sources": {
                "count": len(sources),
                "bindings": dict(sources),
                "all_rehashed": True,
            },
            "probability_calibration_open_count": 0,
            "prior_runtime_output_open_count": 0,
            "rejected_checkpoint_open_count": 0,
            "g2_navigation_heldout_or_sealed_open_count": 0,
            "all_consumed_inputs_rehashed": True,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )

    passed = bool(phase_b is not None and phase_b["gate"]["passed"])
    if not phase_a["gate"]["passed"]:
        phase_a_control = str(phase_a["gate"]["control"])
        if phase_a_control in {
            contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
            contract.CONTROL_PHASE_A_UPDATE_400_FAIL,
        }:
            status = phase_a_control
        else:
            status = "FAIL_PHASE_A_MECHANISM_TERMINATED_NO_PHASE_B"
        terminal_control = phase_a["gate"]
    elif not passed:
        status = "FAIL_PHASE_B_MECHANISM_TERMINATED"
        terminal_control = phase_b["gate"] if phase_b is not None else None
    else:
        status = "PASS_BOUNDED_FALSIFICATION_SEPARATE_QUALIFICATION_ONLY"
        terminal_control = phase_b["gate"]
    progress["stage"] = "result_publication"
    result, result_raw = _publish_json(
        output_root / "result.json",
        {
            "schema": contract.RESULT_SCHEMA,
            "status": status,
            "reservation":
                _binding("reservation.json", reservation, reservation_raw),
            "access": _binding("access.json", access, access_raw),
            "phase_a":
                _binding("phase_a/artifact.json", phase_a_value, phase_a_raw),
            "phase_b": phase_b_binding,
            "phase_b_entered": bool(progress["phase_b_entered"]),
            "terminal_control": terminal_control,
            "operation_counts": {
                "optimizer_updates":
                    phase_a["updates"]
                    + (0 if phase_b is None else phase_b["updates"]),
                "pair_presentations":
                    phase_a["presentations"]
                    + (0 if phase_b is None else phase_b["presentations"]),
                "phase_a_ema_updates": phase_a["ema_update_count"],
                "phase_b_jepa_objectives":
                    0 if phase_b is None else phase_b["jepa_objective_count"],
                "observer_reruns": 0,
            },
            "gpu_active_elapsed_seconds":
                time.monotonic() - gpu_started,
            "checkpoint_qualified": False,
            "pass_authorizes":
                "separate_bounded_qualification_preregistration_only"
                if passed else "nothing",
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    inventory = _terminal_inventory(output_root)
    progress["stage"] = "completion_publication"
    _publish_json(
        output_root / "completed.json",
        {
            "schema": contract.COMPLETION_SCHEMA,
            "status": (
                status
                if status in {
                    contract.CONTROL_PHASE_A_UPDATE_100_FAIL,
                    contract.CONTROL_PHASE_A_UPDATE_400_FAIL,
                }
                else ("TERMINAL_PASS" if passed else "TERMINAL_FAIL")
            ),
            "attempt_identity": reservation["attempt_identity"],
            "result": _binding("result.json", result, result_raw),
            "phase_b_entered": bool(progress["phase_b_entered"]),
            "exact_precompletion_files": inventory["files"],
            "exact_terminal_files":
                sorted([*inventory["files"], "completed.json"]),
            "exact_terminal_directories_including_root":
                inventory["directories_including_root"],
            "all_inputs_rehashed": True,
            "all_terminal_files_sealed_read_only": True,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        },
    )
    progress["stage"] = "terminal_sealing"
    _seal_terminal(output_root)
    return 0 if passed else 2


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
    preflight_file_sha256: str,
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
        "n320_checkpoint_loaded": False,
        "phase_a_updates": 0,
        "phase_a_presentations": 0,
        "phase_a_passed": False,
        "phase_b_entered": False,
        "phase_b_updates": 0,
        "phase_b_presentations": 0,
        "phase_b_passed": False,
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
            preflight_sha256=preflight_file_sha256,
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
                "experiment failed and terminal failure publication also failed"
            ) from receipt_error
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("execution requires --run")
    for name in (
        "review_sha256",
        "authorization_sha256",
        "preflight_sha256",
    ):
        if not contract.is_sha256(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be an exact SHA-256")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
        preflight_file_sha256=args.preflight_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
