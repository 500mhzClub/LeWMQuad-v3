#!/usr/bin/env python3
"""Run the one bounded RGB overlapping-tokenization V1 falsification.

This is a thin additive adapter over the corrected causal-motion V1 custody
lifecycle.  Import remains source-only: Torch, generated inputs, RGB, and
checkpoints are deferred until the inherited authority and reservation gates
have passed.  The scientific path itself is static and carries no temporal or
motion condition.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_OVERLAPPING_TOKENIZATION_V1_PREFLIGHT_JSON"
)
ARM_NAME = "overlapping_tokenization_v1"
INHERITED_TRAIN_ARM_NAME = "causal_temporal_perception_v1"
_INHERITED_SELECTION_SENTINEL = "temporal_population_pre_model"
_FIXED_TRAINING_COMPLETE = (
    "FIXED_TRAINING_COMPLETE_PENDING_FINALIZED_LEDGER_PARSE"
)
_FORBIDDEN_DYNAMIC_FIELDS = (
    "causal_motion_condition",
    "history_valid",
    "nominal_delta_current_frame",
    "relative_se2",
    "temporal_population",
    "warm_scopes_informational_only",
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
    "_lewm_go2_rgb_overlapping_tokenization_v1_contract",
    ROOT / "lewm/benchmarks/go2_rgb_overlapping_tokenization_v1.py",
)
_MOTION = _source_only_module(
    "_lewm_go2_rgb_overlapping_tokenization_v1_corrected_motion_lifecycle",
    ROOT / "scripts/run_go2_rgb_causal_motion_alignment_v1.py",
)
_BASE = _MOTION._BASE

# The corrected lifecycle functions resolve their globals in these two private
# modules.  Replace only the successor identity before any function can run.
_MOTION.contract = contract
_MOTION.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY
_MOTION._FINALIZED_LEDGER_PARSE_RECEIPT = None
_BASE.contract = contract
_BASE.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY

OperationProgress = _BASE.OperationProgress
PartialAccessLedger = _BASE.PartialAccessLedger
_TEMPORAL_TRAIN = _MOTION._TEMPORAL_TRAIN
_FINALIZED_LEDGER_PARSE_RECEIPT: dict[str, Any] | None = None


def _load_post_reservation_stack(
    sources: Mapping[str, str],
) -> tuple[Any, Any, Any, Any]:
    """First Torch-capable import point, after exact reservation."""

    model_source_sha256 = sources.get(contract.MODEL_RELATIVE_PATH)
    matched_source_sha256 = sources.get(
        contract.MATCHED_V1_RUNNER_RELATIVE_PATH
    )
    if (
        not contract.is_sha256(model_source_sha256)
        or not contract.is_sha256(matched_source_sha256)
    ):
        raise PermissionError("reviewed runtime source binding is incomplete")
    matched_path = ROOT / contract.MATCHED_V1_RUNNER_RELATIVE_PATH
    matched_raw = _BASE._read_regular(
        matched_path,
        expected_sha256=matched_source_sha256,
    )
    if not matched_raw:
        raise PermissionError("matched V1 reusable runner is empty")
    matched = _BASE._load_path(
        "_lewm_overlapping_tokenization_matched_v1_loader",
        matched_path,
    )
    base_runtime = matched._load_runtime()

    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_multires_overlapping_tokenization_v1
            as overlap,
        )
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth
            as tail_depth,
        )
    finally:
        sys.path[:] = original_path
    expected_model = ROOT / contract.MODEL_RELATIVE_PATH
    observed_model = Path(overlap.__file__)
    if (
        observed_model.is_symlink()
        or expected_model.is_symlink()
        or observed_model.resolve() != expected_model.resolve()
    ):
        raise PermissionError(
            "imported overlapping-tokenization model source changed"
        )
    _BASE._read_regular(
        observed_model,
        expected_sha256=str(model_source_sha256),
    )
    loss_adapter = SimpleNamespace(
        observable_camera_ray_v4_loss_v4=(
            tail_depth.observable_camera_ray_v4_tail_depth_loss_v4
        )
    )
    runtime = SimpleNamespace(
        **{
            **vars(base_runtime),
            "loss_adapter": loss_adapter,
        }
    )
    schedule_adapter = _BASE._load_path(
        "_lewm_overlapping_tokenization_v1_schedule_adapter",
        ROOT / contract.SCHEDULE_ADAPTER_RELATIVE_PATH,
    )
    return matched, runtime, overlap, schedule_adapter


def _receipt_dict(value: Any) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        observed = to_dict()
    elif is_dataclass(value):
        observed = asdict(value)
    elif type(value) is dict:
        observed = dict(value)
    else:
        raise TypeError("migration receipt is not structured")
    if type(observed) is not dict:
        raise TypeError("migration receipt did not normalize to a dict")
    return observed


def _state_sha(runtime: Any, state_or_model: Any) -> str:
    state = (
        state_or_model.state_dict()
        if hasattr(state_or_model, "state_dict")
        else state_or_model
    )
    return runtime.model_module.tensor_state_dict_sha256(state)


def _validate_migration_receipt(
    runtime: Any,
    overlap: Any,
    model: Any,
    fit: Any,
    value: object,
) -> dict[str, Any]:
    """Validate the one versioned 7x7-to-11x11 N320 migration."""

    receipt = _receipt_dict(value)
    transformed = ["encoder.patch_embed.weight"]
    n320_derived = sorted((
        *(f"encoder.{name}" for name in model.encoder.state_dict()),
        *(
            f"evidence_head.pixel_head.{name}"
            for name in model.evidence_head.pixel_head.state_dict()
        ),
        *(
            f"evidence_head.ground_head.{name}"
            for name in model.evidence_head.ground_head.state_dict()
        ),
    ))
    exact_copied = [
        name for name in n320_derived if name not in transformed
    ]
    old_weight = fit.encoder.patch_embed.weight.detach()
    new_weight = model.encoder.patch_embed.weight.detach()
    outer = new_weight.clone()
    outer[:, :, 2:9, 2:9] = 0
    expected_architecture = (
        contract.overlapping_tokenization_architecture_contract_v1()
    )
    observed_architecture = (
        overlap.overlapping_tokenization_architecture_contract_v1()
    )
    required = {
        "schema",
        "model_family",
        "base_initialization_seed",
        "decoder_initialization_seed",
        "initialization_input_role",
        "n320_checkpoint_file_sha256",
        "n320_checkpoint_content_sha256",
        "fit_model_state_sha256",
        "shared_encoder_state_sha256",
        "pixel_head_state_sha256",
        "ground_head_state_sha256",
        "decoder_state_sha256",
        "evidence_head_state_sha256",
        "copied_state_keys",
        "exact_copy_state_keys",
        "exact_copy_state_entry_count",
        "transformed_state_keys",
        "transformed_state_entry_count",
        "retained_n320_derived_entry_count",
        "source_patch_weight_shape",
        "destination_patch_weight_shape",
        "center_copy_slice",
        "central_weight_scalar_count",
        "outer_ring_scalar_count",
        "patch_bias_scalar_count",
        "central_copy_exact",
        "outer_ring_exact_zero",
        "patch_bias_exact_copy",
        "copied_predecessor_dense_decoder_entry_count",
        "canonical_ground_support_exact",
        "hard_sync_count",
        "caller_cpu_rng_restored",
        "replacement_module_caller_cpu_rng_restored",
        "rejected_adaptation_checkpoint_open_count",
        "torch_version",
    }
    if (
        set(receipt) != required
        or receipt["schema"] != overlap.INITIALIZATION_SCHEMA
        or receipt["model_family"] != overlap.MODEL_FAMILY
        or overlap.MODEL_FAMILY != contract.MODEL_FAMILY
        or receipt["base_initialization_seed"]
        != contract.BASE_INITIALIZATION_SEED
        or receipt["decoder_initialization_seed"]
        != contract.DECODER_INITIALIZATION_SEED
        or receipt["initialization_input_role"]
        != "n320_fit_initialization_only"
        or receipt["n320_checkpoint_file_sha256"]
        != contract.RUNTIME_FILE_SHA256[
            contract.N320_CHECKPOINT_RELATIVE_PATH
        ]
        or receipt["n320_checkpoint_content_sha256"]
        != contract.RUNTIME_CONTENT_SHA256[
            contract.N320_CHECKPOINT_RELATIVE_PATH
        ]
        or receipt["fit_model_state_sha256"] != _state_sha(runtime, fit)
        or receipt["shared_encoder_state_sha256"]
        != _state_sha(runtime, model.encoder)
        or receipt["pixel_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head.pixel_head)
        or receipt["ground_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head.ground_head)
        or receipt["decoder_state_sha256"]
        != _state_sha(runtime, model.evidence_head.dense_decoder)
        or receipt["evidence_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head)
        or receipt["copied_state_keys"] != n320_derived
        or receipt["exact_copy_state_keys"] != exact_copied
        or receipt["exact_copy_state_entry_count"] != 83
        or receipt["transformed_state_keys"] != transformed
        or receipt["transformed_state_entry_count"] != 1
        or receipt["retained_n320_derived_entry_count"] != 84
        or receipt["source_patch_weight_shape"] != [192, 3, 7, 7]
        or receipt["destination_patch_weight_shape"] != [192, 3, 11, 11]
        or receipt["center_copy_slice"] != [2, 9, 2, 9]
        or receipt["central_weight_scalar_count"] != 28_224
        or receipt["outer_ring_scalar_count"] != 41_472
        or receipt["patch_bias_scalar_count"] != 192
        or receipt["central_copy_exact"] is not True
        or receipt["outer_ring_exact_zero"] is not True
        or receipt["patch_bias_exact_copy"] is not True
        or not bool(runtime.torch.equal(
            new_weight[:, :, 2:9, 2:9], old_weight
        ))
        or int(runtime.torch.count_nonzero(outer).item()) != 0
        or not bool(runtime.torch.equal(
            model.encoder.patch_embed.bias.detach(),
            fit.encoder.patch_embed.bias.detach(),
        ))
        or receipt["copied_predecessor_dense_decoder_entry_count"] != 0
        or receipt["canonical_ground_support_exact"] is not True
        or receipt["hard_sync_count"] != 1
        or receipt["caller_cpu_rng_restored"] is not True
        or receipt["replacement_module_caller_cpu_rng_restored"] is not True
        or receipt["rejected_adaptation_checkpoint_open_count"] != 0
        or receipt["torch_version"] != str(runtime.torch.__version__)
        or expected_architecture != observed_architecture
        or contract.canonical_json_sha256(observed_architecture)
        != contract.ARCHITECTURE_CONTRACT_SHA256
        or not bool(getattr(model, "_n320_initialization_complete", False))
        or _state_sha(runtime, model.encoder)
        != _state_sha(runtime, model.target_encoder)
    ):
        raise PermissionError(
            "N320 overlapping-tokenization initialization receipt changed"
        )
    return receipt


class _StaticPartition(dict[str, Any]):
    """Drop the inherited temporal sentinel; serialize only static science."""

    inherited_selection_sentinel_neutralized_count: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.inherited_selection_sentinel_neutralized_count = 0

    def __setitem__(self, key: str, value: Any) -> None:
        if key == _INHERITED_SELECTION_SENTINEL:
            if value is not None:
                raise PermissionError(
                    "inherited selection sentinel carried dynamic state"
                )
            self.inherited_selection_sentinel_neutralized_count += 1
            if self.inherited_selection_sentinel_neutralized_count != 1:
                raise PermissionError(
                    "inherited selection sentinel was applied more than once"
                )
            return
        super().__setitem__(key, value)


def _assert_static_payload(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if any(field in normalized for field in _FORBIDDEN_DYNAMIC_FIELDS):
                raise PermissionError(f"{name} retained dynamic field {key!r}")
            _assert_static_payload(item, name=name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_static_payload(item, name=name)


def _prepare_model(
    runtime: Any,
    overlap: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, list[Any], list[Any], list[Any], dict[str, Any]]:
    caller_rng = runtime.torch.random.get_rng_state().clone()
    model, raw_migration = (
        overlap.SharedObservableCameraRayJepaV5MultiresOverlappingTokenizationV1
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
    if not bool(runtime.torch.equal(
        caller_rng, runtime.torch.random.get_rng_state()
    )):
        raise RuntimeError("N320 model initialization changed caller CPU RNG")
    migration = _validate_migration_receipt(
        runtime, overlap, model, fit, raw_migration
    )
    declared_trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_trainable = {
        name for name, _parameter in model.named_parameters()
        if name.startswith(contract.TRAINABLE_PARAMETER_PREFIXES)
    }
    if declared_trainable != expected_trainable:
        raise PermissionError("constructor trainable partition changed")
    model = model.to(device)
    model.requires_grad_(False)
    groups: dict[str, list[tuple[str, Any]]] = {
        "evidence_head": [],
        "encoder": [],
        "frozen": [],
    }
    for state_name in model.state_dict():
        contract.parameter_partition(state_name)
    for parameter_name, parameter in model.named_parameters():
        component = contract.parameter_partition(parameter_name)
        if component in ("evidence_head", "encoder"):
            parameter.requires_grad_(True)
            groups[component].append((parameter_name, parameter))
        else:
            groups["frozen"].append((parameter_name, parameter))
    counts = {
        name: sum(parameter.numel() for _, parameter in groups[name])
        for name in ("evidence_head", "encoder")
    }
    tensor_counts = {
        name: len(groups[name]) for name in ("evidence_head", "encoder")
    }
    if (
        counts != contract.EXPECTED_PARAMETER_COUNTS
        or tensor_counts != contract.EXPECTED_PARAMETER_TENSOR_COUNTS
        or not groups["frozen"]
        or any(parameter.requires_grad for _, parameter in groups["frozen"])
    ):
        raise PermissionError(
            "overlapping-tokenization trainable/frozen partition changed"
        )
    names = {
        name: [parameter_name for parameter_name, _ in values]
        for name, values in groups.items()
    }
    architecture = (
        contract.overlapping_tokenization_architecture_contract_v1()
    )
    partition: _StaticPartition = _StaticPartition({
        "parameter_counts": counts,
        "parameter_tensor_counts": tensor_counts,
        "parameter_names_sha256": {
            name: contract.canonical_json_sha256(values)
            for name, values in names.items()
        },
        "migration": migration,
        "architecture_contract": architecture,
        "architecture_contract_sha256":
            contract.ARCHITECTURE_CONTRACT_SHA256,
        "model_runtime_version": contract.MODEL_RUNTIME_VERSION,
        "initial_state_sha256": _state_sha(runtime, model),
    })
    _assert_static_payload(partition, name="model partition")
    return (
        model,
        [parameter for _, parameter in groups["evidence_head"]],
        [parameter for _, parameter in groups["encoder"]],
        [parameter for _, parameter in groups["frozen"]],
        partition,
    )


def _neutralize_inherited_selection_sentinel(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any], None]:
    """Validate the static population without constructing temporal history."""

    endpoints = {
        str(pair[key])
        for pair in pairs
        for key in ("current_endpoint_sha256", "next_endpoint_sha256")
    }
    if (
        len(pairs) != contract.SELECTION_ROLE_COUNTS["pairs"]
        or len(endpoints)
        != contract.SELECTION_ROLE_COUNTS["unique_endpoints"]
        or any(
            pair.get("dataset_role") != "checkpoint_selection"
            for pair in pairs
        )
    ):
        raise PermissionError("static selection population changed")
    return (), {}, None


def _visual_only_batch(
    trainer: Any,
    pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: Any,
    *,
    role: str,
    arm: str,
    stage: str,
) -> dict[str, Any]:
    """Materialize only RGB, camera geometry, and frozen supervision."""

    if arm != INHERITED_TRAIN_ARM_NAME:
        raise PermissionError("inherited training arm identity changed")
    selected = [pairs[index] for index in indices]
    if any(item["dataset_role"] != role for item in selected):
        raise PermissionError("static visual-only batch crossed dataset roles")
    current = [
        trainer.inputs.frame(
            str(item["current_endpoint_sha256"]),
            role=role,
            arm=ARM_NAME,
            stage=stage,
        )
        for item in selected
    ]
    next_ = [
        trainer.inputs.frame(
            str(item["next_endpoint_sha256"]),
            role=role,
            arm=ARM_NAME,
            stage=stage,
        )
        for item in selected
    ]

    def stack(frames: Sequence[Mapping[str, Any]], field: str) -> Any:
        return trainer.r.torch.stack(
            [item[field] for item in frames]
        ).to(device)

    batch = {
        "forward": {
            "current_image": stack(current, "image"),
            "next_image": stack(next_, "image"),
            "current_camera_origin_body_m": stack(
                current, "camera_origin"
            ).float(),
            "current_camera_basis_body_fru": stack(
                current, "camera_basis"
            ).float(),
            "current_ground_plane_z_body_m": stack(
                current, "ground"
            ).float(),
            "next_camera_origin_body_m": stack(
                next_, "camera_origin"
            ).float(),
            "next_camera_basis_body_fru": stack(
                next_, "camera_basis"
            ).float(),
            "next_ground_plane_z_body_m": stack(
                next_, "ground"
            ).float(),
        },
        "current_supervision": trainer.supervision(current, device),
        "next_supervision": trainer.supervision(next_, device),
    }
    _assert_static_payload(batch["forward"], name="training model batch")
    return batch


def _camera_pair(runtime: Any, model: Any, batch: Mapping[str, Any]) -> Any:
    """Run two independent static frames under the unchanged Camera loss."""

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


def _snapshot(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    update: int,
    frozen_sha256: str,
    initial_state_sha256: str,
    migration: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    state_sha256 = _state_sha(runtime, state)
    frozen_observed = _state_sha(runtime, {
        name: value
        for name, value in state.items()
        if name.startswith(contract.FROZEN_STATE_PREFIXES)
    })
    if frozen_observed != frozen_sha256:
        raise RuntimeError("frozen state changed before snapshot")
    architecture = (
        contract.overlapping_tokenization_architecture_contract_v1()
    )
    semantic = {
        "schema": contract.SNAPSHOT_SCHEMA,
        "update": update,
        "model_family": contract.MODEL_FAMILY,
        "model_config": model.model_config.to_dict(),
        "architecture_contract": architecture,
        "architecture_contract_sha256":
            contract.ARCHITECTURE_CONTRACT_SHA256,
        "state_sha256": state_sha256,
        "frozen_state_sha256": frozen_sha256,
        "initial_state_sha256": initial_state_sha256,
        "migration": dict(migration),
        "schedule_prefix_indices_sha256":
            contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[update],
        "development_only": True,
        "resume_authorized": False,
        "runtime_ready": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    _assert_static_payload(semantic, name="checkpoint snapshot")
    contract.validate_no_preledger_scientific_control(semantic)
    content_sha256 = contract.canonical_json_sha256(semantic)
    buffer = io.BytesIO()
    runtime.torch.save({
        **semantic,
        "content_sha256": content_sha256,
        "model_state_dict": state,
    }, buffer)
    raw = buffer.getvalue()
    relative = f"checkpoints/update_{update}.pt"
    _BASE._write_exclusive(output_root / relative, raw)
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
        "state_sha256": state_sha256,
        "frozen_state_sha256": frozen_sha256,
    }


def _evaluate(
    runtime: Any,
    trainer: Any,
    model: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    update: int,
    frozen_sha256: str,
) -> dict[str, Any]:
    before = _state_sha(runtime, model)
    if (
        _BASE._subset_sha(
            runtime, model, contract.FROZEN_STATE_PREFIXES
        )
        != frozen_sha256
    ):
        raise RuntimeError("frozen state changed before inline evaluation")
    model.eval()
    physical, camera_loss = trainer.physical_metrics(
        model,
        selection_pairs,
        device,
        arm=ARM_NAME,
        stage=f"inline_checkpoint_selection_update_{update}",
    )
    model.train()
    after = _state_sha(runtime, model)
    frozen_after = _BASE._subset_sha(
        runtime, model, contract.FROZEN_STATE_PREFIXES
    )
    if before != after or frozen_after != frozen_sha256:
        raise RuntimeError("inline evaluation mutated model state")
    evaluation = contract.evaluate_physical_scopes(physical)
    metric = {
        "update": update,
        "role": "checkpoint_selection",
        "pair_count": contract.SELECTION_ROLE_COUNTS["pairs"],
        "unique_endpoint_count":
            contract.SELECTION_ROLE_COUNTS["unique_endpoints"],
        "scopes": physical,
        "aggregate_complete_v4_tail_depth_loss": float(camera_loss),
        "evaluation": evaluation,
        "preledger_model_state_checks_pass": True,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "frozen_state_sha256_before_and_after": frozen_sha256,
        "state_mutation_count": 0,
    }
    _assert_static_payload(metric, name="inline static evaluation")
    return metric


class _DeferredTerminalControl(dict[str, Any]):
    """Materialize scientific control only after the corrected disk parse."""

    def __init__(self, evaluation: Mapping[str, Any]) -> None:
        super().__init__()
        self._evaluation = dict(evaluation)

    def _materialize(self) -> None:
        if self:
            return
        receipt = _FINALIZED_LEDGER_PARSE_RECEIPT
        if (
            type(receipt) is not dict
            or receipt.get("terminal_record_type")
            != "RUNTIME_INPUT_ACCESS_FINALIZED"
            or receipt.get("corrected_parser_pass") is not True
        ):
            raise PermissionError(
                "terminal control requested before finalized-ledger parse"
            )
        self.update(contract.checkpoint_control_decision(
            update=contract.MAXIMUM_UPDATE,
            evaluation=self._evaluation,
            integrity_pass=True,
        ))

    def __getitem__(self, key: str) -> Any:
        self._materialize()
        return super().__getitem__(key)


def _inherited_fixed_flow_control(update: int) -> dict[str, str]:
    if update not in contract.CHECKPOINT_UPDATES:
        raise ValueError("fixed-flow checkpoint update changed")
    return {
        "action": (
            contract.CONTROL_CONTINUE
            if update in (100, 400)
            else _FIXED_TRAINING_COMPLETE
        )
    }


class _InheritedTrainingContract:
    """Expose a neutral terminal marker to the reused fixed training loop."""

    def __getattr__(self, name: str) -> Any:
        if name in {"CONTROL_PASS", "CONTROL_FAIL"}:
            return _FIXED_TRAINING_COMPLETE
        return getattr(contract, name)


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Publish static provisional evidence without computing PASS or FAIL."""

    if metric.get("preledger_model_state_checks_pass") is not True:
        raise PermissionError("preledger model-state checks changed")
    core = {
        "schema": contract.METRIC_SIDECAR_SCHEMA,
        "status":
            "PROVISIONAL_INADMISSIBLE_PENDING_FINALIZED_LEDGER_PARSE",
        "update": update,
        "checkpoint": dict(checkpoint),
        "metric": dict(metric),
        "inline_evaluation_count": 1,
        "state_mutation_count": 0,
        "publication_order": [
            "cpu_snapshot",
            "inline_nonmutating_selection_evaluation",
            "atomic_mode_0444_provisional_sidecar",
            "internal_fixed_training_flow_only",
        ],
        "continuation": contract.provisional_checkpoint_control(update),
        "scientifically_admissible": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    value = contract.with_content_sha256(core)
    raw = contract.canonical_json_bytes(value) + b"\n"
    relative = contract.metric_sidecar_relative_path(update)
    contract.validate_metric_sidecar(value, update=update)
    _BASE._publish_readonly_atomic(output_root / relative, raw)
    return (
        _BASE._binding(relative, value, raw),
        _inherited_fixed_flow_control(update),
    )


def _train(
    runtime: Any,
    trainer: Any,
    model: Any,
    head: Sequence[Any],
    encoder: Sequence[Any],
    frozen: Sequence[Any],
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    device: Any,
    output_root: Path,
    partition: Mapping[str, Any],
    progress: OperationProgress,
) -> dict[str, Any]:
    if (
        not isinstance(partition, _StaticPartition)
        or partition.inherited_selection_sentinel_neutralized_count != 1
        or _INHERITED_SELECTION_SENTINEL in partition
    ):
        raise PermissionError(
            "inherited selection sentinel was not exactly neutralized"
        )
    _assert_static_payload(partition, name="pre-training partition")
    inherited_contract = _BASE.contract
    if inherited_contract is not contract:
        raise PermissionError("inherited training contract identity changed")
    _BASE.contract = _InheritedTrainingContract()
    try:
        result = _TEMPORAL_TRAIN(
            runtime,
            trainer,
            model,
            head,
            encoder,
            frozen,
            train_pairs,
            selection_pairs,
            indices,
            device,
            output_root,
            partition,
            progress,
        )
    finally:
        _BASE.contract = inherited_contract
    internal_controls = result.get("controls")
    if (
        [row["update"] for row in result["metrics"]]
        != list(contract.CHECKPOINT_UPDATES)
        or any("integrity_pass" in row for row in result["metrics"])
        or internal_controls
        != [
            _inherited_fixed_flow_control(update)
            for update in contract.CHECKPOINT_UPDATES
        ]
    ):
        raise PermissionError("preledger checkpoint evidence changed")
    result["controls"] = [
        contract.provisional_checkpoint_control(update)
        for update in contract.CHECKPOINT_UPDATES
    ]
    result["terminal_control"] = _DeferredTerminalControl(
        result["metrics"][-1]["evaluation"]
    )
    _assert_static_payload(result, name="static training result")
    return result


def _ledger_binding(self: Any) -> dict[str, Any]:
    """Mirror the corrected finalized-parse receipt into this adapter."""

    global _FINALIZED_LEDGER_PARSE_RECEIPT
    value = _MOTION._ledger_binding(self)
    receipt = _MOTION._FINALIZED_LEDGER_PARSE_RECEIPT
    if type(receipt) is dict:
        _FINALIZED_LEDGER_PARSE_RECEIPT = dict(receipt)
    return value


# Corrected motion-V1 integrity/publication functions resolve against the
# overlap contract after the identity patch above.
_publish_training_records = _MOTION._publish_training_records
_terminal_failure = _MOTION._terminal_failure
_terminal_pre_ledger_failure = _MOTION._terminal_pre_ledger_failure
_terminal_contract_invalid_ledger_failure = (
    _MOTION._terminal_contract_invalid_ledger_failure
)

PartialAccessLedger.binding = _ledger_binding
_BASE._load_post_reservation_stack = _load_post_reservation_stack
_BASE._receipt_dict = _receipt_dict
_BASE._state_sha = _state_sha
_BASE._validate_migration_receipt = _validate_migration_receipt
_BASE._prepare_model = _prepare_model
_BASE._selection_temporal_index = (
    _neutralize_inherited_selection_sentinel
)
_BASE._visual_only_batch = _visual_only_batch
_BASE._camera_pair = _camera_pair
_BASE._snapshot = _snapshot
_BASE._evaluate = _evaluate
_BASE._publish_metric_sidecar = _publish_metric_sidecar
_BASE._publish_training_records = _publish_training_records
_BASE._train = _train
_BASE._terminal_failure = _terminal_failure
_BASE._terminal_pre_ledger_failure = _terminal_pre_ledger_failure

run_parent = _BASE.run_parent
parse_args = _BASE.parse_args


def main(argv: Sequence[str] | None = None) -> int:
    return _BASE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
