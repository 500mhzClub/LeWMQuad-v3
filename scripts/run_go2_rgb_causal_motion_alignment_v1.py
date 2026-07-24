#!/usr/bin/env python3
"""Run the one bounded RGB causal-motion-alignment V1 falsification.

This module is a deliberately thin additive adapter over the frozen causal
temporal V1 lifecycle.  Import remains source-only: Torch, generated inputs,
RGB, and checkpoints are deferred until the inherited exact authority and
reservation gates have passed.
"""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import hashlib
import importlib.util
import io
import math
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ENVIRONMENT_KEY = "LEWM_CAUSAL_MOTION_ALIGNMENT_V1_PREFLIGHT_JSON"
PRIMITIVE_VOCABULARY = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
ARM_NAME = "causal_motion_alignment_v1"
INHERITED_TRAIN_ARM_NAME = "causal_temporal_perception_v1"


def _source_only_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _source_only_module(
    "_lewm_go2_rgb_causal_motion_alignment_v1_contract",
    ROOT / "lewm/benchmarks/go2_rgb_causal_motion_alignment_v1.py",
)
_BASE = _source_only_module(
    "_lewm_go2_rgb_causal_motion_alignment_v1_temporal_lifecycle",
    ROOT / "scripts/run_go2_rgb_causal_temporal_perception_v1.py",
)

# Every inherited lifecycle function resolves its globals in this private
# module.  Patch the version identity once, before any function can run.
_BASE.contract = contract
_BASE.PREFLIGHT_ENVIRONMENT_KEY = PREFLIGHT_ENVIRONMENT_KEY

OperationProgress = _BASE.OperationProgress
PartialAccessLedger = _BASE.PartialAccessLedger
_ORIGINAL_LEDGER_APPEND_TERMINAL = PartialAccessLedger.append_terminal
_ORIGINAL_LEDGER_BINDING = PartialAccessLedger.binding
_TEMPORAL_TRAIN = _BASE._train
_TEMPORAL_PUBLISH_TRAINING_RECORDS = _BASE._publish_training_records
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
        "_lewm_causal_motion_alignment_matched_v1_loader",
        matched_path,
    )
    base_runtime = matched._load_runtime()

    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_multires_motion_alignment_v1
            as motion,
        )
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth
            as tail_depth,
        )
    finally:
        sys.path[:] = original_path
    expected_model = ROOT / contract.MODEL_RELATIVE_PATH
    observed_model = Path(motion.__file__)
    if (
        observed_model.is_symlink()
        or expected_model.is_symlink()
        or observed_model.resolve() != expected_model.resolve()
    ):
        raise PermissionError("imported motion-alignment model source changed")
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
        "_lewm_causal_motion_alignment_v2_schedule_adapter",
        ROOT / contract.SCHEDULE_ADAPTER_RELATIVE_PATH,
    )
    return matched, runtime, motion, schedule_adapter


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
    motion: Any,
    model: Any,
    fit: Any,
    value: object,
) -> dict[str, Any]:
    """Validate the additive N320 receipt without accepting prior probe state."""

    receipt = _receipt_dict(value)
    required = {
        "schema",
        "model_family",
        "base_initialization_seed",
        "decoder_initialization_seed",
        "temporal_initialization_seed",
        "alignment_initialization_seed",
        "initialization_input_role",
        "n320_checkpoint_file_sha256",
        "n320_checkpoint_content_sha256",
        "fit_model_state_sha256",
        "shared_encoder_state_sha256",
        "pixel_head_state_sha256",
        "ground_head_state_sha256",
        "decoder_state_sha256",
        "temporal_state_sha256",
        "alignment_state_sha256",
        "evidence_head_state_sha256",
        "copied_state_keys",
        "copied_state_entry_count",
        "copied_predecessor_dense_decoder_entry_count",
        "copied_temporal_entry_count",
        "copied_alignment_entry_count",
        "temporal_output_projection_exact_zero",
        "alignment_offset_projection_exact_zero",
        "canonical_ground_support_exact",
        "hard_sync_count",
        "caller_cpu_rng_restored",
        "rejected_adaptation_checkpoint_open_count",
        "torch_version",
    }
    copied = receipt.get("copied_state_keys")
    expected_copied = sorted((
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
    alignment = model.evidence_head.motion_alignment
    temporal = model.evidence_head.temporal_residual
    if (
        set(receipt) != required
        or receipt["schema"] != motion.INITIALIZATION_SCHEMA
        or receipt["model_family"] != motion.MODEL_FAMILY
        or receipt["base_initialization_seed"]
        != contract.BASE_INITIALIZATION_SEED
        or receipt["decoder_initialization_seed"]
        != contract.DECODER_INITIALIZATION_SEED
        or receipt["temporal_initialization_seed"]
        != contract.TEMPORAL_INITIALIZATION_SEED
        or receipt["alignment_initialization_seed"]
        != contract.ALIGNMENT_INITIALIZATION_SEED
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
        or receipt["temporal_state_sha256"]
        != _state_sha(runtime, temporal)
        or receipt["alignment_state_sha256"]
        != _state_sha(runtime, alignment)
        or receipt["evidence_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head)
        or type(copied) is not list
        or copied != expected_copied
        or len(copied) != 84
        or len(set(copied)) != 84
        or receipt["copied_state_entry_count"] != 84
        or receipt["copied_predecessor_dense_decoder_entry_count"] != 0
        or receipt["copied_temporal_entry_count"] != 0
        or receipt["copied_alignment_entry_count"] != 0
        or receipt["temporal_output_projection_exact_zero"] is not True
        or receipt["alignment_offset_projection_exact_zero"] is not True
        or int(runtime.torch.count_nonzero(
            temporal.output_projection.weight
        ).item()) != 0
        or int(runtime.torch.count_nonzero(
            alignment.offset_projection.weight
        ).item()) != 0
        or any(
            "dense_decoder" in name
            or "temporal_residual" in name
            or "motion_alignment" in name
            for name in copied
        )
        or receipt["canonical_ground_support_exact"] is not True
        or receipt["hard_sync_count"] != 1
        or receipt["caller_cpu_rng_restored"] is not True
        or receipt["rejected_adaptation_checkpoint_open_count"] != 0
        or receipt["torch_version"] != str(runtime.torch.__version__)
        or not bool(getattr(model, "_n320_initialization_complete", False))
        or _state_sha(runtime, model.encoder)
        != _state_sha(runtime, model.target_encoder)
    ):
        raise PermissionError(
            "N320 motion-alignment initialization receipt changed"
        )
    return receipt


def _prepare_model(
    runtime: Any,
    motion: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, list[Any], list[Any], list[Any], dict[str, Any]]:
    caller_rng = runtime.torch.random.get_rng_state().clone()
    model, raw_migration = (
        motion.SharedObservableCameraRayJepaV5MultiresMotionAlignmentV1
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
        runtime, motion, model, fit, raw_migration
    )
    if getattr(motion, "MODEL_FAMILY", None) != contract.MODEL_FAMILY:
        raise PermissionError("motion-alignment model runtime identity changed")
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
    for name in model.state_dict():
        contract.parameter_partition(name)
    for name, parameter in model.named_parameters():
        component = contract.parameter_partition(name)
        if component in ("evidence_head", "encoder"):
            parameter.requires_grad_(True)
            groups[component].append((name, parameter))
        else:
            groups["frozen"].append((name, parameter))
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
            "motion-alignment trainable/frozen partition changed"
        )
    names = {
        name: [parameter_name for parameter_name, _ in values]
        for name, values in groups.items()
    }
    partition = {
        "parameter_counts": counts,
        "parameter_tensor_counts": tensor_counts,
        "parameter_names_sha256": {
            name: contract.canonical_json_sha256(values)
            for name, values in names.items()
        },
        "migration": migration,
        "model_runtime_version": contract.MODEL_RUNTIME_VERSION,
        "initial_state_sha256": _state_sha(runtime, model),
    }
    return (
        model,
        [parameter for _, parameter in groups["evidence_head"]],
        [parameter for _, parameter in groups["encoder"]],
        [parameter for _, parameter in groups["frozen"]],
        partition,
    )


def _build_nominal_command_table(
    runtime: Any,
    train_pairs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], Any, dict[str, Any]]:
    """Aggregate realized training labels once; expose only nine medians."""

    if not train_pairs or any(
        pair.get("dataset_role") != "train" for pair in train_pairs
    ):
        raise PermissionError("nominal command table crossed dataset roles")
    vocabulary = tuple(sorted({
        str(pair.get("primitive")) for pair in train_pairs
    }))
    if vocabulary != PRIMITIVE_VOCABULARY:
        raise PermissionError("train primitive vocabulary changed")
    rows = []
    counts: dict[str, int] = {}
    for primitive in vocabulary:
        values = [
            pair.get("relative_se2_current_frame")
            for pair in train_pairs
            if pair.get("primitive") == primitive
        ]
        counts[primitive] = len(values)
        tensor = runtime.torch.tensor(
            values,
            dtype=runtime.torch.float32,
            device="cpu",
        )
        if (
            tensor.ndim != 2
            or tensor.shape[0] <= 0
            or tensor.shape[1] != 3
            or not bool(runtime.torch.isfinite(tensor).all().item())
        ):
            raise PermissionError("train nominal command rows changed")
        rows.append(runtime.torch.quantile(tensor, 0.5, dim=0))
    table = runtime.torch.stack(rows)
    table_values = [
        [float(component) for component in row]
        for row in table.tolist()
    ]
    receipt = {
        "source_role": "train",
        "primitive_vocabulary": list(vocabulary),
        "primitive_row_counts": counts,
        "aggregation": "float32_torch_quantile_0.5_dim_0",
        "table": table_values,
        "table_sha256": contract.canonical_json_sha256(table_values),
        "per_sample_realized_se2_model_batch_count": 0,
        "selection_rows_contributed": 0,
    }
    return vocabulary, table, receipt


def _require_motion_table(trainer: Any) -> tuple[dict[str, int], Any]:
    mapping = getattr(trainer, "_motion_primitive_to_index", None)
    table = getattr(trainer, "_motion_nominal_table", None)
    if (
        type(mapping) is not dict
        or set(mapping) != set(PRIMITIVE_VOCABULARY)
        or table is None
        or tuple(table.shape) != (9, 3)
    ):
        raise RuntimeError("motion command table is not installed")
    return mapping, table


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
    """Materialize RGB/supervision plus only the aggregate nominal delta."""

    if arm != INHERITED_TRAIN_ARM_NAME:
        raise PermissionError("inherited training arm identity changed")
    selected = [pairs[index] for index in indices]
    if any(item["dataset_role"] != role for item in selected):
        raise PermissionError("motion-alignment batch crossed dataset roles")
    primitive_to_index, table = _require_motion_table(trainer)
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

    def stack(frames: Sequence[Mapping[str, Any]], name: str) -> Any:
        return trainer.r.torch.stack([item[name] for item in frames]).to(device)

    primitive_indices = trainer.r.torch.tensor(
        [primitive_to_index[str(item["primitive"])] for item in selected],
        dtype=trainer.r.torch.long,
        device=table.device,
    )
    nominal = table.index_select(0, primitive_indices).to(device)
    if (
        tuple(nominal.shape) != (len(selected), 3)
        or nominal.dtype != trainer.r.torch.float32
        or not bool(trainer.r.torch.isfinite(nominal).all().item())
    ):
        raise PermissionError("nominal motion batch changed")
    forward = {
        "current_image": stack(current, "image"),
        "next_image": stack(next_, "image"),
        "nominal_delta_current_frame": nominal,
        "current_camera_origin_body_m": stack(
            current, "camera_origin"
        ).float(),
        "current_camera_basis_body_fru": stack(
            current, "camera_basis"
        ).float(),
        "current_ground_plane_z_body_m": stack(current, "ground").float(),
        "next_camera_origin_body_m": stack(
            next_, "camera_origin"
        ).float(),
        "next_camera_basis_body_fru": stack(next_, "camera_basis").float(),
        "next_ground_plane_z_body_m": stack(next_, "ground").float(),
    }
    if any("realized" in key or "relative_se2" in key for key in forward):
        raise RuntimeError("per-sample realized motion entered model batch")
    return {
        "forward": forward,
        "current_supervision": trainer.supervision(current, device),
        "next_supervision": trainer.supervision(next_, device),
    }


def _camera_pair(runtime: Any, model: Any, batch: Mapping[str, Any]) -> Any:
    forward = batch["forward"]
    current, next_frame = model.forward_camera_pair(
        previous_image=forward["current_image"],
        current_image=forward["next_image"],
        previous_camera_origin_body_m=(
            forward["current_camera_origin_body_m"]
        ),
        previous_camera_basis_body_fru=(
            forward["current_camera_basis_body_fru"]
        ),
        previous_ground_plane_z_body_m=(
            forward["current_ground_plane_z_body_m"]
        ),
        current_camera_origin_body_m=(
            forward["next_camera_origin_body_m"]
        ),
        current_camera_basis_body_fru=(
            forward["next_camera_basis_body_fru"]
        ),
        current_ground_plane_z_body_m=(
            forward["next_ground_plane_z_body_m"]
        ),
        nominal_delta_current_frame=(
            forward["nominal_delta_current_frame"]
        ),
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
    semantic = {
        "schema": contract.SNAPSHOT_SCHEMA,
        "update": update,
        "model_family": contract.MODEL_FAMILY,
        "model_config": model.model_config.to_dict(),
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


def _motion_physical_metrics(
    runtime: Any,
    trainer: Any,
    model: Any,
    pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    arm: str,
    stage: str,
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    """Run unchanged physical math with target-owned causal motion context."""

    torch = runtime.torch
    correct = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    wrong = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    warm_correct = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    warm_wrong = {
        scope: runtime.MetricAccumulator() for scope in contract.SCOPES
    }
    ids_by_family, predecessor_by_target, population = (
        _BASE._selection_temporal_index(pairs)
    )
    primitive_to_index, table = _require_motion_table(trainer)
    loss_sum = 0.0
    frame_count = 0
    warm_frame_count = 0

    def packet(
        endpoint_id: str,
        *,
        family: str,
    ) -> tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any] | None,
    ]:
        current = trainer.inputs.frame(
            endpoint_id,
            role="checkpoint_selection",
            arm=arm,
            stage=stage,
        )
        pair = predecessor_by_target.get(endpoint_id)
        if pair is None:
            return current, current, None
        if (
            str(pair["family"]) != family
            or str(pair["next_endpoint_sha256"]) != endpoint_id
        ):
            raise PermissionError(
                "motion-alignment predecessor crossed family or direction"
            )
        previous = trainer.inputs.frame(
            str(pair["current_endpoint_sha256"]),
            role="checkpoint_selection",
            arm=arm,
            stage=stage,
        )
        return previous, current, pair

    def stack(frames: Sequence[Mapping[str, Any]], name: str) -> Any:
        return torch.stack([item[name] for item in frames]).to(device)

    with torch.no_grad():
        for family, ids in ids_by_family.items():
            wrong_ids = ids[1:] + ids[:1]
            for start in range(0, len(ids), contract.MICROBATCH_SIZE):
                target_ids = ids[
                    start : start + contract.MICROBATCH_SIZE
                ]
                mapped_ids = wrong_ids[
                    start : start + contract.MICROBATCH_SIZE
                ]
                target_packets = [
                    packet(item, family=family) for item in target_ids
                ]
                mapped_packets = [
                    packet(item, family=family) for item in mapped_ids
                ]
                target_previous = [item[0] for item in target_packets]
                target_current = [item[1] for item in target_packets]
                mapped_previous = [item[0] for item in mapped_packets]
                mapped_current = [item[1] for item in mapped_packets]
                incoming_pairs = [item[2] for item in target_packets]
                history_valid = torch.tensor(
                    [item is not None for item in incoming_pairs],
                    dtype=torch.bool,
                    device=device,
                )
                nominal_rows = []
                for incoming in incoming_pairs:
                    if incoming is None:
                        nominal_rows.append(
                            torch.zeros(
                                3,
                                dtype=torch.float32,
                                device=table.device,
                            )
                        )
                    else:
                        primitive = str(incoming["primitive"])
                        nominal_rows.append(
                            table[primitive_to_index[primitive]]
                        )
                nominal = torch.stack(nominal_rows).to(device)
                if (
                    tuple(nominal.shape) != (len(target_ids), 3)
                    or not bool(torch.isfinite(nominal).all().item())
                    or (
                        bool((~history_valid).any().item())
                        and int(torch.count_nonzero(
                            nominal[~history_valid]
                        ).item()) != 0
                    )
                ):
                    raise PermissionError(
                        "selection causal-motion condition changed"
                    )
                previous_basis = stack(
                    target_previous, "camera_basis"
                ).float()
                origin = stack(target_current, "camera_origin").float()
                basis = stack(target_current, "camera_basis").float()
                ground = stack(target_current, "ground").float()
                supervision = trainer.supervision(target_current, device)
                targets = runtime.derive_targets(
                    pixel_hit_mask=supervision.pixel_hit_mask,
                    pixel_first_hit_distance_m=(
                        supervision.pixel_first_hit_distance_m
                    ),
                    ground_support_in_frustum=(
                        supervision.ground_support_in_frustum
                    ),
                    ground_support_clear_to_target=(
                        supervision.ground_support_clear_to_target
                    ),
                )
                observations = (
                    (target_previous, target_current),
                    (mapped_previous, mapped_current),
                )
                outputs = []
                for previous_frames, current_frames in observations:
                    online = model.forward_temporal_frame(
                        previous_image=stack(previous_frames, "image"),
                        current_image=stack(current_frames, "image"),
                        previous_camera_basis_body_fru=previous_basis,
                        target_camera_origin_body_m=origin,
                        target_camera_basis_body_fru=basis,
                        target_ground_plane_z_body_m=ground,
                        nominal_delta_current_frame=nominal,
                        history_valid=history_valid,
                    )
                    soft = runtime.soft_rasterize(
                        online.evidence,
                        camera_origin_body_m=origin,
                        camera_basis_body_fru=basis,
                        pixel_ray_chunk_size=(
                            model.model_config.v4_pixel_ray_chunk_size
                        ),
                    )
                    outputs.append((online, soft))

                for accumulator_set, warm_set, output in zip(
                    (correct, wrong),
                    (warm_correct, warm_wrong),
                    outputs,
                    strict=True,
                ):
                    online, soft = output
                    for scope in ("aggregate", family):
                        accumulator_set[scope].update(
                            raw_output=online.evidence,
                            targets=targets,
                            soft_raster=soft,
                            target_raster_labels=(
                                supervision.target_raster_labels
                            ),
                            families=[family] * len(target_ids),
                        )
                    for index, is_warm in enumerate(
                        history_valid.detach().cpu().tolist()
                    ):
                        if not is_warm:
                            continue
                        selected = slice(index, index + 1)
                        for scope in ("aggregate", family):
                            warm_set[scope].update(
                                raw_output=_BASE._slice_batch_dataclass(
                                    online.evidence, selected
                                ),
                                targets=_BASE._slice_batch_dataclass(
                                    targets, selected
                                ),
                                soft_raster=_BASE._slice_batch_dataclass(
                                    soft, selected
                                ),
                                target_raster_labels=(
                                    supervision.target_raster_labels[selected]
                                ),
                                families=[family],
                            )

                camera = (
                    runtime.loss_adapter.observable_camera_ray_v4_loss_v4(
                        model,
                        trainer._single_frame_pair(outputs[0][0]),
                        supervision,
                        supervision,
                        require_b4=False,
                    )
                )
                loss_sum += float(camera.total.cpu()) * len(target_ids)
                frame_count += len(target_ids)
                warm_frame_count += int(history_valid.sum().item())

    if (
        frame_count != contract.SELECTION_ROLE_COUNTS["unique_endpoints"]
        or warm_frame_count
        != contract.SELECTION_ROLE_COUNTS["warm_endpoints"]
    ):
        raise PermissionError("motion-alignment evaluator frame counts changed")
    metrics = {
        scope: trainer._flatten_physical(
            correct[scope].finalize(), wrong[scope].finalize()
        )
        for scope in contract.SCOPES
    }
    warm_metrics = {
        scope: trainer._flatten_physical(
            warm_correct[scope].finalize(),
            warm_wrong[scope].finalize(),
        )
        for scope in contract.SCOPES
    }
    return metrics, warm_metrics, loss_sum / frame_count, population


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
    physical, warm_physical, camera_loss, population = (
        _motion_physical_metrics(
            runtime,
            trainer,
            model,
            selection_pairs,
            device,
            arm="causal_motion_alignment_v1",
            stage=f"inline_checkpoint_selection_update_{update}",
        )
    )
    model.train()
    after = _state_sha(runtime, model)
    frozen_after = _BASE._subset_sha(
        runtime, model, contract.FROZEN_STATE_PREFIXES
    )
    if before != after or frozen_after != frozen_sha256:
        raise RuntimeError("inline evaluation mutated model state")
    evaluation = contract.evaluate_physical_scopes(physical)
    return {
        "update": update,
        "role": "checkpoint_selection",
        "pair_count": contract.SELECTION_ROLE_COUNTS["pairs"],
        "unique_endpoint_count":
            contract.SELECTION_ROLE_COUNTS["unique_endpoints"],
        "temporal_population": population,
        "scopes": physical,
        "warm_scopes_informational_only": warm_physical,
        "aggregate_complete_v4_tail_depth_loss": float(camera_loss),
        "evaluation": evaluation,
        "preledger_model_state_checks_pass": True,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "frozen_state_sha256_before_and_after": frozen_sha256,
        "state_mutation_count": 0,
    }


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish metrics without pre-claiming ledger integrity or PASS/FAIL."""

    if metric.get("preledger_model_state_checks_pass") is not True:
        raise PermissionError("preledger model-state checks changed")
    internal_control = contract.checkpoint_control_decision(
        update=update,
        evaluation=metric["evaluation"],
        integrity_pass=True,
    )
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
    return _BASE._binding(relative, value, raw), internal_control


def _publish_training_records(
    output_root: Path,
    training: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collate only provisional checkpoint evidence before ledger finalization."""

    trace = training["trace"]
    contract.validate_no_preledger_scientific_control(trace)
    trace_raw = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in trace
    )
    _BASE._write_exclusive(output_root / "training_trace.jsonl", trace_raw)
    trace_binding = {
        "path": "training_trace.jsonl",
        "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
        "content_sha256": contract.canonical_json_sha256(trace),
        "byte_count": len(trace_raw),
        "row_count": len(trace),
    }
    controls = list(training["controls"])
    if controls != [
        contract.provisional_checkpoint_control(update)
        for update in contract.CHECKPOINT_UPDATES
    ]:
        raise PermissionError("provisional checkpoint flows changed")
    metrics = list(training["metrics"])
    if len(metrics) != len(contract.CHECKPOINT_UPDATES):
        raise PermissionError("provisional checkpoint metric count changed")
    for update, metric in zip(
        contract.CHECKPOINT_UPDATES, metrics, strict=True
    ):
        contract.validate_provisional_metric(metric, update=update)
    checkpoint_core = {
        "schema": contract.CHECKPOINT_METRICS_SCHEMA,
        "status":
            "PROVISIONAL_INADMISSIBLE_PENDING_FINALIZED_LEDGER_PARSE",
        "checkpoint_updates": list(contract.CHECKPOINT_UPDATES),
        "rows": metrics,
        "sidecars": list(training["sidecars"]),
        "controls": controls,
        "inline_evaluation_count": 3,
        "observer_evaluation_rerun_count": 0,
        "threshold_equality_passes": False,
        "scientifically_admissible": False,
        "integrity_or_pass_fail_control_emitted": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    contract.validate_no_preledger_scientific_control(checkpoint_core)
    value, raw = _BASE._publish_json(
        output_root / "checkpoint_metrics.json",
        checkpoint_core,
    )
    return trace_binding, _BASE._binding(
        "checkpoint_metrics.json", value, raw
    )


class _DeferredTerminalControl(dict[str, Any]):
    """Materialize terminal scientific control only after the full disk parse."""

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
    vocabulary, table_cpu, table_receipt = _build_nominal_command_table(
        runtime, train_pairs
    )
    if type(partition) is not dict:
        raise TypeError("motion-alignment partition must be mutable")
    partition["causal_motion_condition"] = table_receipt
    trainer._motion_primitive_to_index = {
        primitive: index for index, primitive in enumerate(vocabulary)
    }
    trainer._motion_nominal_table = table_cpu.to(device)
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
        del trainer._motion_primitive_to_index
        del trainer._motion_nominal_table
    if (
        [row["update"] for row in result["metrics"]]
        != list(contract.CHECKPOINT_UPDATES)
        or any("integrity_pass" in row for row in result["metrics"])
    ):
        raise PermissionError("preledger checkpoint evidence changed")
    result["controls"] = [
        contract.provisional_checkpoint_control(update)
        for update in contract.CHECKPOINT_UPDATES
    ]
    result["terminal_control"] = _DeferredTerminalControl(
        result["metrics"][-1]["evaluation"]
    )
    return result


def _ledger_append_terminal(
    self: Any,
    *,
    record_type: str,
    stage: Mapping[str, Any],
    operation_counts: Mapping[str, Any],
    error: BaseException | None,
) -> None:
    """Dry-parse a success terminal before making it durable."""

    if record_type == "RUNTIME_INPUT_ACCESS_FINALIZED":
        if error is not None:
            raise ValueError("successful ledger terminal cannot carry an error")
        candidate = self._record_value({
            "record_type": record_type,
            "stage": dict(stage),
            "operation_counts": dict(operation_counts),
            "error": None,
        })
        candidate_raw = contract.canonical_json_bytes(candidate) + b"\n"
        parsed = contract.parse_partial_access_ledger(
            b"".join(self.raw_parts) + candidate_raw
        )
        if (
            len(parsed) != len(self.records) + 1
            or parsed[-1] != candidate
        ):
            raise PermissionError(
                "corrected full-ledger parser rejected success terminal"
            )
    _ORIGINAL_LEDGER_APPEND_TERMINAL(
        self,
        record_type=record_type,
        stage=stage,
        operation_counts=operation_counts,
        error=error,
    )


def _ledger_binding(self: Any) -> dict[str, Any]:
    global _FINALIZED_LEDGER_PARSE_RECEIPT

    value = _ORIGINAL_LEDGER_BINDING(self)
    raw = b"".join(self.raw_parts)
    parsed = contract.parse_partial_access_ledger(raw)
    if (
        len(parsed) != value["record_count"]
        or contract.canonical_json_sha256(parsed)
        != value["records_content_sha256"]
        or parsed[-1]["record_type"]
        not in {
            "RUNTIME_INPUT_ACCESS_FINALIZED",
            "ATTEMPT_TERMINATING",
        }
    ):
        raise PermissionError("finalized full-ledger parser receipt changed")
    if parsed[-1]["record_type"] == "RUNTIME_INPUT_ACCESS_FINALIZED":
        _FINALIZED_LEDGER_PARSE_RECEIPT = {
            "corrected_parser_pass": True,
            "full_on_disk_ledger_checked": True,
            "ledger_file_sha256": value["file_sha256"],
            "record_count": value["record_count"],
            "last_record_content_sha256":
                value["last_record_content_sha256"],
            "terminal_record_type": parsed[-1]["record_type"],
        }
    return value


def _error_evidence(error: BaseException) -> dict[str, str]:
    return PartialAccessLedger._error(error)


def _failure_core(
    *,
    schema: str,
    status: str,
    scientific_result_status: str,
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    ledger_binding: Mapping[str, Any],
    runtime_opens: Sequence[Mapping[str, Any]],
    progress: OperationProgress,
    operation_counts: Mapping[str, Any],
    error: BaseException,
    ledger_parser_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prefix = _BASE._terminal_file_bindings(
        output_root,
        exclude=("failed.json", ".failed.json.publishing"),
    )
    _files, directories = _BASE._terminal_inventory(
        output_root,
        exclude=("failed.json", ".failed.json.publishing"),
    )
    core = {
        "schema": schema,
        "status": status,
        "attempt_identity": reservation["attempt_identity"],
        "reservation": _BASE._binding(
            "reservation.json", reservation, reservation_raw
        ),
        "partial_access_ledger": dict(ledger_binding),
        "runtime_opens": [dict(row) for row in runtime_opens],
        "runtime_opens_sha256":
            contract.canonical_json_sha256(runtime_opens),
        "failure_stage": progress.location(),
        "operation_counts": dict(operation_counts),
        "published_prefix": prefix,
        "published_prefix_sha256":
            contract.canonical_json_sha256(prefix),
        "directories_including_root": directories,
        "error": _error_evidence(error),
        "scientific_result": None,
        "scientific_result_status": scientific_result_status,
        "retry_authorized": False,
        "g2_navigation_or_heldout_attempted": False,
        "prior_runtime_output_open_count": 0,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
        "terminalization": {
            "failure_publication": "exclusive_atomic_fsync",
            "terminal_file_mode": "0444",
            "terminal_directory_mode": "0555",
            "seal_after_publication": True,
        },
    }
    if ledger_parser_failure is not None:
        core["ledger_parser_failure"] = dict(ledger_parser_failure)
    return core


def _terminal_contract_invalid_ledger_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    ledger: Any,
    progress: OperationProgress,
    *,
    error: BaseException,
    operation_counts: Mapping[str, Any],
) -> None:
    try:
        if not ledger.closed:
            _ORIGINAL_LEDGER_APPEND_TERMINAL(
                ledger,
                record_type="ATTEMPT_TERMINATING",
                stage=progress.location(),
                operation_counts=operation_counts,
                error=error,
            )
        ledger_binding = _ORIGINAL_LEDGER_BINDING(ledger)
        ledger_raw = b"".join(ledger.raw_parts)
        structural_records = (
            contract.parse_structural_partial_access_ledger(ledger_raw)
        )
        if structural_records[-1]["record_type"] not in {
            "ATTEMPT_TERMINATING",
            "RUNTIME_INPUT_ACCESS_FINALIZED",
        }:
            raise PermissionError(
                "contract-invalid ledger lacks a structural terminal"
            )
        try:
            contract.parse_partial_access_ledger(ledger_raw)
        except BaseException as parser_error:
            parser_evidence = _error_evidence(parser_error)
        else:
            raise PermissionError(
                "contract-invalid ledger unexpectedly passed corrected parser"
            )
        runtime_opens = ledger.runtime_opens()
        value = contract.with_content_sha256(_failure_core(
            schema=contract.CONTRACT_INVALID_LEDGER_FAILURE_SCHEMA,
            status=contract.CONTRACT_INVALID_LEDGER_FAILURE_STATUS,
            scientific_result_status=(
                "NOT_OBSERVED_CONTRACT_INVALID_ACCESS_LEDGER"
            ),
            output_root=output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
            ledger_binding=ledger_binding,
            runtime_opens=runtime_opens,
            progress=progress,
            operation_counts=operation_counts,
            error=error,
            ledger_parser_failure={
                "validator": "parse_partial_access_ledger",
                "full_on_disk_ledger_checked": True,
                "accepted": False,
                "ledger_file_sha256": ledger_binding["file_sha256"],
                "error": parser_evidence,
            },
        ))
        contract.validate_contract_invalid_ledger_failure_receipt(
            value,
            reservation_binding=_BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
            ledger_raw=ledger_raw,
        )
        raw = contract.canonical_json_bytes(value) + b"\n"
        _BASE._publish_readonly_atomic(output_root / "failed.json", raw)
    finally:
        ledger.close()
        _BASE._seal_terminal(output_root)


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    ledger: Any,
    progress: OperationProgress,
    *,
    error: BaseException,
) -> None:
    """Terminalize normal failures or preserve corrected-parser rejection."""

    operation_counts = progress.snapshot()
    candidate_raw = b"".join(ledger.raw_parts)
    if not ledger.closed:
        candidate = ledger._record_value({
            "record_type": "ATTEMPT_TERMINATING",
            "stage": progress.location(),
            "operation_counts": dict(operation_counts),
            "error": _error_evidence(error),
        })
        candidate_raw += contract.canonical_json_bytes(candidate) + b"\n"
    try:
        contract.parse_partial_access_ledger(candidate_raw)
    except BaseException:
        _terminal_contract_invalid_ledger_failure(
            output_root,
            reservation,
            reservation_raw,
            ledger,
            progress,
            error=error,
            operation_counts=operation_counts,
        )
        return

    try:
        if not ledger.closed:
            _ORIGINAL_LEDGER_APPEND_TERMINAL(
                ledger,
                record_type="ATTEMPT_TERMINATING",
                stage=progress.location(),
                operation_counts=operation_counts,
                error=error,
            )
        ledger_binding = ledger.binding()
        runtime_opens = ledger.runtime_opens()
        ledger_records = contract.parse_partial_access_ledger(
            b"".join(ledger.raw_parts)
        )
        if (
            ledger_binding["records_content_sha256"]
            != contract.canonical_json_sha256(ledger_records)
        ):
            raise PermissionError("partial-access ledger summary changed")
        value = contract.with_content_sha256(_failure_core(
            schema=contract.FAILURE_SCHEMA,
            status=contract.NORMAL_FAILURE_STATUS,
            scientific_result_status=(
                "NOT_OBSERVED_TERMINAL_OPERATIONAL_OR_INTEGRITY_FAILURE"
            ),
            output_root=output_root,
            reservation=reservation,
            reservation_raw=reservation_raw,
            ledger_binding=ledger_binding,
            runtime_opens=runtime_opens,
            progress=progress,
            operation_counts=operation_counts,
            error=error,
        ))
        contract.validate_failure_receipt(
            value,
            reservation_binding=_BASE._binding(
                "reservation.json", reservation, reservation_raw
            ),
        )
        raw = contract.canonical_json_bytes(value) + b"\n"
        _BASE._publish_readonly_atomic(output_root / "failed.json", raw)
    finally:
        ledger.close()
        _BASE._seal_terminal(output_root)


def _terminal_pre_ledger_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    progress: OperationProgress,
    *,
    failure: Any,
) -> None:
    """Publish the frozen pre-ledger receipt with successor identity."""

    reservation_binding = _BASE._binding(
        "reservation.json", reservation, reservation_raw
    )
    operation_counts = progress.snapshot()
    if operation_counts != contract.empty_partial_operation_counts():
        raise PermissionError("pre-ledger failure performed an operation")
    try:
        header: dict[str, Any] | None = None
        header_prefix: dict[str, Any] | None = None
        ledger_status = "NOT_PUBLISHED"
        if failure.durable_header_raw is not None:
            header_raw = failure.durable_header_raw
            observed = _BASE._read_pre_ledger_prefix(
                output_root / PartialAccessLedger.RELATIVE_PATH
            )
            if observed != header_raw:
                raise PermissionError(
                    "pre-ledger durable header bytes changed"
                )
            header = contract.validate_pre_ledger_header(
                header_raw,
                reservation_binding=reservation_binding,
                attempt_identity=str(reservation["attempt_identity"]),
            )
            ledger_status = "DURABLE_NOT_CONSTRUCTOR_ACCEPTED"
        elif failure.unaccepted_header_prefix_raw is not None:
            prefix_raw = failure.unaccepted_header_prefix_raw
            observed = _BASE._read_pre_ledger_prefix(
                output_root / PartialAccessLedger.RELATIVE_PATH
            )
            if observed != prefix_raw:
                raise PermissionError(
                    "pre-ledger header prefix bytes changed"
                )
            expected_header = PartialAccessLedger.header_value(
                reservation=reservation,
                reservation_raw=reservation_raw,
            )
            expected_header_raw = (
                contract.canonical_json_bytes(expected_header) + b"\n"
            )
            header_prefix = {
                "path": PartialAccessLedger.RELATIVE_PATH,
                "file_sha256": hashlib.sha256(prefix_raw).hexdigest(),
                "byte_count": len(prefix_raw),
                "matches_expected_header":
                    prefix_raw == expected_header_raw,
                "constructor_accepted": False,
                "complete_ledger": False,
            }
            ledger_status = "UNACCEPTED_HEADER_PREFIX"
        prefix = _BASE._terminal_file_bindings(
            output_root,
            exclude=("failed.json", ".failed.json.publishing"),
        )
        _files, directories = _BASE._terminal_inventory(
            output_root,
            exclude=("failed.json", ".failed.json.publishing"),
        )
        core = {
            "schema": contract.PRE_LEDGER_FAILURE_SCHEMA,
            "status": contract.PRE_LEDGER_FAILURE_STATUS,
            "attempt_identity": reservation["attempt_identity"],
            "reservation": reservation_binding,
            "ledger_state": {
                "status": ledger_status,
                "header": header,
                "header_prefix": header_prefix,
                "runtime_input_open_count": 0,
                "standard_ledger_complete": False,
                "standard_failure_validator_applicable": False,
            },
            "failure_stage": {
                "name": "partial_access_ledger_initialization",
                "boundary": failure.boundary,
            },
            "operation_counts": operation_counts,
            "published_prefix": prefix,
            "published_prefix_sha256":
                contract.canonical_json_sha256(prefix),
            "directories_including_root": directories,
            "error": _error_evidence(failure.error),
            "scientific_result": None,
            "scientific_result_status":
                "NOT_OBSERVED_TERMINAL_PRE_LEDGER_FAILURE",
            "retry_authorized": False,
            "g2_navigation_or_heldout_attempted": False,
            "prior_runtime_output_open_count": 0,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
            "terminalization": {
                "failure_publication": "exclusive_atomic_fsync",
                "terminal_file_mode": "0444",
                "terminal_directory_mode": "0555",
                "seal_after_publication": True,
            },
        }
        value = contract.with_content_sha256(core)
        contract.validate_pre_ledger_failure_receipt(
            value,
            reservation_binding=reservation_binding,
            attempt_identity=str(reservation["attempt_identity"]),
        )
        raw = contract.canonical_json_bytes(value) + b"\n"
        _BASE._publish_readonly_atomic(output_root / "failed.json", raw)
    finally:
        _BASE._seal_terminal(output_root)


# Install the small scientific and integrity deltas into the private frozen
# lifecycle.  All other schedule, optimizer, loss, checkpoint, failure, and
# sealing behavior remains byte-for-byte sourced from temporal V1.
PartialAccessLedger.append_terminal = _ledger_append_terminal
PartialAccessLedger.binding = _ledger_binding
_BASE._load_post_reservation_stack = _load_post_reservation_stack
_BASE._receipt_dict = _receipt_dict
_BASE._state_sha = _state_sha
_BASE._validate_migration_receipt = _validate_migration_receipt
_BASE._prepare_model = _prepare_model
_BASE._visual_only_batch = _visual_only_batch
_BASE._camera_pair = _camera_pair
_BASE._snapshot = _snapshot
_BASE._temporal_physical_metrics = _motion_physical_metrics
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
