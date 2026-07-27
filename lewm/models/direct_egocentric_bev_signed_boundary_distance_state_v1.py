"""RGB-only Direct-BEV signed-boundary-distance perception V1.

This additive model preserves the frozen Direct-BEV V10/V8 RGB encoder,
learned-query BEV decoder, predictor construction, and target EMA.  It
replaces the prototype state head and class-NLL training leaf with two learned
signed fields:

``K``
    Positive for known (FREE or OCCUPIED) and negative for UNKNOWN.
``O``
    Positive for FREE and negative for OCCUPIED.  UNKNOWN is not supervised
    in this field.

The learned fields are converted to normalized UNKNOWN/FREE/OCCUPIED stable
log probabilities by a fixed, parameter-free hierarchical adapter.  Only the
per-row sign-macro signed-distance Huber objective is optimized here; the
three-class representation remains available to the frozen observation and
future predictor interfaces.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F


_V10_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_state_jepa_v10_final_class_macro_grounding.py"
)
_V10_SPEC = importlib.util.spec_from_file_location(
    "_lewm_signed_boundary_distance_v1_frozen_v10_model",
    _V10_SOURCE_PATH,
)
if _V10_SPEC is None or _V10_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V10 model source")
_v10 = importlib.util.module_from_spec(_V10_SPEC)
sys.modules[_V10_SPEC.name] = _v10
_V10_SPEC.loader.exec_module(_v10)

# Preserve the complete frozen public API.  The class symbol and the new
# mechanism helpers below deliberately replace or extend selected names.
for _name in _v10.__all__:
    globals()[_name] = getattr(_v10, _name)


SIGNED_BOUNDARY_FIELD_COUNT_V1 = 2
KNOWN_FIELD_INDEX_V1 = 0
FREE_OCCUPIED_FIELD_INDEX_V1 = 1
SIGNED_BOUNDARY_FIELD_ORDER_V1 = (
    "K_known_signed_boundary_distance",
    "O_free_occupied_signed_boundary_distance",
)
SIGNED_BOUNDARY_RADIUS_CELLS_V1 = 8.0
SIGNED_BOUNDARY_HALF_CELL_CORRECTION_V1 = 0.5
BOUNDARY_HUBER_DELTA_V1 = 0.125
HIERARCHICAL_ADAPTER_SCALE_V1 = 16.0
MAXIMUM_PERCEPTION_UPDATES_SIGNED_BOUNDARY_V1 = 1000
SIGNED_BOUNDARY_HEAD_PARAMETER_COUNT_V1 = 130
SIGNED_BOUNDARY_HEAD_PARAMETER_TENSOR_COUNT_V1 = 2

_V8_DECODER_PARAMETER_COUNT = (
    _v10.ONLINE_DECODER_PROTOTYPE_PARAMETER_COUNT_V8
    - _v10.STATE_CLASS_COUNT_V8 * _v10.BEV_FEATURE_DIMENSION_V8
)
_V8_DECODER_PARAMETER_TENSOR_COUNT = (
    _v10.ONLINE_DECODER_PROTOTYPE_PARAMETER_TENSOR_COUNT_V8 - 1
)
ONLINE_DECODER_SIGNED_BOUNDARY_HEAD_PARAMETER_COUNT_V1 = (
    _V8_DECODER_PARAMETER_COUNT + SIGNED_BOUNDARY_HEAD_PARAMETER_COUNT_V1
)
ONLINE_DECODER_SIGNED_BOUNDARY_HEAD_PARAMETER_TENSOR_COUNT_V1 = (
    _V8_DECODER_PARAMETER_TENSOR_COUNT
    + SIGNED_BOUNDARY_HEAD_PARAMETER_TENSOR_COUNT_V1
)


def _validate_labels_v1(target_labels: torch.Tensor) -> None:
    if target_labels.ndim != 3:
        raise ValueError("target_labels must have shape (B,H,W)")
    if any(int(value) < 1 for value in target_labels.shape):
        raise ValueError("target_labels dimensions must be positive")
    if (
        target_labels.is_floating_point()
        or target_labels.is_complex()
        or target_labels.dtype == torch.bool
    ):
        raise TypeError("target_labels must use an integer dtype")
    supported = (
        (target_labels == _v10.UNKNOWN_CLASS_V1)
        | (target_labels == _v10.FREE_CLASS_V1)
        | (target_labels == _v10.OCCUPIED_CLASS_V1)
    )
    if not bool(supported.all()):
        raise ValueError(
            "target_labels contain a class outside UNKNOWN/FREE/OCCUPIED"
        )


def _half_cell_corrected_magnitude_v1(
    source_mask: np.ndarray,
) -> np.ndarray:
    """Return normalized distance to in-raster ``True`` source cells.

    The explicit empty-set branch is scientifically important: SciPy's
    behavior for a raster with no background/source cells must not introduce
    an implicit out-of-raster distance source.
    """

    if source_mask.ndim != 2 or source_mask.dtype != np.bool_:
        raise TypeError("source_mask must be a two-dimensional bool array")
    if not bool(source_mask.any()):
        return np.ones(source_mask.shape, dtype=np.float64)
    # EDT input zeroes are distance sources.  The array contains at least one
    # explicit in-raster source because the empty case was handled above.
    center_distance = distance_transform_edt(~source_mask)
    corrected = np.maximum(
        0.0,
        center_distance - SIGNED_BOUNDARY_HALF_CELL_CORRECTION_V1,
    )
    return np.minimum(
        SIGNED_BOUNDARY_RADIUS_CELLS_V1,
        corrected,
    ) / SIGNED_BOUNDARY_RADIUS_CELLS_V1


def signed_boundary_distance_targets_v1(
    target_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct float32 K/O targets and boolean loss masks.

    The returned tensors have shape ``(B,2,H,W)`` in K,O order and are placed
    on the label device.  Geometry is constructed independently per raster in
    float64 on a unit-spaced row/column grid, then the completed target is cast
    exactly once to the registered float32 model loss dtype.
    """

    _validate_labels_v1(target_labels)
    labels = target_labels.detach().to(device="cpu").numpy()
    batch, height, width = labels.shape
    fields = np.zeros(
        (batch, SIGNED_BOUNDARY_FIELD_COUNT_V1, height, width),
        dtype=np.float64,
    )
    masks = np.zeros(fields.shape, dtype=np.bool_)
    masks[:, KNOWN_FIELD_INDEX_V1, :, :] = True

    for row in range(batch):
        unknown = labels[row] == _v10.UNKNOWN_CLASS_V1
        free = labels[row] == _v10.FREE_CLASS_V1
        occupied = labels[row] == _v10.OCCUPIED_CLASS_V1
        known = free | occupied

        # K: known is positive and UNKNOWN is negative.  Opposite sets are
        # restricted to this raster; neither padding nor an exterior class is
        # introduced.
        distance_to_unknown = _half_cell_corrected_magnitude_v1(unknown)
        distance_to_known = _half_cell_corrected_magnitude_v1(known)
        fields[row, KNOWN_FIELD_INDEX_V1, known] = distance_to_unknown[known]
        fields[row, KNOWN_FIELD_INDEX_V1, unknown] = -distance_to_known[unknown]

        # O: FREE is positive and OCCUPIED is negative.  UNKNOWN is exactly a
        # zero serialization witness and is absent from both search sets and
        # the complete O loss mask.
        distance_to_occupied = _half_cell_corrected_magnitude_v1(occupied)
        distance_to_free = _half_cell_corrected_magnitude_v1(free)
        fields[row, FREE_OCCUPIED_FIELD_INDEX_V1, free] = (
            distance_to_occupied[free]
        )
        fields[row, FREE_OCCUPIED_FIELD_INDEX_V1, occupied] = (
            -distance_to_free[occupied]
        )
        masks[row, FREE_OCCUPIED_FIELD_INDEX_V1, known] = True

    return (
        torch.from_numpy(fields).to(
            device=target_labels.device,
            dtype=torch.float32,
        ),
        torch.from_numpy(masks).to(device=target_labels.device),
    )


def _validate_field_tensor_v1(
    fields: torch.Tensor,
    *,
    name: str,
) -> None:
    if fields.ndim != 4 or fields.shape[1] != SIGNED_BOUNDARY_FIELD_COUNT_V1:
        raise ValueError(f"{name} must have shape (B,2,H,W)")
    if fields.shape[0] < 1 or fields.shape[2] < 1 or fields.shape[3] < 1:
        raise ValueError(f"{name} dimensions must be positive")
    if not fields.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype")
    if not bool(torch.isfinite(fields).all()):
        raise FloatingPointError(f"{name} is nonfinite")


def hierarchical_class_log_probabilities_v1(
    fields: torch.Tensor,
) -> torch.Tensor:
    """Map K/O fields to stable UNKNOWN/FREE/OCCUPIED log probabilities."""

    _validate_field_tensor_v1(fields, name="fields")
    if bool(((fields < -1.0) | (fields > 1.0)).any()):
        raise ValueError("fields must lie in the closed interval [-1,1]")
    known = fields[:, KNOWN_FIELD_INDEX_V1]
    free_occupied = fields[:, FREE_OCCUPIED_FIELD_INDEX_V1]
    scaled_known = known * HIERARCHICAL_ADAPTER_SCALE_V1
    scaled_free_occupied = (
        free_occupied * HIERARCHICAL_ADAPTER_SCALE_V1
    )
    return torch.stack(
        (
            F.logsigmoid(-scaled_known),
            F.logsigmoid(scaled_known)
            + F.logsigmoid(scaled_free_occupied),
            F.logsigmoid(scaled_known)
            + F.logsigmoid(-scaled_free_occupied),
        ),
        dim=1,
    )


def _boundary_huber_per_row_v1(
    predicted_fields: torch.Tensor,
    target_fields: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    """Return one exact present-sign macro boundary Huber loss per raster."""

    _validate_field_tensor_v1(predicted_fields, name="predicted_fields")
    _validate_field_tensor_v1(target_fields, name="target_fields")
    if target_fields.shape != predicted_fields.shape:
        raise ValueError("predicted and target field shapes differ")
    if target_fields.device != predicted_fields.device:
        raise TypeError("predicted and target fields must share a device")
    if target_fields.dtype != predicted_fields.dtype:
        raise TypeError("predicted and target fields must share a dtype")
    _validate_labels_v1(target_labels)
    expected_labels = predicted_fields.shape[:1] + predicted_fields.shape[2:]
    if target_labels.shape != expected_labels:
        raise ValueError("target_labels must match field batch and grid shape")
    if target_labels.device != predicted_fields.device:
        raise TypeError("target labels and predicted fields must share a device")

    pointwise = F.huber_loss(
        predicted_fields,
        target_fields,
        reduction="none",
        delta=BOUNDARY_HUBER_DELTA_V1,
    )
    rows: list[torch.Tensor] = []
    for row in range(predicted_fields.shape[0]):
        labels = target_labels[row]
        unknown = labels == _v10.UNKNOWN_CLASS_V1
        free = labels == _v10.FREE_CLASS_V1
        occupied = labels == _v10.OCCUPIED_CLASS_V1
        known = free | occupied

        k_groups = [
            pointwise[row, KNOWN_FIELD_INDEX_V1][mask].mean()
            for mask in (unknown, known)
            if bool(mask.any())
        ]
        if not k_groups:
            raise ValueError("each raster must contain at least one K sign group")
        k_macro = torch.stack(k_groups).mean()

        o_groups = [
            pointwise[row, FREE_OCCUPIED_FIELD_INDEX_V1][mask].mean()
            for mask in (free, occupied)
            if bool(mask.any())
        ]
        if o_groups:
            o_macro = torch.stack(o_groups).mean()
            rows.append(0.5 * k_macro + 0.5 * o_macro)
        else:
            # An all-UNKNOWN raster has no O availability whatsoever.
            rows.append(k_macro)
    return torch.stack(rows)


class SignedBoundaryDistanceStateHeadV1(nn.Module):
    """Biased 1x1 projection followed by bounded K/O ``tanh`` fields."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = _v10.BEV_FEATURE_DIMENSION_V8
        self.out_channels = SIGNED_BOUNDARY_FIELD_COUNT_V1
        self.projection = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )

    def forward(self, cell_features: torch.Tensor) -> torch.Tensor:
        expected = (
            _v10.BEV_FEATURE_DIMENSION_V8,
            _v10.BEV_QUERY_ROWS_V8,
            _v10.BEV_QUERY_COLUMNS_V8,
        )
        if cell_features.ndim != 4 or tuple(cell_features.shape[1:]) != expected:
            raise ValueError("cell_features must have shape (B,64,64,64)")
        if not cell_features.is_floating_point():
            raise TypeError("cell_features must use a floating dtype")
        if not bool(torch.isfinite(cell_features).all()):
            raise FloatingPointError("cell_features is nonfinite")
        projected = self.projection(cell_features)
        if not bool(torch.isfinite(projected).all()):
            raise FloatingPointError("signed-boundary head projection is nonfinite")
        return torch.tanh(projected)


class DirectEgocentricBevStateJepaV1(
    _v10.DirectEgocentricBevStateJepaV1
):
    """Frozen Direct-BEV stack with signed-distance perception throughout."""

    def __init__(
        self,
        n320_encoder_state_dict: Mapping[str, torch.Tensor],
        config: DirectEgocentricBevStateJepaV1Config | None = None,
    ) -> None:
        # The frozen constructor provides the exact encoder, V8 decoder, V3
        # predictor, phase-accounting attributes, target modules, and EMA
        # buffer.  Its prototype head is immediately and completely replaced.
        super().__init__(n320_encoder_state_dict, config=config)

        # Reproduce the registered seed and construction position immediately
        # after the frozen V8 decoder.  The temporary decoder consumes exactly
        # the frozen decoder draws and is checked against the retained module;
        # it is never registered on this model.
        caller_cpu_rng = torch.random.get_rng_state().clone()
        try:
            torch.random.default_generator.manual_seed(
                self.config.initialization_seed
            )
            initialization_witness = _v10.LearnedBevQueryDecoderV8()
            retained_state = self.bev_decoder.state_dict()
            witness_state = initialization_witness.state_dict()
            if retained_state.keys() != witness_state.keys() or any(
                not torch.equal(retained_state[name], witness_state[name])
                for name in retained_state
            ):
                raise RuntimeError("frozen V8 decoder initialization changed")
            self.state_head = SignedBoundaryDistanceStateHeadV1()
        finally:
            torch.random.set_rng_state(caller_cpu_rng)

        self.target_state_head = copy.deepcopy(self.state_head)
        # The frozen constructor already performed the sole registered initial
        # hard sync for encoder/decoder/head.  Replacing both head copies with
        # the same freshly initialized state preserves exact online/target
        # equality without introducing a successor-level sync call.
        self._freeze_target()

        decoder_parameters = tuple(self.bev_decoder.parameters())
        head_parameters = tuple(self.state_head.parameters())
        if (
            len(decoder_parameters) + len(head_parameters)
            != ONLINE_DECODER_SIGNED_BOUNDARY_HEAD_PARAMETER_TENSOR_COUNT_V1
            or sum(
                value.numel()
                for value in (*decoder_parameters, *head_parameters)
            )
            != ONLINE_DECODER_SIGNED_BOUNDARY_HEAD_PARAMETER_COUNT_V1
            or len(head_parameters)
            != SIGNED_BOUNDARY_HEAD_PARAMETER_TENSOR_COUNT_V1
            or sum(value.numel() for value in head_parameters)
            != SIGNED_BOUNDARY_HEAD_PARAMETER_COUNT_V1
        ):
            raise RuntimeError("signed-boundary decoder/head inventory changed")

    @property
    def active_phase_v6(self) -> str:
        """Keep every registered callback in perception-only phase one."""

        if not self.phase_policy_armed_v6:
            raise RuntimeError("V6 phase policy is not armed")
        if self._v6_phase_override not in (None, _v10.PHASE_ONE_V6):
            raise RuntimeError("signed-boundary V1 has no predictor phase")
        callback_count = int(self.ema_update_count.detach().cpu().item())
        if callback_count > MAXIMUM_PERCEPTION_UPDATES_SIGNED_BOUNDARY_V1:
            raise RuntimeError("signed-boundary V1 exceeded its update cap")
        return _v10.PHASE_ONE_V6

    def phase_counters_v6(self) -> dict[str, int | bool]:
        """Report all registered updates as perception and EMA updates."""

        callback_buffer = int(self.ema_update_count.detach().cpu().item())
        if (
            callback_buffer > MAXIMUM_PERCEPTION_UPDATES_SIGNED_BOUNDARY_V1
            or self._v6_target_update_callback_count != callback_buffer
            or self._v6_ema_arithmetic_update_count != callback_buffer
            or self._v6_boundary_hard_sync_count != 0
            or self._v6_phase_two_target_noop_count != 0
        ):
            raise RuntimeError("signed-boundary target accounting diverged")
        return {
            "phase_policy_armed": self.phase_policy_armed_v6,
            "global_target_update_callback_count": callback_buffer,
            "target_update_callback_count": callback_buffer,
            "ema_arithmetic_update_count": callback_buffer,
            "boundary_hard_sync_count": 0,
            "phase_two_target_noop_count": 0,
            "perception_optimizer_update_count": callback_buffer,
            "predictor_optimizer_update_count": 0,
        }

    def apply_phase_policy_v6(self) -> None:
        """Keep online perception trainable and the predictor frozen to 1000."""

        if not self.phase_policy_armed_v6:
            return
        callback_count = int(self.ema_update_count.detach().cpu().item())
        if callback_count > MAXIMUM_PERCEPTION_UPDATES_SIGNED_BOUNDARY_V1:
            raise RuntimeError("signed-boundary V1 exceeded its update cap")
        for module in self._online_modules():
            module.requires_grad_(True)
            module.train(bool(self.training))
        self.predictor.requires_grad_(False)
        self.predictor.eval()
        self._freeze_target()

    @torch.no_grad()
    def update_target_ema_after_optimizer_step(self) -> None:
        """Apply exact inherited EMA once per perception update through 1000."""

        if not self.phase_policy_armed_v6:
            raise RuntimeError("target update used before phase policy was armed")
        if self._v6_phase_override is not None:
            raise RuntimeError("target update is forbidden during a probe")
        before = int(self.ema_update_count.detach().cpu().item())
        if before >= MAXIMUM_PERCEPTION_UPDATES_SIGNED_BOUNDARY_V1:
            raise RuntimeError("signed-boundary V1 exceeded its update cap")
        if (
            self._v6_target_update_callback_count != before
            or self._v6_ema_arithmetic_update_count != before
            or self._v6_boundary_hard_sync_count != 0
            or self._v6_phase_two_target_noop_count != 0
        ):
            raise RuntimeError("signed-boundary target accounting diverged")

        momentum = self.config.target_ema_momentum
        for target_module, online_module in zip(
            self._target_modules(), self._online_modules(), strict=True
        ):
            target_parameters = dict(target_module.named_parameters())
            online_parameters = dict(online_module.named_parameters())
            if target_parameters.keys() != online_parameters.keys():
                raise RuntimeError("online and target parameter inventories differ")
            for name, target in target_parameters.items():
                target.mul_(momentum).add_(
                    online_parameters[name], alpha=1.0 - momentum
                )
            target_buffers = dict(target_module.named_buffers())
            online_buffers = dict(online_module.named_buffers())
            if target_buffers.keys() != online_buffers.keys():
                raise RuntimeError("online and target buffer inventories differ")
            for name, target in target_buffers.items():
                target.copy_(online_buffers[name])
        self.ema_update_count.add_(1)
        object.__setattr__(
            self,
            "_v6_target_update_callback_count",
            self._v6_target_update_callback_count + 1,
        )
        object.__setattr__(
            self,
            "_v6_ema_arithmetic_update_count",
            self._v6_ema_arithmetic_update_count + 1,
        )
        self._freeze_target()
        self.apply_phase_policy_v6()

    def online_state_fields(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return learned RGB-only online K/O fields before adaptation."""

        self._validate_rgb(rgb, name="online_rgb")
        return self._encode_state(
            rgb,
            self.encoder,
            self.bev_decoder,
            self.state_head,
        )

    @torch.no_grad()
    def target_state_fields(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return detached EMA-target K/O fields before adaptation."""

        self._validate_rgb(rgb, name="target_rgb")
        return self._encode_state(
            rgb,
            self.target_encoder,
            self.target_bev_decoder,
            self.target_state_head,
        ).detach()

    def online_state(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return fixed-adapter UNKNOWN/FREE/OCCUPIED online log probabilities."""

        return hierarchical_class_log_probabilities_v1(
            self.online_state_fields(rgb)
        )

    @torch.no_grad()
    def target_state(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return detached fixed-adapter target log probabilities."""

        return hierarchical_class_log_probabilities_v1(
            self.target_state_fields(rgb)
        ).detach()

    def training_objective(
        self,
        *,
        current_rgb: torch.Tensor,
        next_rgb: torch.Tensor,
        fixed_negative_rgb: torch.Tensor,
        action_one_hot: torch.Tensor,
        non_hold_mask: torch.Tensor,
        current_labels: torch.Tensor,
        next_labels: torch.Tensor,
    ) -> DirectBevStateObjectiveV1:
        """Evaluate the exact two-online/three-target perception call graph."""

        if not self.phase_policy_armed_v6:
            raise RuntimeError("objective used before phase policy was armed")
        callback_count = int(self.ema_update_count.detach().cpu().item())
        if callback_count > MAXIMUM_PERCEPTION_UPDATES_SIGNED_BOUNDARY_V1:
            raise RuntimeError("signed-boundary V1 exceeded its update cap")
        if self.active_phase_v6 != _v10.PHASE_ONE_V6:
            raise RuntimeError("signed-boundary V1 has no predictor objective")
        if (
            next_rgb.shape != current_rgb.shape
            or fixed_negative_rgb.shape != current_rgb.shape
        ):
            raise ValueError("current, next, and fixed-negative RGB shapes differ")
        executed = _v10._v8._v6._v3._v1._validate_action_one_hot(
            action_one_hot,
            batch=current_rgb.shape[0],
            reference=current_rgb,
        )
        if (
            non_hold_mask.shape != (current_rgb.shape[0],)
            or non_hold_mask.dtype != torch.bool
        ):
            raise TypeError("non_hold_mask must be boolean with shape (B,)")
        if not torch.equal(non_hold_mask, executed != _v10.HOLD_ACTION_INDEX_V1):
            raise ValueError("non_hold_mask differs from executed actions")

        current_fields = self.online_state_fields(current_rgb)
        next_online_fields = self.online_state_fields(next_rgb)
        target_next_fields = self.target_state_fields(next_rgb)
        target_current_fields = self.target_state_fields(current_rgb)
        target_mapped_negative_fields = self.target_state_fields(
            fixed_negative_rgb
        )

        current_state = hierarchical_class_log_probabilities_v1(current_fields)
        next_online_state = hierarchical_class_log_probabilities_v1(
            next_online_fields
        )
        target_next = hierarchical_class_log_probabilities_v1(
            target_next_fields
        )
        target_current = hierarchical_class_log_probabilities_v1(
            target_current_fields
        )
        target_mapped_negative = hierarchical_class_log_probabilities_v1(
            target_mapped_negative_fields
        )
        persistence = current_state[:, None].expand(
            -1,
            len(_v10.ACTION_VOCABULARY_V1),
            -1,
            -1,
            -1,
        )
        base = _v10.direct_bev_state_objective_v1(
            current_state_logits=current_state,
            next_online_state_logits=next_online_state,
            all_action_prediction_logits=persistence,
            target_next_logits=target_next,
            target_current_logits=target_current,
            target_mapped_negative_logits=target_mapped_negative,
            current_labels=current_labels,
            next_labels=next_labels,
            executed_action_indices=executed,
            non_hold_mask=non_hold_mask,
        )

        current_targets, _current_masks = (
            signed_boundary_distance_targets_v1(current_labels)
        )
        next_targets, _next_masks = (
            signed_boundary_distance_targets_v1(next_labels)
        )
        if (
            current_targets.device != current_fields.device
            or current_targets.dtype != current_fields.dtype
            or next_targets.device != next_online_fields.device
            or next_targets.dtype != next_online_fields.dtype
        ):
            raise RuntimeError(
                "signed-boundary targets must share model device and dtype"
            )
        current_rows = _boundary_huber_per_row_v1(
            current_fields,
            current_targets,
            current_labels,
        )
        next_rows = _boundary_huber_per_row_v1(
            next_online_fields,
            next_targets,
            next_labels,
        )
        g_current = current_rows.mean()
        g_next = next_rows.mean()
        grounding = 0.5 * g_current + 0.5 * g_next
        return base._replace(
            total=grounding,
            G=grounding,
            G_current=g_current,
            G_next=g_next,
        )


__all__ = sorted({
    *_v10.__all__,
    "BOUNDARY_HUBER_DELTA_V1",
    "DirectEgocentricBevStateJepaV1",
    "FREE_OCCUPIED_FIELD_INDEX_V1",
    "HIERARCHICAL_ADAPTER_SCALE_V1",
    "KNOWN_FIELD_INDEX_V1",
    "MAXIMUM_PERCEPTION_UPDATES_SIGNED_BOUNDARY_V1",
    "ONLINE_DECODER_SIGNED_BOUNDARY_HEAD_PARAMETER_COUNT_V1",
    "ONLINE_DECODER_SIGNED_BOUNDARY_HEAD_PARAMETER_TENSOR_COUNT_V1",
    "SIGNED_BOUNDARY_FIELD_COUNT_V1",
    "SIGNED_BOUNDARY_FIELD_ORDER_V1",
    "SIGNED_BOUNDARY_HALF_CELL_CORRECTION_V1",
    "SIGNED_BOUNDARY_HEAD_PARAMETER_COUNT_V1",
    "SIGNED_BOUNDARY_HEAD_PARAMETER_TENSOR_COUNT_V1",
    "SIGNED_BOUNDARY_RADIUS_CELLS_V1",
    "SignedBoundaryDistanceStateHeadV1",
    "_boundary_huber_per_row_v1",
    "hierarchical_class_log_probabilities_v1",
    "signed_boundary_distance_targets_v1",
})
