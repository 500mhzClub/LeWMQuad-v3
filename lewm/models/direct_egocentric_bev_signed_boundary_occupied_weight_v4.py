"""Objective-only V4 successor: fixed 5:1 occupied-to-free O-field weighting.

The complete RGB encoder, learned-query decoder, two-field K/O head, fixed
scale-16 adapter, target EMA, predictor isolation, initialization, semantic
anchor, and public state interfaces come directly from the semantic-anchor V1
model.  This module adds no parameter or buffer.

Its sole scientific change replaces the O-field class-macro reduction with a
fixed per-cell weighting over known cells:

* ``w_free = 0.8778``, ``w_occupied = 4.3890`` -- a 5:1 ratio, renormalised so
  the mean O-field weight over known cells is 1 at the measured occupied share
  of observable cells.

This is a **reduction** in occupied emphasis, not an increase.  The inherited
reduction ``0.5*mean(free) + 0.5*mean(occupied)`` implies a per-cell
occupied-to-free ratio of ``(1-p)/p``, about 28x at the measured ``p = 0.0348``.
The hypothesis under test is that this macro balance encourages a diffuse
occupied representation, and that reducing occupied influence sharpens
occupied localisation and improves held-out precision and IoU.

Everything else is unchanged: the K field keeps its ``(unknown, known)`` macro
reduction, ``UNKNOWN`` remains excluded from the O field entirely, the per-row
reduction and the ``0.5*K + 0.5*O`` combination are preserved, and the Huber
delta, semantic anchor weight, schedule, optimizer, data, and seed are
untouched.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


_ANCHOR_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
)
_ANCHOR_SPEC = importlib.util.spec_from_file_location(
    "_lewm_signed_boundary_occupied_weight_v4_frozen_anchor_model",
    _ANCHOR_SOURCE_PATH,
)
if _ANCHOR_SPEC is None or _ANCHOR_SPEC.loader is None:
    raise ImportError("cannot load frozen semantic-anchor V1 model source")
_anchor = importlib.util.module_from_spec(_ANCHOR_SPEC)
sys.modules[_ANCHOR_SPEC.name] = _anchor
_ANCHOR_SPEC.loader.exec_module(_anchor)

_signed_boundary = sys.modules[
    "_lewm_signed_boundary_semantic_anchor_frozen_distance_v1_model"
]
_v10 = _signed_boundary._v10

for _name in getattr(_anchor, "__all__", ()):
    globals()[_name] = getattr(_anchor, _name)

# The sole registered training-science delta.
OCCUPIED_TO_FREE_WEIGHT_RATIO_V4 = 5.0
FREE_CELL_WEIGHT_V4 = 0.8778
OCCUPIED_CELL_WEIGHT_V4 = 4.3890
TRAINING_SCIENCE_DELTA_COUNT_V4 = 1


def boundary_huber_per_row_v4(
    predicted_fields: torch.Tensor,
    target_fields: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    """Per-raster boundary Huber with fixed 5:1 O-field cell weighting.

    Identical to the inherited implementation except that the O reduction is a
    fixed-weight mean over known cells rather than a class-macro mean.  K keeps
    its ``(unknown, known)`` macro reduction exactly.
    """

    _signed_boundary._validate_field_tensor_v1(predicted_fields, name="predicted_fields")
    _signed_boundary._validate_field_tensor_v1(target_fields, name="target_fields")
    if target_fields.shape != predicted_fields.shape:
        raise ValueError("predicted and target field shapes differ")
    if target_fields.device != predicted_fields.device:
        raise TypeError("predicted and target fields must share a device")
    if target_fields.dtype != predicted_fields.dtype:
        raise TypeError("predicted and target fields must share a dtype")
    _signed_boundary._validate_labels_v1(target_labels)
    expected_labels = predicted_fields.shape[:1] + predicted_fields.shape[2:]
    if target_labels.shape != expected_labels:
        raise ValueError("target_labels must match field batch and grid shape")
    if target_labels.device != predicted_fields.device:
        raise TypeError("target labels and predicted fields must share a device")

    pointwise = F.huber_loss(
        predicted_fields,
        target_fields,
        reduction="none",
        delta=_signed_boundary.BOUNDARY_HUBER_DELTA_V1,
    )
    known_index = _signed_boundary.KNOWN_FIELD_INDEX_V1
    o_index = _signed_boundary.FREE_OCCUPIED_FIELD_INDEX_V1

    rows: list[torch.Tensor] = []
    for row in range(predicted_fields.shape[0]):
        labels = target_labels[row]
        unknown = labels == _v10.UNKNOWN_CLASS_V1
        free = labels == _v10.FREE_CLASS_V1
        occupied = labels == _v10.OCCUPIED_CLASS_V1
        known = free | occupied

        k_groups = [
            pointwise[row, known_index][mask].mean()
            for mask in (unknown, known)
            if bool(mask.any())
        ]
        if not k_groups:
            raise ValueError("each raster must contain at least one K sign group")
        k_macro = torch.stack(k_groups).mean()

        if bool(known.any()):
            # Fixed 5:1 weighting over known cells; UNKNOWN contributes nothing.
            weights = torch.where(
                occupied,
                torch.full_like(pointwise[row, o_index], OCCUPIED_CELL_WEIGHT_V4),
                torch.full_like(pointwise[row, o_index], FREE_CELL_WEIGHT_V4),
            )
            weights = weights * known.to(weights.dtype)
            denominator = weights.sum()
            if not bool(torch.isfinite(denominator)) or float(denominator) <= 0.0:
                raise ValueError("O-field weight normaliser is degenerate")
            o_weighted = (pointwise[row, o_index] * weights).sum() / denominator
            rows.append(0.5 * k_macro + 0.5 * o_weighted)
        else:
            # An all-UNKNOWN raster has no O availability whatsoever.
            rows.append(k_macro)
    return torch.stack(rows)


class DirectEgocentricBevStateJepaV1(_anchor.DirectEgocentricBevStateJepaV1):
    """Unchanged semantic-anchor model under the V4 O-field weighting."""

    def training_objective_with_components(self, *args, **kwargs):
        original = _signed_boundary._boundary_huber_per_row_v1
        _signed_boundary._boundary_huber_per_row_v1 = boundary_huber_per_row_v4
        try:
            return super().training_objective_with_components(*args, **kwargs)
        finally:
            _signed_boundary._boundary_huber_per_row_v1 = original


__all__ = [
    *getattr(_anchor, "__all__", ()),
    "DirectEgocentricBevStateJepaV1",
    "FREE_CELL_WEIGHT_V4",
    "OCCUPIED_CELL_WEIGHT_V4",
    "OCCUPIED_TO_FREE_WEIGHT_RATIO_V4",
    "TRAINING_SCIENCE_DELTA_COUNT_V4",
    "boundary_huber_per_row_v4",
]
