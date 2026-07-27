"""Direct BEV V10 final-class macro-grounding adapter.

The complete frozen V8 RGB encoder, learned-query BEV decoder, normalized
prototype head, EMA target, and persistence-only predictor diagnostics are
retained.  V10 changes only grounding: each raster contributes the mean of
the target NLL means for the UNKNOWN, FREE, and OCCUPIED classes present in
that raster.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


_V8_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_state_jepa_v8_"
    "learned_bev_query_prototype_decoder.py"
)
_V8_SPEC = importlib.util.spec_from_file_location(
    "_lewm_direct_bev_v10_macro_grounding_frozen_v8_model",
    _V8_SOURCE_PATH,
)
if _V8_SPEC is None or _V8_SPEC.loader is None:
    raise ImportError("cannot load frozen Direct BEV V8 model source")
_v8 = importlib.util.module_from_spec(_V8_SPEC)
sys.modules[_V8_SPEC.name] = _v8
_V8_SPEC.loader.exec_module(_v8)

# Preserve the complete public V8 model API.  The V10 class below replaces
# only the class symbol and introduces no constructor, parameter, or buffer.
for _name in _v8.__all__:
    globals()[_name] = getattr(_v8, _name)


FINAL_CLASS_COUNT_V10 = 3
GROUNDING_PUBLIC_SCALE_V10 = math.log(2.0) / math.log(3.0)


def _final_class_macro_nll_per_row_v10(
    state_logits: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    """Return one equal-present-final-class macro NLL per raster."""

    if state_logits.ndim != 4 or state_logits.shape[1] != FINAL_CLASS_COUNT_V10:
        raise ValueError("state_logits must have shape (B,3,H,W)")
    if not state_logits.is_floating_point():
        raise TypeError("state_logits must use a floating dtype")
    expected = state_logits.shape[:1] + state_logits.shape[2:]
    if target_labels.shape != expected:
        raise ValueError("target_labels must have shape (B,H,W)")
    if target_labels.device != state_logits.device:
        raise TypeError("labels and state logits must share a device")
    if target_labels.is_floating_point() or target_labels.dtype == torch.bool:
        raise TypeError("target_labels must use an integer dtype")
    supported = (
        (target_labels == _v8.UNKNOWN_CLASS_V1)
        | (target_labels == _v8.FREE_CLASS_V1)
        | (target_labels == _v8.OCCUPIED_CLASS_V1)
    )
    if not bool(supported.all()):
        raise ValueError(
            "target_labels contain a class outside UNKNOWN/FREE/OCCUPIED"
        )

    target_nll = -F.log_softmax(state_logits, dim=1).gather(
        1,
        target_labels.to(dtype=torch.long).unsqueeze(1),
    ).squeeze(1)
    rows: list[torch.Tensor] = []
    for row in range(state_logits.shape[0]):
        present_class_means: list[torch.Tensor] = []
        for state_class in (
            _v8.UNKNOWN_CLASS_V1,
            _v8.FREE_CLASS_V1,
            _v8.OCCUPIED_CLASS_V1,
        ):
            mask = target_labels[row] == state_class
            if bool(mask.any()):
                present_class_means.append(target_nll[row][mask].mean())
        # Label validation and non-empty H/W dimensions guarantee at least one
        # present class for every raster.
        if not present_class_means:
            raise ValueError("each raster must contain at least one state cell")
        rows.append(torch.stack(present_class_means).mean())
    return torch.stack(rows)


class DirectEgocentricBevStateJepaV1(_v8.DirectEgocentricBevStateJepaV1):
    """Frozen V8 architecture with final-class macro grounding only."""

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
        # The frozen method performs the complete validation and exact
        # 2-online/3-target state call graph, constructs persistence for all
        # actions, and never calls the predictor.
        base = super().training_objective(
            current_rgb=current_rgb,
            next_rgb=next_rgb,
            fixed_negative_rgb=fixed_negative_rgb,
            action_one_hot=action_one_hot,
            non_hold_mask=non_hold_mask,
            current_labels=current_labels,
            next_labels=next_labels,
        )
        raw_current = _final_class_macro_nll_per_row_v10(
            base.current_state_logits,
            current_labels,
        ).mean()
        raw_next = _final_class_macro_nll_per_row_v10(
            base.next_online_state_logits,
            next_labels,
        ).mean()
        raw_grounding = 0.5 * (raw_current + raw_next)
        public_current = raw_current * GROUNDING_PUBLIC_SCALE_V10
        public_next = raw_next * GROUNDING_PUBLIC_SCALE_V10
        public_grounding = raw_grounding * GROUNDING_PUBLIC_SCALE_V10
        return base._replace(
            total=public_grounding / math.log(2.0),
            G=public_grounding,
            G_current=public_current,
            G_next=public_next,
        )

    @torch.no_grad()
    def wrong_rgb_grounding_control(
        self,
        *,
        next_rgb: torch.Tensor,
        fixed_negative_rgb: torch.Tensor,
        next_labels: torch.Tensor,
    ) -> WrongRgbGroundingControlV1:
        # Preserve the frozen validation and exact two-online-state call graph,
        # then replace only the diagnostic loss vectors with raw V10 macro NLL.
        base = super().wrong_rgb_grounding_control(
            next_rgb=next_rgb,
            fixed_negative_rgb=fixed_negative_rgb,
            next_labels=next_labels,
        )
        return base._replace(
            correct_next_loss_per_row=_final_class_macro_nll_per_row_v10(
                base.correct_next_state_logits,
                next_labels,
            ),
            mapped_negative_loss_per_row=_final_class_macro_nll_per_row_v10(
                base.mapped_negative_state_logits,
                next_labels,
            ),
        )


__all__ = sorted({
    *_v8.__all__,
    "FINAL_CLASS_COUNT_V10",
    "GROUNDING_PUBLIC_SCALE_V10",
    "_final_class_macro_nll_per_row_v10",
})
