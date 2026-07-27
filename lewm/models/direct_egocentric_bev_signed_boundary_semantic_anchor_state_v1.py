"""Objective-only semantic-anchor successor to signed-boundary V1.

The complete RGB encoder, learned-query decoder, two-field K/O head, fixed
scale-16 adapter, target EMA, predictor isolation, initialization, and public
state interfaces come directly from signed-boundary V1.  This module adds no
parameter or buffer.  Its sole scientific change adds one sixty-fourth of the
existing present-final-class macro NLL to the signed-distance objective.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import NamedTuple

import torch


_SIGNED_BOUNDARY_SOURCE_PATH = Path(__file__).with_name(
    "direct_egocentric_bev_signed_boundary_distance_state_v1.py"
)
_SIGNED_BOUNDARY_SPEC = importlib.util.spec_from_file_location(
    "_lewm_signed_boundary_semantic_anchor_frozen_distance_v1_model",
    _SIGNED_BOUNDARY_SOURCE_PATH,
)
if _SIGNED_BOUNDARY_SPEC is None or _SIGNED_BOUNDARY_SPEC.loader is None:
    raise ImportError("cannot load frozen signed-boundary V1 model source")
_signed_boundary = importlib.util.module_from_spec(_SIGNED_BOUNDARY_SPEC)
sys.modules[_SIGNED_BOUNDARY_SPEC.name] = _signed_boundary
_SIGNED_BOUNDARY_SPEC.loader.exec_module(_signed_boundary)

for _name in _signed_boundary.__all__:
    globals()[_name] = getattr(_signed_boundary, _name)


SEMANTIC_ANCHOR_WEIGHT_V1 = 1.0 / 64.0


class SignedBoundarySemanticAnchorObjectiveComponentsV1(NamedTuple):
    """Expose the exact optimized objective and its registered components."""

    objective: DirectBevStateObjectiveV1
    G_distance_current: torch.Tensor
    G_distance_next: torch.Tensor
    G_distance: torch.Tensor
    G_semantic_current: torch.Tensor
    G_semantic_next: torch.Tensor
    G_semantic_macro_nll: torch.Tensor
    G_combined: torch.Tensor


class DirectEgocentricBevStateJepaV1(
    _signed_boundary.DirectEgocentricBevStateJepaV1
):
    """Unchanged signed-boundary model with one fixed semantic loss anchor."""

    def training_objective_with_components(
        self,
        *,
        current_rgb: torch.Tensor,
        next_rgb: torch.Tensor,
        fixed_negative_rgb: torch.Tensor,
        action_one_hot: torch.Tensor,
        non_hold_mask: torch.Tensor,
        current_labels: torch.Tensor,
        next_labels: torch.Tensor,
    ) -> SignedBoundarySemanticAnchorObjectiveComponentsV1:
        """Evaluate the parent call graph once and add the exact anchor."""

        distance_objective = super().training_objective(
            current_rgb=current_rgb,
            next_rgb=next_rgb,
            fixed_negative_rgb=fixed_negative_rgb,
            action_one_hot=action_one_hot,
            non_hold_mask=non_hold_mask,
            current_labels=current_labels,
            next_labels=next_labels,
        )

        semantic_current = (
            _signed_boundary._v10._final_class_macro_nll_per_row_v10(
                distance_objective.current_state_logits,
                current_labels,
            ).mean()
        )
        semantic_next = (
            _signed_boundary._v10._final_class_macro_nll_per_row_v10(
                distance_objective.next_online_state_logits,
                next_labels,
            ).mean()
        )
        semantic_macro_nll = 0.5 * semantic_current + 0.5 * semantic_next

        distance_current = distance_objective.G_current
        distance_next = distance_objective.G_next
        distance = distance_objective.G
        combined_current = (
            distance_current + SEMANTIC_ANCHOR_WEIGHT_V1 * semantic_current
        )
        combined_next = (
            distance_next + SEMANTIC_ANCHOR_WEIGHT_V1 * semantic_next
        )
        combined = distance + (
            SEMANTIC_ANCHOR_WEIGHT_V1 * semantic_macro_nll
        )
        if not bool(
            torch.isfinite(
                torch.stack(
                    (
                        distance_current,
                        distance_next,
                        distance,
                        semantic_current,
                        semantic_next,
                        semantic_macro_nll,
                        combined_current,
                        combined_next,
                        combined,
                    )
                )
            ).all()
        ):
            raise FloatingPointError("semantic-anchor objective is nonfinite")

        objective = distance_objective._replace(
            total=combined,
            G=combined,
            G_current=combined_current,
            G_next=combined_next,
        )
        if objective.G is not combined or objective.total is not combined:
            raise RuntimeError("combined objective result identity changed")
        return SignedBoundarySemanticAnchorObjectiveComponentsV1(
            objective=objective,
            G_distance_current=distance_current,
            G_distance_next=distance_next,
            G_distance=distance,
            G_semantic_current=semantic_current,
            G_semantic_next=semantic_next,
            G_semantic_macro_nll=semantic_macro_nll,
            G_combined=combined,
        )

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
        """Return the unchanged public result interface with combined G."""

        return self.training_objective_with_components(
            current_rgb=current_rgb,
            next_rgb=next_rgb,
            fixed_negative_rgb=fixed_negative_rgb,
            action_one_hot=action_one_hot,
            non_hold_mask=non_hold_mask,
            current_labels=current_labels,
            next_labels=next_labels,
        ).objective


__all__ = sorted({
    *_signed_boundary.__all__,
    "DirectEgocentricBevStateJepaV1",
    "SEMANTIC_ANCHOR_WEIGHT_V1",
    "SignedBoundarySemanticAnchorObjectiveComponentsV1",
})
