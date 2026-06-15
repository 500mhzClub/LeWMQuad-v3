"""Phase 2V RGB-to-swept-state distillation contracts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .phase2d_training import image_tensor
from .phase2s_swept_geometry_affordance import (
    PHASE2S_SWEPT_FEATURE_SCHEMA,
    Phase2SSweptGeometryAffordanceExample,
)

PHASE2V_SWEPT_STATE_DISTILLATION_SCHEMA = (
    "phase2v_source_rgb_to_phase2s_swept_state_v0"
)


@dataclass
class Phase2VSourceSweptStateBatch:
    """Materialized source-RGB batch with Phase 2S swept-state targets."""

    example_indices: tuple[int, ...]
    examples: tuple[Phase2SSweptGeometryAffordanceExample, ...]
    start_vision: torch.Tensor
    swept_geometry_features: torch.Tensor
    primitive_utility_targets: torch.Tensor
    primitive_utility_mask: torch.Tensor
    factor_targets: torch.Tensor
    factor_mask: torch.Tensor

    def to(self, device: torch.device) -> "Phase2VSourceSweptStateBatch":
        return Phase2VSourceSweptStateBatch(
            example_indices=self.example_indices,
            examples=self.examples,
            start_vision=self.start_vision.to(device),
            swept_geometry_features=self.swept_geometry_features.to(device),
            primitive_utility_targets=self.primitive_utility_targets.to(device),
            primitive_utility_mask=self.primitive_utility_mask.to(device),
            factor_targets=self.factor_targets.to(device),
            factor_mask=self.factor_mask.to(device),
        )


def materialize_phase2v_source_swept_state_batch(
    examples: Sequence[Phase2SSweptGeometryAffordanceExample],
    indices: Sequence[int],
    *,
    image_size: int = 224,
    image_cache: dict[Path, torch.Tensor] | None = None,
) -> Phase2VSourceSweptStateBatch:
    """Build one source-RGB to swept-state distillation batch."""

    example_indices = tuple(int(index) for index in indices)
    if not example_indices:
        raise ValueError("cannot materialize an empty Phase 2V batch")
    selected = tuple(examples[index] for index in example_indices)
    base_examples = tuple(example.factorized_example for example in selected)
    primitive_names = base_examples[0].primitive_names
    if any(example.primitive_names != primitive_names for example in base_examples):
        raise ValueError("all Phase 2V examples in a batch must share vocabulary")
    cache = image_cache if image_cache is not None else {}

    def cached_image(path: Path) -> torch.Tensor:
        cached = cache.get(path)
        if cached is None:
            cached = image_tensor(path, image_size=image_size)
            cache[path] = cached
        return cached

    return Phase2VSourceSweptStateBatch(
        example_indices=example_indices,
        examples=selected,
        start_vision=torch.stack(
            [cached_image(Path(example.start_frame)) for example in base_examples]
        ),
        swept_geometry_features=torch.tensor(
            [example.swept_geometry_features for example in selected],
            dtype=torch.float32,
        ),
        primitive_utility_targets=torch.tensor(
            [example.utility_targets for example in base_examples],
            dtype=torch.float32,
        ),
        primitive_utility_mask=torch.tensor(
            [example.utility_mask for example in base_examples],
            dtype=torch.bool,
        ),
        factor_targets=torch.tensor(
            [example.factor_targets for example in base_examples],
            dtype=torch.float32,
        ),
        factor_mask=torch.tensor(
            [example.factor_mask for example in base_examples],
            dtype=torch.bool,
        ),
    )


def phase2v_batch_contract_audit(batch: Phase2VSourceSweptStateBatch) -> dict:
    """Return compact evidence for one materialized Phase 2V batch."""

    return {
        "schema": "jepa_phase2v_swept_state_batch_contract_v0",
        "distillation_schema": PHASE2V_SWEPT_STATE_DISTILLATION_SCHEMA,
        "teacher_feature_schema": PHASE2S_SWEPT_FEATURE_SCHEMA,
        "examples": len(batch.examples),
        "primitive_count": int(batch.swept_geometry_features.shape[1]),
        "feature_dim": int(batch.swept_geometry_features.shape[2]),
        "factor_count": int(batch.factor_targets.shape[2]),
        "primitive_utility_targets": int(batch.primitive_utility_mask.sum()),
        "factor_targets": int(batch.factor_mask.sum()),
        "all_start_frames_finite": bool(torch.isfinite(batch.start_vision).all()),
        "all_swept_targets_finite": bool(
            torch.isfinite(batch.swept_geometry_features).all()
        ),
    }


def phase2v_swept_state_error_summary(
    *,
    predicted_features: torch.Tensor,
    target_features: torch.Tensor,
    feature_names: Sequence[str],
) -> dict:
    """Summarize raw feature reconstruction error for Phase 2V reports."""

    if predicted_features.shape != target_features.shape:
        raise ValueError("predicted_features and target_features must align")
    if predicted_features.ndim != 3:
        raise ValueError(
            "predicted_features must have shape (B, primitive_count, feature_dim)"
        )
    if predicted_features.shape[2] != len(feature_names):
        raise ValueError("feature_names length must match feature dimension")
    predicted = predicted_features.detach().cpu().float()
    target = target_features.detach().cpu().float()
    diff = predicted - target
    per_feature_mae = diff.abs().mean(dim=(0, 1))
    per_feature_mse = diff.square().mean(dim=(0, 1))
    worst_index = int(torch.argmax(per_feature_mae)) if per_feature_mae.numel() else -1
    return {
        "schema": "jepa_phase2v_swept_state_error_summary_v0",
        "finite_predictions": bool(torch.isfinite(predicted).all()),
        "finite_targets": bool(torch.isfinite(target).all()),
        "mean_absolute_error": float(diff.abs().mean()),
        "mean_squared_error": float(diff.square().mean()),
        "max_feature_mean_absolute_error": (
            float(per_feature_mae[worst_index]) if worst_index >= 0 else None
        ),
        "max_feature_mean_squared_error": (
            float(per_feature_mse[worst_index]) if worst_index >= 0 else None
        ),
        "worst_feature_name": (
            str(tuple(feature_names)[worst_index]) if worst_index >= 0 else None
        ),
    }
