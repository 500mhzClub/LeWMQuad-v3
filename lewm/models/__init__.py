"""LeWM base world-model components (ported from LeWMQuad-v2, paper-faithful).

The H-JEPA superstructure (BeliefEncoder, GoalAdapter, LoopClosure,
Reachability, MemoryGraph, hierarchical planner) is net-new for v3 and lands
in sibling modules — see docs/v3_topological_nav_plan.md.
"""
from __future__ import annotations

from .encoders import JointEncoder, Projector, VisionEncoder, ProprioEncoder
from .predictor import TransformerPredictor, ActionEmbedder
from .sigreg import sigreg, sigreg_stepwise
from .lewm import LeWorldModel
from .spatial_predictor import SpatialTokenPredictor, trainable_parameter_count
from .spatial_lewm import SpatialLeWorldModel, TokenProjector, spatial_variance_floor_loss
from .phase2d_spatial_lewm import (
    IdentityTokenGeometry,
    LearnedSlotGeometry,
    LinearTokenProjector,
    Phase2DSpatialLeWorldModel,
    PredictionInputMode,
    TargetGeometry,
    action_identifiability_losses,
    normalize_spatial_tokens,
)

__all__ = [
    "LeWorldModel",
    "TransformerPredictor",
    "ActionEmbedder",
    "JointEncoder",
    "VisionEncoder",
    "ProprioEncoder",
    "Projector",
    "sigreg",
    "sigreg_stepwise",
    "SpatialTokenPredictor",
    "trainable_parameter_count",
    "SpatialLeWorldModel",
    "TokenProjector",
    "spatial_variance_floor_loss",
    "IdentityTokenGeometry",
    "LearnedSlotGeometry",
    "LinearTokenProjector",
    "Phase2DSpatialLeWorldModel",
    "PredictionInputMode",
    "TargetGeometry",
    "action_identifiability_losses",
    "normalize_spatial_tokens",
]
