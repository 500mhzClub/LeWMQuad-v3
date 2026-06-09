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
]
