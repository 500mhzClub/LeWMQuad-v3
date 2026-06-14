"""Level-3 local controller — warm-started primitive MPC over the LeWM model.

``LocalMPC`` is the v3 contract ``(current_state, goal_spec) -> action_block``
(spec §4.1). The state is a *bundle* (``PlannerState``) rather than a bare latent
so downstream cost variants (sub-goal cost, multi-step rollout, history
conditioning) do not require interface changes; the ``KeyframeMemory`` baseline
uses only ``image`` + ``goal_image`` and is behaviourally identical to the
pre-refactor benchmark planner.

``choose`` reproduces ``_choose_lewm_primitive`` (energy/plan_cost, no pose head);
``candidate_costs`` reproduces ``_lewm_primitive_costs`` (pose/energy/plan_cost).
The module-level ``choose_primitive`` / ``primitive_costs`` keep the benchmark's
exact call signatures so the refactor is a drop-in delegation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from .costs import rollout_costs


@dataclass
class PlannerState:
    """Bundle of everything a cost function might need about the current step."""

    image: torch.Tensor
    z_history: Optional[torch.Tensor] = None
    action_history: Optional[list[Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalSpec:
    """Goal handed to the controller: a goal/sub-goal image, optionally a node id."""

    goal_image: torch.Tensor
    subgoal_node_id: Optional[int] = None


class LocalMPC:
    """Receding-horizon controller: roll out the candidate primitive bank in
    latent space, score against the (sub-)goal image, return the best first
    primitive."""

    def __init__(self, model, sequences: list[tuple[str, ...]], action_tensor: torch.Tensor):
        self.model = model
        self.sequences = sequences
        self.action_tensor = action_tensor

    @torch.no_grad()
    def choose(self, state: PlannerState, goal: GoalSpec) -> tuple[str, float]:
        """Best first primitive + its cost (servoing cost: energy head or plan_cost)."""
        cost = rollout_costs(
            self.model, state.image, goal.goal_image, self.action_tensor, allow_pose_head=False
        )
        best_idx = int(torch.argmin(cost).item())
        return self.sequences[best_idx][0], float(cost[best_idx].detach().cpu().item())

    @torch.no_grad()
    def candidate_costs(self, state: PlannerState, goal: GoalSpec) -> tuple[np.ndarray, list[str]]:
        """Full per-candidate cost vector + first-primitive names (allows pose-head metric cost)."""
        cost = rollout_costs(
            self.model, state.image, goal.goal_image, self.action_tensor, allow_pose_head=True
        )
        return cost.detach().cpu().numpy(), [seq[0] for seq in self.sequences]


# --- thin module helpers matching the benchmark's pre-refactor signatures ---

def choose_primitive(
    model,
    image: torch.Tensor,
    goal_image: torch.Tensor,
    sequences: list[tuple[str, ...]],
    action_tensor: torch.Tensor,
) -> tuple[str, float]:
    return LocalMPC(model, sequences, action_tensor).choose(PlannerState(image), GoalSpec(goal_image))


def primitive_costs(
    model,
    image: torch.Tensor,
    goal_image: torch.Tensor,
    sequences: list[tuple[str, ...]],
    action_tensor: torch.Tensor,
) -> tuple[np.ndarray, list[str]]:
    return LocalMPC(model, sequences, action_tensor).candidate_costs(
        PlannerState(image), GoalSpec(goal_image)
    )
