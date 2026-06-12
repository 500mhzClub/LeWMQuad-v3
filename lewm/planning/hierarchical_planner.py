"""Hierarchical planner orchestrator (spec §4.1, the v3 seam).

Contract: ``step(observation, goal) -> primitive``. Initially it routes through
``LocalMPC`` with ``KeyframeMemory`` and is behaviourally identical to the v2
planner (the goal image is the sub-goal). This is the single place where the
later stages plug in: swap ``KeyframeMemory`` for the learned topological memory
(Level 1 routing + ReachabilityHead sub-goal selection at Level 2), and ``step``
already hands the chosen sub-goal to ``LocalMPC`` at Level 3.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from lewm.memory.topological_memory import KeyframeMemory, Memory

from .local_mpc import GoalSpec, LocalMPC, PlannerState


class HierarchicalPlanner:
    def __init__(self, local_mpc: LocalMPC, memory: Optional[Memory] = None) -> None:
        self.local_mpc = local_mpc
        self.memory = memory if memory is not None else KeyframeMemory()

    def step(
        self,
        observation: torch.Tensor,
        goal: GoalSpec,
        *,
        action_history: Optional[list[Any]] = None,
    ) -> tuple[str, float]:
        """One closed-loop decision: update belief, pick the sub-goal, run LocalMPC."""
        self.memory.update(observation)
        subgoal = self.memory.select_subgoal(goal)
        state = PlannerState(observation, action_history=action_history)
        return self.local_mpc.choose(state, subgoal)
