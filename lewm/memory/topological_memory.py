"""Topological memory interface + the v2-equivalent ``KeyframeMemory`` baseline.

The abstract ``Memory`` contract (spec §4.1) is what the hierarchical planner
talks to, so the learned topological memory can be swapped in later without
touching the planner. ``KeyframeMemory`` is the behaviour-lock baseline: it keeps
no graph and no belief filter — ``select_subgoal`` returns the goal unchanged, so
routing through it is identical to the pre-refactor planner that servoed straight
at the goal image.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from lewm.planning.local_mpc import GoalSpec


class Memory(ABC):
    """Belief/memory over places. The planner queries it for the current
    sub-goal; concrete subclasses range from ``KeyframeMemory`` (no map) to the
    learned topological graph + top-k Bayes filter."""

    @abstractmethod
    def update(self, observation: Any, action_block: Optional[Any] = None) -> None:
        """Ingest the latest observation (and the action just taken)."""

    @abstractmethod
    def current_belief(self) -> Any:
        """Return the current belief over place (baseline: the last observation)."""

    @abstractmethod
    def select_subgoal(self, goal: GoalSpec) -> GoalSpec:
        """Return the immediate sub-goal to hand to the local controller."""


class KeyframeMemory(Memory):
    """v2-equivalent memory: the goal image *is* the sub-goal. No graph, no
    filter. Routing through this is behaviourally identical to calling
    ``LocalMPC.choose`` directly on the final goal."""

    def __init__(self) -> None:
        self._last_observation: Any = None

    def update(self, observation: Any, action_block: Optional[Any] = None) -> None:
        self._last_observation = observation

    def current_belief(self) -> Any:
        return self._last_observation

    def select_subgoal(self, goal: GoalSpec) -> GoalSpec:
        return goal
