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

from collections.abc import Callable, Sequence
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


CandidateScorer = Callable[
    [PlannerState, GoalSpec, Sequence[tuple[str, ...]], torch.Tensor],
    Any,
]


@dataclass(frozen=True)
class DecisionTrace:
    """Immutable record of the candidate ranking used by ``choose``."""

    scorer: str
    candidate_costs: tuple[float, ...]
    sequences: tuple[tuple[str, ...], ...]
    best_index: int
    best_sequence: tuple[str, ...]
    primitive: str
    cost: float


class LocalMPC:
    """Receding-horizon controller: roll out the candidate primitive bank in
    latent space, score against the (sub-)goal image, return the best first
    primitive."""

    def __init__(
        self,
        model,
        sequences: list[tuple[str, ...]],
        action_tensor: torch.Tensor,
        *,
        candidate_scorer: Optional[CandidateScorer] = None,
    ):
        self.model = model
        self.sequences = sequences
        self.action_tensor = action_tensor
        self.candidate_scorer = candidate_scorer
        self._last_decision: Optional[DecisionTrace] = None

    @property
    def last_decision(self) -> Optional[DecisionTrace]:
        """Most recent ``choose`` trace, or ``None`` before the first decision."""
        return self._last_decision

    def _scorer_name(self) -> str:
        if self.candidate_scorer is None:
            return "rollout_costs"
        return str(
            getattr(
                self.candidate_scorer,
                "__name__",
                type(self.candidate_scorer).__name__,
            )
        )

    def _validate_candidate_costs(self, values: Any) -> torch.Tensor:
        """Require exactly one finite real scalar for every candidate."""
        if isinstance(values, torch.Tensor):
            cost = values
        else:
            try:
                cost = torch.as_tensor(values, device=self.action_tensor.device)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError("candidate scorer must return real numeric costs") from exc

        expected = int(self.action_tensor.shape[0])
        if len(self.sequences) != expected:
            raise ValueError(
                "candidate sequences and action tensor disagree: "
                f"{len(self.sequences)} sequences for {expected} action rows"
            )
        if cost.ndim != 1 or int(cost.shape[0]) != expected:
            raise ValueError(
                "candidate scorer must return exactly one scalar per candidate: "
                f"expected shape ({expected},), got {tuple(cost.shape)}"
            )
        if cost.dtype == torch.bool or torch.is_complex(cost):
            raise TypeError("candidate scorer must return finite real numeric costs")
        try:
            finite = bool(torch.isfinite(cost).all().item())
        except RuntimeError as exc:
            raise TypeError("candidate scorer must return finite real numeric costs") from exc
        if not finite:
            raise ValueError("candidate scorer returned a non-finite cost")
        return cost

    def _candidate_cost_tensor(
        self,
        state: PlannerState,
        goal: GoalSpec,
        *,
        allow_pose_head: bool,
    ) -> torch.Tensor:
        if self.candidate_scorer is None:
            return rollout_costs(
                self.model,
                state.image,
                goal.goal_image,
                self.action_tensor,
                allow_pose_head=allow_pose_head,
            )
        values = self.candidate_scorer(
            state,
            goal,
            tuple(self.sequences),
            self.action_tensor,
        )
        return self._validate_candidate_costs(values)

    def _trace(self, cost: torch.Tensor, best_idx: int) -> DecisionTrace:
        costs = tuple(float(value) for value in cost.detach().cpu().tolist())
        sequences = tuple(tuple(seq) for seq in self.sequences)
        best_sequence = sequences[best_idx]
        return DecisionTrace(
            scorer=self._scorer_name(),
            candidate_costs=costs,
            sequences=sequences,
            best_index=best_idx,
            best_sequence=best_sequence,
            primitive=best_sequence[0],
            cost=costs[best_idx],
        )

    @torch.no_grad()
    def choose(self, state: PlannerState, goal: GoalSpec) -> tuple[str, float]:
        """Best first primitive + its cost (servoing cost: energy head or plan_cost)."""
        cost = self._candidate_cost_tensor(
            state,
            goal,
            allow_pose_head=False,
        )
        best_idx = int(torch.argmin(cost).item())
        self._last_decision = self._trace(cost, best_idx)
        return self._last_decision.primitive, self._last_decision.cost

    @torch.no_grad()
    def candidate_costs(self, state: PlannerState, goal: GoalSpec) -> tuple[np.ndarray, list[str]]:
        """Full per-candidate cost vector + first-primitive names (allows pose-head metric cost)."""
        cost = self._candidate_cost_tensor(
            state,
            goal,
            allow_pose_head=True,
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
