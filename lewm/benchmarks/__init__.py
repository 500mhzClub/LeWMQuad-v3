"""Model-agnostic benchmark contracts for LeWM research."""

from .counterfactual import (
    CandidateTrajectory,
    Pose2D,
    integrate_action_blocks,
    oracle_sort_key,
    simulate_candidate_trajectory,
)

__all__ = [
    "CandidateTrajectory",
    "Pose2D",
    "integrate_action_blocks",
    "oracle_sort_key",
    "simulate_candidate_trajectory",
]
