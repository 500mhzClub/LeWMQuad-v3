"""Goal-image planning costs over candidate latent rollouts — pure functions.

Single source of truth for the cost math that was inlined in the benchmark's
``_choose_lewm_primitive`` (energy / plan_cost) and ``_lewm_primitive_costs``
(pose / energy / plan_cost). The two callers differ only in whether a
``_pose_head`` metric cost is allowed; that asymmetry is preserved via
``allow_pose_head`` so the refactor is behaviour-locked.

No side effects. Cost is lower = closer to goal.

Head-selection priority (matches the pre-refactor benchmark):
  - if ``allow_pose_head`` and ``model._pose_head`` exists: metric cost =
    predicted ``||dxy||`` to goal (min over goal views);
  - elif ``model._energy_head`` exists: learned contrastive energy (min over views);
  - else: ``model.plan_cost`` = projected-latent L2 (recognition cost).
See ``docs/lewm_energy_head_vs_plancost_2026-06-09.md`` for why plan_cost is the
default servoing cost.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _encode_frame(model, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(C,H,W) -> (z_raw (1,D), z_proj (1,D)), mirroring the benchmark helper."""
    return model.encode(image[None, ...], None)


@torch.no_grad()
def rollout_costs(
    model,
    image: torch.Tensor,
    goal_image: torch.Tensor,
    action_tensor: torch.Tensor,
    *,
    allow_pose_head: bool,
) -> torch.Tensor:
    """Cost per candidate action sequence (lower = closer to the goal image).

    ``goal_image`` may be a single ``(C,H,W)`` frame or a ``(V,C,H,W)`` multi-view
    stack; multi-view costs take the per-view minimum ("match the goal from
    whichever side I arrive").
    """
    z_start_raw, _z_start_proj = _encode_frame(model, image)
    goal_views = goal_image if goal_image.dim() == 4 else goal_image[None]
    z_goal_views = torch.cat([_encode_frame(model, gv)[1] for gv in goal_views], dim=0)  # (V, D)
    n_cand = action_tensor.shape[0]
    z_pred = model.plan_rollout(z_start_raw.repeat(n_cand, 1), action_tensor)
    z_pred_last = z_pred[:, -1, :] if z_pred.dim() == 3 else z_pred

    pose_head = getattr(model, "_pose_head", None) if allow_pose_head else None
    energy_head = getattr(model, "_energy_head", None)
    n_views = z_goal_views.shape[0]

    if pose_head is not None:
        per_view = torch.stack(
            [
                pose_head(z_pred_last, z_goal_views[v : v + 1].repeat(n_cand, 1))[:, :2].norm(dim=-1)
                for v in range(n_views)
            ],
            dim=0,
        )
        return per_view.min(dim=0).values
    if energy_head is not None:
        per_view = torch.stack(
            [
                energy_head(z_pred_last, z_goal_views[v : v + 1].repeat(n_cand, 1))
                for v in range(n_views)
            ],
            dim=0,
        )
        return per_view.min(dim=0).values
    return model.plan_cost(z_pred, z_goal_views[0:1].repeat(n_cand, 1))
