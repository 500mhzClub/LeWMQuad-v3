"""Behaviour-lock for the Stage 0 planner refactor (v3 spec §4.1).

The extracted ``lewm.planning`` / ``lewm.memory`` modules must reproduce the
pre-refactor benchmark planner math *exactly*. We assert that against a
deterministic ``FakeLeWM`` and reference re-implementations copied from the
original ``_choose_lewm_primitive`` / ``_lewm_primitive_costs``:

  * ``LocalMPC.choose`` / ``choose_primitive`` == reference (energy/plan_cost);
  * ``LocalMPC.candidate_costs`` / ``primitive_costs`` == reference
    (pose/energy/plan_cost);
  * the head-selection *asymmetry* is preserved (``choose`` ignores ``_pose_head``,
    ``candidate_costs`` uses it);
  * ``HierarchicalPlanner`` + ``KeyframeMemory`` == calling ``choose`` directly;
  * ``candidate_action_tensor`` is deterministic under a fixed seed.

Runs under pytest or standalone (``python lewm/tests/test_planning_refactor.py``)
so it needs no pytest dependency in the GPU venv.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lewm.memory.topological_memory import KeyframeMemory  # noqa: E402
from lewm.planning.costs import rollout_costs  # noqa: E402
from lewm.planning.hierarchical_planner import HierarchicalPlanner  # noqa: E402
from lewm.planning.local_mpc import (  # noqa: E402
    GoalSpec,
    LocalMPC,
    PlannerState,
    choose_primitive,
    primitive_costs,
)
from lewm.planning.primitive_bank import candidate_action_tensor  # noqa: E402

D = 6  # latent dim
_FLAT = 3 * 4 * 4
_W_RAW = torch.linspace(-1.0, 1.0, _FLAT * D).reshape(_FLAT, D)
_W_PROJ = torch.linspace(1.0, -1.0, _FLAT * D).reshape(_FLAT, D)


class _Head:
    """Deterministic callable head ``(z_pred_last, z_goal) -> tensor``."""

    def __init__(self, kind: str):
        self.kind = kind

    def __call__(self, z_pred_last: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        if self.kind == "energy":  # -> (N,)
            return (z_pred_last - z_goal).abs().sum(dim=-1) * 0.3
        # pose -> (N, 3); cost code takes [:, :2].norm(-1)
        a = z_pred_last.mean(dim=-1) - z_goal.mean(dim=-1)
        b = z_pred_last.sum(dim=-1) * 0.01
        return torch.stack([a, b, torch.zeros_like(a)], dim=-1)


class FakeLeWM:
    """Deterministic stand-in exposing the LeWM planner surface."""

    def __init__(self, *, energy_head=False, pose_head=False, rollout_3d=False):
        self.rollout_3d = rollout_3d
        if energy_head:
            self._energy_head = _Head("energy")
        if pose_head:
            self._pose_head = _Head("pose")

    def encode(self, images: torch.Tensor, _aux):
        flat = images.reshape(images.shape[0], -1)
        return flat @ _W_RAW, flat @ _W_PROJ

    def plan_rollout(self, z_start: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        n = action_tensor.shape[0]
        scalar = action_tensor.reshape(n, -1).sum(dim=1, keepdim=True)  # (N,1)
        z_pred = z_start + scalar  # (N, D)
        return z_pred[:, None, :] if self.rollout_3d else z_pred

    def plan_cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        zp = z_pred[:, -1, :] if z_pred.dim() == 3 else z_pred
        return ((zp - z_goal) ** 2).sum(dim=-1)


# --- reference re-implementations (verbatim original math) -------------------

def _ref_choose(model, image, goal_image, sequences, action_tensor):
    z_start_raw, _ = model.encode(image[None, ...], None)
    goal_views = goal_image if goal_image.dim() == 4 else goal_image[None]
    z_goal_views = torch.cat([model.encode(gv[None, ...], None)[1] for gv in goal_views], dim=0)
    n_cand = action_tensor.shape[0]
    z_pred = model.plan_rollout(z_start_raw.repeat(n_cand, 1), action_tensor)
    head = getattr(model, "_energy_head", None)
    if head is not None:
        z_pred_last = z_pred[:, -1, :] if z_pred.dim() == 3 else z_pred
        per_view = torch.stack(
            [head(z_pred_last, z_goal_views[v : v + 1].repeat(n_cand, 1)) for v in range(z_goal_views.shape[0])],
            dim=0,
        )
        cost = per_view.min(dim=0).values
    else:
        cost = model.plan_cost(z_pred, z_goal_views[0:1].repeat(n_cand, 1))
    best = int(torch.argmin(cost).item())
    return sequences[best][0], float(cost[best].detach().cpu().item())


def _ref_costs(model, image, goal_image, sequences, action_tensor):
    z_start_raw, _ = model.encode(image[None, ...], None)
    goal_views = goal_image if goal_image.dim() == 4 else goal_image[None]
    z_goal_views = torch.cat([model.encode(gv[None, ...], None)[1] for gv in goal_views], dim=0)
    n_cand = action_tensor.shape[0]
    z_pred = model.plan_rollout(z_start_raw.repeat(n_cand, 1), action_tensor)
    z_pred_last = z_pred[:, -1, :] if z_pred.dim() == 3 else z_pred
    pose_head = getattr(model, "_pose_head", None)
    head = getattr(model, "_energy_head", None)
    if pose_head is not None:
        per_view = torch.stack(
            [pose_head(z_pred_last, z_goal_views[v : v + 1].repeat(n_cand, 1))[:, :2].norm(dim=-1)
             for v in range(z_goal_views.shape[0])],
            dim=0,
        )
        cost = per_view.min(dim=0).values
    elif head is not None:
        per_view = torch.stack(
            [head(z_pred_last, z_goal_views[v : v + 1].repeat(n_cand, 1)) for v in range(z_goal_views.shape[0])],
            dim=0,
        )
        cost = per_view.min(dim=0).values
    else:
        cost = model.plan_cost(z_pred, z_goal_views[0:1].repeat(n_cand, 1))
    return cost.detach().cpu().numpy(), [seq[0] for seq in sequences]


# --- fixtures ----------------------------------------------------------------

def _fixtures(device="cpu", horizon=1, multiview=False, seed=7):
    names = ["hold", "fwd", "left", "right"]
    blocks = {
        "hold": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "fwd": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "left": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "right": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    seqs, action_tensor = candidate_action_tensor(
        blocks, names, horizon, max_candidates=None, rng=random.Random(seed), device=torch.device(device)
    )
    g = torch.Generator().manual_seed(123)
    image = torch.rand(3, 4, 4, generator=g)
    if multiview:
        goal = torch.rand(2, 3, 4, 4, generator=g)
    else:
        goal = torch.rand(3, 4, 4, generator=g)
    return names, seqs, action_tensor, image, goal


# --- tests -------------------------------------------------------------------

def test_choose_matches_reference_plan_cost():
    for multiview in (False, True):
        for r3d in (False, True):
            model = FakeLeWM(rollout_3d=r3d)
            _n, seqs, at, image, goal = _fixtures(multiview=multiview)
            ref = _ref_choose(model, image, goal, seqs, at)
            got_cls = LocalMPC(model, seqs, at).choose(PlannerState(image), GoalSpec(goal))
            got_fn = choose_primitive(model, image, goal, seqs, at)
            assert got_cls[0] == ref[0] == got_fn[0]
            assert np.isclose(got_cls[1], ref[1]) and np.isclose(got_fn[1], ref[1])


def test_choose_matches_reference_energy_head():
    model = FakeLeWM(energy_head=True)
    _n, seqs, at, image, goal = _fixtures(multiview=True)
    ref = _ref_choose(model, image, goal, seqs, at)
    got = LocalMPC(model, seqs, at).choose(PlannerState(image), GoalSpec(goal))
    assert got[0] == ref[0] and np.isclose(got[1], ref[1])


def test_candidate_costs_match_reference():
    # pose-head priority, energy-only, and plan_cost-only models
    for kwargs in ({"pose_head": True, "energy_head": True}, {"energy_head": True}, {}):
        model = FakeLeWM(**kwargs)
        _n, seqs, at, image, goal = _fixtures(multiview=True)
        ref_cost, ref_names = _ref_costs(model, image, goal, seqs, at)
        got_cost, got_names = LocalMPC(model, seqs, at).candidate_costs(PlannerState(image), GoalSpec(goal))
        got_cost_fn, got_names_fn = primitive_costs(model, image, goal, seqs, at)
        assert got_names == ref_names == got_names_fn
        assert np.allclose(got_cost, ref_cost) and np.allclose(got_cost_fn, ref_cost)


def test_head_selection_asymmetry():
    # choose() must ignore _pose_head (energy path); candidate_costs() must use it (pose path).
    model = FakeLeWM(pose_head=True, energy_head=True)
    _n, seqs, at, image, goal = _fixtures(multiview=True)
    choose_cost = rollout_costs(model, image, goal, at, allow_pose_head=False)
    full_cost = rollout_costs(model, image, goal, at, allow_pose_head=True)
    # energy head present -> choose uses energy; pose head present -> full uses pose; they differ.
    assert not np.allclose(choose_cost.numpy(), full_cost.numpy())
    # choose path == reference (energy); full path == reference (pose)
    ref_choose = _ref_choose(model, image, goal, seqs, at)
    assert seqs[int(torch.argmin(choose_cost))][0] == ref_choose[0]


def test_hierarchical_planner_matches_direct_choose():
    for kwargs in ({}, {"energy_head": True}):
        model = FakeLeWM(**kwargs)
        _n, seqs, at, image, goal = _fixtures(multiview=True)
        direct = LocalMPC(model, seqs, at).choose(PlannerState(image), GoalSpec(goal))
        planner = HierarchicalPlanner(LocalMPC(model, seqs, at), KeyframeMemory())
        routed = planner.step(image, GoalSpec(goal))
        assert routed == direct


def test_candidate_action_tensor_deterministic():
    names = ["hold", "fwd", "left", "right"]
    blocks = {n: np.arange(3, dtype=np.float32) + i for i, n in enumerate(names)}
    s1, t1 = candidate_action_tensor(blocks, names, 2, max_candidates=8, rng=random.Random(99), device=torch.device("cpu"))
    s2, t2 = candidate_action_tensor(blocks, names, 2, max_candidates=8, rng=random.Random(99), device=torch.device("cpu"))
    assert s1 == s2 and torch.equal(t1, t2)
    assert t1.shape == (8, 2, 3)
    # full product (no truncation) is order-stable
    s3, _ = candidate_action_tensor(blocks, names, 1, max_candidates=None, rng=random.Random(0), device=torch.device("cpu"))
    assert s3 == [("hold",), ("fwd",), ("left",), ("right",)]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} behaviour-lock tests passed.")


if __name__ == "__main__":
    _run_all()
