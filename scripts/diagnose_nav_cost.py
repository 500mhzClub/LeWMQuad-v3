#!/usr/bin/env python3
"""Phase-0 nav-cost diagnostic: is the energy-head cost a usable metric field?

For each visible-beacon scene, at the START pose, score ALL horizon-length
candidate primitive sequences with any supplied learned cost plus bare latent L2:
  - head cost, single front-view goal   (exactly what navH deploys)
  - head cost, multi-view min goal       (the unused --goal-views lever)
  - pose-head predicted distance          (what lewm_pose deploys)
  - bare latent-L2 plan_cost              (what navL2 deploys)
against the TRUE final distance-to-goal each candidate reaches under the kinematic
model. Then report, per scene and aggregate:
  - Spearman rho(cost, true_distance)              -- is the cost monotone in reality?
  - regret = true_dist(argmin cost) - true_dist(oracle-best candidate)
  - oracle-best vs random-pick vs chosen distance  -- candidate-set capability vs ranking

This disambiguates: bad candidate set (planner/horizon) vs bad cost ranking (head
objective / unused multi-view) vs non-metric latent (representation). CPU-only; it
reuses the benchmark harness so the start pose / goal / candidates match the
deployed navH first decision exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_lewm_closed_loop_mpc as B  # noqa: E402  (sets sys.path + heavy imports)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _kinematic_endpoint(seq, registry, grid, start_xy, start_yaw, dt) -> tuple[float, float]:
    """Pure-numpy mirror of _execute_kinematic_primitive over a full candidate
    sequence: integrate every tick, stop the current primitive on collision."""
    x, y = float(start_xy[0]), float(start_xy[1])
    yaw = float(start_yaw)
    for name in seq:
        block = B.expand_primitive_to_block(registry, name)
        for vx, vy, yaw_rate in block:
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            nx = x + (float(vx) * cos_y - float(vy) * sin_y) * dt
            ny = y + (float(vx) * sin_y + float(vy) * cos_y) * dt
            if grid is not None and not grid.is_free((nx, ny)):
                break
            x, y = nx, ny
            yaw = B.wrap_angle_pi(yaw + float(yaw_rate) * dt)
    return x, y


@torch.no_grad()
def _costs(model, energy_head, pose_head, start_image, goal_single, goal_multi, action_tensor):
    n_cand = action_tensor.shape[0]
    z_start_raw, _ = B._encode_frame(model, start_image)
    z_pred = model.plan_rollout(z_start_raw.repeat(n_cand, 1), action_tensor)
    z_pred_last = z_pred[:, -1, :] if z_pred.dim() == 3 else z_pred

    zg_single = B._encode_frame(model, goal_single)[1]  # (1, D) z_proj space
    l2 = model.plan_cost(z_pred, zg_single.repeat(n_cand, 1))  # bare latent-L2 (navL2 path)
    costs = {"l2": l2.detach().cpu().numpy()}
    if energy_head is not None:
        costs["head_single"] = energy_head(
            z_pred_last, zg_single.repeat(n_cand, 1)
        ).detach().cpu().numpy()
        zg_multi = torch.cat([B._encode_frame(model, gv)[1] for gv in goal_multi], dim=0)
        per_view = torch.stack(
            [
                energy_head(z_pred_last, zg_multi[v : v + 1].repeat(n_cand, 1))
                for v in range(zg_multi.shape[0])
            ],
            dim=0,
        )
        costs["head_multi"] = per_view.min(dim=0).values.detach().cpu().numpy()
    if pose_head is not None:
        costs["pose_single"] = pose_head(
            z_pred_last, zg_single.repeat(n_cand, 1)
        )[:, :2].norm(dim=-1).detach().cpu().numpy()
    return costs, int(len(goal_multi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--head-ckpt", type=Path, default=None)
    ap.add_argument("--pose-head-ckpt", type=Path, default=None)
    ap.add_argument("--scene-corpus", type=Path,
                    default=B.REPO_ROOT / ".generated" / "scene_corpus" / "minimum_20260520T080420Z")
    ap.add_argument("--platform-manifest", type=Path, default=B.REPO_ROOT / "config" / "go2_platform_manifest.yaml")
    ap.add_argument("--primitive-registry", type=Path, default=B.REPO_ROOT / "config" / "go2_primitive_registry.yaml")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--family", default="open_obstacle_field")
    ap.add_argument("--scene-limit", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--goal-views", type=int, default=8)
    ap.add_argument("--goal-standoff-m", type=float, default=0.85)
    ap.add_argument("--beacon-approach-distance-m", type=float, default=1.5)
    ap.add_argument("--primitive-names",
                    default="hold,forward_medium,arc_left,arc_right,yaw_left,yaw_right,backward")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.head_ckpt is None and args.pose_head_ckpt is None:
        ap.error("at least one of --head-ckpt or --pose-head-ckpt is required")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)

    model, model_config = B.load_model(
        SimpleNamespace(checkpoint=args.checkpoint.resolve(), max_seq_len=None, sigreg_lambda=None),
        device,
    )
    head_ck = None
    head = None
    if args.head_ckpt is not None:
        head_ck = torch.load(args.head_ckpt.resolve(), map_location=device, weights_only=False)
        head = B.GoalEnergyHead(
            latent_dim=int(head_ck.get("latent_dim", model.latent_dim)),
            hidden=int(head_ck.get("hidden", 1024)),
            dropout=0.0,
        ).to(device)
        head.load_state_dict(head_ck["head_state_dict"])
        head.eval()
        print(f"[head] ranking acc {head_ck.get('best_eval_ranking_acc', '?')}", flush=True)
    pose_head = None
    if args.pose_head_ckpt is not None:
        pose_ck = torch.load(args.pose_head_ckpt.resolve(), map_location=device, weights_only=False)
        pose_head = B.RelPoseHead(
            latent_dim=int(pose_ck.get("latent_dim", model.latent_dim)),
            hidden=int(pose_ck.get("hidden", 512)),
        ).to(device)
        pose_head.load_state_dict(pose_ck["head_state_dict"])
        pose_head.eval()

    primitive_names = [s.strip() for s in args.primitive_names.split(",") if s.strip()]
    platform = B.load_platform_manifest(args.platform_manifest.resolve())
    registry = B.PrimitiveRegistry.from_yaml(args.primitive_registry.resolve())
    dt = float(platform.get("timing", {}).get("command_dt_s", 0.10))
    primitive_blocks = B._primitive_active_blocks(registry, primitive_names)
    sequences, action_tensor = B._candidate_action_tensor(
        primitive_blocks, primitive_names, int(args.horizon),
        max_candidates=None, rng=random.Random(args.seed), device=device,
    )

    scene_dirs = sorted(
        B.find_scene_dirs(args.scene_corpus.resolve(), split=args.split, family=args.family),
        key=lambda p: p.name,
    )[: int(args.scene_limit)]
    if not scene_dirs:
        raise SystemExit(f"no scenes for split={args.split!r} family={args.family!r}")

    rows: list[dict] = []
    for si, sd in enumerate(scene_dirs):
        pack = B.load_scene_pack(sd, platform_manifest=platform, workspace_root=B.REPO_ROOT)
        try:
            build = B.build_scene_from_pack(pack, n_envs=1, backend=args.backend,
                                            show_viewer=False, render_robot=False)
            grid = B.InflatedOccupancyGrid(pack.scene_graph.manifest, cell_size_m=0.05, inflation_m=0.20)
            # Same seed convention as navH (trial 0) -> identical start/goal.
            start_pos, start_quat, goal = B._select_visible_beacon_setup(
                pack, random.Random(int(args.seed) + si * 1000),
                device=device, build=build, grid=grid,
                approach_distance_m=float(args.beacon_approach_distance_m),
                goal_standoff_m=float(args.goal_standoff_m),
                start_yaw_jitter_rad=0.0, n_goal_views=int(args.goal_views),
            )
            B._set_pose(build=build, runner=None, pos_xyz=start_pos, quat_wxyz=start_quat)
            start_image = B._render_tensor_from_base(
                build, pack, base_xyz_m=start_pos, base_quat_wxyz=start_quat, device=device)
            goal_multi = goal.approach_images if goal.approach_images is not None else goal.image[None]

            # single-view cost uses goal.image; multi-view min uses goal_multi.
            costs, n_views = _costs(
                model, head, pose_head, start_image, goal.image, goal_multi, action_tensor
            )

            start_yaw = B._yaw_from_quat_wxyz(start_quat)
            # endpoint distance (full candidate) and first-primitive-only distance
            # (= what closed-loop nav actually executes before replanning).
            true_d = np.array([
                B._xy_distance(_kinematic_endpoint(seq, registry, grid, start_pos[:2], start_yaw, dt), goal.target_xy)
                for seq in sequences
            ])
            first_d = np.array([
                B._xy_distance(_kinematic_endpoint(seq[:1], registry, grid, start_pos[:2], start_yaw, dt), goal.target_xy)
                for seq in sequences
            ])
            init_d = float(B._xy_distance(start_pos[:2], goal.target_xy))

            row = {
                "scene_id": str(pack.scene_id),
                "n_cand": len(sequences),
                "n_goal_views": int(n_views),
                "initial_distance_m": init_d,
                "oracle_best_dist_m": float(true_d.min()),
                "random_mean_dist_m": float(true_d.mean()),
                "oracle_first_dist_m": float(first_d.min()),
                "random_first_dist_m": float(first_d.mean()),
                "true_dist_min_med_max_m": [float(true_d.min()), float(np.median(true_d)), float(true_d.max())],
            }
            for label, cost in costs.items():
                ci = int(np.argmin(cost))
                row[label] = {
                    "spearman_vs_true": _spearman(cost, true_d),
                    "chosen_dist_m": float(true_d[ci]),
                    "regret_m": float(true_d[ci] - true_d.min()),
                    # first-primitive view: does the argmin pick a good FIRST move?
                    "first_spearman": _spearman(cost, first_d),
                    "first_chosen_dist_m": float(first_d[ci]),
                    "first_regret_m": float(first_d[ci] - first_d.min()),
                    "chosen_seq": list(sequences[ci]),
                }
            rows.append(row)
            print(
                f"[{si + 1}/{len(scene_dirs)}] {pack.scene_id} init={init_d:.2f} "
                f"oracle(end/1st)={true_d.min():.2f}/{first_d.min():.2f} | "
                + " | ".join(
                    f"{label} end_reg={row[label]['regret_m']:.2f} "
                    f"1st_reg={row[label]['first_regret_m']:.2f}"
                    for label in costs
                ),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {sd.name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def _agg(label: str, key: str) -> float:
        vals = [r[label][key] for r in rows
                if not (isinstance(r[label][key], float) and math.isnan(r[label][key]))]
        return float(np.mean(vals)) if vals else float("nan")

    labels = [label for label in ("pose_single", "head_single", "head_multi", "l2")
              if rows and label in rows[0]]
    summary = {
        "schema": "nav_cost_diagnosis_v0",
        "checkpoint": str(args.checkpoint.resolve()),
        "head_ckpt": str(args.head_ckpt.resolve()) if args.head_ckpt else None,
        "pose_head_ckpt": str(args.pose_head_ckpt.resolve()) if args.pose_head_ckpt else None,
        "head_ranking_acc": head_ck.get("best_eval_ranking_acc") if head_ck else None,
        "horizon": int(args.horizon),
        "goal_views": int(args.goal_views),
        "n_scenes": len(rows),
        "mean_initial_dist_m": float(np.mean([r["initial_distance_m"] for r in rows])) if rows else float("nan"),
        "mean_oracle_best_dist_m": float(np.mean([r["oracle_best_dist_m"] for r in rows])) if rows else float("nan"),
        "mean_random_pick_dist_m": float(np.mean([r["random_mean_dist_m"] for r in rows])) if rows else float("nan"),
        "mean_oracle_first_dist_m": float(np.mean([r["oracle_first_dist_m"] for r in rows])) if rows else float("nan"),
        "mean_random_first_dist_m": float(np.mean([r["random_first_dist_m"] for r in rows])) if rows else float("nan"),
        "aggregate": {
            label: {
                "mean_spearman_vs_true": _agg(label, "spearman_vs_true"),
                "mean_regret_m": _agg(label, "regret_m"),
                "mean_chosen_dist_m": _agg(label, "chosen_dist_m"),
                "mean_first_spearman": _agg(label, "first_spearman"),
                "mean_first_regret_m": _agg(label, "first_regret_m"),
                "mean_first_chosen_dist_m": _agg(label, "first_chosen_dist_m"),
            }
            for label in labels
        },
        "scenes": rows,
    }

    print("\n=== AGGREGATE ===")
    print(f"init={summary['mean_initial_dist_m']:.2f}m | endpoint oracle={summary['mean_oracle_best_dist_m']:.2f} "
          f"rand={summary['mean_random_pick_dist_m']:.2f} | first-step oracle={summary['mean_oracle_first_dist_m']:.2f} "
          f"rand={summary['mean_random_first_dist_m']:.2f}   (lower = closer)")
    print(f"  {'variant':12s} {'end_rho':>8s} {'end_reg':>8s} {'end_chosen':>10s} | {'1st_rho':>8s} {'1st_reg':>8s} {'1st_chosen':>10s}")
    for label in labels:
        a = summary["aggregate"][label]
        print(f"  {label:12s} {a['mean_spearman_vs_true']:>+8.3f} {a['mean_regret_m']:>8.3f} {a['mean_chosen_dist_m']:>10.3f} | "
              f"{a['mean_first_spearman']:>+8.3f} {a['mean_first_regret_m']:>8.3f} {a['mean_first_chosen_dist_m']:>10.3f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
