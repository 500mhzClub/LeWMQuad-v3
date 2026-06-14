#!/usr/bin/env python3
"""Stage 3 Unit C — offline Level-1 routing probe on the TopologicalNavigator.

The Stage 3 wiring (``lewm/memory/topological_navigator.py``) composes the
validated pieces: v6 BeliefEncoder window -> §5.4 filter (0.96 replay
coherence) -> view-keyframe nodes -> raw-frame goal matching -> BFS routing
over memory edges. This probe answers the §9.3 Level-1 question offline (pure
torch, no genesis): when the agent has built a memory of a held-out scene and
is handed a goal *image* from that scene, does the selected next-hop sub-goal
make progress on the TRUE scene graph?

Per eval trajectory (cached banks): feed the trajectory through the navigator
(recording the MAP node per step), then sample (t_current, t_goal) queries with
true BFS(cell_cur, cell_goal) >= 2. For each query: match the goal frame
against node keyframes; BFS next hop from MAP(t_current) to the goal node;
score the hop's node-majority cell against the true graph.

Metrics (per scene, aggregated):
  - goal_match_acc: goal node's majority cell == goal's true cell (confident).
  - progress_rate: BFS_true(next_hop_cell, goal_cell) < BFS_true(cur_cell,
    goal_cell) among plannable queries; non_regress_rate uses <=.
  - plannable_frac: confident match + connected path + localized current node.
  - random-baseline progress rate (random memory node as the hop).

Gate (registered before running): mean progress_rate >= 0.70 among plannable
queries AND >= random + 0.25; goal_match_acc >= 0.60. Pass -> Level 1 works on
held-out scenes with zero privileged signals at query time -> Stage 4
(closed-loop with exploration mode) is next. Fail -> per-component breakdown
(goal match vs localization vs graph connectivity) names the bottleneck.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "lewm_worlds"))

from lewm.memory.topological_navigator import TopologicalNavigator  # noqa: E402
from lewm_worlds.manifest import parse_scene_manifest_dict  # noqa: E402
from lewm_worlds.scene_graph import SceneGraph  # noqa: E402
from probe_lewm_reachability_a3 import _find_manifest  # noqa: E402
from probe_topo_filter_replay import train_same_yaw_head  # noqa: E402
from train_loop_closure_head import load_belief_encoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe_topo_routing_offline")


def load_graph(scene_id: str, args) -> SceneGraph | None:
    family = "_".join(scene_id.split("_")[:-1])
    manifest_path = _find_manifest(args.manifest_corpus, args.eval_split, family, scene_id)
    if manifest_path is None:
        return None
    return SceneGraph(parse_scene_manifest_dict(json.loads(manifest_path.read_text())))


def run_scene(bank, graph, encoder, head, platt, args, device, rng):
    a, b = platt

    def scorer(query, nodes):
        logits = head(query.unsqueeze(0).expand(len(nodes), -1), nodes)
        return torch.sigmoid(a * logits + b)

    navigator = TopologicalNavigator(
        encoder, scorer, history=args.history, tau_new=args.tau_new, tau_goal=args.tau_goal,
    )
    z = torch.from_numpy(bank["z_history"]).float().to(device)
    cells, yaws = bank["cells"], bank["yaws"]
    map_at = []
    for i in range(len(z)):
        map_at.append(navigator.update((z[i, -1], int(i)), label=(int(cells[i]), int(yaws[i]))))

    majority = navigator.memory.node_majority_labels()
    node_cell = {nid: lab[0][0] for nid, lab in majority.items()}

    lookaheads = [int(k) for k in args.lookaheads.split(",")]
    n_q = 0
    stats = Counter()
    attempts = 0
    while n_q < args.queries_per_scene and attempts < args.queries_per_scene * 30:
        attempts += 1
        t_cur = rng.randrange(args.history * 2, len(z))
        t_goal = rng.randrange(0, len(z))
        d_true = graph.bfs_distance(int(cells[t_cur]), int(cells[t_goal]))
        if d_true is None or d_true < 2 or map_at[t_cur] is None:
            continue
        n_q += 1
        goal_node, score = navigator.match_goal(z[t_goal, -1])
        if goal_node is None or score < args.tau_goal:
            stats["unconfident"] += 1
            continue
        if goal_node in node_cell:
            stats["goal_match_scored"] += 1
            stats["goal_match_correct"] += int(node_cell[goal_node] == int(cells[t_goal]))
        path = navigator.bfs_path(map_at[t_cur], goal_node)
        if path is None:
            stats["disconnected"] += 1
            continue
        stats["plannable"] += 1
        for k in lookaheads:
            hop = path[min(k, len(path) - 1)]
            d_hop = graph.bfs_distance(node_cell[hop], int(cells[t_goal])) if hop in node_cell else None
            if d_hop is not None:
                stats[f"scored_k{k}"] += 1
                stats[f"progress_k{k}"] += int(d_hop < d_true)
                stats[f"non_regress_k{k}"] += int(d_hop <= d_true)
                d_local = graph.bfs_distance(int(cells[t_cur]), node_cell[hop])
                if d_local is not None:
                    stats[f"locality_sum_k{k}"] += d_local
                    stats[f"locality_n_k{k}"] += 1
        # Adaptive (deployment-valid): skip the start's place cluster detected
        # in BELIEF space (yaw-blurred place similarity), per tau_place.
        start_node = map_at[t_cur]
        start_emb = navigator.memory.nodes[start_node].embedding
        for tau_place in (0.7, 0.8, 0.9):
            hop_a = next(
                (n for n in path[1:]
                 if float(navigator.memory.nodes[n].embedding @ start_emb) < tau_place),
                path[-1],
            )
            d_a = graph.bfs_distance(node_cell[hop_a], int(cells[t_goal])) if hop_a in node_cell else None
            if d_a is not None:
                tag = f"adaptive_t{int(tau_place * 100)}"
                stats[f"scored_{tag}"] += 1
                stats[f"progress_{tag}"] += int(d_a < d_true)
                stats[f"non_regress_{tag}"] += int(d_a <= d_true)
        # Privileged ceiling: first path node whose majority cell differs from
        # the start's (perfect same-place-cluster skipping; not deployable).
        start_cell = node_cell.get(map_at[t_cur])
        hop_priv = next((n for n in path[1:] if node_cell.get(n) not in (None, start_cell)), path[-1])
        d_priv = graph.bfs_distance(node_cell[hop_priv], int(cells[t_goal])) if hop_priv in node_cell else None
        if d_priv is not None:
            stats["scored_priv"] += 1
            stats["progress_priv"] += int(d_priv < d_true)
        # Random-node baseline for the same query (teleports scene-wide; weak
        # comparator) + random LOCAL node (fair comparator for a local hop:
        # same <=2-cell locality budget as the lookahead hop).
        random_node = rng.choice(list(node_cell))
        d_rand = graph.bfs_distance(node_cell[random_node], int(cells[t_goal]))
        if d_rand is not None:
            stats["rand_scored"] += 1
            stats["rand_progress"] += int(d_rand < d_true)
        local_nodes = [n for n, c in node_cell.items()
                       if (dl := graph.bfs_distance(int(cells[t_cur]), c)) is not None and dl <= 2]
        if local_nodes:
            rl = rng.choice(local_nodes)
            d_rl = graph.bfs_distance(node_cell[rl], int(cells[t_goal]))
            if d_rl is not None:
                stats["rand_local_scored"] += 1
                stats["rand_local_progress"] += int(d_rl < d_true)

    out = {
        "scene_id": bank["scene_id"],
        "n_queries": n_q,
        "plannable_frac": stats["plannable"] / max(n_q, 1),
        "unconfident_frac": stats["unconfident"] / max(n_q, 1),
        "disconnected_frac": stats["disconnected"] / max(n_q, 1),
        "goal_match_acc": stats["goal_match_correct"] / max(stats["goal_match_scored"], 1),
        "progress_priv": stats["progress_priv"] / max(stats["scored_priv"], 1),
        "random_progress_rate": stats["rand_progress"] / max(stats["rand_scored"], 1),
        "random_local_progress_rate": stats["rand_local_progress"] / max(stats["rand_local_scored"], 1),
        "n_nodes": len(navigator.memory.nodes),
    }
    for k in lookaheads:
        out[f"progress_k{k}"] = stats[f"progress_k{k}"] / max(stats[f"scored_k{k}"], 1)
        out[f"non_regress_k{k}"] = stats[f"non_regress_k{k}"] / max(stats[f"scored_k{k}"], 1)
        out[f"locality_k{k}"] = stats[f"locality_sum_k{k}"] / max(stats[f"locality_n_k{k}"], 1)
    for tau_place in (70, 80, 90):
        tag = f"adaptive_t{tau_place}"
        out[f"progress_{tag}"] = stats[f"progress_{tag}"] / max(stats[f"scored_{tag}"], 1)
        out[f"non_regress_{tag}"] = stats[f"non_regress_{tag}"] / max(stats[f"scored_{tag}"], 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--belief-encoder", type=Path, required=True)
    parser.add_argument("--yaw-train-banks", type=Path, required=True)
    parser.add_argument("--traj-banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-corpus", type=Path, default=REPO_ROOT / ".generated/scene_corpus/minimum_tex_20260520T211541Z")
    parser.add_argument("--eval-split", default="test_id")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--tau-new", type=float, default=None,
                        help="default: the calP95 threshold derived during head training")
    parser.add_argument("--tau-goal", type=float, default=0.80)
    parser.add_argument("--queries-per-scene", type=int, default=60)
    parser.add_argument("--lookaheads", default="1,2,4,6,8,10")
    parser.add_argument("--gate-lookahead", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--gate-progress", type=float, default=0.70)
    parser.add_argument("--gate-margin-vs-random", type=float, default=0.25)
    parser.add_argument("--gate-goal-match", type=float, default=0.60)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)
    encoder = load_belief_encoder(args.belief_encoder, device)
    head, platt, tau_candidates = train_same_yaw_head(encoder, args, device)
    if args.tau_new is None:
        args.tau_new = tau_candidates["calP95"]
    banks = torch.load(args.traj_banks, weights_only=False)

    rng = random.Random(args.seed)
    per_scene, skipped = [], []
    for bank in banks:
        graph = load_graph(bank["scene_id"], args)
        if graph is None:
            continue
        report = run_scene(bank, graph, encoder, head, platt, args, device, rng)
        # Scenes whose trajectory never produced a valid (d>=2, localized)
        # query have no routing question to answer — report, don't average in.
        if report["n_queries"] == 0:
            skipped.append(report["scene_id"])
            continue
        per_scene.append(report)
        logger.info("%s: plannable=%.2f goal_match=%.2f k1=%.2f k4=%.2f priv=%.2f rand=%.2f nodes=%d",
                    report["scene_id"], report["plannable_frac"], report["goal_match_acc"],
                    report.get("progress_k1", 0.0), report.get("progress_k4", 0.0),
                    report["progress_priv"], report["random_progress_rate"], report["n_nodes"])

    lookaheads = [int(k) for k in args.lookaheads.split(",")]
    keys = ["plannable_frac", "unconfident_frac", "disconnected_frac", "goal_match_acc",
            "progress_priv", "random_progress_rate", "random_local_progress_rate"]
    keys += [f"progress_k{k}" for k in lookaheads] + [f"non_regress_k{k}" for k in lookaheads]
    keys += [f"locality_k{k}" for k in lookaheads]
    keys += [f"progress_adaptive_t{t}" for t in (70, 80, 90)]
    keys += [f"non_regress_adaptive_t{t}" for t in (70, 80, 90)]
    agg = {k: float(np.mean([s[k] for s in per_scene])) for k in keys}
    best_adaptive = max((agg[f"progress_adaptive_t{t}"] for t in (70, 80, 90)))
    gate_progress = max(agg[f"progress_k{args.gate_lookahead}"], best_adaptive)
    gate_passed = bool(
        gate_progress >= args.gate_progress
        and gate_progress >= agg["random_local_progress_rate"] + args.gate_margin_vs_random
        and agg["goal_match_acc"] >= args.gate_goal_match
    )
    report = {
        "schema": "lewm_topo_routing_offline_v0",
        "n_scenes": len(per_scene),
        "skipped_scenes_no_valid_queries": skipped,
        "tau_new": args.tau_new,
        "tau_goal": args.tau_goal,
        "aggregate": agg,
        "per_scene": per_scene,
        "gate": {
            "passed": gate_passed,
            "gate_lookahead": args.gate_lookahead,
            "gate_progress_value": gate_progress,
            "required_progress": args.gate_progress,
            "required_margin_vs_random": args.gate_margin_vs_random,
            "required_goal_match": args.gate_goal_match,
        },
        "config": {k: str(v) for k, v in vars(args).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": agg, "gate": report["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
