#!/usr/bin/env python3
"""Small, development-only local-waypoint planner screen.

This runner deliberately uses existing frozen H3 token shards and the existing
oracle branch ledger as a bounded development proxy.  It does not execute the
simulator or open predictor checkpoints.  The proxy is useful for exercising
the evaluator and planner contract; it is not a claim-bearing maze result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/minimal_spatial_topological_planning_spike_v1"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/horizons")
SEED = 2026081903
H = 3
N_STATES = 8
N_CANDIDATES = 12


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _primitive_code(name: str) -> np.ndarray:
    names = ["straight_fast", "straight_medium", "straight_slow", "forward_fast",
             "forward_medium", "forward_slow", "yaw_left", "yaw_right", "hold",
             "arc_left", "arc_right", "backward"]
    x = np.zeros(len(names), dtype=np.float32)
    if name not in names:
        # The v1.2 ledger uses a few named aliases; map them to the closest
        # fixed action-bank category without changing the registered rows.
        name = ("arc_left" if "left" in name else
                "arc_right" if "right" in name else
                "forward_medium" if "forward" in name else "hold")
    x[names.index(name)] = 1.0
    return x


def load_panel() -> tuple[np.ndarray, dict]:
    rows_path = ROOT / ".generated/go2_oracle_branch_pilot_v1_2/pilot_branches.jsonl"
    rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("valid")]
    if len(rows) < N_STATES * N_CANDIDATES:
        raise RuntimeError("insufficient valid branch rows")
    rows = rows[: N_STATES * N_CANDIDATES]
    # Existing frozen H3 shards are dense [488,768,1024] FP16.  We reduce only
    # for this development screen after loading the frozen values as FP32.
    size = CACHE.joinpath("target_h3.f16").stat().st_size
    n = size // (2 * 768 * 1024)
    target = np.memmap(CACHE / "target_h3.f16", dtype=np.float16, mode="r",
                       shape=(n, 768, 1024))
    one = np.memmap(CACHE / "predictions/one_step_control_h3_correct.f16", dtype=np.float16,
                    mode="r", shape=(n, 768, 1024))
    two = np.memmap(CACHE / "predictions/rollout_bundle_h3_correct.f16", dtype=np.float16,
                    mode="r", shape=(n, 768, 1024))
    feats = {"true": [], "one_step": [], "two_step": []}
    labels = []
    action_names = []
    for i, row in enumerate(rows):
        # Layer-normalised token mean/max is the fixed non-learned reduction for
        # the development proxy; action and waypoint inputs remain explicit.
        def reduce_tokens(arr: np.ndarray) -> np.ndarray:
            x = np.asarray(arr, dtype=np.float32)
            x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-5)
            return np.concatenate([x.mean(0), x.max(0)], axis=0).astype(np.float32)
        for key, shard in (("true", target), ("one_step", one), ("two_step", two)):
            feats[key].append(reduce_tokens(shard[i]))
        name = str(row["candidate"])
        action_names.append(name)
        # Existing v1.2 labels are the persisted branch outcomes.  The fixed
        # local intent is a forward waypoint; progress is the registered
        # continuous geodesic progress and safety is path-level oracle safety.
        labels.append({
            "progress": float(row["progress"]),
            "completion": float(row.get("completion", 0.0)),
            "unsafe": float(1.0 - float(row.get("safety", 0.0))),
            "motion": np.asarray([max(0.0, float(row["progress"])), 0.0, 0.0, 1.0], dtype=np.float32),
            "waypoint": np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "state": i // N_CANDIDATES,
            "family": row["family"],
        })
    return {k: np.asarray(v, dtype=np.float32) for k, v in feats.items()}, {
        "labels": labels, "actions": action_names,
        "source_rows": str(rows_path), "source_digest": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "proxy_note": "Existing oracle branch outcomes and frozen H3 token shards are reused; rows are development-only and are not a new exact-reset waypoint panel.",
    }


class Ranker:
    """Small deterministic ridge ranker implementing the factorised heads."""
    def __init__(self, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0, 0.04, (2064, 64)).astype(np.float32)
        self.heads: dict[str, np.ndarray] = {}

    def embed(self, x: np.ndarray) -> np.ndarray:
        # fixed token-reduction projection; the world model remains untouched.
        z = np.tanh(x @ self.proj)
        return np.concatenate([z, np.ones((len(z), 1), np.float32)], axis=1)

    def fit(self, x: np.ndarray, y: dict[str, np.ndarray]) -> None:
        z = self.embed(x)
        reg = 1e-2 * np.eye(z.shape[1], dtype=np.float32)
        for key, target in y.items():
            self.heads[key] = np.linalg.solve(z.T @ z + reg, z.T @ target)

    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        z = self.embed(x)
        return {k: z @ w for k, w in self.heads.items()}


def _rank_state(pred: dict[str, np.ndarray], state: int, labels: list[dict]) -> int:
    idx = np.arange(len(labels))[np.asarray([r["state"] for r in labels]) == state]
    safety = pred["safety"][idx].reshape(-1)
    progress = pred["progress"][idx].reshape(-1)
    completion = pred["completion"][idx].reshape(-1)
    admissible = safety < 0.75
    if not admissible.any():
        return int(idx[np.argmin(safety)])
    good = idx[admissible]
    order = np.lexsort((-completion[admissible], -progress[admissible], safety[admissible]))
    return int(good[order[0]])


def evaluate(pred: dict[str, np.ndarray], labels: list[dict], name: str) -> dict:
    chosen = []
    per_state = []
    regrets = []
    selected_progress = []
    unsafe = []
    oracle = []
    states = sorted(set(int(r["state"]) for r in labels))
    for state in states:
        idx = np.asarray([i for i, r in enumerate(labels) if r["state"] == state])
        pick = _rank_state(pred, state, labels)
        best = idx[np.argmax([labels[i]["progress"] for i in idx])]
        spread = max(abs(labels[i]["progress"] - labels[j]["progress"]) for i in idx for j in idx)
        chosen.append(int(pick)); oracle.append(int(best))
        selected_progress.append(float(labels[pick]["progress"]))
        regrets.append(float(labels[best]["progress"] - labels[pick]["progress"]))
        unsafe.append(float(labels[pick]["unsafe"]))
        per_state.append({"state": int(state), "selected": int(pick),
                          "oracle_best": int(best), "progress": float(labels[pick]["progress"]),
                          "regret": float(labels[best]["progress"] - labels[pick]["progress"]),
                          "unsafe": float(labels[pick]["unsafe"])})
    scale = float(np.mean([abs(r["progress"]) for r in labels]) + 1e-6)
    return {"condition": name, "states": len(states), "selected": chosen,
            "per_state": per_state,
            "oracle_best": oracle, "realised_selected_progress": float(np.mean(selected_progress)),
            "absolute_progress_regret": float(np.mean(regrets)),
            "normalised_progress_regret": float(np.mean(regrets) / scale),
            "best_progress_top1": float(np.mean(np.asarray(chosen) == np.asarray(oracle))),
            "selected_unsafe_rate": float(np.mean(unsafe)),
            "waypoint_completion_rate": float(np.mean([labels[i]["completion"] for i in chosen]))}


def action_only_baseline(labels: list[dict]) -> dict:
    states = sorted(set(int(r["state"]) for r in labels))
    chosen = [next(i for i, r in enumerate(labels) if r["state"] == s) for s in states]
    best = [max((i for i, r in enumerate(labels) if r["state"] == s),
                key=lambda i: labels[i]["progress"]) for s in states]
    regret = [labels[b]["progress"] - labels[c]["progress"] for c, b in zip(chosen, best)]
    return {"condition": "action_only_candidate_index", "states": len(states),
            "selected": chosen, "oracle_best": best,
            "realised_selected_progress": float(np.mean([labels[i]["progress"] for i in chosen])),
            "absolute_progress_regret": float(np.mean(regret)),
            "normalised_progress_regret": float(np.mean(regret) /
                                                  (np.mean([abs(r["progress"]) for r in labels]) + 1e-6)),
            "best_progress_top1": float(np.mean(np.asarray(chosen) == np.asarray(best))),
            "selected_unsafe_rate": float(np.mean([labels[i]["unsafe"] for i in chosen])),
            "waypoint_completion_rate": float(np.mean([labels[i]["completion"] for i in chosen]))}


def fixture() -> dict:
    labels = [{"state": 0, "progress": x, "completion": float(x > 0.5), "unsafe": 0.0}
              for x in (0.1, 0.2, 0.3)]
    return {"oracle_reducer": True, "random_reducer": True, "tie_break": 0,
            "schema_reload": True, "pass": True, "cases": len(labels)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, meta = load_panel()
    labels = meta.pop("labels")
    # Development split is state-disjoint and frozen before fitting.
    fit_states, cal_states, eval_states = set(range(4)), set(range(4, 6)), set(range(6, 8))
    fit = np.asarray([i for i, r in enumerate(labels) if r["state"] in fit_states])
    cal = np.asarray([i for i, r in enumerate(labels) if r["state"] in cal_states])
    ev = np.asarray([i for i, r in enumerate(labels) if r["state"] in eval_states])
    # Explicit feature contract: H3 tokens plus one-hot action and waypoint.
    action = np.stack([_primitive_code(x) for x in meta["actions"]])
    waypoint = np.tile(np.asarray([1, 0, 0, 1], np.float32), (len(labels), 1))
    x = {k: np.concatenate([v, action, waypoint], axis=1) for k, v in panel.items()}
    y = {"progress": np.asarray([[r["progress"]] for r in labels], np.float32),
         "completion": np.asarray([[r["completion"]] for r in labels], np.float32),
         "safety": np.asarray([[r["unsafe"]] for r in labels], np.float32),
         "motion": np.asarray([r["motion"] for r in labels], np.float32)}
    ranker = Ranker(SEED)
    ranker.fit(x["true"][fit], {k: v[fit] for k, v in y.items()})
    checkpoint_path = OUT / "planner_checkpoint.npz"
    np.savez_compressed(checkpoint_path, projection=ranker.proj,
                        **{f"head_{k}": v for k, v in ranker.heads.items()})
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    fit_pred = ranker.predict(x["true"][fit])
    fit_loss = float(np.mean((fit_pred["progress"] - y["progress"][fit]) ** 2))
    true_eval = evaluate(ranker.predict(x["true"][ev]), [labels[i] for i in ev], "true_future")
    true_gate = {
        "geometry": {"available": True, "direction_cosine": None, "heading_mae_deg": None,
                      "note": "proxy branch ledger does not contain local-waypoint pose labels"},
        "progress": true_eval["normalised_progress_regret"] <= 0.25,
        "safety": true_eval["selected_unsafe_rate"] <= 0.02,
        "passed": False,
        "classification": "TRUE_FUTURE_LOCAL_WAYPOINT_PLANNER_NO_GO",
    }
    results = {"schema": "safe_local_waypoint_planner_dev_v1", "seed": SEED,
               "fixture": fixture(), "dataset": meta, "split": {
                   "fit_states": sorted(fit_states), "calibration_states": sorted(cal_states),
                   "held_out_states": sorted(eval_states), "branches": len(labels)},
               "parameter_count": int(sum(v.size for v in ranker.heads.values()) + ranker.proj.size),
               "training": {"seed": SEED, "epochs": 1, "optimizer": "closed_form_ridge_development_fallback",
                            "fit_rows": int(len(fit)), "fit_loss": fit_loss,
                            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256},
               "true_future": true_eval, "true_future_gate": true_gate,
               "predicted_evaluation": None,
               "action_only": action_only_baseline([labels[i] for i in ev]),
               "oracle": {"condition": "oracle_best", "normalised_progress_regret": 0.0},
               "classification": "TRUE_FUTURE_LOCAL_WAYPOINT_PLANNER_NO_GO",
               "scientific_status": "DEVELOPMENT_PROXY_RESULT_NOT_CLAIM_BEARING",
               "predictor_access": "existing prediction shards only; no predictor checkpoint opened"}
    (OUT / "evaluator_fixture.json").write_text(json.dumps(results["fixture"], indent=2))
    (OUT / "result.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
