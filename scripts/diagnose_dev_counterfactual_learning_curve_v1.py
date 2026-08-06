#!/usr/bin/env python3
"""Learning-curve diagnostic: data scaling versus training-regime memorisation.

DEVELOPMENT ONLY.  NOT CLAIM BEARING.  The encoder stays frozen; no corpus is
collected.

The input-sufficiency diagnostic recorded SCENE_SPECIFIC_OVERFITTING_DOMINATES:
stateful arms nearly solve the 96 training groups yet stay worse than uniform on
scene-disjoint selection, and neither temporal context nor recorded
proprioception closes that.  Missing state is therefore not the main
explanation -- but a final-only evaluation cannot separate insufficient scene
diversity from overtraining or excessive predictor capacity.

This runs the temporal-context arm alone (best selection soft-label CE, and
vision-only) over **nested** family-balanced subsets of the existing V3 train
role:

    8 scenes  = 1 per family = 32 groups
    16 scenes = 2 per family = 64 groups
    24 scenes = 3 per family = 96 groups

Subsets are nested by construction: scenes are ranked within family by the same
deterministic hash used for the selection split, and subset ``k`` takes ranks
``0..k-1``.  Presentations per group are held approximately constant instead of
giving every size the full 10,680-step budget, so a difference between sizes is
a difference in scene diversity rather than in gradient steps per group.

The unchanged 8-scene selection set is evaluated periodically, and both the
best-selection and final checkpoints are saved.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import lewm.models.direct_egocentric_bev_state_jepa_v1 as _preload  # noqa: F401,E402

from lewm.datasets import go2_v3_counterfactual_branches_v1 as V3  # noqa: E402
from lewm.models.token_primary_action_conditioned_jepa_v1 import (  # noqa: E402
    TOKEN_DIM, initialize_token_predictor_v1,
)
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
import scripts.run_dev_token_counterfactual_matching_h1 as RUN  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402
from scripts.diagnose_dev_counterfactual_input_sufficiency_v1 import (  # noqa: E402
    ARMS, InputAdapterV1, LN, matching_stats, normalised_ce, quat_to_body,
)

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
NINE = V3.BRANCHES_PER_GROUP
ARM = "temporal_context"
FULL_STEPS = 10_680
FULL_GROUPS = 96
GROUPS_PER_STEP = RUN.GROUPS_PER_STEP
LR = RUN.PRED_LR
EVAL_POINTS = 40
DERANGEMENT_SEEDS = (11, 23, 37)
SUBSETS = (1, 2, 3)          # scenes per family -> 8, 16, 24 scenes
OUT = RUN.OUT / "learning_curve"


def scene_rank_within_family(scene_id: str, family: str) -> str:
    return hashlib.sha256(f"{V3.SPLIT_SEED_V1}|{family}|{scene_id}".encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    cache = torch.load(RUN.CACHE, map_location="cpu", weights_only=False)
    order = cache["state_ids"]
    train_ids = set(cache["train_state_ids"])
    selection_ids = set(cache["selection_state_ids"])
    idx = {"train": [i for i, s in enumerate(order) if s in train_ids],
           "selection": [i for i, s in enumerate(order) if s in selection_ids]}

    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    by_id = {g.state_id: g for g in V3.load_branch_groups_v1("train", ceiling, ledger)}
    _g, records = ceiling.load_role_v1("train", ledger=ledger)

    encoder = P.load_stack(device)[0]
    encoder.eval().requires_grad_(False)

    panels = {}
    for split, rows in idx.items():
        groups = [by_id[order[i]] for i in rows]
        frames = torch.empty(len(groups), 3, 256, TOKEN_DIM)
        proprio = torch.zeros(len(groups), 30)
        for gi, group in enumerate(groups):
            artifacts = records[group.state_id]["context"]["rgb_artifact_ids"]
            batch = torch.stack([
                V3.preprocess_v3_frame_v1(ceiling.rgb_path_v1("train", a))
                for a in artifacts]).to(device)
            with torch.no_grad():
                frames[gi] = LN(encoder.forward_tokens(batch)[:, 1:, :]).cpu()
        tbar_flat = F.normalize(cache["successor_flat_unit"][rows].to(device), dim=-1)
        q = torch.softmax(RUN.group_cosine(tbar_flat, tbar_flat) / RUN.TAU_T, dim=-1)
        panels[split] = {
            "groups": groups,
            "frames": frames.to(device),
            "proprio": proprio.to(device),
            "commands": cache["commands"][rows].to(device),
            "tbar_tokens": cache["successor_normalised_tokens"][rows].to(device),
            "tbar_flat": tbar_flat,
            "q": q,
            "entropy": float(-(q * q.clamp_min(1e-12).log()).sum(-1).mean()),
            "scenes": [g.scene_id for g in groups],
            "families": [g.family for g in groups],
        }

    # Frozen changed-token threshold from the FULL 96-group train panel, reused
    # unchanged by every subset so the prediction metric is comparable.
    full = panels["train"]
    n_full = full["frames"].shape[0]
    adapter_probe = InputAdapterV1(ARM, proprio_dim=30).to(device)
    with torch.no_grad():
        base_state = adapter_probe(full["frames"], full["proprio"])
    current_full = base_state.repeat_interleave(NINE, dim=0)
    target_full = full["tbar_tokens"].reshape(n_full * NINE, 256, TOKEN_DIM)
    THRESHOLD = float(torch.quantile(
        (target_full - current_full).pow(2).mean(-1).flatten().float(), 0.75))
    del adapter_probe, base_state, current_full, target_full

    # nested family-balanced subsets of the 24 train scenes
    by_family = collections.defaultdict(set)
    for group in panels["train"]["groups"]:
        by_family[group.family].add(group.scene_id)
    ranked = {f: sorted(sorted(s), key=lambda x: scene_rank_within_family(x, f))
              for f, s in by_family.items()}

    one_hot_all = torch.eye(NINE, device=device)
    report = {"status": STATUS, "claim_bearing": False, "arm": ARM,
              "encoder": "frozen (diagnostic only)",
              "reference": {"ce_uniform": math.log(9), "top1_chance": 1.0 / 9.0,
                            "ce_floor_entropy_of_Q": {k: panels[k]["entropy"]
                                                      for k in panels}},
              "changed_token_threshold": THRESHOLD,
              "selection_set": sorted(set(panels["selection"]["scenes"])),
              "subsets": {}}

    def evaluate(adapter, predictor, panel):
        n = panel["frames"].shape[0]
        adapter.eval(); predictor.eval()
        with torch.no_grad():
            state = adapter(panel["frames"], panel["proprio"])
            expanded = state.repeat_interleave(NINE, dim=0)
            pred = LN(predictor(expanded, one_hot_all.repeat(n, 1),
                                panel["commands"].reshape(-1, 3)))
        stats = matching_stats(pred.reshape(n, NINE, -1), panel["tbar_flat"],
                               panel["q"], RUN.TAU_P)
        stats["normalised_ce"] = normalised_ce(stats["matching_ce"], panel["entropy"])
        target = panel["tbar_tokens"].reshape(n * NINE, 256, TOKEN_DIM)
        mask = ((target - expanded).pow(2).mean(-1) > THRESHOLD)
        correct = R.metrics(pred, target, expanded, mask)
        shuffled = []
        for seed in DERANGEMENT_SEEDS:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(NINE, generator=g).to(device)
            sh = pred.reshape(n, NINE, 256, TOKEN_DIM)[:, perm].reshape(
                n * NINE, 256, TOKEN_DIM)
            shuffled.append(R.metrics(sh, target, expanded, mask)["cosine"])
        stats["correct_minus_shuffled"] = correct["cosine"] - float(np.mean(shuffled))
        adapter.train(); predictor.train()
        return stats

    for per_family in SUBSETS:
        scenes = {s for f, ranked_scenes in ranked.items() for s in ranked_scenes[:per_family]}
        rows = [i for i, sc in enumerate(panels["train"]["scenes"]) if sc in scenes]
        sub = {k: (panels["train"][k][rows] if torch.is_tensor(panels["train"][k])
                   else [panels["train"][k][i] for i in rows])
               for k in ("frames", "proprio", "commands", "tbar_tokens", "scenes",
                         "families", "groups")}
        sub["tbar_flat"] = panels["train"]["tbar_flat"][rows]
        sub["q"] = panels["train"]["q"][rows]
        sub["entropy"] = float(-(sub["q"] * sub["q"].clamp_min(1e-12).log()).sum(-1).mean())
        n_groups = len(rows)
        steps = int(round(FULL_STEPS * n_groups / FULL_GROUPS))
        eval_every = max(1, steps // EVAL_POINTS)
        label = f"{len(scenes)}_scenes"

        torch.manual_seed(RUN.SEED)
        adapter = InputAdapterV1(ARM, proprio_dim=30).to(device)
        predictor = initialize_token_predictor_v1(RUN.SEED).to(device)
        optimiser = torch.optim.AdamW(
            list(adapter.parameters()) + list(predictor.parameters()),
            lr=LR, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(RUN.SEED)
        cursor, group_order = 0, torch.randperm(n_groups, generator=generator).tolist()
        adapter.train(); predictor.train()

        curve, best = [], None
        for step in range(1, steps + 1):
            picked = []
            for _ in range(GROUPS_PER_STEP):
                if cursor >= n_groups:
                    group_order = torch.randperm(n_groups, generator=generator).tolist()
                    cursor = 0
                picked.append(group_order[cursor]); cursor += 1
            sel = torch.tensor(picked, device=device)
            state = adapter(sub["frames"][sel], sub["proprio"][sel])
            state = state.repeat_interleave(NINE, dim=0)
            pred = LN(predictor(state, one_hot_all.repeat(len(picked), 1),
                                sub["commands"][sel].reshape(-1, 3)))
            branch_jepa = F.mse_loss(
                pred, sub["tbar_tokens"][sel].reshape(-1, 256, TOKEN_DIM))
            logits = torch.einsum(
                "gid,gjd->gij",
                F.normalize(pred.reshape(len(picked), NINE, -1), dim=-1),
                sub["tbar_flat"][sel]) / RUN.TAU_P
            match = -(sub["q"][sel] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = branch_jepa + RUN.LAMBDA_MATCH * match
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(adapter.parameters()) + list(predictor.parameters()), 1.0)
            optimiser.step()

            if step % eval_every == 0 or step == steps:
                selection = evaluate(adapter, predictor, panels["selection"])
                curve.append({"step": step,
                              "presentations_per_group":
                                  step * GROUPS_PER_STEP / n_groups,
                              **{f"selection_{k}": v for k, v in selection.items()}})
                if best is None or selection["normalised_ce"] < best["normalised_ce"]:
                    best = {"step": step, **selection}
                    torch.save({"adapter": adapter.state_dict(),
                                "predictor": predictor.state_dict(),
                                "step": step, "selection": selection},
                               OUT / f"best_selection_{label}.pt")

        final_selection = evaluate(adapter, predictor, panels["selection"])
        final_train = evaluate(adapter, predictor, sub)
        torch.save({"adapter": adapter.state_dict(), "predictor": predictor.state_dict(),
                    "step": steps, "selection": final_selection},
                   OUT / f"final_{label}.pt")

        best_state = torch.load(OUT / f"best_selection_{label}.pt",
                                map_location=device, weights_only=False)
        adapter.load_state_dict(best_state["adapter"])
        predictor.load_state_dict(best_state["predictor"])
        best_train = evaluate(adapter, predictor, sub)

        per_scene = {}
        for scene in sorted(set(panels["selection"]["scenes"])):
            srows = torch.tensor([i for i, s in enumerate(panels["selection"]["scenes"])
                                  if s == scene], device=device)
            view = {"frames": panels["selection"]["frames"][srows],
                    "proprio": panels["selection"]["proprio"][srows],
                    "commands": panels["selection"]["commands"][srows],
                    "tbar_tokens": panels["selection"]["tbar_tokens"][srows],
                    "tbar_flat": panels["selection"]["tbar_flat"][srows],
                    "q": panels["selection"]["q"][srows]}
            view["entropy"] = float(
                -(view["q"] * view["q"].clamp_min(1e-12).log()).sum(-1).mean())
            per_scene[scene] = evaluate(adapter, predictor, view)

        report["subsets"][label] = {
            "scenes": sorted(scenes), "scenes_per_family": per_family,
            "train_groups": n_groups, "steps": steps, "eval_every": eval_every,
            "presentations_per_group": steps * GROUPS_PER_STEP / n_groups,
            "best_selection": {"step": best["step"],
                               **{k: v for k, v in best.items() if k != "step"}},
            "final_selection": final_selection,
            "train_at_best": best_train,
            "train_at_final": final_train,
            "gap_at_best": best_train["normalised_ce"] - best["normalised_ce"],
            "gap_at_final": final_train["normalised_ce"] - final_selection["normalised_ce"],
            "per_selection_scene_at_best": per_scene,
            "curve": curve,
        }
        del adapter, predictor
        torch.cuda.empty_cache()

    (RUN.OUT / "learning_curve_diagnostic.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                      for k, v in report["subsets"].items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
