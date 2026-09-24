#!/usr/bin/env python3
"""Evaluate one seed's four epoch-21 checkpoints on the frozen selection set.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING as code; it produces scientific results.

BLINDING.  This script writes its full result to disk and prints ONLY operational
status.  It never prints cell-to-cell differences, interaction values, family
contrasts or any inferential quantity, so running it during the initial stage
cannot unblind the operator.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import run_dev_v03_two_step_rollout_v1 as R  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import eval_dev_proprio_factorial_v1 as E  # noqa: E402
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
MAX_H = 4
DERANGEMENT_SEED = E.DERANGEMENT_SEED
HORIZONS = D.CACHE / "horizons"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def deranged(count: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(DERANGEMENT_SEED)
    order = torch.randperm(count, generator=generator)
    while bool((order == torch.arange(count)).any()):
        order = torch.randperm(count, generator=generator)
    return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-index", type=int, required=True)
    ap.add_argument("--authorisation", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from scripts import authorise_dev_proprio_launch_v1 as AUTH
    receipt = AUTH.verify(args.seed_index, Path(args.authorisation))
    seed = D.SEED_REGISTRY[args.seed_index]
    seed_out = D.OUT / f"seed_{seed}"
    device = D.resolve_device()

    map_record = MAP.load()
    factorial = FM.load()
    rows = [json.loads(l) for l in
            (D.PROPRIO / "proprio_rows.jsonl").read_text().splitlines() if l.strip()]
    stats = json.loads((D.PROPRIO / "proprio_norm_stats.json").read_text())

    loader = D.CanonicalLoader(map_record, rows, stats, split="checkpoint_selection",
                               expected_digest=map_record["digest"], factorial=factorial,
                               expected_factorial_digest=factorial["digest"])
    n = len(loader)
    clusters = [e["episode_cluster"] for e in loader.entries]
    families = [e["family"] for e in loader.entries]
    thresholds = factorial["horizon_masks"]["thresholds"]

    # ---- frozen inputs, identical for every cell ---------------------------
    positions = list(range(n))
    batch = loader.batch(positions, device, stats)
    now = batch["context"][:, 2]
    targets = {1: batch["y1"], 2: batch["y2"]}
    masks = {1: (targets[1] - now).pow(2).mean(-1) >= thresholds["step1"],
             2: (targets[2] - now).pow(2).mean(-1) >= thresholds["step2"]}

    horizon_rows = [json.loads(l) for l in
                    (HORIZONS / "FINAL" / "FINAL_horizon_rows_479.jsonl").read_text().splitlines()
                    if l.strip()]
    horizon_rows = [r for r in horizon_rows if r["max_horizon"] >= MAX_H]
    horizon_position = {r["pair_sha256"]: i for i, r in enumerate(horizon_rows)}
    covered = [i for i, e in enumerate(loader.entries) if e["pair_sha256"] in horizon_position]
    picks = [horizon_position[loader.entries[i]["pair_sha256"]] for i in covered]
    for h in (3, 4):
        blob = T.normalise(R.load_cache(HORIZONS / f"target_h{h}.f16",
                                        len(horizon_rows))[picks].float()).to(device)
        targets[h] = blob
        masks[h] = (blob - now[covered]).pow(2).mean(-1) >= thresholds["step2"]

    actions = []
    for h in range(MAX_H):
        actions.append(torch.tensor(
            [r["action_blocks"][min(h, len(r["action_blocks"]) - 1)] for r in loader.rows],
            dtype=torch.float32, device=device))
    order = deranged(n).to(device)

    results, operational = {}, []
    for cell in D.CELLS:
        path = seed_out / f"seed_{seed}_{cell}_epoch{D.CHECKPOINT_EPOCH}.pt"
        if not path.is_file():
            raise SystemExit(f"missing epoch-{D.CHECKPOINT_EPOCH} checkpoint for {cell}")
        spec = D.CELL_SPEC[cell]
        model = P.build_paired(seed, use_proprio=spec["use_proprio"]).to(device)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        def unroll(action_list):
            outs = []
            for start in range(0, n, args.batch):
                stop = min(start + args.batch, n)
                sub = {k: (v[start:stop] if torch.is_tensor(v) else v)
                       for k, v in batch.items()}
                steps = P.unroll(model, sub["context"],
                                 [a[start:stop] for a in action_list],
                                 sub["proprio"] if spec["use_proprio"] else None,
                                 sub["control"], max_h=MAX_H)
                outs.append([s.cpu() for s in steps])
            return [torch.cat([o[h] for o in outs], 0) for h in range(MAX_H)]

        with torch.no_grad():
            correct = unroll(actions)
            shuffled = unroll([a[order] for a in actions])

        per_h = {}
        for h in range(1, MAX_H + 1):
            index = covered if h >= 3 else list(range(n))
            p = correct[h - 1][index].to(device)
            q = shuffled[h - 1][index].to(device)
            t, m = targets[h], masks[h]
            cosine = F.cosine_similarity(p, t, dim=-1)
            shuffled_cosine = F.cosine_similarity(q, t, dim=-1)
            sub_clusters = [clusters[i] for i in index]
            sub_families = [families[i] for i in index]
            aggregate = E.episode_then_family(
                E.row_scores(cosine.cpu(), m.cpu()), sub_clusters, sub_families)
            shuffled_aggregate = E.episode_then_family(
                E.row_scores(shuffled_cosine.cpu(), m.cpu()), sub_clusters, sub_families)
            per_h[str(h)] = {
                "equal_family_cosine": aggregate["equal_family"],
                "per_family_cosine": aggregate["per_family"],
                "episode_clusters": aggregate["episode_clusters"],
                "equal_family_shuffled_cosine": shuffled_aggregate["equal_family"],
                "correct_minus_shuffled_margin":
                    aggregate["equal_family"] - shuffled_aggregate["equal_family"],
                "secondary_token_pooled_cosine": E.token_pooled(cosine.cpu(), m.cpu()),
                "occupied": E.occupied_metrics(p.cpu(), t.cpu(), m.cpu()),
                "rows": len(index), "changed_tokens": int(m.sum()),
            }
        results[cell] = {
            "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
            "epoch": payload["epoch"], "per_horizon": per_h,
        }
        operational.append({"cell": cell, "checkpoint_epoch": payload["epoch"],
                            "checkpoint_sha256": sha256_file(path)[:16],
                            "finite": all(np.isfinite(v["equal_family_cosine"])
                                          for v in per_h.values())})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    record = {
        "status": STATUS, "claim_bearing": False,
        "seed": seed, "seed_index": args.seed_index,
        "authorisation_receipt_digest": receipt["receipt_digest"],
        "factorial_manifest_digest": factorial["digest"],
        "selection_rows": n,
        "mask_digest": factorial["horizon_masks"]["mask_digest"],
        "changed_token_counts": factorial["horizon_masks"]["changed_token_counts"],
        "derangement_seed": DERANGEMENT_SEED,
        "estimator": ("valid tokens within a row -> rows within an episode cluster -> "
                      "episodes within a family -> unweighted mean of eight families"),
        "cells": results,
    }
    out = seed_out / "selection_result.json"
    out.write_text(json.dumps(record, indent=2))
    # BLINDED: operational status only.
    print(json.dumps({"seed": seed, "selection_rows": n,
                      "cells_evaluated": operational,
                      "result_written": str(out),
                      "blinded": "no comparative quantity is printed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
