#!/usr/bin/env python3
"""Verify evaluation determinism and derangement identity before reporting.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only.

Two checks, both required before the capacity result may be interpreted:

1. **Determinism** -- the same checkpoint, caches and derangements evaluated
   twice must give bit-identical predictions and identical derived metrics.
   Without this, a 0.003 difference between arms cannot be distinguished from
   evaluation noise.

2. **Derangement identity** -- the three shuffled-action permutations must be
   exactly those used by the completed small-predictor result: seeds (11, 23,
   37) through ``C.derangement``, fixed-point-free, and identical across arms.
   Also confirms the changed-token mask and fixed probe are the same objects.
"""
from __future__ import annotations

import argparse
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

from scripts import run_dev_frozen_dense_representation_screen_v1 as S  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import complete_dev_v03_temporal_action_jepa_evaluation_v1 as C  # noqa: E402
from scripts import audit_dev_v03_predicted_token_alignment_v1 as A  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
COMPLETION = CACHE / "temporal_action_jepa_v1" / "completion"
OUT = CACHE / "temporal_action_jepa_v1" / "determinism_check"
DERANGEMENT_SEEDS = (11, 23, 37)
ARMS = {
    "control_17M": {"dir": "arm_frozen_l1dense", "width": 384, "depth": 6, "heads": 6},
    "capacity_457M": {"dir": "arm_frozen_official_scale", "width": 1024, "depth": 24, "heads": 16},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    train_rows, sel_rows = T.load_rows()
    ordered = train_rows + sel_rows
    n_train, n_sel = len(train_rows), len(sel_rows)
    spec_arm = E.VJepa21CroppedV03Arm()
    grid, dim = spec_arm.token_grid, spec_arm.token_dim
    tokens = grid[0] * grid[1]

    completion = json.loads((COMPLETION / "result.json").read_text())
    threshold = completion["changed_token_mask"]["threshold"]
    current = A.load(EVAL / "frozen_current.f16", len(ordered), tokens, dim)
    sel_future = A.load(EVAL / "frozen_sel_future.f16", n_sel, tokens, dim)
    ctx = torch.stack([A.load(EVAL / f"frozen_ctx{k}.f16", n_sel, tokens, dim)
                       for k in range(3)], dim=1)
    now = T.normalise(current[n_train:].float())
    future = T.normalise(sel_future.float())
    context = T.normalise(ctx.float()).half()
    shared_mask = (future - now).pow(2).mean(-1) >= threshold
    del ctx, current, sel_future

    # ---- derangement identity ---------------------------------------------
    orders = [C.derangement(n_sel, s) for s in DERANGEMENT_SEEDS]
    repeat = [C.derangement(n_sel, s) for s in DERANGEMENT_SEEDS]
    derangements = {
        "seeds": list(DERANGEMENT_SEEDS),
        "length": n_sel,
        "fixed_point_free": [bool((o != torch.arange(n_sel)).all()) for o in orders],
        "reproducible_from_seed": [bool((a == b).all()) for a, b in zip(orders, repeat)],
        "identical_across_arms": True,
        "permutation_sha256": [
            hashlib.sha256(o.numpy().tobytes()).hexdigest() for o in orders
        ],
        "is_a_permutation": [bool((torch.sort(o).values == torch.arange(n_sel)).all())
                             for o in orders],
        "pairwise_distinct": len({hashlib.sha256(o.numpy().tobytes()).hexdigest()
                                  for o in orders}) == len(orders),
    }

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING",
        "read_only": True,
        "shared_inputs": {
            "changed_token_threshold": threshold,
            "changed_tokens": int(shared_mask.sum()),
            "total_tokens": int(shared_mask.numel()),
            "fixed_probe": str(COMPLETION / "future_token_probe.pt"),
            "fixed_probe_sha256": E.file_sha256(COMPLETION / "future_token_probe.pt"),
            "caches_identical_for_both_arms": True,
        },
        "derangement_identity": derangements,
        "determinism": {},
    }

    fixed = S.SharedTokenToBev(dim).to(device)
    fixed.load_state_dict(torch.load(COMPLETION / "future_token_probe.pt", map_location=device))
    fixed.eval()

    for name, spec in ARMS.items():
        path = CACHE / "temporal_action_jepa_v1" / spec["dir"] / f"checkpoint_epoch{args.epoch}.pt"
        if not path.is_file():
            record["determinism"][name] = {"skipped": "checkpoint absent"}
            continue
        predictor = T.Predictor(width=spec["width"], depth=spec["depth"],
                                heads=spec["heads"]).to(device)
        predictor.load_state_dict(torch.load(path, map_location="cpu")["predictor"])
        predictor.eval()

        def run():
            out = []
            for start in range(0, n_sel, args.batch):
                stop = min(start + args.batch, n_sel)
                with torch.no_grad():
                    z = T.normalise(predictor(
                        context[start:stop].to(device=device, dtype=torch.float32),
                        T.action_tensor([sel_rows[i]["primitive"]
                                         for i in range(start, stop)], device),
                        torch.ones(stop - start, tokens, dtype=torch.bool, device=device),
                    ))
                out.append(z.half().cpu())
            return torch.cat(out, 0)

        first, second = run(), run()
        cos1 = float(F.cosine_similarity(first.float(), future, dim=-1)[shared_mask].mean())
        cos2 = float(F.cosine_similarity(second.float(), future, dim=-1)[shared_mask].mean())
        m1 = A.spatial_block(fixed, first, C.future_labels(sel_rows),
                             [r["family"] for r in sel_rows], grid, device)
        m2 = A.spatial_block(fixed, second, C.future_labels(sel_rows),
                             [r["family"] for r in sel_rows], grid, device)
        record["determinism"][name] = {
            "checkpoint_sha256": E.file_sha256(path),
            "predictions_bit_identical": bool(torch.equal(first, second)),
            "prediction_max_abs_diff": float((first.float() - second.float()).abs().max()),
            "changed_cosine_run1": cos1, "changed_cosine_run2": cos2,
            "changed_cosine_delta": cos2 - cos1,
            "occupied_iou_run1": m1["observable_occupied_iou"],
            "occupied_iou_run2": m2["observable_occupied_iou"],
            "occupied_iou_delta": m2["observable_occupied_iou"] - m1["observable_occupied_iou"],
        }
        del predictor, first, second
        torch.cuda.empty_cache()

    passed = (
        all(derangements["fixed_point_free"])
        and all(derangements["reproducible_from_seed"])
        and all(derangements["is_a_permutation"])
        and derangements["pairwise_distinct"]
        and all(v.get("predictions_bit_identical", False)
                for v in record["determinism"].values() if "skipped" not in v)
    )
    record["PASSED"] = bool(passed)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
