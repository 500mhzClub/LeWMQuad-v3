#!/usr/bin/env python3
"""Evaluate the L1-dense supervision successor against the registered gate.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  The encoder is frozen and byte-identical to
the original frozen arm, so every encoder feature cache is reused unchanged and
only the predictor differs.

Acceptance requires ALL of:
  1. predicted future occupied IoU exceeds persistence under the fixed
     true-future probe;
  2. correct-minus-shuffled margin retains or improves +0.0586;
  3. open_obstacle_field predicted geometry improves;
  4. not achieved through diffuse occupied over-prediction;
  5. predicted tokens remain closer to the canonical true-future interface, not
     merely decodable by a fresh supervised probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

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
OUT = CACHE / "temporal_action_jepa_v1" / "supervision_successor"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
DERANGEMENT_SEEDS = (11, 23, 37)

BASELINE_MARGIN = 0.0585573514302572
BASELINE_PERSISTENCE_IOU = 0.3133053567020489
BASELINE_RAW_PREDICTED_IOU = 0.2653421175494112
BASELINE_OFIELD_PREDICTED_IOU = 0.1385


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm-dir", default="arm_frozen_l1dense")
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pred-width", type=int, default=384)
    ap.add_argument("--pred-depth", type=int, default=6)
    ap.add_argument("--pred-heads", type=int, default=6)
    ap.add_argument("--normalise-output", action="store_true", default=True,
                    help="the successor normalises its predictor output, as the official loss block does")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows, sel_rows = T.load_rows()
    ordered = train_rows + sel_rows
    n_train, n_sel = len(train_rows), len(sel_rows)
    arm = E.VJepa21CroppedV03Arm()
    grid, dim = arm.token_grid, arm.token_dim
    tokens = grid[0] * grid[1]
    families = [r["family"] for r in sel_rows]
    labels_future = C.future_labels(sel_rows)

    fixed = S.SharedTokenToBev(dim).to(device)
    fixed.load_state_dict(torch.load(COMPLETION / "future_token_probe.pt", map_location=device))
    fixed.eval()
    completion = json.loads((COMPLETION / "result.json").read_text())
    threshold = completion["changed_token_mask"]["threshold"]

    # frozen-encoder caches, reused unchanged
    current = A.load(EVAL / "frozen_current.f16", len(ordered), tokens, dim)
    sel_future = A.load(EVAL / "frozen_sel_future.f16", n_sel, tokens, dim)
    ctx = torch.stack([A.load(EVAL / f"frozen_ctx{k}.f16", n_sel, tokens, dim)
                       for k in range(3)], dim=1)
    now = T.normalise(current[n_train:].float())
    future = T.normalise(sel_future.float())
    context = T.normalise(ctx.float()).half()
    shared_mask = (future - now).pow(2).mean(-1) >= threshold
    del ctx, current, sel_future

    checkpoint = torch.load(
        CACHE / "temporal_action_jepa_v1" / args.arm_dir / f"checkpoint_epoch{args.epoch}.pt",
        map_location="cpu",
    )
    predictor = T.Predictor(width=args.pred_width, depth=args.pred_depth, heads=args.pred_heads).to(device)
    predictor.load_state_dict(checkpoint["predictor"])
    predictor.eval()
    if checkpoint.get("encoder_trainable"):
        raise RuntimeError("successor must have a frozen encoder")

    actions = T.action_tensor([r["primitive"] for r in sel_rows], torch.device("cpu"))

    def run(action_tensor):
        out = []
        for start in range(0, n_sel, args.batch):
            stop = min(start + args.batch, n_sel)
            with torch.no_grad():
                z = predictor(
                    context[start:stop].to(device=device, dtype=torch.float32),
                    action_tensor[start:stop].to(device),
                    torch.ones(stop - start, tokens, dtype=torch.bool, device=device),
                )
                if args.normalise_output:
                    z = T.normalise(z)
            out.append(z.half().cpu())
        return torch.cat(out, 0)

    predicted = run(actions)
    orders = [C.derangement(n_sel, s) for s in DERANGEMENT_SEEDS]
    correct_cos = float(F.cosine_similarity(
        predicted.float(), future, dim=-1)[shared_mask].mean())
    shuffled_cos = float(np.mean([
        float(F.cosine_similarity(run(actions[o]).float(), future, dim=-1)[shared_mask].mean())
        for o in orders
    ]))
    persistence_cos = float(F.cosine_similarity(now, future, dim=-1)[shared_mask].mean())

    variants = {
        "predicted": predicted,
        "persistence": now.half(),
        "true_future_reference": future.half(),
    }
    spatial = {
        tag: A.spatial_block(fixed, block, labels_future, families, grid, device)
        for tag, block in variants.items()
    }

    # clause 5: distance to the canonical interface, no fresh probe involved
    interface = {
        "predicted_mean_squared_error_to_true_future": float(
            (predicted.float() - future).pow(2).mean()),
        "persistence_mean_squared_error_to_true_future": float((now - future).pow(2).mean()),
        "baseline_smooth_l1_masked_predicted_mse": completion["arms"]["frozen"][
            "prediction"]["correct_action"]["normalised_error_vs_persistence"],
    }

    record = {
        "status": STATUS, "claim_bearing": False,
        "successor": {
            "intervention": "supervision only: official DROID loss block minus auto_steps",
            "changed": ["loss_exp 1.0 (L1) instead of smooth-L1",
                        "dense over every future token instead of a sampled 50% mask",
                        "normalize_reps applied to the predictor output as well as the target"],
            "unchanged": ["predictor architecture and capacity", "three-frame context",
                          "frozen official V-JEPA 2.1 ViT-L encoder", "rows, ordering, seed",
                          "schedule, batch, learning rate", "action conditioning", "scene split"],
            "checkpoint": str(CACHE / "temporal_action_jepa_v1" / args.arm_dir /
                              f"checkpoint_epoch{args.epoch}.pt"),
            "encoder_moved": False,
        },
        "rows": {"train": n_train, "checkpoint_selection": n_sel},
        "changed_token_mask": {"threshold": threshold, "changed": int(shared_mask.sum()),
                               "total": int(shared_mask.numel()),
                               "derived_from": "frozen arm train representation, unchanged"},
        "prediction": {
            "correct_changed_cosine": correct_cos,
            "shuffled_changed_cosine": shuffled_cos,
            "persistence_changed_cosine": persistence_cos,
            "correct_minus_shuffled": correct_cos - shuffled_cos,
            "correct_minus_persistence": correct_cos - persistence_cos,
        },
        "spatial_fixed_probe": spatial,
        "interface_distance": interface,
        "baseline_smooth_l1_masked": {
            "margin": BASELINE_MARGIN,
            "predicted_occupied_iou": BASELINE_RAW_PREDICTED_IOU,
            "persistence_occupied_iou": BASELINE_PERSISTENCE_IOU,
            "open_obstacle_field_predicted_iou": BASELINE_OFIELD_PREDICTED_IOU,
        },
    }

    p = spatial["predicted"]; q = spatial["persistence"]
    of_p = p["open_obstacle_field"]; of_q = q["open_obstacle_field"]
    clauses = {
        "1_predicted_beats_persistence": p["observable_occupied_iou"] > q["observable_occupied_iou"],
        "2_margin_retained_or_improved": (correct_cos - shuffled_cos) >= BASELINE_MARGIN,
        "3_open_obstacle_field_improves":
            of_p["observable_occupied_iou"] > BASELINE_OFIELD_PREDICTED_IOU,
        "4_not_diffuse_over_prediction": (
            p["predicted_occupied_fraction_all_cells"]
            <= q["predicted_occupied_fraction_all_cells"]
            or p["observable_occupied_precision"] >= q["observable_occupied_precision"]
        ),
        "5_closer_to_canonical_interface": (
            interface["predicted_mean_squared_error_to_true_future"]
            < interface["persistence_mean_squared_error_to_true_future"]
        ),
    }
    record["acceptance_clauses"] = clauses
    record["ACCEPTED"] = all(clauses.values())
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({
        "margin": record["prediction"]["correct_minus_shuffled"],
        "predicted_occupied_iou": p["observable_occupied_iou"],
        "persistence_occupied_iou": q["observable_occupied_iou"],
        "open_obstacle_field_predicted_iou": of_p["observable_occupied_iou"],
        "predicted_occupied_fraction": p["predicted_occupied_fraction_all_cells"],
        "clauses": clauses, "ACCEPTED": record["ACCEPTED"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
