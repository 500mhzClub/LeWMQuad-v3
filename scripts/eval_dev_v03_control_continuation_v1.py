#!/usr/bin/env python3
"""Evaluate ONLY the one-step control's continuation epochs.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The rollout arm is frozen at epoch 22 and is neither trained nor re-evaluated
here; its curve is read from the preserved matched-duration result.  Only the
control's new epochs are evaluated, against the identical cached-feature path,
fixed probe, frozen changed-token masks, derangements and metric definitions.

Prospective convergence test for the continuation block, fixed before the block
was read:

    early_best  = max occupied IoU over the first three epochs of the block
    late_best   = max occupied IoU over the last three
    early_margin, late_margin = mean margin over the same windows

    converged iff |late_best - early_best| <= 0.005
                  and |late_margin - early_margin| <= 0.003

    d_iou > +0.005  -> still improving
    d_iou < -0.005  -> late deterioration
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
from scripts import run_dev_v03_two_step_rollout_v1 as R  # noqa: E402
from scripts import eval_dev_v03_two_step_rollout_v1 as V  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
COMPLETION = CACHE / "temporal_action_jepa_v1" / "completion"
TWO = CACHE / "two_step"
MATCHED = TWO / "evaluation" / "MATCHED_24_EPOCH_result_epochs_0_23.json"
ARM = None  # set from --arm-dir
OUT = TWO / "continuation"
DERANGEMENT_SEEDS = (11, 23, 37)
GATE_MARGIN = 0.0585573514302572


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--arm-dir", default="arm_one_step")
    ap.add_argument("--arm-name", default="one_step")
    ap.add_argument("--baseline-curve", default="one_step",
                    help="curve in the matched result to prepend; use 'none' for a fresh arm")
    ap.add_argument("--block", type=int, nargs=2, required=True,
                    help="inclusive epoch range of the continuation block, e.g. 24 29")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lo, hi = args.block
    arm_dir = TWO / "arms" / args.arm_dir
    early_window = tuple(range(lo, lo + 3))
    late_window = tuple(range(hi - 2, hi + 1))

    matched = json.loads(MATCHED.read_text())
    base = [json.loads(l) for l in (CACHE / "temporal_rows.jsonl").read_text().splitlines() if l.strip()]
    two = [json.loads(l) for l in (TWO / "two_step_rows.jsonl").read_text().splitlines() if l.strip()]
    base_train = [r for r in base if r["role"] == "train"]
    base_sel = [r for r in base if r["role"] == "checkpoint_selection"]
    pos_sel = {r["pair_sha256"]: i for i, r in enumerate(base_sel)}
    sel_rows = [r for r in two if r["role"] == "checkpoint_selection"]
    sel_idx = np.array([pos_sel[r["pair_sha256"]] for r in sel_rows])
    n_bt, n_bs = len(base_train), len(base_sel)
    families = [r["family"] for r in sel_rows]
    scene_ids = [r["scene"] for r in sel_rows]

    ctx = torch.stack([R.load_cache(EVAL / f"frozen_ctx{k}.f16", n_bs)[sel_idx] for k in range(3)], 1)
    current = R.load_cache(EVAL / "frozen_current.f16", n_bt + n_bs)[n_bt:][sel_idx]
    y1 = R.load_cache(EVAL / "frozen_sel_future.f16", n_bs)[sel_idx]
    y2 = R.load_cache(TWO / "frozen_sel_step2.f16", len(sel_rows))
    now, t1, t2 = (T.normalise(current.float()), T.normalise(y1.float()), T.normalise(y2.float()))
    context1 = T.normalise(ctx.float()).half()
    del ctx

    masks = matched["masks"]
    mask1 = (t1 - now).pow(2).mean(-1) >= masks["step1_threshold"]
    mask2 = (t2 - now).pow(2).mean(-1) >= masks["step2_threshold"]

    fixed = S.SharedTokenToBev(R.DIM).to(device)
    fixed.load_state_dict(torch.load(COMPLETION / "future_token_probe.pt", map_location=device))
    fixed.eval()
    labels1 = C.future_labels(sel_rows)
    persistence = matched["reference"]["persistence_step1_spatial"]

    orders = [C.derangement(len(sel_rows), s) for s in DERANGEMENT_SEEDS]
    a0 = T.action_tensor([r["action_step1"] for r in sel_rows], torch.device("cpu"))
    a1 = T.action_tensor([r["action_step2"] for r in sel_rows], torch.device("cpu"))
    training = json.loads((arm_dir / "result.json").read_text())

    curve = ([] if args.baseline_curve == "none"
             else [e for e in matched["curves"][args.baseline_curve] if e["epoch"] < lo])
    final = {}
    scan = range(0, hi + 1) if args.baseline_curve == "none" else range(lo, hi + 1)
    for epoch in scan:
        path = arm_dir / f"checkpoint_epoch{epoch}.pt"
        if not path.is_file():
            continue
        predictor = T.Predictor(**R.PRED).to(device)
        predictor.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
        predictor.eval()

        def rollout(x0, x1):
            o1, o2 = [], []
            for s in range(0, len(sel_rows), args.batch):
                e = min(s + args.batch, len(sel_rows))
                m = torch.ones(e - s, R.TOKENS, dtype=torch.bool, device=device)
                with torch.no_grad():
                    c1 = context1[s:e].to(device=device, dtype=torch.float32)
                    q1 = T.normalise(predictor(c1, x0[s:e].to(device), m))
                    c2 = torch.stack([c1[:, 1], c1[:, 2], q1], dim=1)
                    q2 = T.normalise(predictor(c2, x1[s:e].to(device), m))
                o1.append(q1.half().cpu()); o2.append(q2.half().cpu())
            return torch.cat(o1, 0), torch.cat(o2, 0)

        p1, p2 = rollout(a0, a1)
        shuffled = [rollout(a0[o], a1[o]) for o in orders]
        spatial1 = A.spatial_block(fixed, p1, labels1, families, (24, 32), device)
        entry = {
            "epoch": epoch,
            "train_e1": next((e["e1"] for e in training["epochs"] if e["epoch"] == epoch), None),
            "train_e2": 0.0,
            "step1": V.latent(p1, now, t1, mask1),
            "step1_shuffled": {k: float(np.mean([V.latent(s[0], now, t1, mask1)[k] for s in shuffled]))
                               for k in ("changed_cosine", "normalised_error_vs_persistence")},
            "step2": V.latent(p2, now, t2, mask2),
            "step2_shuffled": {k: float(np.mean([V.latent(s[1], now, t2, mask2)[k] for s in shuffled]))
                               for k in ("changed_cosine", "normalised_error_vs_persistence")},
            "step1_occupied_iou": spatial1["observable_occupied_iou"],
            "step1_occupied_precision": spatial1["observable_occupied_precision"],
            "step1_occupied_recall": spatial1["observable_occupied_recall"],
            "step1_occupied_fraction": spatial1["predicted_occupied_fraction_all_cells"],
        }
        entry["step1_margin"] = entry["step1"]["changed_cosine"] - entry["step1_shuffled"]["changed_cosine"]
        entry["step2_margin"] = entry["step2"]["changed_cosine"] - entry["step2_shuffled"]["changed_cosine"]
        entry["step1_to_step2_degradation"] = entry["step1"]["changed_cosine"] - entry["step2"]["changed_cosine"]
        curve.append(entry)

        if epoch in late_window:
            combos = {"correct_a0_correct_a1": (a0, a1),
                      "shuffled_a0_correct_a1": (a0[orders[0]], a1),
                      "correct_a0_shuffled_a1": (a0, a1[orders[0]]),
                      "shuffled_a0_shuffled_a1": (a0[orders[0]], a1[orders[0]])}
            sequence, per_scene = {}, {}
            for tag, (x0, x1) in combos.items():
                q1, q2 = rollout(x0, x1)
                sequence[tag] = {"step1": V.latent(q1, now, t1, mask1),
                                 "step2": V.latent(q2, now, t2, mask2)}
                if tag == "correct_a0_correct_a1":
                    for sc in sorted(set(scene_ids)):
                        pick = torch.tensor([i for i, v in enumerate(scene_ids) if v == sc])
                        per_scene[sc] = V.latent(q2[pick], now[pick], t2[pick],
                                                 mask2[pick])["changed_cosine"]
                del q1, q2
            final[str(epoch)] = {
                "epoch": epoch, "step1_spatial": spatial1,
                "step1_spatial_per_family": spatial1["per_family"],
                "open_obstacle_field": spatial1["open_obstacle_field"],
                "action_sequence_conditions": sequence,
                "per_scene_step2_cosine": per_scene,
                "step1_to_step2_degradation": entry["step1_to_step2_degradation"],
                "step1_margin": entry["step1_margin"], "step2": entry["step2"],
            }
        del predictor, p1, p2, shuffled
        torch.cuda.empty_cache()
        print(f"  [{args.arm_name}] epoch {epoch}: s1cos {entry['step1']['changed_cosine']:.4f} "
              f"margin {entry['step1_margin']:+.4f} occIoU {entry['step1_occupied_iou']:.4f} | "
              f"s2cos {entry['step2']['changed_cosine']:.4f}", flush=True)

    iou = {e["epoch"]: e["step1_occupied_iou"] for e in curve}
    mar = {e["epoch"]: e["step1_margin"] for e in curve}
    ei = [iou[k] for k in early_window if k in iou]
    li = [iou[k] for k in late_window if k in iou]
    em = [mar[k] for k in early_window if k in mar]
    lm = [mar[k] for k in late_window if k in mar]
    d_iou = max(li) - max(ei)
    d_mar = abs(float(np.mean(lm)) - float(np.mean(em)))
    convergence = {
        "block": [lo, hi], "early_window": list(early_window), "late_window": list(late_window),
        "early_best_iou": max(ei), "late_best_iou": max(li),
        "late_minus_early_iou": d_iou, "abs_late_minus_early_iou": abs(d_iou),
        "early_mean_margin": float(np.mean(em)), "late_mean_margin": float(np.mean(lm)),
        "abs_margin_change": d_mar,
        "still_improving": bool(d_iou > 0.005),
        "late_deterioration": bool(d_iou < -0.005),
        "converged": bool(abs(d_iou) <= 0.005 and d_mar <= 0.003),
        "rule": "|late_best - early_best| <= 0.005 AND |late_margin - early_margin| <= 0.003",
    }

    # checkpoint selection, unchanged rule, applied within the first converged window
    selection = None
    if convergence["converged"]:
        p_iou = persistence["observable_occupied_iou"]
        p_of = persistence["open_obstacle_field"]["observable_occupied_iou"]
        p_frac = persistence["predicted_occupied_fraction_all_cells"]
        p_prec = persistence["observable_occupied_precision"]
        eligible = []
        for ep in late_window:
            fin = final.get(str(ep))
            e = iou.get(ep)
            if fin is None or e is None:
                continue
            sp = fin["step1_spatial"]
            conds = {
                "beats_persistence": e > p_iou,
                "margin_at_least_gate": mar[ep] >= GATE_MARGIN,
                "beats_ofield_persistence":
                    fin["open_obstacle_field"]["observable_occupied_iou"] > p_of,
                "occupied_volume_calibrated": (
                    sp["predicted_occupied_fraction_all_cells"] <= p_frac
                    or sp["observable_occupied_precision"] >= p_prec),
            }
            eligible.append({"epoch": ep, "step1_occupied_iou": e, "conditions": conds,
                             "all_conditions_met": all(conds.values())})
        passing = [x for x in eligible if x["all_conditions_met"]]
        selection = {
            "window": list(late_window), "candidates": eligible,
            "selected_epoch": (max(passing, key=lambda x: x["step1_occupied_iou"])["epoch"]
                               if passing else None),
            "one_step_gate": "PASS" if passing else "FAIL",
            "rule": "unchanged; applied within the first converged window; not retrospective",
        }

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "arm": args.arm_name,
        "rollout_arm_frozen": True, "rollout_arm_re_evaluated": False,
        "matched_duration_result_preserved": str(MATCHED),
        "curves": {args.arm_name: curve},
        "final": {args.arm_name: final},
        "convergence": convergence,
        "checkpoint_selection": selection,
        "next": ("compare against the frozen rollout checkpoint as CONVERGED-MODEL SELECTION "
                 "WITH UNEQUAL TRAINING DURATION, not a compute-matched causal comparison"
                 if convergence["converged"] else
                 f"not converged: run one final block {hi+1}-{hi+6} under the identical rule"),
        "wall_seconds": round(time.time() - started, 1),
    }
    (OUT / f"{args.arm_name}_block_{lo}_{hi}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"convergence": convergence,
                      "selection": (selection or {}).get("selected_epoch"),
                      "next": record["next"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
