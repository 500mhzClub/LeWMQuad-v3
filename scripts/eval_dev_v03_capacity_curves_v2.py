#!/usr/bin/env python3
"""Per-epoch capacity comparison: 457M shape-matched successor vs 17.2M control.

v2: each action arm is predicted exactly once per epoch and every scene metric is
obtained by indexing.  v1 recomputed the three shuffled arms inside the 8-scene
loop -- 24 full 491-row passes where 3 suffice, ~4.6 h of avoidable compute at
457M parameters.  Numerically identical; the fix is cost only.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

Both predictor checkpoints are evaluated through the SAME verified fp16 frozen
encoder feature caches and an identical evaluation path.  The encoder is never
executed here.

For every saved epoch of both arms:
  * train dense L1        (from each arm's own training record)
  * checkpoint_selection dense L1
  * correct-minus-shuffled action margin
  * predicted-future occupied IoU under the fixed true-future probe
  * predicted occupied fraction

At the final epoch the full battery is reported: per-scene margins, per-family
spatial results, occupied-volume calibration and the absolute operational gates.

Naming, fixed: the successor is the **24x1024x16 shape-matched AdaLN capacity
successor**, 457,309,184 parameters, 26.6x the 17.2M control.  It is NOT
parameter-matched to the official 305M V-JEPA robot predictor, which is smaller
because it inserts actions and proprioception as tokens rather than through a
1024->6144 AdaLN projection in every block.
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
from scripts import run_dev_v03_official_scale_predictor_v1 as R  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
COMPLETION = CACHE / "temporal_action_jepa_v1" / "completion"
OUT = CACHE / "temporal_action_jepa_v1" / "capacity_curves_v2"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
DERANGEMENT_SEEDS = (11, 23, 37)
EPOCHS = 6

ARMS = {
    "control_17M": {"dir": "arm_frozen_l1dense", "width": 384, "depth": 6, "heads": 6},
    "capacity_457M": {"dir": "arm_frozen_official_scale", "width": 1024, "depth": 24, "heads": 16},
}
GATE_MARGIN = 0.0585573514302572
MATERIAL_IMPROVEMENT = 0.005          # declared before reading: IoU still rising by >this


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows, sel_rows = T.load_rows()
    ordered = train_rows + sel_rows
    n_train, n_sel = len(train_rows), len(sel_rows)
    arm_spec = E.VJepa21CroppedV03Arm()
    grid, dim = arm_spec.token_grid, arm_spec.token_dim
    tokens = grid[0] * grid[1]
    families = [r["family"] for r in sel_rows]
    scenes = [r["scene"] for r in sel_rows]
    labels_future = C.future_labels(sel_rows)

    fixed = S.SharedTokenToBev(dim).to(device)
    fixed.load_state_dict(torch.load(COMPLETION / "future_token_probe.pt", map_location=device))
    fixed.eval()
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

    actions = T.action_tensor([r["primitive"] for r in sel_rows], torch.device("cpu"))
    orders = [C.derangement(n_sel, s) for s in DERANGEMENT_SEEDS]
    persistence_spatial = A.spatial_block(fixed, now.half(), labels_future, families, grid, device)
    true_future_spatial = A.spatial_block(fixed, future.half(), labels_future, families, grid, device)

    record = {
        "status": STATUS, "claim_bearing": False,
        "model_identity": {
            "capacity_457M": {
                "name": "24x1024x16 shape-matched AdaLN capacity successor",
                "parameters": 457309184,
                "ratio_to_control": 26.6,
                "parameter_matched_to_official_305M": False,
                "why_not": (
                    "the official V-JEPA robot predictor is smaller because it inserts "
                    "actions and proprioception as TOKENS rather than through a "
                    "1024->6144 AdaLN projection in every block; preserving AdaLN is "
                    "correct here because this is a capacity-only intervention"
                ),
            },
            "control_17M": {"name": "17.2M dense-L1 / output-normalised predictor",
                            "parameters": 17198080},
        },
        "evaluation_path": {
            "feature_caches": "identical verified fp16 frozen-encoder caches for both arms",
            "encoder_executed": False,
            "fixed_probe": str(COMPLETION / "future_token_probe.pt"),
            "changed_token_threshold": threshold,
            "changed_tokens": int(shared_mask.sum()), "total_tokens": int(shared_mask.numel()),
            "derangement_seeds": list(DERANGEMENT_SEEDS),
        },
        "reference": {
            "persistence": persistence_spatial,
            "true_future": true_future_spatial,
            "gate_margin": GATE_MARGIN,
        },
        "curves": {}, "final": {},
    }

    for name, spec in ARMS.items():
        arm_dir = CACHE / "temporal_action_jepa_v1" / spec["dir"]
        training = json.loads((arm_dir / "result.json").read_text())
        train_losses = {e["epoch"]: e["train_loss"] for e in training["epochs"]}
        curve = []
        for epoch in range(EPOCHS):
            path = arm_dir / f"checkpoint_epoch{epoch}.pt"
            if not path.is_file():
                continue
            predictor = T.Predictor(width=spec["width"], depth=spec["depth"],
                                    heads=spec["heads"]).to(device)
            predictor.load_state_dict(torch.load(path, map_location="cpu")["predictor"])
            predictor.eval()

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
                        z = T.normalise(z)
                    out.append(z.half().cpu())
                return torch.cat(out, 0)

            # each action arm is predicted EXACTLY ONCE; every scene metric below
            # is obtained by indexing these tensors.  v1 recomputed the shuffled
            # arms once per scene -- 24 full passes where 3 suffice.
            predicted = run(actions)
            shuffled_predictions = [run(actions[o]) for o in orders]
            correct = float(F.cosine_similarity(predicted.float(), future, dim=-1)[shared_mask].mean())
            shuffled = float(np.mean([
                float(F.cosine_similarity(s.float(), future, dim=-1)[shared_mask].mean())
                for s in shuffled_predictions
            ]))
            spatial = A.spatial_block(fixed, predicted, labels_future, families, grid, device)
            entry = {
                "epoch": epoch,
                "train_dense_l1": train_losses.get(epoch),
                "selection_dense_l1": float((predicted.float() - future).abs().mean()),
                "correct_changed_cosine": correct,
                "shuffled_changed_cosine": shuffled,
                "correct_minus_shuffled": correct - shuffled,
                "predicted_future_occupied_iou": spatial["observable_occupied_iou"],
                "predicted_occupied_fraction": spatial["predicted_occupied_fraction_all_cells"],
            }
            curve.append(entry)
            if epoch == EPOCHS - 1:
                per_scene = {}
                for scene in sorted(set(scenes)):
                    pick = torch.tensor([i for i, s in enumerate(scenes) if s == scene])
                    c = float(F.cosine_similarity(
                        predicted[pick].float(), future[pick], dim=-1)[shared_mask[pick]].mean())
                    sh = float(np.mean([
                        float(F.cosine_similarity(
                            s[pick].float(), future[pick],
                            dim=-1)[shared_mask[pick]].mean())
                        for s in shuffled_predictions
                    ]))
                    per_scene[scene] = {"family": families[int(pick[0])],
                                        "correct_minus_shuffled": c - sh}
                record["final"][name] = {
                    "spatial": spatial,
                    "per_scene_margin": per_scene,
                    "persistence_changed_cosine": float(
                        F.cosine_similarity(now, future, dim=-1)[shared_mask].mean()),
                    "interface_mse_to_true_future": float(
                        (predicted.float() - future).pow(2).mean()),
                    "persistence_mse_to_true_future": float((now - future).pow(2).mean()),
                }
            del predictor, predicted, shuffled_predictions
            torch.cuda.empty_cache()
            print(f"  [{name}] epoch {epoch}: selL1 {entry['selection_dense_l1']:.5f} "
                  f"margin {entry['correct_minus_shuffled']:+.4f} "
                  f"occIoU {entry['predicted_future_occupied_iou']:.4f} "
                  f"occFrac {entry['predicted_occupied_fraction']:.5f}", flush=True)
        record["curves"][name] = curve
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    # ------------------------------------------------------------- decision
    large = record["curves"]["capacity_457M"]
    small = record["curves"]["control_17M"]
    final_large = record["final"]["capacity_457M"]
    persistence_iou = persistence_spatial["observable_occupied_iou"]
    of_large = final_large["spatial"]["open_obstacle_field"]
    of_persist = persistence_spatial["open_obstacle_field"]
    of_small = record["final"]["control_17M"]["spatial"]["open_obstacle_field"]

    last_delta_iou = (large[-1]["predicted_future_occupied_iou"]
                      - large[-2]["predicted_future_occupied_iou"]) if len(large) > 1 else 0.0
    still_improving = last_delta_iou > MATERIAL_IMPROVEMENT

    gates = {
        "1_beats_persistence": final_large["spatial"]["observable_occupied_iou"] > persistence_iou,
        "2_margin_retained": large[-1]["correct_minus_shuffled"] >= GATE_MARGIN,
        "3_ofield_beats_persistence_and_control": (
            of_large["observable_occupied_iou"] > of_persist["observable_occupied_iou"]
            and of_large["observable_occupied_iou"] > of_small["observable_occupied_iou"]
        ),
        "4_not_diffuse_overprediction": (
            final_large["spatial"]["predicted_occupied_fraction_all_cells"]
            <= persistence_spatial["predicted_occupied_fraction_all_cells"]
            or final_large["spatial"]["observable_occupied_precision"]
            >= persistence_spatial["observable_occupied_precision"]
        ),
        "5_canonical_interface_compatible": (
            final_large["interface_mse_to_true_future"]
            < final_large["persistence_mse_to_true_future"]
        ),
    }
    record["gates"] = gates
    record["capacity_control_comparison"] = {
        "selection_dense_l1": {"control": small[-1]["selection_dense_l1"],
                               "capacity": large[-1]["selection_dense_l1"]},
        "margin": {"control": small[-1]["correct_minus_shuffled"],
                   "capacity": large[-1]["correct_minus_shuffled"]},
        "predicted_future_occupied_iou": {"control": small[-1]["predicted_future_occupied_iou"],
                                          "capacity": large[-1]["predicted_future_occupied_iou"]},
        "predicted_occupied_fraction": {"control": small[-1]["predicted_occupied_fraction"],
                                        "capacity": large[-1]["predicted_occupied_fraction"]},
    }
    record["convergence"] = {
        "final_epoch_iou_delta": last_delta_iou,
        "material_improvement_threshold": MATERIAL_IMPROVEMENT,
        "still_materially_improving_at_epoch_six": bool(still_improving),
    }
    if still_improving:
        record["DECISION"] = "CAPACITY TEST INCONCLUSIVE"
        record["decision_reason"] = (
            "checkpoint_selection occupied IoU was still materially improving at epoch six; "
            "an undertrained model is not evidence that capacity does not matter"
        )
    elif all(gates.values()):
        record["DECISION"] = "ACCEPT OFFICIAL-SCALE PREDICTOR CAPACITY"
    else:
        record["DECISION"] = "REJECT CAPACITY AS SUFFICIENT"
        record["decision_reason"] = (
            "predictor capacity alone is insufficient within the current one-step "
            "AdaLN architecture; this does NOT reject the official 305M action-token "
            "predictor architecture, which was not tested"
        )
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"gates": gates, "convergence": record["convergence"],
                      "comparison": record["capacity_control_comparison"],
                      "DECISION": record["DECISION"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
