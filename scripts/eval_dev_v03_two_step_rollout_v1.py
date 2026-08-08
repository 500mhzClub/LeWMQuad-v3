#!/usr/bin/env python3
"""Evaluate the rollout-supervision bundle against the matched one-step control.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

**What is being compared.** The rollout arm optimises `1.5*e1 + 0.5*e2`; the
control optimises `e1`. The rollout arm therefore also carries 1.5x the weight on
the one-step term, so this is an **official-inspired rollout-supervision bundle**,
not a pure rollout-only ablation. Total losses are NOT comparable across arms and
are never compared here. The comparison is on `e1`, operational one-step geometry,
action sensitivity, and step-two performance.

Each action arm is predicted exactly once per epoch; every family and subset
metric is obtained by indexing (the v1 capacity evaluator's redundancy is not
repeated).

Two-step spatial metrics are DESCRIPTIVE ONLY: native t+480 rasters exist for 82
selection rows and a single open_obstacle_field row, which cannot support family
claims or an open-field gate.
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

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
COMPLETION = CACHE / "temporal_action_jepa_v1" / "completion"
TWO = CACHE / "two_step"
OUT = TWO / "evaluation"
DERANGEMENT_SEEDS = (11, 23, 37)
EPOCHS = 6
GATE_MARGIN = 0.0585573514302572
GATE_PERSISTENCE_IOU = 0.3133053567020489
ARMS = ("one_step", "rollout")


@torch.no_grad()
def extract(rows, key, device, blob: Path, batch=16):
    arm = E.VJepa21CroppedV03Arm()
    shape = (len(rows), R.TOKENS, R.DIM)
    if blob.is_file() and blob.stat().st_size == int(np.prod(shape) * 2):
        return
    module = arm.build(device, torch.float32)
    memory = np.memmap(blob, dtype=np.float16, mode="w+", shape=shape)
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        pixels = torch.stack([arm.preprocess(r[key]) for r in chunk]).to(device, torch.float32)
        memory[start : start + len(chunk)] = module(pixels.unsqueeze(2)).half().cpu().numpy()
    memory.flush()
    del module
    torch.cuda.empty_cache()


def latent(pred, base, truth, mask):
    pred = pred.float()
    cos = F.cosine_similarity(pred, truth, dim=-1)[mask]
    err = (pred - truth).pow(2).mean(-1)[mask]
    ref = (base - truth).pow(2).mean(-1)[mask]
    return {"changed_cosine": float(cos.mean()),
            "normalised_error_vs_persistence": float(err.mean() / ref.mean().clamp_min(1e-12)),
            "tokens": int(cos.numel())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base = [json.loads(l) for l in (CACHE / "temporal_rows.jsonl").read_text().splitlines() if l.strip()]
    two = [json.loads(l) for l in (TWO / "two_step_rows.jsonl").read_text().splitlines() if l.strip()]
    base_train = [r for r in base if r["role"] == "train"]
    base_sel = [r for r in base if r["role"] == "checkpoint_selection"]
    pos_sel = {r["pair_sha256"]: i for i, r in enumerate(base_sel)}
    pos_train = {r["pair_sha256"]: i for i, r in enumerate(base_train)}
    train_rows = [r for r in two if r["role"] == "train"]
    sel_rows = [r for r in two if r["role"] == "checkpoint_selection"]
    sel_idx = np.array([pos_sel[r["pair_sha256"]] for r in sel_rows])
    tr_idx = np.array([pos_train[r["pair_sha256"]] for r in train_rows])
    n_bt, n_bs = len(base_train), len(base_sel)
    families = [r["family"] for r in sel_rows]
    scene_ids = [r["scene"] for r in sel_rows]

    # selection-side frozen caches, indexed onto the two-step subset
    ctx = torch.stack([R.load_cache(EVAL / f"frozen_ctx{k}.f16", n_bs)[sel_idx] for k in range(3)], 1)
    current = R.load_cache(EVAL / "frozen_current.f16", n_bt + n_bs)[n_bt:][sel_idx]
    y1 = R.load_cache(EVAL / "frozen_sel_future.f16", n_bs)[sel_idx]
    extract(sel_rows, "step2_path", device, TWO / "frozen_sel_step2.f16")
    y2 = R.load_cache(TWO / "frozen_sel_step2.f16", len(sel_rows))

    now = T.normalise(current.float())
    t1 = T.normalise(y1.float())
    t2 = T.normalise(y2.float())
    context1 = T.normalise(ctx.float()).half()

    # step-1 mask: the existing frozen definition, unchanged
    completion = json.loads((COMPLETION / "result.json").read_text())
    threshold1 = completion["changed_token_mask"]["threshold"]
    mask1 = (t1 - now).pow(2).mean(-1) >= threshold1

    # step-2 mask: defined on TRAIN ONLY and frozen before selection is touched
    tr_now = R.load_cache(EVAL / "frozen_current.f16", n_bt + n_bs)[:n_bt][tr_idx]
    tr_y2 = R.load_cache(TWO / "frozen_train_step2.f16", len(train_rows))
    chunks = []
    for s in range(0, len(train_rows), 256):
        e = min(s + 256, len(train_rows))
        chunks.append((T.normalise(tr_y2[s:e].float()) - T.normalise(tr_now[s:e].float()))
                      .pow(2).mean(-1))
    threshold2 = float(torch.quantile(torch.cat(chunks, 0).flatten().float(), 0.75))
    mask2 = (t2 - now).pow(2).mean(-1) >= threshold2
    del tr_now, tr_y2, chunks

    fixed = S.SharedTokenToBev(R.DIM).to(device)
    fixed.load_state_dict(torch.load(COMPLETION / "future_token_probe.pt", map_location=device))
    fixed.eval()
    labels1 = C.future_labels(sel_rows)

    # native t+480 labels: descriptive subset only
    native = [i for i, r in enumerate(sel_rows) if r["native_step2_labels"]]
    labels2 = None
    if native:
        shards, arr = {}, np.empty((len(native), 64, 64), dtype=np.uint8)
        for j, i in enumerate(native):
            r = sel_rows[i]
            sh = r["step2_shard_dir"]
            if sh not in shards:
                shards[sh] = np.fromfile(Path(sh) / "raster_labels.u1", dtype=np.uint8).reshape(-1, 64, 64)
            arr[j] = shards[sh][r["step2_shard_row"]]
        labels2 = arr

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "comparison_contract": {
            "rollout_objective": "1.5*e1 + 0.5*e2 (jloss + sloss, official reduction)",
            "control_objective": "e1",
            "bundle_not_pure_ablation": (
                "the rollout arm also carries 1.5x the weight on the one-step term, so this "
                "is an official-inspired rollout-supervision BUNDLE, not a pure rollout-only "
                "ablation"
            ),
            "total_losses_never_compared_across_arms": True,
            "compared_on": ["e1", "one-step operational geometry", "action sensitivity",
                            "step-two performance"],
            "attribution_control_required_if_passing": "a later 1.5*e1 arm",
        },
        "rows": {"train": len(train_rows), "checkpoint_selection": len(sel_rows)},
        "masks": {
            "step1_threshold": threshold1, "step1_changed": int(mask1.sum()),
            "step1_source": "existing frozen definition, unchanged",
            "step2_threshold": threshold2, "step2_changed": int(mask2.sum()),
            "step2_source": "75th percentile of |LN(y2)-LN(now)| on TRAIN only, frozen before selection",
            "total_tokens": int(mask1.numel()),
        },
        "step2_spatial_scope": {
            "native_labelled_selection_rows": len(native),
            "open_obstacle_field_rows": sum(1 for i in native
                                            if sel_rows[i]["family"] == "open_obstacle_field"),
            "descriptive_only": True,
            "supports_family_claims": False, "supports_open_field_gate": False,
        },
        "curves": {}, "final": {},
    }

    persistence1 = A.spatial_block(fixed, now.half(), labels1, families, (24, 32), device)
    record["reference"] = {
        "persistence_step1_spatial": persistence1,
        "true_future_step1_spatial": A.spatial_block(fixed, t1.half(), labels1, families, (24, 32), device),
        "persistence_step1_latent": latent(now, now, t1, mask1),
        "persistence_step2_latent": latent(now, now, t2, mask2),
        "gate_margin": GATE_MARGIN, "gate_persistence_iou": GATE_PERSISTENCE_IOU,
    }

    orders = [C.derangement(len(sel_rows), s) for s in DERANGEMENT_SEEDS]
    a0 = T.action_tensor([r["action_step1"] for r in sel_rows], torch.device("cpu"))
    a1 = T.action_tensor([r["action_step2"] for r in sel_rows], torch.device("cpu"))

    for name in ARMS:
        arm_dir = TWO / "arms" / f"arm_{name}"
        training = json.loads((arm_dir / "result.json").read_text())
        curve = []
        for epoch in range(args.epochs):
            path = arm_dir / f"checkpoint_epoch{epoch}.pt"
            if not path.is_file():
                continue
            predictor = T.Predictor(**R.PRED).to(device)
            predictor.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"])
            predictor.eval()

            def rollout(action0, action1):
                """One pass; returns (p1, p2). Each action arm is run exactly once."""
                o1, o2 = [], []
                for s in range(0, len(sel_rows), args.batch):
                    e = min(s + args.batch, len(sel_rows))
                    m = torch.ones(e - s, R.TOKENS, dtype=torch.bool, device=device)
                    with torch.no_grad():
                        c1 = context1[s:e].to(device=device, dtype=torch.float32)
                        q1 = T.normalise(predictor(c1, action0[s:e].to(device), m))
                        c2 = torch.stack([c1[:, 1], c1[:, 2], q1], dim=1)
                        q2 = T.normalise(predictor(c2, action1[s:e].to(device), m))
                    o1.append(q1.half().cpu()); o2.append(q2.half().cpu())
                return torch.cat(o1, 0), torch.cat(o2, 0)

            p1, p2 = rollout(a0, a1)
            shuffled = [rollout(a0[o], a1[o]) for o in orders]
            spatial1 = A.spatial_block(fixed, p1, labels1, families, (24, 32), device)

            entry = {
                "epoch": epoch,
                "train_e1": training["epochs"][epoch]["e1"],
                "train_e2": training["epochs"][epoch]["e2"],
                "step1": latent(p1, now, t1, mask1),
                "step1_shuffled": {k: float(np.mean([latent(s[0], now, t1, mask1)[k]
                                                     for s in shuffled]))
                                   for k in ("changed_cosine", "normalised_error_vs_persistence")},
                "step2": latent(p2, now, t2, mask2),
                "step2_shuffled": {k: float(np.mean([latent(s[1], now, t2, mask2)[k]
                                                     for s in shuffled]))
                                   for k in ("changed_cosine", "normalised_error_vs_persistence")},
                "step1_occupied_iou": spatial1["observable_occupied_iou"],
                "step1_occupied_precision": spatial1["observable_occupied_precision"],
                "step1_occupied_recall": spatial1["observable_occupied_recall"],
                "step1_occupied_fraction": spatial1["predicted_occupied_fraction_all_cells"],
            }
            entry["step1_margin"] = entry["step1"]["changed_cosine"] - entry["step1_shuffled"]["changed_cosine"]
            entry["step2_margin"] = entry["step2"]["changed_cosine"] - entry["step2_shuffled"]["changed_cosine"]
            entry["step1_to_step2_degradation"] = (
                entry["step1"]["changed_cosine"] - entry["step2"]["changed_cosine"])
            curve.append(entry)

            if epoch >= args.epochs - 3:
                # the four action-sequence conditions, each run exactly once
                combos = {
                    "correct_a0_correct_a1": (a0, a1),
                    "shuffled_a0_correct_a1": (a0[orders[0]], a1),
                    "correct_a0_shuffled_a1": (a0, a1[orders[0]]),
                    "shuffled_a0_shuffled_a1": (a0[orders[0]], a1[orders[0]]),
                }
                sequence = {}
                per_scene_step2 = {}
                for tag, (x0, x1) in combos.items():
                    q1, q2 = rollout(x0, x1)
                    sequence[tag] = {"step1": latent(q1, now, t1, mask1),
                                     "step2": latent(q2, now, t2, mask2)}
                    if tag == "correct_a0_correct_a1":
                        for sc in sorted(set(scene_ids)):
                            pick = torch.tensor([i for i, v in enumerate(scene_ids) if v == sc])
                            per_scene_step2[sc] = latent(q2[pick], now[pick], t2[pick],
                                                         mask2[pick])["changed_cosine"]
                    del q1, q2
                final = {"epoch": epoch,
                         "step1_spatial": spatial1,
                         "step1_spatial_per_family": spatial1["per_family"],
                         "open_obstacle_field": spatial1["open_obstacle_field"],
                         "action_sequence_conditions": sequence,
                         "per_scene_step2_cosine": per_scene_step2,
                         "step1_to_step2_degradation": entry["step1_to_step2_degradation"],
                         "step1_margin": entry["step1_margin"],
                         "step2": entry["step2"]}
                if labels2 is not None:
                    pick = torch.tensor(native)
                    final["step2_spatial_descriptive"] = {
                        "rows": len(native),
                        "predicted": A.spatial_block(fixed, p2[pick], labels2,
                                                     [families[i] for i in native], (24, 32), device),
                        "persistence": A.spatial_block(fixed, now[pick].half(), labels2,
                                                       [families[i] for i in native], (24, 32), device),
                        "caveat": "82 rows, one open_obstacle_field row: descriptive only",
                    }
                record["final"].setdefault(name, {})[str(epoch)] = final
            del predictor, p1, p2, shuffled
            torch.cuda.empty_cache()
            print(f"  [{name}] epoch {epoch}: e1(train) {entry['train_e1']:.5f} "
                  f"s1cos {entry['step1']['changed_cosine']:.4f} margin {entry['step1_margin']:+.4f} "
                  f"occIoU {entry['step1_occupied_iou']:.4f} | s2cos {entry['step2']['changed_cosine']:.4f} "
                  f"s2margin {entry['step2_margin']:+.4f}", flush=True)
        record["curves"][name] = curve
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    c = record["curves"]["one_step"][-1]
    r = record["curves"]["rollout"][-1]
    of_r = record["final"]["rollout"]["open_obstacle_field"]
    of_p = persistence1["open_obstacle_field"]
    gates = {
        "1_step1_occupied_iou_beats_persistence":
            r["step1_occupied_iou"] > persistence1["observable_occupied_iou"],
        "2_step1_margin_at_least_gate": r["step1_margin"] >= GATE_MARGIN,
        "3_ofield_step1_beats_persistence":
            of_r["observable_occupied_iou"] > of_p["observable_occupied_iou"],
        "4_not_diffuse_overprediction": (
            r["step1_occupied_fraction"] <= persistence1["predicted_occupied_fraction_all_cells"]
            or record["final"]["rollout"]["step1_spatial"]["observable_occupied_precision"]
            >= persistence1["observable_occupied_precision"]),
        "5_step2_beats_persistence_and_shuffled_sequence": (
            r["step2"]["normalised_error_vs_persistence"] < 1.0
            and r["step2_margin"] > 0.0),
        "6_step2_degradation_bounded_and_in_interface": (
            r["step1_to_step2_degradation"] < r["step1"]["changed_cosine"] * 0.5
            and r["step2"]["normalised_error_vs_persistence"] < 1.0),
    }
    record["gates"] = gates
    # Predeclared, fixed BEFORE resuming: middle window 18-20 vs late window 21-23.
    MIDDLE, LATE = (18, 19, 20), (21, 22, 23)

    def converged(curve):
        iou = {e["epoch"]: e["step1_occupied_iou"] for e in curve}
        mar = {e["epoch"]: e["step1_margin"] for e in curve}
        mid_i = [iou[k] for k in MIDDLE if k in iou]
        late_i = [iou[k] for k in LATE if k in iou]
        mid_m = [mar[k] for k in MIDDLE if k in mar]
        late_m = [mar[k] for k in LATE if k in mar]
        if not (mid_i and late_i and mid_m and late_m):
            return None
        d_iou = max(late_i) - max(mid_i)
        d_mar = abs(float(np.mean(late_m)) - float(np.mean(mid_m)))
        return {"middle_best_iou_18_20": max(mid_i), "late_best_iou_21_23": max(late_i),
                "late_minus_middle_iou": d_iou,
                "middle_mean_margin_18_20": float(np.mean(mid_m)),
                "late_mean_margin_21_23": float(np.mean(late_m)),
                "abs_margin_change": d_mar,
                "converged": (d_iou <= 0.005) and (d_mar <= 0.003)}

    per_arm = {a: converged(record["curves"][a]) for a in ARMS}
    record["convergence"] = {
        "rule": ("late_best_IoU(21,22,23) - middle_best_IoU(18,19,20) <= 0.005 AND "
                 "|mean margin(21-23) - mean margin(18-20)| <= 0.003, for BOTH arms"),
        "iou_threshold": 0.005, "margin_threshold": 0.003,
        "per_arm": per_arm,
        "both_converged": bool(all(c is not None and c["converged"] for c in per_arm.values())),
        "primary_endpoint": "epoch 23", "retrospective_early_stopping": False,
        "final_bounded_extension": True,
    }

    # Checkpoint selection, fixed before resuming: within 21-23, highest step-one
    # occupied IoU that ALSO clears all four one-step conditions.
    persist_iou = persistence1["observable_occupied_iou"]
    persist_of = persistence1["open_obstacle_field"]["observable_occupied_iou"]
    persist_frac = persistence1["predicted_occupied_fraction_all_cells"]
    persist_prec = persistence1["observable_occupied_precision"]
    selection = {}
    for name in ARMS:
        eligible = []
        for e in record["curves"][name]:
            if e["epoch"] not in LATE:
                continue
            fin = record["final"].get(name, {}).get(str(e["epoch"]))
            if fin is None:
                continue
            of = fin["open_obstacle_field"]["observable_occupied_iou"]
            sp = fin["step1_spatial"]
            calibrated = (sp["predicted_occupied_fraction_all_cells"] <= persist_frac
                          or sp["observable_occupied_precision"] >= persist_prec)
            conds = {
                "beats_persistence": e["step1_occupied_iou"] > persist_iou,
                "margin_at_least_gate": e["step1_margin"] >= GATE_MARGIN,
                "beats_ofield_persistence": of > persist_of,
                "occupied_volume_calibrated": calibrated,
            }
            eligible.append({"epoch": e["epoch"], "step1_occupied_iou": e["step1_occupied_iou"],
                             "conditions": conds, "all_conditions_met": all(conds.values())})
        passing = [x for x in eligible if x["all_conditions_met"]]
        selection[name] = {
            "window": list(LATE), "candidates": eligible,
            "selected_epoch": (max(passing, key=lambda x: x["step1_occupied_iou"])["epoch"]
                               if passing else None),
            "one_step_gate": "PASS" if passing else "FAIL",
        }
    record["checkpoint_selection"] = selection

    record["control_vs_rollout"] = {
        "train_e1": {"control": c["train_e1"], "rollout": r["train_e1"]},
        "step1_margin": {"control": c["step1_margin"], "rollout": r["step1_margin"]},
        "step1_occupied_iou": {"control": c["step1_occupied_iou"], "rollout": r["step1_occupied_iou"]},
        "step1_occupied_fraction": {"control": c["step1_occupied_fraction"],
                                    "rollout": r["step1_occupied_fraction"]},
        "step2_changed_cosine": {"control": c["step2"]["changed_cosine"],
                                 "rollout": r["step2"]["changed_cosine"]},
        "step2_margin": {"control": c["step2_margin"], "rollout": r["step2_margin"]},
        "note": "total losses are not comparable across arms and are not compared",
    }
    if not record["convergence"]["both_converged"]:
        record["DECISION"] = "ROLLOUT TEST INCONCLUSIVE"
        record["decision_reason"] = (
            "the predeclared prospective convergence rule was not met; no further automatic "
            "schedule extension is taken"
        )
    elif selection["rollout"]["selected_epoch"] is None:
        record["DECISION"] = "REJECT ROLLOUT OBJECTIVE AS SUFFICIENT"
        record["rejection_scope"] = (
            "the rollout arm cleared no checkpoint in 21-23 satisfying all four one-step "
            "conditions; scoped to this official-inspired rollout-supervision bundle with "
            "fixed sliding context at 17.2M"
        )
    else:
        re_ = selection["rollout"]["selected_epoch"]
        ce_ = selection["one_step"]["selected_epoch"]
        rf = record["final"]["rollout"][str(re_)]
        cf = (record["final"]["one_step"].get(str(ce_)) if ce_ is not None
              else record["final"]["one_step"][str(max(LATE))])
        seq = rf["action_sequence_conditions"]
        full = seq["correct_a0_correct_a1"]["step2"]["changed_cosine"]
        partials = [seq["shuffled_a0_correct_a1"]["step2"]["changed_cosine"],
                    seq["correct_a0_shuffled_a1"]["step2"]["changed_cosine"],
                    seq["shuffled_a0_shuffled_a1"]["step2"]["changed_cosine"]]
        scenes_better = sum(
            1 for sc, v in rf["per_scene_step2_cosine"].items()
            if v > cf["per_scene_step2_cosine"].get(sc, float("inf")))
        step2 = {
            "cosine_advantage_at_least_0.005":
                rf["step2"]["changed_cosine"] - cf["step2"]["changed_cosine"] >= 0.005,
            "lower_normalised_error":
                rf["step2"]["normalised_error_vs_persistence"]
                < cf["step2"]["normalised_error_vs_persistence"],
            "lower_degradation":
                rf["step1_to_step2_degradation"] < cf["step1_to_step2_degradation"],
            "full_sequence_beats_all_shuffles": all(full > x for x in partials),
            "improvement_not_confined_to_one_or_two_scenes": scenes_better >= 3,
            "scenes_where_rollout_step2_better": scenes_better,
        }
        record["step2_superiority"] = step2
        if all(v for k, v in step2.items() if isinstance(v, bool)):
            record["DECISION"] = "ACCEPT TWO-STEP AUTOREGRESSIVE PREDICTOR"
            record["attribution_caveat"] = (
                "acceptance is of the BUNDLE (1.5*e1 + 0.5*e2) with fixed sliding context. "
                "The benefit is NOT yet attributed to autoregressive feedback: a 1.5*e1 "
                "attribution control remains required."
            )
        else:
            record["DECISION"] = "REJECT ROLLOUT OBJECTIVE AS SUFFICIENT"
            record["rejection_scope"] = (
                "the rollout arm cleared the one-step gates but did not materially outperform "
                "the matched control at step two; scoped to this official-inspired "
                "rollout-supervision bundle with fixed sliding context at 17.2M"
            )
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"convergence": record["convergence"],
                      "checkpoint_selection": {k: {"selected_epoch": v["selected_epoch"],
                                                   "one_step_gate": v["one_step_gate"]}
                                               for k, v in selection.items()},
                      "step2_superiority": record.get("step2_superiority"),
                      "DECISION": record["DECISION"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
