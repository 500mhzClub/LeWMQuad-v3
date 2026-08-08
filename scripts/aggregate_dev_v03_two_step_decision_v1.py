#!/usr/bin/env python3
"""Aggregate the two-step decision from the saved per-epoch evaluation.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only over
``two_step/evaluation/result.json``; no model is run and no metric is recomputed.

The per-epoch evaluation completed all 48 checkpoints and wrote its curves and
epoch 21-23 batteries before a stale block in the evaluator raised.  This applies
the predeclared rules to that saved output rather than re-running four hours of
evaluation.

All rules are exactly as fixed before the resumed epochs were read:

  convergence   |late_best_IoU(21,22,23) - middle_best_IoU(18,19,20)| <= 0.005
                AND |mean margin(21-23) - mean margin(18-20)| <= 0.003, both arms;
                a decline beyond 0.005 is late-window deterioration, never
                convergence
  selection     within 21-23, highest step-one occupied IoU that ALSO beats
                persistence, has margin >= +0.0586, beats open_obstacle_field
                persistence, and passes occupied-volume calibration
  step-two      cosine advantage >= 0.005, lower normalised error, lower
                degradation, full sequence beats all shuffles, improvement spread
                over >= 3 selection scenes
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

EVAL = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/two_step/evaluation/result.json")
OUT = EVAL.parent / "decision.json"
ARMS = ("one_step", "rollout")
MIDDLE, LATE = (18, 19, 20), (21, 22, 23)
GATE_MARGIN = 0.0585573514302572


def main() -> int:
    record = json.loads(EVAL.read_text())
    curves, final = record["curves"], record["final"]
    persistence = record["reference"]["persistence_step1_spatial"]
    p_iou = persistence["observable_occupied_iou"]
    p_of = persistence["open_obstacle_field"]["observable_occupied_iou"]
    p_frac = persistence["predicted_occupied_fraction_all_cells"]
    p_prec = persistence["observable_occupied_precision"]

    def converged(curve):
        iou = {e["epoch"]: e["step1_occupied_iou"] for e in curve}
        mar = {e["epoch"]: e["step1_margin"] for e in curve}
        mi, li = [iou[k] for k in MIDDLE], [iou[k] for k in LATE]
        mm, lm = [mar[k] for k in MIDDLE], [mar[k] for k in LATE]
        d_iou = max(li) - max(mi)
        d_mar = abs(float(np.mean(lm)) - float(np.mean(mm)))
        return {
            "middle_best_iou_18_20": max(mi), "late_best_iou_21_23": max(li),
            "late_minus_middle_iou": d_iou, "abs_late_minus_middle_iou": abs(d_iou),
            "middle_mean_margin_18_20": float(np.mean(mm)),
            "late_mean_margin_21_23": float(np.mean(lm)), "abs_margin_change": d_mar,
            "still_improving": bool(d_iou > 0.005),
            "late_window_deterioration": bool(d_iou < -0.005),
            "converged": bool(abs(d_iou) <= 0.005 and d_mar <= 0.003),
        }

    per_arm = {a: converged(curves[a]) for a in ARMS}
    out = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "derived_from": str(EVAL), "recomputed_any_metric": False,
        "comparison_contract": record["comparison_contract"],
        "convergence": {
            "rule": ("|late_best_IoU(21,22,23) - middle_best_IoU(18,19,20)| <= 0.005 AND "
                     "|mean margin(21-23) - mean margin(18-20)| <= 0.003, for BOTH arms; "
                     "a decline beyond 0.005 is late-window deterioration, never convergence"),
            "iou_threshold": 0.005, "margin_threshold": 0.003, "per_arm": per_arm,
            "both_converged": bool(all(c["converged"] for c in per_arm.values())),
            "primary_endpoint": "epoch 23", "retrospective_early_stopping": False,
            "final_bounded_extension": True,
        },
    }

    selection = {}
    for name in ARMS:
        eligible = []
        for e in curves[name]:
            if e["epoch"] not in LATE:
                continue
            fin = final[name][str(e["epoch"])]
            sp = fin["step1_spatial"]
            conds = {
                "beats_persistence": e["step1_occupied_iou"] > p_iou,
                "margin_at_least_gate": e["step1_margin"] >= GATE_MARGIN,
                "beats_ofield_persistence":
                    fin["open_obstacle_field"]["observable_occupied_iou"] > p_of,
                "occupied_volume_calibrated": (
                    sp["predicted_occupied_fraction_all_cells"] <= p_frac
                    or sp["observable_occupied_precision"] >= p_prec),
            }
            eligible.append({"epoch": e["epoch"],
                             "step1_occupied_iou": e["step1_occupied_iou"],
                             "conditions": conds, "all_conditions_met": all(conds.values())})
        passing = [x for x in eligible if x["all_conditions_met"]]
        selection[name] = {
            "window": list(LATE), "candidates": eligible,
            "selected_epoch": (max(passing, key=lambda x: x["step1_occupied_iou"])["epoch"]
                               if passing else None),
            "one_step_gate": "PASS" if passing else "FAIL",
        }
    out["checkpoint_selection"] = selection

    selected = sorted({v["selected_epoch"] for v in selection.values()
                       if v["selected_epoch"] is not None})
    matched = {}
    for ep in selected:
        row = {}
        for name in ARMS:
            fin, cur = final[name][str(ep)], next(e for e in curves[name] if e["epoch"] == ep)
            sp = fin["step1_spatial"]
            row[name] = {
                "step1_occupied_iou": cur["step1_occupied_iou"],
                "step1_occupied_precision": sp["observable_occupied_precision"],
                "step1_occupied_recall": sp["observable_occupied_recall"],
                "step1_occupied_fraction": sp["predicted_occupied_fraction_all_cells"],
                "step1_margin": cur["step1_margin"],
                "open_obstacle_field": fin["open_obstacle_field"],
                "step2_changed_cosine": cur["step2"]["changed_cosine"],
                "step2_normalised_error": cur["step2"]["normalised_error_vs_persistence"],
                "step1_to_step2_degradation": cur["step1_to_step2_degradation"],
                "action_sequence_conditions": fin["action_sequence_conditions"],
                "per_family": fin["step1_spatial_per_family"],
                "step2_spatial_descriptive": fin.get("step2_spatial_descriptive"),
                "selected_by_its_own_arm": selection[name]["selected_epoch"] == ep,
            }
        row["rollout_minus_control"] = {
            k: row["rollout"][k] - row["one_step"][k]
            for k in ("step1_occupied_iou", "step1_margin", "step2_changed_cosine",
                      "step2_normalised_error", "step1_to_step2_degradation")
        }
        matched[str(ep)] = row
    out["both_arms_at_each_selected_epoch"] = {
        "purpose": "prevent rollout attribution being confounded by different training durations",
        "epochs": matched,
    }

    if not out["convergence"]["both_converged"]:
        bad = {a: ("late-window deterioration" if per_arm[a]["late_window_deterioration"]
                   else "still improving" if per_arm[a]["still_improving"]
                   else "margin not stable") for a in ARMS if not per_arm[a]["converged"]}
        out["DECISION"] = "ROLLOUT TEST INCONCLUSIVE"
        out["decision_reason"] = {
            "rule_not_met_for": bad,
            "no_further_automatic_extension": True,
        }
    elif selection["rollout"]["selected_epoch"] is None:
        out["DECISION"] = "REJECT ROLLOUT OBJECTIVE AS SUFFICIENT"
        out["rejection_scope"] = (
            "no checkpoint in 21-23 satisfied all four one-step conditions; scoped to this "
            "official-inspired rollout-supervision bundle with fixed sliding context at 17.2M")
    else:
        re_ = selection["rollout"]["selected_epoch"]
        ce_ = selection["one_step"]["selected_epoch"] or max(LATE)
        rf, cf = final["rollout"][str(re_)], final["one_step"][str(ce_)]
        rc = next(e for e in curves["rollout"] if e["epoch"] == re_)
        cc = next(e for e in curves["one_step"] if e["epoch"] == ce_)
        seq = rf["action_sequence_conditions"]
        full = seq["correct_a0_correct_a1"]["step2"]["changed_cosine"]
        shuffles = [seq[k]["step2"]["changed_cosine"] for k in
                    ("shuffled_a0_correct_a1", "correct_a0_shuffled_a1", "shuffled_a0_shuffled_a1")]
        better = sum(1 for sc, v in rf["per_scene_step2_cosine"].items()
                     if v > cf["per_scene_step2_cosine"].get(sc, float("inf")))
        step2 = {
            "cosine_advantage": rc["step2"]["changed_cosine"] - cc["step2"]["changed_cosine"],
            "cosine_advantage_at_least_0.005":
                rc["step2"]["changed_cosine"] - cc["step2"]["changed_cosine"] >= 0.005,
            "lower_normalised_error":
                rc["step2"]["normalised_error_vs_persistence"]
                < cc["step2"]["normalised_error_vs_persistence"],
            "lower_degradation":
                rc["step1_to_step2_degradation"] < cc["step1_to_step2_degradation"],
            "full_sequence_beats_all_shuffles": all(full > x for x in shuffles),
            "scenes_where_rollout_step2_better": better,
            "improvement_not_confined_to_one_or_two_scenes": better >= 3,
        }
        out["step2_superiority"] = step2
        if all(v for v in step2.values() if isinstance(v, bool)):
            out["DECISION"] = "ACCEPT TWO-STEP AUTOREGRESSIVE PREDICTOR"
            out["attribution_caveat"] = (
                "acceptance is of the BUNDLE (1.5*e1 + 0.5*e2) with fixed sliding context; a "
                "1.5*e1 attribution control remains required before crediting autoregressive "
                "feedback specifically")
        else:
            out["DECISION"] = "REJECT ROLLOUT OBJECTIVE AS SUFFICIENT"
            out["rejection_scope"] = (
                "one-step gates cleared but the bundle did not materially outperform the "
                "matched control at step two; scoped to this official-inspired "
                "rollout-supervision bundle with fixed sliding context at 17.2M")

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({"convergence": out["convergence"]["per_arm"],
                      "both_converged": out["convergence"]["both_converged"],
                      "checkpoint_selection": {k: {"selected_epoch": v["selected_epoch"],
                                                   "one_step_gate": v["one_step_gate"]}
                                               for k, v in selection.items()},
                      "step2_superiority": out.get("step2_superiority"),
                      "DECISION": out["DECISION"],
                      "reason": out.get("decision_reason") or out.get("rejection_scope")
                      or out.get("attribution_caveat")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
