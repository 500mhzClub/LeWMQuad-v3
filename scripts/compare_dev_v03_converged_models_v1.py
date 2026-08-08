#!/usr/bin/env python3
"""Final comparison of exactly two converged models.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only; no model is trained.

    1. rollout epoch 22 -- already frozen and converged
    2. the newly selected converged one-step control checkpoint

**This is CONVERGED-MODEL SELECTION WITH UNEQUAL TRAINING DURATION.  It is not a
compute-matched causal estimate of rollout supervision.**  The matched-duration
causal result is the separate epoch-0-23 comparison, whose verdict is ROLLOUT
TEST INCONCLUSIVE and is not restated or revised here.

Selection is NOT primarily by one-step occupied IoU.  One-step geometry, action
margin, occupied-volume calibration and open_obstacle_field are **eligibility /
non-regression gates**; among eligible models the winner is chosen on
autoregressive planning capability, using the step2_superiority test and
thresholds exactly as already encoded:

    cosine advantage >= 0.005
    lower step-two normalised error
    lower step-one-to-step-two degradation
    correct/correct beats all three shuffled sequences
    advantage spread over >= 3 selection scenes

No rule is added or relaxed after the control continuation is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
TWO = CACHE / "two_step"
FROZEN = TWO / "rollout_frozen" / "frozen_receipt.json"
MATCHED = TWO / "evaluation" / "MATCHED_24_EPOCH_result_epochs_0_23.json"
CONT = TWO / "control_continuation"
OUT = TWO / "converged_model_selection"

# thresholds, unchanged from the already-encoded step2_superiority test
COSINE_ADVANTAGE = 0.005
MIN_SCENES = 3
GATE_MARGIN = 0.0585573514302572


def gates(entry, persistence):
    sp = entry["step1_spatial"]
    return {
        "beats_persistence":
            sp["observable_occupied_iou"] > persistence["observable_occupied_iou"],
        "margin_at_least_gate": entry["step1_margin"] >= GATE_MARGIN,
        "beats_ofield_persistence":
            entry["open_obstacle_field"]["observable_occupied_iou"]
            > persistence["open_obstacle_field"]["observable_occupied_iou"],
        "occupied_volume_calibrated": (
            sp["predicted_occupied_fraction_all_cells"]
            <= persistence["predicted_occupied_fraction_all_cells"]
            or sp["observable_occupied_precision"]
            >= persistence["observable_occupied_precision"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-block", type=int, nargs=2, required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lo, hi = args.control_block

    frozen = json.loads(FROZEN.read_text())
    matched = json.loads(MATCHED.read_text())
    block = json.loads((CONT / f"block_{lo}_{hi}.json").read_text())
    persistence = matched["reference"]["persistence_step1_spatial"]

    if not block["convergence"]["converged"]:
        raise SystemExit(
            f"control block {lo}-{hi} is not converged "
            f"({'still improving' if block['convergence']['still_improving'] else 'late deterioration'}); "
            "the final comparison must not run until the control converges")
    sel = block["checkpoint_selection"]
    if sel is None or sel["selected_epoch"] is None:
        raise SystemExit("control converged but no checkpoint satisfied the one-step gate")

    c_ep = sel["selected_epoch"]
    r_ep = frozen["selected_epoch"]
    control = block["final"]["one_step"][str(c_ep)]
    control_curve = next(e for e in block["curves"]["one_step"] if e["epoch"] == c_ep)
    rollout = frozen["battery_at_selected_epoch"]
    rollout_full = matched["final"]["rollout"][str(r_ep)]
    rollout_curve = frozen["curve_at_selected_epoch"]

    control_entry = {"step1_spatial": control["step1_spatial"],
                     "step1_margin": control["step1_margin"],
                     "open_obstacle_field": control["open_obstacle_field"]}
    rollout_entry = {"step1_spatial": rollout_full["step1_spatial"],
                     "step1_margin": rollout_full["step1_margin"],
                     "open_obstacle_field": rollout_full["open_obstacle_field"]}
    eligibility = {"control": gates(control_entry, persistence),
                   "rollout": gates(rollout_entry, persistence)}
    eligible = {k: all(v.values()) for k, v in eligibility.items()}

    # ---- autoregressive planning capability, primary criterion ---------------
    c2, r2 = control_curve["step2"], rollout_curve["step2"]
    c_seq = control["action_sequence_conditions"]
    r_seq = rollout_full["action_sequence_conditions"]
    r_full = r_seq["correct_a0_correct_a1"]["step2"]["changed_cosine"]
    r_shuffles = {k: r_seq[k]["step2"]["changed_cosine"] for k in
                  ("shuffled_a0_correct_a1", "correct_a0_shuffled_a1", "shuffled_a0_shuffled_a1")}
    c_full = c_seq["correct_a0_correct_a1"]["step2"]["changed_cosine"]
    c_shuffles = {k: c_seq[k]["step2"]["changed_cosine"] for k in r_shuffles}

    per_scene = {}
    for sc, rv in rollout_full["per_scene_step2_cosine"].items():
        cv = control["per_scene_step2_cosine"].get(sc)
        if cv is not None:
            per_scene[sc] = {"rollout": rv, "control": cv, "difference": rv - cv}
    better = sum(1 for v in per_scene.values() if v["difference"] > 0)
    differences = [v["difference"] for v in per_scene.values()]

    step2 = {
        "cosine_advantage": r2["changed_cosine"] - c2["changed_cosine"],
        "cosine_advantage_at_least_0.005":
            (r2["changed_cosine"] - c2["changed_cosine"]) >= COSINE_ADVANTAGE,
        "lower_normalised_error":
            r2["normalised_error_vs_persistence"] < c2["normalised_error_vs_persistence"],
        "normalised_error_difference":
            r2["normalised_error_vs_persistence"] - c2["normalised_error_vs_persistence"],
        "lower_degradation": (rollout_curve["step1_to_step2_degradation"]
                              < control_curve["step1_to_step2_degradation"]),
        "degradation_difference": (rollout_curve["step1_to_step2_degradation"]
                                   - control_curve["step1_to_step2_degradation"]),
        "full_sequence_beats_all_shuffles": all(r_full > v for v in r_shuffles.values()),
        "scenes_where_rollout_step2_better": better,
        "scenes_total": len(per_scene),
        "improvement_not_confined_to_one_or_two_scenes": better >= MIN_SCENES,
        "paired_scene_difference_mean": float(np.mean(differences)) if differences else None,
        "paired_scene_difference_min": float(np.min(differences)) if differences else None,
        "paired_scene_difference_max": float(np.max(differences)) if differences else None,
    }
    step2_pass = all(v for k, v in step2.items() if isinstance(v, bool))

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "comparison_type": "CONVERGED-MODEL SELECTION WITH UNEQUAL TRAINING DURATION",
        "not_a_causal_estimate": (
            "the two models were trained for different numbers of epochs; this is model "
            "selection, NOT a compute-matched causal estimate of rollout supervision. The "
            "matched-duration causal result is the separate epoch-0-23 comparison, verdict "
            "ROLLOUT TEST INCONCLUSIVE, which is neither restated nor revised here."
        ),
        "selection_policy": (
            "one-step geometry, action margin, occupied-volume calibration and "
            "open_obstacle_field are ELIGIBILITY / NON-REGRESSION GATES; among eligible "
            "models the winner is chosen on autoregressive planning capability. The winner "
            "is NOT chosen primarily by one-step occupied IoU."
        ),
        "models": {
            "rollout": {"epoch": r_ep, "checkpoint_sha256": frozen["checkpoint"]["sha256"],
                        "frozen": True, "objective": "1.5*e1 + 0.5*e2, fixed sliding context",
                        "converged": frozen["convergence"]["converged"]},
            "control": {"epoch": c_ep,
                        "checkpoint": str(TWO / "arms" / "arm_one_step" / f"checkpoint_epoch{c_ep}.pt"),
                        "objective": "e1", "converged": True,
                        "convergence": block["convergence"]},
        },
        "eligibility_gates": {"per_model": eligibility, "eligible": eligible},
        "one_step_comparison": {
            "control": {"occupied_iou": control["step1_spatial"]["observable_occupied_iou"],
                        "precision": control["step1_spatial"]["observable_occupied_precision"],
                        "recall": control["step1_spatial"]["observable_occupied_recall"],
                        "occupied_fraction": control["step1_spatial"]["predicted_occupied_fraction_all_cells"],
                        "margin": control["step1_margin"],
                        "open_obstacle_field": control["open_obstacle_field"]},
            "rollout": {"occupied_iou": rollout_full["step1_spatial"]["observable_occupied_iou"],
                        "precision": rollout_full["step1_spatial"]["observable_occupied_precision"],
                        "recall": rollout_full["step1_spatial"]["observable_occupied_recall"],
                        "occupied_fraction": rollout_full["step1_spatial"]["predicted_occupied_fraction_all_cells"],
                        "margin": rollout_full["step1_margin"],
                        "open_obstacle_field": rollout_full["open_obstacle_field"]},
            "note": "reported for non-regression, not as the selection criterion",
        },
        "step_two_planning_capability": {
            "persistence_step2_latent_cosine":
                matched["reference"]["persistence_step2_latent"]["changed_cosine"],
            "control": {"changed_cosine": c2["changed_cosine"],
                        "normalised_error_vs_persistence": c2["normalised_error_vs_persistence"],
                        "degradation": control_curve["step1_to_step2_degradation"],
                        "action_sequences": {"correct_a0_correct_a1": c_full, **c_shuffles}},
            "rollout": {"changed_cosine": r2["changed_cosine"],
                        "normalised_error_vs_persistence": r2["normalised_error_vs_persistence"],
                        "degradation": rollout_curve["step1_to_step2_degradation"],
                        "action_sequences": {"correct_a0_correct_a1": r_full, **r_shuffles}},
            "paired_scene_differences": per_scene,
        },
        "step2_superiority": step2,
        "step2_superiority_passed": bool(step2_pass),
        "descriptive_step2_spatial": {
            "control": control.get("step2_spatial_descriptive"),
            "rollout": rollout_full.get("step2_spatial_descriptive"),
            "caveat": "82 native-labelled rows, one open_obstacle_field row: descriptive only",
        },
    }

    if not eligible["rollout"]:
        record["SELECTED"] = "one-step control"
        record["selection_reason"] = (
            "the rollout model failed an eligibility / non-regression gate; the simpler "
            "one-step control is selected")
    elif not eligible["control"]:
        record["SELECTED"] = "rollout bundle"
        record["selection_reason"] = (
            "the control failed an eligibility / non-regression gate while the rollout model "
            "passed")
    elif step2_pass:
        record["SELECTED"] = "rollout bundle"
        record["selection_reason"] = (
            "both models are eligible; the rollout bundle gives a material and broad "
            "step-two planning advantage without a meaningful one-step regression, so it is "
            "selected as the practical planning predictor even though the control's one-step "
            "occupied IoU is "
            + ("higher" if record["one_step_comparison"]["control"]["occupied_iou"]
               > record["one_step_comparison"]["rollout"]["occupied_iou"] else "lower"))
        record["attribution_caveat"] = (
            "selection is of the BUNDLE (1.5*e1 + 0.5*e2) at unequal training duration; the "
            "benefit is NOT attributed to autoregressive feedback. A 1.5*e1 attribution "
            "control remains required.")
    else:
        failed = [k for k, v in step2.items() if isinstance(v, bool) and not v]
        record["SELECTED"] = "one-step control"
        record["selection_reason"] = (
            "step-two performance is tied or the advantage is unstable "
            f"(failed: {failed}); the simpler one-step control is selected")

    record["conclusions"] = {
        "A_matched_duration_causal_through_epoch_23": "ROLLOUT TEST INCONCLUSIVE",
        "B_converged_model_selection": record["SELECTED"],
        "B_caveat": record["not_a_causal_estimate"],
    }
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({
        "models": {k: v.get("epoch") for k, v in record["models"].items()},
        "eligible": eligible,
        "step2_superiority": step2,
        "step2_superiority_passed": step2_pass,
        "SELECTED": record["SELECTED"],
        "reason": record["selection_reason"],
        "conclusions": record["conclusions"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
