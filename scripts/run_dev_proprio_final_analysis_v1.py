#!/usr/bin/env python3
"""The frozen final unblinded analysis, run once over eight complete quadruplets.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING as code; it produces the study's results.

Structure is fixed in advance and enforced here:

  CONFIRMATORY   the interaction I_s under the frozen equal-family H=2 estimator,
                 with the seed quadruplet as the replication unit
  SECONDARY      corpus-weighted (token-pooled) estimates, H=1-4 correct-future
                 cosine, correct-versus-shuffled margin, occupied co-outcomes
  DIAGNOSTIC     terminal-window stability, per-family results, and the
                 prospectively declared local_composite_motifs contrast

No combined H=2-3 endpoint is formed, no checkpoint is selected, and no success
threshold is introduced after unblinding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import dev_seed_reestimation_v1 as S  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402
from scripts import eval_dev_proprio_factorial_v1 as E  # noqa: E402
from scripts import dev_proprio_experiment_config_v1 as C  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT = D.CACHE / "factorial_v1" / "final_analysis.json"
FROZEN_N = 8
DIAGNOSTIC_FAMILY = "local_composite_motifs"

SEED_4_INCIDENT = (
    "Seed index 4 (seed 2026080905): the first evaluation attempt was REFUSED by the "
    "launch guard because the source tree was dirty -- a new, unimported module had been "
    "created in the repository while the stage was running. No bound scientific artefact "
    "and no executed scientific source changed, and the first attempt produced no "
    "evaluation result. The pinned launch state was restored and the read-only evaluation "
    "was re-run from the preserved epoch-21 checkpoints using the byte-identical evaluator "
    "that scored every other seed."
)


def h2(result: dict, cell: str) -> float:
    return result["cells"][cell]["per_horizon"]["2"]["equal_family_cosine"]


def interaction(result: dict) -> float:
    return ((h2(result, "proprio_rollout") - h2(result, "proprio_one_step"))
            - (h2(result, "rgb_rollout") - h2(result, "rgb_one_step")))


def mean_sd(values):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else float("nan")
    return mean, sd


def t_interval(values, alpha=0.05):
    from scipy import stats
    n = len(values)
    mean, sd = mean_sd(values)
    half = stats.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n)
    return mean - half, mean + half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    seeds = list(D.SEED_REGISTRY[:FROZEN_N])
    lineage, results, runs = [], {}, {}
    for index, seed in enumerate(seeds):
        seed_dir = D.OUT / f"seed_{seed}"
        run = json.loads((seed_dir / "run_record.json").read_text())
        result = json.loads((seed_dir / "selection_result.json").read_text())
        cells = run["cells_run"]
        entry = {
            "seed_index": index, "seed": seed,
            "completed": bool(run.get("completed")),
            "all_cells_valid": all(c["validity"] == "valid" for c in cells),
            "all_cells_24_epochs": all(c["epochs_trained"] == 24 for c in cells),
            "all_checkpoints_epoch_21": all(c["checkpoint_epoch"] == 21 for c in cells),
            "execution_order": run["execution_order"],
            "shared_parameters_bit_identical": run["shared_parameters_bit_identical"],
            "batch_plan_identical_across_cells": run["batch_plan_identical_across_cells"],
            "authorisation_receipt_digest": run["authorisation_receipt_digest"],
            "factorial_manifest_digest": run["factorial_manifest_digest"],
            "checkpoint_sha256": {c["cell"]: c["checkpoint_sha256"] for c in cells},
            "training_wall_hours": round(run.get("wall_seconds_total", 0) / 3600, 2),
            "attempts": 1, "restarts": 0,
        }
        attempts = seed_dir / "attempt_records.jsonl"
        if attempts.is_file():
            records = [json.loads(l) for l in attempts.read_text().splitlines() if l.strip()]
            entry["preserved_attempt_records"] = records
            entry["attempts"] = 1 + len(records)
        if not all((entry["completed"], entry["all_cells_valid"],
                    entry["all_cells_24_epochs"], entry["all_checkpoints_epoch_21"])):
            raise SystemExit(f"seed {seed} is not a complete, technically valid quadruplet")
        lineage.append(entry)
        results[seed] = result
        runs[seed] = run

    factorial = FM.load()
    interim = json.loads(
        (D.CACHE / "factorial_v1" / "variance_only_interim.json").read_text())

    # ---------------------------------------------------- CONFIRMATORY -----
    values = [interaction(results[seed]) for seed in seeds]
    mean, sd = mean_sd(values)
    low, high = t_interval(values)
    cell_means = {}
    for cell in D.CELLS:
        per_seed = [h2(results[seed], cell) for seed in seeds]
        cell_mean, cell_sd = mean_sd(per_seed)
        cell_means[cell] = {"mean": cell_mean, "sd": cell_sd, "per_seed": per_seed}

    delta_rgb = [h2(results[s], "rgb_rollout") - h2(results[s], "rgb_one_step") for s in seeds]
    delta_prop = [h2(results[s], "proprio_rollout") - h2(results[s], "proprio_one_step")
                  for s in seeds]
    rgb_mean, rgb_sd = mean_sd(delta_rgb)
    prop_mean, prop_sd = mean_sd(delta_prop)
    rgb_low, rgb_high = t_interval(delta_rgb)
    prop_low, prop_high = t_interval(delta_prop)

    confirmatory = {
        "estimand": "I_s = (PropRoll_s - PropOne_s) - (RGBRoll_s - RGBOne_s)",
        "estimator": ("frozen equal-family H=2: valid tokens within a row -> rows within an "
                      "episode cluster -> episodes within a family -> unweighted mean of the "
                      "eight family scores"),
        "replication_unit": "training seed quadruplet",
        "final_seed_count": len(values),
        "individual_interactions": {str(seeds[i]): values[i] for i in range(len(values))},
        "mean_interaction": mean,
        "sample_standard_deviation": sd,
        "t_interval_95": [low, high],
        "interval_excludes_zero": bool(low > 0 or high < 0),
        "delta_rgb": {"per_seed": delta_rgb, "mean": rgb_mean, "sd": rgb_sd,
                      "t_interval_95": [rgb_low, rgb_high]},
        "delta_prop": {"per_seed": delta_prop, "mean": prop_mean, "sd": prop_sd,
                       "t_interval_95": [prop_low, prop_high]},
        "cell_means_h2_equal_family": cell_means,
        "variance_reestimation_record": {
            "sample_sd_at_interim": interim["sample_standard_deviation_s_I"],
            "upper_bound_sigma_U": interim["upper_bound_sigma_U_90pc_one_sided"],
            "minimally_relevant_interaction": interim["power_inputs"]["minimally_relevant_interaction"],
            "alpha": interim["power_inputs"]["alpha"],
            "target_power": interim["power_inputs"]["target_power"],
            "power_by_total_N": interim["power_by_total_N"],
            "frozen_total_N": interim["frozen_total_N"],
            "power_at_frozen_N": interim["power_at_frozen_N"],
            "precision_limited": interim["precision_limited"],
            "recalculated_after_freezing": False,
        },
    }

    # ------------------------------------------------------- SECONDARY -----
    secondary = {"note": "secondary to the confirmatory equal-family result; never mixed with it"}
    for horizon in ("1", "2", "3", "4"):
        block = {}
        for cell in D.CELLS:
            equal = [results[s]["cells"][cell]["per_horizon"][horizon]["equal_family_cosine"]
                     for s in seeds]
            pooled = [results[s]["cells"][cell]["per_horizon"][horizon]["secondary_token_pooled_cosine"]
                      for s in seeds]
            margin = [results[s]["cells"][cell]["per_horizon"][horizon]["correct_minus_shuffled_margin"]
                      for s in seeds]
            occupied = [results[s]["cells"][cell]["per_horizon"][horizon]["occupied"]["occupied_iou"]
                        for s in seeds]
            block[cell] = {
                "equal_family_cosine_mean": mean_sd(equal)[0],
                "equal_family_cosine_sd": mean_sd(equal)[1],
                "corpus_weighted_token_pooled_mean": mean_sd(pooled)[0],
                "correct_minus_shuffled_margin_mean": mean_sd(margin)[0],
                "correct_minus_shuffled_margin_sd": mean_sd(margin)[1],
                "occupied_iou_mean": mean_sd(occupied)[0],
            }
        block["interaction_equal_family"] = mean_sd([
            ((results[s]["cells"]["proprio_rollout"]["per_horizon"][horizon]["equal_family_cosine"]
              - results[s]["cells"]["proprio_one_step"]["per_horizon"][horizon]["equal_family_cosine"])
             - (results[s]["cells"]["rgb_rollout"]["per_horizon"][horizon]["equal_family_cosine"]
                - results[s]["cells"]["rgb_one_step"]["per_horizon"][horizon]["equal_family_cosine"]))
            for s in seeds])[0]
        secondary[f"H{horizon}"] = block
    secondary["co_outcome_status"] = E.non_inferiority_status(C.CONFIG)
    secondary["combined_h2_h3_endpoint"] = "NOT FORMED -- prohibited in advance"

    # ------------------------------------------------------ DIAGNOSTIC -----
    families = sorted(factorial["rows_by_split_and_family"])
    family_names = sorted({name.split("/", 1)[1] for name in families})
    per_family = {}
    for family in family_names:
        per_family[family] = {}
        for cell in D.CELLS:
            vals = [results[s]["cells"][cell]["per_horizon"]["2"]["per_family_cosine"][family]
                    for s in seeds]
            per_family[family][cell] = mean_sd(vals)[0]
        per_family[family]["interaction"] = mean_sd([
            ((results[s]["cells"]["proprio_rollout"]["per_horizon"]["2"]["per_family_cosine"][family]
              - results[s]["cells"]["proprio_one_step"]["per_horizon"]["2"]["per_family_cosine"][family])
             - (results[s]["cells"]["rgb_rollout"]["per_horizon"]["2"]["per_family_cosine"][family]
                - results[s]["cells"]["rgb_one_step"]["per_horizon"]["2"]["per_family_cosine"][family]))
            for s in seeds])[0]

    terminal = {}
    for cell in D.CELLS:
        entries = []
        for seed in seeds:
            for c in runs[seed]["cells_run"]:
                if c["cell"] == cell:
                    entries.append(c["terminal_window"])
        terminal[cell] = {
            "mean_of_terminal_window_means": mean_sd([e["mean"] for e in entries])[0],
            "mean_terminal_window_sd": mean_sd([e["sd"] for e in entries])[0],
            "mean_slope_epochs_14_23": mean_sd([e["slope"] for e in entries])[0],
            "used_for_selection": False, "used_for_exclusion": False,
        }

    diagnostic = {
        "per_family_h2_equal_family": per_family,
        "terminal_window_stability": terminal,
        DIAGNOSTIC_FAMILY: {
            "status": "prospectively declared family-level diagnostic, not a primary endpoint",
            "h2_interaction": per_family[DIAGNOSTIC_FAMILY]["interaction"],
            "cells": {cell: per_family[DIAGNOSTIC_FAMILY][cell] for cell in D.CELLS},
            "tuned_to_this_family": False,
        },
    }

    report = {
        "status": STATUS, "claim_bearing": False,
        "analysis": "frozen final unblinded analysis, run once",
        "quadruplets": len(seeds), "seeds": seeds,
        "selection_rows": factorial["rows_by_split"]["checkpoint_selection"],
        "horizon_mask_digest": factorial["horizon_masks"]["mask_digest"],
        "factorial_manifest_digest": factorial["digest"],
        "new_checkpoint_selection_performed": False,
        "new_success_threshold_introduced": False,
        "attempt_lineage": lineage,
        "seed_4_refused_evaluation_incident": SEED_4_INCIDENT,
        "confirmatory": confirmatory,
        "secondary": secondary,
        "diagnostic": diagnostic,
    }
    report["report_digest"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()).hexdigest()
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"report_digest": report["report_digest"],
                      "quadruplets": len(seeds),
                      "confirmatory": {k: v for k, v in confirmatory.items()
                                       if k not in ("cell_means_h2_equal_family",
                                                    "variance_reestimation_record")}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
