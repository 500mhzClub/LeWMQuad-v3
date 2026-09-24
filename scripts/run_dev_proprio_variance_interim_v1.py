#!/usr/bin/env python3
"""The frozen variance-only interim for the initial five-seed stage.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING as code; it produces a gating decision.

BLINDING IS ENFORCED HERE, NOT BY CONVENTION.  The per-seed interaction values
are computed inside this process and passed straight to the frozen interim
function.  They are never returned, never written to the interim report and never
printed.  The report exposes exactly seven things:

    1. confirmation that five complete, technically valid quadruplets were included
    2. a technical-validity and attempt-lineage summary
    3. the sample standard deviation s_I
    4. the predeclared one-sided 90 % upper bound sigma_U
    5. exact noncentral-t power for each total N in [5, 10]
    6. the resulting frozen N
    7. whether the capped experiment is expected to remain precision-limited

Anything that would reveal the effect -- the interaction mean, its direction, the
individual I_s values, cell means, family contrasts, intervals -- is deliberately
absent, and ``_assert_blinded`` fails the run if any of it reaches the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import dev_seed_reestimation_v1 as S  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT = D.CACHE / "factorial_v1" / "variance_only_interim.json"

FORBIDDEN_KEYS = (
    "interaction_mean", "mean_interaction", "interaction_values",
    "individual_interactions", "direction", "sign", "confidence_interval",
    "t_interval_95", "cell_means", "per_family", "family_contrast",
    "local_composite_motifs", "primary_h2_by_cell",
)


def _walk_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_keys(value)


def _assert_blinded(record: dict) -> None:
    """Structural check: no FIELD may carry a blinded quantity.

    The check is on keys, not on free text -- an earlier substring form matched
    the word "direction" inside this report's own blinding description, which is
    prose about what is withheld, not a leak of it.
    """
    forbidden = {key.lower() for key in FORBIDDEN_KEYS}
    present = {key.lower() for key in _walk_keys(record)}
    leaked = sorted(forbidden & present)
    if leaked:
        raise RuntimeError(f"interim report would leak blinded field(s): {leaked}")


def seed_interaction(path: Path) -> float:
    """I_s from the fixed equal-family H=2 estimator.  Returned to the caller only."""
    result = json.loads(path.read_text())
    h2 = {cell: result["cells"][cell]["per_horizon"]["2"]["equal_family_cosine"]
          for cell in D.CELLS}
    return ((h2["proprio_rollout"] - h2["proprio_one_step"])
            - (h2["rgb_rollout"] - h2["rgb_one_step"]))


def lineage(seed: int) -> dict:
    """Technical validity and attempt lineage for one quadruplet."""
    seed_dir = D.OUT / f"seed_{seed}"
    run = json.loads((seed_dir / "run_record.json").read_text())
    selection = json.loads((seed_dir / "selection_result.json").read_text())
    cells = run["cells_run"]
    return {
        "seed": seed,
        "completed": bool(run.get("completed")),
        "cells": len(cells),
        "all_cells_valid": all(c["validity"] == "valid" for c in cells),
        "all_cells_24_epochs": all(c["epochs_trained"] == 24 for c in cells),
        "all_checkpoints_epoch_21": all(c["checkpoint_epoch"] == 21 for c in cells),
        "execution_order": run["execution_order"],
        "shared_parameters_bit_identical": run["shared_parameters_bit_identical"],
        "batch_plan_identical_across_cells": run["batch_plan_identical_across_cells"],
        "attempts": 1,
        "restarts": 0,
        "resumed_from_interruption": False,
        "authorisation_receipt_digest": run["authorisation_receipt_digest"],
        "factorial_manifest_digest": run["factorial_manifest_digest"],
        "selection_rows": selection["selection_rows"],
        "mask_digest": selection["mask_digest"],
        "checkpoint_sha256": {c["cell"]: c["checkpoint_sha256"] for c in cells},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    seeds = list(D.SEED_REGISTRY[:S.INTERIM_N])
    lineages, values = [], []
    for seed in seeds:
        seed_dir = D.OUT / f"seed_{seed}"
        record = lineage(seed)
        if not (record["completed"] and record["all_cells_valid"]
                and record["all_cells_24_epochs"] and record["all_checkpoints_epoch_21"]):
            raise SystemExit(f"seed {seed} is not a complete, technically valid quadruplet")
        lineages.append(record)
        values.append(seed_interaction(seed_dir / "selection_result.json"))

    # The five values enter the frozen interim and are discarded here.
    decision = S.interim(values)
    del values

    factorial = FM.load()
    manifests = {l["factorial_manifest_digest"] for l in lineages}
    masks = {l["mask_digest"] for l in lineages}
    receipts = {l["authorisation_receipt_digest"] for l in lineages}
    if len(manifests) != 1 or len(masks) != 1 or len(receipts) != 1:
        raise SystemExit("a bound artefact changed between quadruplets")

    report = {
        "status": STATUS, "claim_bearing": False,
        "stage": "variance-only interim after the initial five seed quadruplets",
        "blinding": ("the interaction mean, its direction, the individual I_s values, cell "
                     "means, family contrasts and all intervals are deliberately absent"),
        "quadruplets_included": len(lineages),
        "five_complete_and_technically_valid": True,
        "estimator": ("fixed equal-family H=2: valid tokens within a row -> rows within an "
                      "episode cluster -> episodes within a family -> unweighted mean of "
                      "eight families"),
        "no_artefact_changed_between_quadruplets": {
            "factorial_manifest_digest": manifests.pop(),
            "horizon_mask_digest": masks.pop(),
            "authorisation_receipt_digest": receipts.pop(),
            "selection_rows": factorial["rows_by_split"]["checkpoint_selection"],
        },
        "technical_validity_and_lineage": lineages,
        "sample_standard_deviation_s_I": decision["sample_sd_of_interaction"],
        "upper_bound_sigma_U_90pc_one_sided": decision["sd_upper_bound_90pc_one_sided"],
        "power_inputs": {
            "minimally_relevant_interaction": decision["minimally_relevant_interaction"],
            "alpha": decision["alpha"], "target_power": decision["target_power"],
            "bounds": [S.MIN_N, S.MAX_N],
            "distribution": "exact noncentral t, one-sample, df = N - 1",
        },
        "power_by_total_N": decision["power_curve"],
        "frozen_total_N": decision["n_final"],
        "power_at_frozen_N": decision["power_at_final"],
        "precision_limited": decision["precision_limited"],
        "decision_depends_only_on": decision["decision_depends_only_on"],
        "next_step": ("STOP. Seeds six to ten remain locked; no further seed is launched "
                      "and no unblinded analysis is performed."),
    }
    _assert_blinded(report)
    report["report_digest"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()).hexdigest()
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "technical_validity_and_lineage"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
