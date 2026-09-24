#!/usr/bin/env python3
"""Freeze the scientific run package: one digest over every binding artefact.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The model configuration, rows manifest and normalisation contract keep their own
hashes -- their contents have not changed -- but the run PACKAGE is a new object
and gets its own digest, covering:

    model configuration        582e7088...
    rows manifest              7b79d128...
    normalisation contract     f5ea58b2...
    canonical cache-index map
    ten-seed registry
    cell-order schedule (with the prefix-balance table)
    device policy
    metric aggregation contract
    software environment

Both the trainer and the evaluator must refuse to run if the package digest, or
the canonical-map digest inside it, differs from what they were given.
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

from scripts import dev_proprio_experiment_config_v1 as C  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402
from scripts import eval_dev_proprio_factorial_v1 as E  # noqa: E402
from scripts import dev_seed_reestimation_v1 as S  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT = D.CACHE / "proprio_v1" / "scientific_run_package.json"


def build() -> dict:
    manifest = json.loads((D.PROPRIO / "proprio_manifest.json").read_text())
    map_record = MAP.load()
    factorial = FM.load()
    registry = D.register_seeds(D.CACHE / "factorial_v1")
    if factorial["canonical_cache_map_digest"] != map_record["digest"]:
        raise RuntimeError("factorial manifest and canonical map disagree")
    if factorial["base_manifest_rows_sha256"] != manifest["rows_sha256"]:
        raise RuntimeError("factorial manifest was built from a different base manifest")

    package = {
        "status": STATUS, "claim_bearing": False,
        "name": "proprioception x rollout factorial -- scientific run package",
        "model_configuration_sha256": C.config_sha256(),
        "base_manifest_rows_sha256": manifest["rows_sha256"],
        "factorial_manifest_digest": factorial["digest"],
        "factorial_manifest_totals": {
            "rows_total": factorial["rows_total"],
            "rows_by_split": factorial["rows_by_split"],
            "episode_clusters": factorial["episode_clusters"],
            "exclusions_total": factorial["exclusions_total"],
            "exclusion_reason_codes": list(factorial["exclusion_reason_codes"]),
            "order": factorial["order"],
        },
        "step_two_index_mapping": {
            "space": map_record["step2_cache_indexing"],
            "rows_with_step2_target": map_record["rows_with_step2_target"],
            "rows_with_step2_by_split": map_record["rows_with_step2_by_split"],
        },
        "horizon_masks": factorial["horizon_masks"],
        "normalisation_sha256": manifest["normalisation_sha256"],
        "canonical_cache_map_digest": map_record["digest"],
        "canonical_cache_map_totals": {
            "retained_rows": map_record["retained_rows"],
            "retained_by_split": map_record["retained_by_split"],
            "rows_with_step2_target": map_record["rows_with_step2_target"],
            "rows_with_step2_by_split": map_record["rows_with_step2_by_split"],
            "excluded_rows": map_record["excluded_rows"],
            "episode_clusters": map_record["episode_clusters"],
            "verification": map_record["verification"],
        },
        "seed_registry_sha256": registry["sha256"],
        "seed_identifiers": registry["seed_identifiers"],
        "cell_order_schedule": registry["cell_order_schedule"],
        "cell_order_type": registry["cell_order_type"],
        "prefix_balance_all_within_one": all(
            entry["balanced_within_one"] for entry in registry["prefix_balance"].values()),
        "device_policy": D.DEVICE_POLICY,
        "software_environment": D.environment_record(),
        "metric_aggregation_contract": {
            "primary": ("H=2 correct-future cosine: valid tokens within a row -> rows within "
                        "an episode cluster -> episodes within a family -> unweighted mean of "
                        "the eight family scores"),
            "estimand": "I_s = (PropRoll_s - PropOne_s) - (RGBRoll_s - RGBOne_s)",
            "secondary": "corpus-weighted / token-pooled, reported separately, never mixed",
            "h3": "beyond-trained-horizon transfer",
            "h4": "longer-horizon diagnostic",
            "combined_h2_h3": "prohibited",
            "co_outcomes": E.non_inferiority_status(C.CONFIG),
            "replication_unit": "training seed quadruplet",
            "episode_bootstrap": "within-seed evaluation uncertainty only",
        },
        "seed_design": {
            "interim_quadruplets": S.INTERIM_N,
            "minimally_relevant_interaction": S.MINIMALLY_RELEVANT,
            "alpha": S.ALPHA, "target_power": S.TARGET_POWER,
            "bounds": [S.MIN_N, S.MAX_N],
            "upper_bound": "90% one-sided chi-square bound on sigma",
            "prefix_rule": registry["prefix_rule"],
        },
        "budget": {"epochs": D.EPOCHS, "checkpoint_epoch": D.CHECKPOINT_EPOCH,
                   "selection_permitted": False},
        "launch_state": "LOCKED -- --seed-index refuses to run",
        "superseded_configurations": [
            entry["sha256"] for entry in C.SUPERSEDED_CANDIDATE_CONFIGURATIONS],
    }
    package["package_digest"] = hashlib.sha256(
        json.dumps(package, sort_keys=True).encode()).hexdigest()
    return package


def verify(path: Path = OUT) -> dict:
    package = json.loads(Path(path).read_text())
    stored = package.pop("package_digest")
    recomputed = hashlib.sha256(json.dumps(package, sort_keys=True).encode()).hexdigest()
    if recomputed != stored:
        raise RuntimeError(f"run package digest mismatch: {recomputed} != {stored}")
    package["package_digest"] = stored
    return package


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    package = build()
    Path(args.out).write_text(json.dumps(package, indent=2))
    print(json.dumps({k: v for k, v in package.items()
                      if k not in ("cell_order_schedule", "software_environment")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
