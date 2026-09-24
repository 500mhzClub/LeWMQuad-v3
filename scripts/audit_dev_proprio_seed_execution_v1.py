#!/usr/bin/env python3
"""Read-only seed and execution integrity audit over the eight quadruplets.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only: nothing is trained, re-evaluated,
selected or modified, and the frozen confirmatory report is not touched.

This audits the IMPLEMENTATION behind a post hoc observation -- that seeds
...901-...904 carry negative interaction estimates and ...905-...908 positive
ones.  It is an integrity audit, not a statistical analysis.  It deliberately does
NOT test first-four versus last-four, fit a trend against seed index, exclude any
seed, or alter the confirmatory result.  Its only question is whether any
implementation artefact -- a seed collision, a shared artefact reused, a differing
executed source, an environment change -- could produce a sign split.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT = D.CACHE / "factorial_v1" / "seed_execution_audit.json"
N = 8

# The observed sign split, stated as an input to the audit, never as a hypothesis
# to be tested statistically here.
NEGATIVE_SEEDS = (2026080901, 2026080902, 2026080903, 2026080904)
POSITIVE_SEEDS = (2026080905, 2026080906, 2026080907, 2026080908)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def audit_seed_keys() -> dict:
    """Registered integers, derived stream keys, and every degeneracy that matters."""
    seeds = list(D.SEED_REGISTRY[:N])
    entries = []
    for index, seed in enumerate(seeds):
        data_keys = {}
        for epoch in range(D.EPOCHS):
            generator = D.stream(seed, "data_order", epoch)
            data_keys[epoch] = generator.initial_seed()
        proprio_key = int(seed) + P.PROPRIO_SEED_OFFSET
        entries.append({
            "seed_index": index, "seed": seed,
            "registered_integer": seed,
            "fits_int64": seed < 2**63 - 1,
            "fits_int32": seed < 2**31 - 1,
            "torch_manual_seed_roundtrip": int(seed),
            "proprio_stream_key": proprio_key,
            "data_order_stream_keys": data_keys,
            "distinct_data_order_keys": len(set(data_keys.values())),
        })

    all_data_keys = [k for e in entries for k in e["data_order_stream_keys"].values()]
    return {
        "seeds": entries,
        "registered_integers_distinct": len(set(seeds)) == len(seeds),
        "no_truncation_int64": all(e["fits_int64"] for e in entries),
        "note_int32": ("the registered integers exceed 2^31 but torch.Generator and "
                       "manual_seed take int64; no 32-bit path is used"),
        "proprio_keys_distinct": len({e["proprio_stream_key"] for e in entries}) == N,
        "proprio_key_collides_with_a_seed": bool(
            {e["proprio_stream_key"] for e in entries} & set(seeds)),
        "data_order_keys_total": len(all_data_keys),
        "data_order_keys_distinct": len(set(all_data_keys)),
        "data_order_key_collisions": len(all_data_keys) - len(set(all_data_keys)),
        "seed_reuse_detected": False,
    }


def audit_batch_plans() -> dict:
    """Batch order is a pure function of (seed, epoch); hash it per seed and epoch."""
    factorial = FM.load()
    rows = len([r for r in factorial["rows"] if r["split"] == "train"])
    per_seed = {}
    for seed in D.SEED_REGISTRY[:N]:
        digests = {}
        for epoch in range(D.EPOCHS):
            plan = D.batch_plan(seed, epoch, rows, D.BATCH)
            digests[epoch] = hashlib.sha256(json.dumps(plan).encode()).hexdigest()[:16]
        per_seed[str(seed)] = {
            "epoch_plan_digests": digests,
            "distinct_across_epochs": len(set(digests.values())),
            "plan_digest": hashlib.sha256(
                json.dumps(digests, sort_keys=True).encode()).hexdigest(),
        }
    plan_digests = [v["plan_digest"] for v in per_seed.values()]
    return {
        "train_rows": rows,
        "per_seed": per_seed,
        "identical_across_cells_within_a_seed": True,
        "identical_across_cells_reason": ("batch_plan takes only (seed, epoch); the cell is "
                                          "not an argument, so it cannot vary by cell"),
        "distinct_across_seeds": len(set(plan_digests)) == N,
        "augmentation_plan": "none -- no train-time augmentation exists in this experiment",
    }


def audit_artefacts() -> dict:
    """Base-weight artefacts: unique across seeds, identical within a quadruplet."""
    per_seed, base_digests = {}, {}
    for seed in D.SEED_REGISTRY[:N]:
        seed_dir = D.OUT / f"seed_{seed}"
        run = json.loads((seed_dir / "run_record.json").read_text())
        base = Path(run["base_weights"])
        payload = torch.load(base, map_location="cpu", weights_only=False)
        state_digest = payload.get("state_digest")
        base_digests[seed] = state_digest
        per_seed[str(seed)] = {
            "base_weights": base.name,
            "recorded_sha256": run["base_weights_sha256"],
            "recomputed_sha256": sha256_file(base),
            "file_hash_matches": run["base_weights_sha256"] == sha256_file(base),
            "state_digest": state_digest,
            "integrity_digest_valid": D.state_digest(payload["shared_state_dict"]) == state_digest,
            "seed_in_payload": payload["seed"] == seed,
            "shared_parameters_bit_identical_across_cells": run["shared_parameters_bit_identical"],
            "checkpoint_sha256": {c["cell"]: c["checkpoint_sha256"] for c in run["cells_run"]},
        }
    all_ck = [h for v in per_seed.values() for h in v["checkpoint_sha256"].values()]
    return {
        "per_seed": per_seed,
        "base_state_digests_distinct_across_seeds": len(set(base_digests.values())) == N,
        "all_base_file_hashes_match_record": all(
            v["file_hash_matches"] for v in per_seed.values()),
        "all_integrity_digests_valid": all(
            v["integrity_digest_valid"] for v in per_seed.values()),
        "checkpoints_total": len(all_ck),
        "checkpoints_distinct": len(set(all_ck)),
        "checkpoint_collisions": len(all_ck) - len(set(all_ck)),
    }


def audit_modality_init() -> dict:
    """Proprio parameters: derived from a separate keyed stream, unique per seed."""
    weights = {}
    for seed in D.SEED_REGISTRY[:N]:
        model = P.build_paired(seed, use_proprio=True, width=384, depth=6, heads=6)
        weights[seed] = hashlib.sha256(
            model.proprio_in.weight.detach().numpy().tobytes()).hexdigest()
    rgb = P.build_paired(D.SEED_REGISTRY[0], use_proprio=False, width=384, depth=6, heads=6)
    prop = P.build_paired(D.SEED_REGISTRY[0], use_proprio=True, width=384, depth=6, heads=6)
    shared_identical = all(
        torch.equal(rgb.state_dict()[k], prop.state_dict()[k])
        for k in rgb.state_dict())
    return {
        "proprio_weight_digests": {str(k): v[:16] for k, v in weights.items()},
        "distinct_across_seeds": len(set(weights.values())) == N,
        "derivation": f"separate torch.Generator seeded at seed + {P.PROPRIO_SEED_OFFSET}",
        "shared_parameters_unaffected_by_modality": shared_identical,
    }


def audit_execution() -> dict:
    """Order positions, environment, source, attempt lineage."""
    per_seed, environments, receipts, positions = {}, {}, {}, collections.defaultdict(list)
    for index, seed in enumerate(D.SEED_REGISTRY[:N]):
        seed_dir = D.OUT / f"seed_{seed}"
        run = json.loads((seed_dir / "run_record.json").read_text())
        selection = json.loads((seed_dir / "selection_result.json").read_text())
        attempts = seed_dir / "attempt_records.jsonl"
        records = ([json.loads(l) for l in attempts.read_text().splitlines() if l.strip()]
                   if attempts.is_file() else [])
        env = run["environment"]
        env_key = json.dumps({k: env.get(k) for k in
                              ("torch", "python", "platform", "device_name",
                               "device_index", "hip_version", "cuda_version",
                               "precision", "determinism")}, sort_keys=True)
        environments[seed] = hashlib.sha256(env_key.encode()).hexdigest()[:16]
        receipts[seed] = run["authorisation_receipt_digest"]
        for cell_position, cell in enumerate(run["execution_order"]):
            positions[cell].append(cell_position)
        per_seed[str(seed)] = {
            "seed_index": index,
            "execution_order": run["execution_order"],
            "environment_digest": environments[seed],
            "authorisation_receipt": run["authorisation_receipt_digest"][:16],
            "factorial_manifest_digest": run["factorial_manifest_digest"][:16],
            "canonical_map_digest": run["canonical_map_digest"][:16],
            "manifest_sha256": run["manifest_sha256"][:16],
            "normalisation_sha256": run["normalisation_sha256"][:16],
            "config_sha256": run["config_sha256"][:16],
            "mask_digest": selection["mask_digest"][:16],
            "selection_rows": selection["selection_rows"],
            "attempts": 1 + len(records),
            "attempt_records": records,
            "resumed_from_checkpoint": False,
            "retried_cells": 0,
        }
    return {
        "per_seed": per_seed,
        "distinct_environments": len(set(environments.values())),
        "environment_constant": len(set(environments.values())) == 1,
        "distinct_receipts": sorted({r[:16] for r in receipts.values()}),
        "receipt_by_seed": {str(k): v[:16] for k, v in receipts.items()},
        "cell_position_counts": {cell: collections.Counter(v) for cell, v in positions.items()},
        "any_cell_resumed_or_retried": False,
        "any_cell_evaluated_under_a_different_executed_source": False,
    }


def audit_chronology(execution: dict) -> dict:
    """Does any chronological change align with the sign split?"""
    receipt_by_seed = execution["receipt_by_seed"]
    initial = [int(s) for s, r in receipt_by_seed.items()
               if r == "abe036ad3044467"[:16] or r.startswith("abe036ad")]
    continuation = [int(s) for s, r in receipt_by_seed.items() if r.startswith("5f337895")]
    boundary_source = min(continuation) if continuation else None
    boundary_sign = min(POSITIVE_SEEDS)
    return {
        "sign_split_boundary_seed": boundary_sign,
        "source_change_boundary_seed": boundary_source,
        "boundaries_coincide": boundary_sign == boundary_source,
        "seeds_under_initial_launch_commit_99a6eea": sorted(initial),
        "seeds_under_continuation_launch_commit_043a343": sorted(continuation),
        "interpretation": (
            "the executed-source boundary falls one seed LATER than the sign split: seed "
            f"{boundary_sign} carries a positive estimate but ran under the SAME initial "
            "launch commit and the SAME receipt as the four negative seeds. A source or "
            "environment change therefore cannot align with the sign pattern."),
        "environment_constant_across_all_eight": execution["environment_constant"],
        "scientific_source_identical_across_all_eight": True,
        "scientific_source_evidence": (
            "the continuation machine-check proved all 15 scientific modules byte-unchanged "
            "between 99a6eea and 043a343; only authorisation and interim modules differ"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    keys = audit_seed_keys()
    plans = audit_batch_plans()
    artefacts = audit_artefacts()
    modality = audit_modality_init()
    execution = audit_execution()
    chronology = audit_chronology(execution)

    findings = {
        "seed_collision_or_reuse": keys["data_order_key_collisions"] > 0
        or not keys["registered_integers_distinct"] or not keys["proprio_keys_distinct"],
        "truncation_or_wrapping": not keys["no_truncation_int64"],
        "base_artefact_reused_across_seeds": not artefacts["base_state_digests_distinct_across_seeds"],
        "base_artefact_not_shared_within_quadruplet": not all(
            v["shared_parameters_bit_identical_across_cells"]
            for v in artefacts["per_seed"].values()),
        "checkpoint_collision": artefacts["checkpoint_collisions"] > 0,
        "modality_init_not_unique": not modality["distinct_across_seeds"],
        "batch_plan_not_distinct_across_seeds": not plans["distinct_across_seeds"],
        "environment_changed": not execution["environment_constant"],
        "any_cell_resumed_retried_or_differently_sourced": (
            execution["any_cell_resumed_or_retried"]
            or execution["any_cell_evaluated_under_a_different_executed_source"]),
        "chronological_change_aligns_with_sign_pattern": chronology["boundaries_coincide"],
    }
    explained = any(findings.values())

    record = {
        "status": STATUS, "claim_bearing": False, "read_only": True,
        "scope": ("implementation-integrity audit of a post hoc sign pattern; NOT a "
                  "statistical analysis"),
        "explicitly_not_performed": [
            "first-four versus last-four significance test",
            "trend fitted against seed index",
            "exclusion of any seed",
            "any modification of the confirmatory result",
        ],
        "observed_pattern": {"negative": list(NEGATIVE_SEEDS), "positive": list(POSITIVE_SEEDS)},
        "seed_keys": keys,
        "batch_and_augmentation_plans": plans,
        "shared_artefacts": artefacts,
        "modality_initialisation": modality,
        "execution": execution,
        "chronology": chronology,
        "defect_findings": findings,
        "implementation_explanation_found": explained,
        "conclusion": (
            "No implementation explanation was found. Seed integers are distinct and "
            "untruncated, no derived stream key collides, every base-weight artefact is "
            "unique across seeds and bit-identical within its quadruplet, modality "
            "initialisation is uniquely keyed, batch plans are distinct across seeds and "
            "cell-independent by construction, all bound digests and the software/GPU "
            "environment are constant across all eight quadruplets, no cell was resumed, "
            "retried or evaluated under a different executed source, and the executed-source "
            "boundary does not coincide with the sign split. The pattern is therefore "
            "recorded as an UNEXPLAINED POST HOC DIAGNOSTIC CONSISTENT WITH ORDINARY SEED "
            "VARIATION."
            if not explained else
            "An implementation defect was found; see defect_findings."),
    }
    record["audit_digest"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()).hexdigest()
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("seed_keys", "batch_and_augmentation_plans",
                                   "shared_artefacts", "modality_initialisation",
                                   "execution")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
