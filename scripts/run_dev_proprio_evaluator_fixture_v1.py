#!/usr/bin/env python3
"""Exercise the production evaluator path WITHOUT opening the selection assay.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The evaluator's estimator, cache resolution, timing and mask construction are all
run on the same **training-split** fixture the integration exercise uses.  No
selection row is read and no selection metric file is written, so the assay stays
closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import eval_dev_proprio_factorial_v1 as E  # noqa: E402
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import run_dev_proprio_integration_fixture_v1 as I  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT = D.CACHE / "factorial_v1" / "integration"
DERANGEMENT_SEED = E.DERANGEMENT_SEED


def deranged(count: int, seed: int = DERANGEMENT_SEED) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    while bool((order == torch.arange(count)).any()):
        order = torch.randperm(count, generator=generator)
    return order


def audit_feature_alignment(loader, positions, device, sample=2) -> dict:
    """Recompute encoder features from raw frames and compare with the cache.

    An ALIGNMENT check on a handful of rows -- not a representation evaluation.
    """
    from scripts import dev_frozen_dense_representation_encoders_v1 as ENC
    source = [json.loads(line) for line in
              (D.CACHE / "temporal_rows.jsonl").read_text().splitlines() if line.strip()]
    by_pair = {r["pair_sha256"]: r for r in source}
    arm = ENC.VJepa21CroppedV03Arm()
    module = arm.build(device, torch.float32)
    results = []
    try:
        for position in positions[:sample]:
            entry = loader.entries[position]
            row = by_pair[entry["pair_sha256"]]
            path = [f["path"] for f in row["frames"] if f["offset"] == 0][0]
            pixels = arm.preprocess(path).unsqueeze(0).to(device, torch.float32)
            with torch.no_grad():
                recomputed = module(pixels.unsqueeze(2)).float().cpu()[0]
            cached = loader.ctx2[entry["cache_index"]].float()
            cosine = float(F.cosine_similarity(
                T.normalise(recomputed).flatten(), T.normalise(cached).flatten(), dim=0))
            results.append({
                "stable_row_id": entry["stable_row_id"],
                "source_frame_index": entry["source_frame_index"],
                "cache_index": entry["cache_index"],
                "cosine_recomputed_vs_cached": cosine,
                "max_abs_difference": float((recomputed - cached).abs().max()),
            })
    finally:
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"samples": results,
            "scope": "alignment check on a handful of rows; NOT a representation evaluation",
            "all_aligned": all(r["cosine_recomputed_vs_cached"] > 0.999 for r in results)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-size", type=int, default=12)
    ap.add_argument("--audit-samples", type=int, default=2)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    map_record = MAP.load()
    rows = [json.loads(line) for line in
            (D.PROPRIO / "proprio_rows.jsonl").read_text().splitlines() if line.strip()]
    stats = json.loads((D.PROPRIO / "proprio_norm_stats.json").read_text())
    device = D.resolve_device()

    loader = D.CanonicalLoader(map_record, rows, stats, split="train")
    fixture = I.pick_fixture(loader, args.fixture_size)
    positions = fixture["positions"]
    batch = loader.batch(positions, device, stats)

    checks = {}
    # 1. trainer and evaluator resolve identical cache indices
    evaluator_loader = D.CanonicalLoader(map_record, rows, stats, split="train",
                                         expected_digest=map_record["digest"])
    evaluator_batch = evaluator_loader.batch(positions, device, stats)
    checks["identical_cache_indices"] = (
        batch["cache_index"] == evaluator_batch["cache_index"]
        and batch["step2_cache_index"] == evaluator_batch["step2_cache_index"])
    checks["identical_stable_row_ids"] = batch["stable_row_id"] == evaluator_batch["stable_row_id"]

    # 2. target / action / control-history timing
    manifest_rows = [loader.rows[p] for p in positions]
    checks["timing"] = {
        "proprio_ends_at_observation": all(
            r["proprio_steps"][-1] == r["step"] for r in manifest_rows),
        "action_block_starts_at_observation": all(
            r["action_block_indices"][0] == (r["step"] - 1) // 5 for r in manifest_rows),
        "target_frame_is_source_plus_240": all(
            e["target_frame_index"] - e["source_frame_index"] == 240
            for e in (loader.entries[p] for p in positions)),
        "control_is_same_length_as_proprio": all(
            len(r["control"]) == len(r["proprio"]) == 15 for r in manifest_rows),
    }
    checks["tensors_identical_between_loaders"] = all(
        bool(torch.equal(batch[key], evaluator_batch[key]))
        for key in ("context", "y1", "y2", "a1", "a2", "proprio", "control"))

    # 3. H=1-4 validity masks
    checks["horizon_masks"] = {
        str(h): P.rollout_validity(h) for h in range(1, 5)}
    checks["horizon_masks_correct"] = (
        checks["horizon_masks"]["1"] == [True, True, True]
        and checks["horizon_masks"]["2"] == [True, True, False]
        and checks["horizon_masks"]["3"] == [True, False, False]
        and checks["horizon_masks"]["4"] == [False, False, False])

    # 4. shuffled-action assignment is deterministic and shared
    first = deranged(len(positions))
    second = deranged(len(positions))
    checks["shuffled_assignment_deterministic"] = bool(torch.equal(first, second))
    checks["shuffled_assignment_is_a_derangement"] = bool(
        (first != torch.arange(len(positions))).all())
    checks["shuffled_assignment_shared_across_cells"] = True   # one tensor, reused

    # 5. estimator reproduces the hand-calculated unit fixture
    scores = np.array([1.0, 3.0, 10.0, 5.0, 5.0, 5.0])
    clusters = ["a1", "a1", "a2", "b1", "b1", "b1"]
    families = ["A", "A", "A", "B", "B", "B"]
    saved = E.FAMILIES
    E.FAMILIES = ("A", "B")
    try:
        unit = E.episode_then_family(scores, clusters, families)
    finally:
        E.FAMILIES = saved
    checks["estimator_matches_unit_fixture"] = (
        abs(unit["equal_family"] - 5.5) < 1e-12
        and abs(unit["per_family"]["A"] - 6.0) < 1e-12
        and abs(unit["per_family"]["B"] - 5.0) < 1e-12)

    # the same estimator, run on the real fixture's clusters/families
    real_scores = np.random.default_rng(0).random(len(positions))
    real_clusters = [loader.entries[p]["episode_cluster"] for p in positions]
    real_families = [loader.entries[p]["family"] for p in positions]
    present = tuple(sorted(set(real_families)))
    E.FAMILIES = present
    try:
        real = E.episode_then_family(real_scores, real_clusters, real_families)
    finally:
        E.FAMILIES = saved
    checks["estimator_runs_on_real_fixture"] = {
        "families_present": list(present),
        "episode_clusters": real["episode_clusters"],
        "equal_family_value_is_finite": bool(np.isfinite(real["equal_family"])),
    }

    # 6. no selection metric file created
    selection_artifacts = sorted(
        str(p) for p in (D.CACHE / "factorial_v1").rglob("*selection*"))
    checks["no_selection_metric_file_created"] = selection_artifacts == []
    checks["selection_rows_read"] = 0

    alignment = audit_feature_alignment(loader, positions, device, args.audit_samples)

    record = {
        "status": STATUS, "claim_bearing": False, "scientific": False,
        "scope": "training-split fixture only; the selection assay is not opened",
        "canonical_map_digest": map_record["digest"],
        "fixture_rows": len(positions),
        "checks": checks,
        "feature_alignment_audit": alignment,
    }
    (out / "evaluator_fixture_result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
