#!/usr/bin/env python3
"""The explicit ordered factorial manifest: the one artefact every cell iterates.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The broader proprioceptive manifest holds 4,444 rows, but a rollout cell needs a
step-two target and all four cells must see IDENTICAL rows or the objective
comparison is confounded.  This artefact therefore fixes the factorial row set
once, in an explicit order, and both the one-step and the rollout cells iterate
it directly -- neither re-derives a row set from a filter.

    3,922 training rows + 475 selection rows, every one with a step-two target
    47 rows excluded from the broader manifest, reason ``missing_step2_target``

The H = 1..4 changed-token masks and their counts are computed here from the
FROZEN thresholds and stored, so no cell or evaluator can refit a threshold.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import run_dev_v03_two_step_rollout_v1 as R  # noqa: E402
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
PROPRIO = CACHE / "proprio_v1"
EVAL_CACHE = CACHE / "temporal_action_jepa_v1" / "evaluation"
TWO = CACHE / "two_step"
HORIZONS = CACHE / "horizons"
OUT = PROPRIO / "factorial_manifest.json"

EXPECTED = {"train": 3922, "checkpoint_selection": 475}
EXPECTED_EXCLUSIONS = 47
MAX_H = 4


class ManifestViolation(RuntimeError):
    """The factorial manifest must not be written."""


def frozen_thresholds() -> dict:
    matched = json.loads(
        (TWO / "evaluation" / "MATCHED_24_EPOCH_result_epochs_0_23.json").read_text())
    return matched["masks"]


def _load(path: Path, count: int) -> torch.Tensor:
    return R.load_cache(path, count)


def horizon_masks(entries, map_record, thresholds) -> dict:
    """Changed-token masks at H = 1..4 for the selection split, frozen thresholds only.

    H=1 uses the frozen step-1 threshold; H>=2 the frozen step-2 threshold.  No
    threshold is fitted here.  H=3 and H=4 need the horizon target encodings, which
    exist for the rows of the frozen 479-row horizon manifest; rows outside it are
    reported rather than silently dropped.
    """
    selection = [e for e in entries if e["split"] == "checkpoint_selection"]
    n_train = map_record["source_train"]
    n_sel = map_record["source_selection"]
    cache = [e["cache_index"] for e in selection]
    step2 = [e["step2_cache_index"] for e in selection]

    now = T.normalise(_load(EVAL_CACHE / "frozen_current.f16",
                            n_train + n_sel)[n_train:][cache].float())
    y1 = T.normalise(_load(EVAL_CACHE / "frozen_sel_future.f16", n_sel)[cache].float())
    sel_step2_rows = (TWO / "frozen_sel_step2.f16").stat().st_size // (R.TOKENS * R.DIM * 2)
    y2 = T.normalise(_load(TWO / "frozen_sel_step2.f16", sel_step2_rows)[step2].float())

    masks = {
        "1": (y1 - now).pow(2).mean(-1) >= thresholds["step1_threshold"],
        "2": (y2 - now).pow(2).mean(-1) >= thresholds["step2_threshold"],
    }

    # H=3 and H=4 come from the frozen horizon target encodings.
    horizon_rows = [json.loads(line) for line in
                    (HORIZONS / "FINAL" / "FINAL_horizon_rows_479.jsonl").read_text().splitlines()
                    if line.strip()]
    horizon_rows = [r for r in horizon_rows if r["max_horizon"] >= MAX_H]
    horizon_position = {r["pair_sha256"]: i for i, r in enumerate(horizon_rows)}
    covered = [i for i, e in enumerate(selection) if e["pair_sha256"] in horizon_position]
    picks = [horizon_position[selection[i]["pair_sha256"]] for i in covered]

    counts = {"1": int(masks["1"].sum()), "2": int(masks["2"].sum())}
    detail = {"1": {"rows": len(selection)}, "2": {"rows": len(selection)}}
    for h in (3, 4):
        targets = T.normalise(_load(HORIZONS / f"target_h{h}.f16",
                                    len(horizon_rows))[picks].float())
        sub = (targets - now[covered]).pow(2).mean(-1) >= thresholds["step2_threshold"]
        masks[str(h)] = sub
        counts[str(h)] = int(sub.sum())
        detail[str(h)] = {"rows": len(covered),
                          "rows_without_horizon_targets": len(selection) - len(covered)}

    return {
        "policy": ("H=1 uses the frozen step-1 threshold; H>=2 reuse the frozen step-2 "
                   "threshold. No threshold is fitted on the factorial rows."),
        "thresholds": {"step1": thresholds["step1_threshold"],
                       "step2": thresholds["step2_threshold"]},
        "changed_token_counts": counts,
        "coverage": detail,
        "tokens_per_row": R.TOKENS,
        "mask_digest": hashlib.sha256(
            b"".join(masks[str(h)].numpy().tobytes() for h in range(1, MAX_H + 1))).hexdigest(),
    }


def build() -> dict:
    map_record = MAP.load()
    entries = [e for e in map_record["entries"] if e["has_step2_target"]]
    excluded = [e for e in map_record["entries"] if not e["has_step2_target"]]

    counts = collections.Counter(e["split"] for e in entries)
    if dict(counts) != EXPECTED:
        raise ManifestViolation(f"factorial row counts {dict(counts)} != {EXPECTED}")
    if len(excluded) != EXPECTED_EXCLUSIONS:
        raise ManifestViolation(
            f"{len(excluded)} exclusions != the expected {EXPECTED_EXCLUSIONS}")
    if any(e["step2_cache_index"] is None for e in entries):
        raise ManifestViolation("a factorial row carries no step-two cache index")
    if not all(e["has_action"] and e["has_control"] and e["has_proprio"] for e in entries):
        raise ManifestViolation("a factorial row is missing action, control or proprioception")

    # Explicit, stable order: split, then family, then source frame index.
    order = sorted(entries, key=lambda e: (e["split"] != "train", e["family"],
                                           e["source_frame_index"], e["stable_row_id"]))
    rows = [{
        "position": position,
        "stable_row_id": e["stable_row_id"],
        "split": e["split"], "family": e["family"],
        "episode_cluster": e["episode_cluster"],
        "manifest_row_index": e["manifest_row_index"],
        "cache_index": e["cache_index"],
        "step2_cache_index": e["step2_cache_index"],
        "source_frame_index": e["source_frame_index"],
        "target_frame_index": e["target_frame_index"],
        "pair_sha256": e["pair_sha256"],
        "action_blocks_available": e["action_blocks_available"],
    } for position, e in enumerate(order)]

    exclusion_rows = [{
        "stable_row_id": e["stable_row_id"], "split": e["split"], "family": e["family"],
        "manifest_row_index": e["manifest_row_index"],
        "reason": "missing_step2_target",
    } for e in sorted(excluded, key=lambda e: (e["split"] != "train", e["family"],
                                               e["source_frame_index"]))]

    thresholds = frozen_thresholds()
    masks = horizon_masks(order, map_record, thresholds)

    record = {
        "status": STATUS, "claim_bearing": False,
        "purpose": ("the single ordered row set every cell iterates directly; neither the "
                    "one-step nor the rollout cell re-derives it from a filter"),
        "rows_total": len(rows),
        "rows_by_split": dict(counts),
        "rows_by_split_and_family": dict(
            collections.Counter(f"{e['split']}/{e['family']}" for e in order)),
        "episode_clusters": len({e["episode_cluster"] for e in order}),
        "order": "split (train first), then family, then source_frame_index, then stable_row_id",
        "exclusions": exclusion_rows,
        "exclusions_total": len(exclusion_rows),
        "exclusions_by_split_and_family": dict(
            collections.Counter(f"{e['split']}/{e['family']}" for e in excluded)),
        "exclusion_reason_codes": {
            "missing_step2_target": ("no step-two target exists for this row, so a rollout "
                                     "cell could not train on it; excluding it from every "
                                     "cell keeps the four cells on identical rows"),
        },
        "horizon_masks": masks,
        "base_manifest_rows_sha256": map_record["manifest_rows_sha256"],
        "normalisation_sha256": map_record["normalisation_sha256"],
        "canonical_cache_map_digest": map_record["digest"],
        "rows": rows,
    }
    record["digest"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def load(path: Path = OUT) -> dict:
    record = json.loads(Path(path).read_text())
    stored = record.pop("digest")
    recomputed = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    if recomputed != stored:
        raise ManifestViolation(f"factorial manifest digest mismatch: {recomputed} != {stored}")
    record["digest"] = stored
    return record


def positions(record: dict, split: str):
    """The ONLY iteration order a cell may use."""
    return [row["position"] for row in record["rows"] if row["split"] == split]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    record = build()
    Path(args.out).write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items()
                      if k not in ("rows", "exclusions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
