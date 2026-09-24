#!/usr/bin/env python3
"""Post-confirmatory H=1 spatial-retention assay, executed against a frozen spec.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

This is NOT the originally registered occupied co-outcome, which is invalid by
construction and is neither amended nor repaired here.  It measures spatial
retention at **H=1 only**; the corpus carries no usable occupancy label at H=2-4.

Order is enforced by the code:

    1. fit the probe on the probe-fit split, true target latents only
    2. evaluate the FINAL-epoch probe on the calibration split
    3. gate on the frozen qualification criterion
    4. only then hash the probe package and load predictor checkpoints

The 475 factorial selection rows are never touched during fit, calibration, epoch
choice, architecture choice or the qualification decision -- ``_forbid_selection``
makes that a hard failure rather than a promise.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_frozen_dense_representation_screen_v1 as S  # noqa: E402
from scripts import complete_dev_v03_temporal_action_jepa_evaluation_v1 as C  # noqa: E402
from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import run_dev_v03_two_step_rollout_v1 as R  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import eval_dev_proprio_factorial_v1 as E  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
SPEC = D.CACHE / "factorial_v1" / "spatial_retention_spec.json"
OUT = D.CACHE / "factorial_v1" / "spatial_retention"
SPEC_DIGEST = "646073a9b0a43d7a6c3230f55b3d68026d0632af70726c196603cb7ccf182478"

PROBE_SEED = 20_260_810
PROBE_EPOCHS = 12            # fixed budget; FINAL epoch weights are taken
PROBE_BATCH = 32
PROBE_LR = 1e-3
PROBE_WD = 0.01
PROBE_CLIP = 1.0
FIT_FRACTION = 0.8           # of the 3,922 factorial train rows
QUALIFICATION_IOU = 0.35
GRID = (24, 32)


class SpecViolation(RuntimeError):
    """A frozen-specification violation: the assay must not proceed."""


def _forbid_selection(entries, stage: str) -> None:
    bad = [e for e in entries if e["split"] != "train"]
    if bad:
        raise SpecViolation(
            f"{stage}: {len(bad)} non-train row(s) reached a fit/calibration stage; "
            "the 475 selection rows must never be inspected here")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-batch", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    spec = json.loads(SPEC.read_text())
    if spec["specification_digest"] != SPEC_DIGEST:
        raise SpecViolation("frozen specification digest mismatch")
    device = D.resolve_device()

    factorial = FM.load()
    train_entries = [r for r in factorial["rows"] if r["split"] == "train"]
    selection_entries = [r for r in factorial["rows"] if r["split"] == "checkpoint_selection"]

    # ---- deterministic, frozen fit / calibration split of TRAIN rows only ----
    generator = torch.Generator().manual_seed(PROBE_SEED)
    order = torch.randperm(len(train_entries), generator=generator).tolist()
    cut = int(len(order) * FIT_FRACTION)
    fit_rows = [train_entries[i] for i in order[:cut]]
    cal_rows = [train_entries[i] for i in order[cut:]]
    _forbid_selection(fit_rows, "probe fit")
    _forbid_selection(cal_rows, "probe calibration")

    # ---- labels: independent simulator rasters, evaluation labels only -------
    def labels_for(entries):
        return C.future_labels([{"pair_sha256": e["pair_sha256"]} for e in entries])

    fit_labels = labels_for(fit_rows)
    cal_labels = labels_for(cal_rows)

    # ---- TRUE target latents ------------------------------------------------
    n_train = 4075
    future = R.load_cache(
        D.CACHE / "temporal_action_jepa_v1" / "evaluation" / "frozen_train_future.f16", n_train)

    def latents(entries):
        idx = [e["cache_index"] for e in entries]
        return T.normalise(future[idx].float())

    fit_latents = latents(fit_rows)
    cal_latents = latents(cal_rows)

    # ---- fit: fixed budget, FINAL epoch taken, no best-epoch selection -------
    torch.manual_seed(PROBE_SEED)
    probe = S.SharedTokenToBev(P.TOKEN_DIM).to(device)
    optimiser = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR,
                                  weight_decay=PROBE_WD, foreach=False)
    weight = S.class_weights(fit_labels, device)
    targets = torch.from_numpy(fit_labels).long()
    shuffler = torch.Generator().manual_seed(PROBE_SEED + 1)
    history = []
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(len(fit_rows), generator=shuffler)
        total, batches = 0.0, 0
        for start in range(0, len(perm), PROBE_BATCH):
            sel = perm[start:start + PROBE_BATCH]
            batch = fit_latents[sel].to(device, torch.float32)
            optimiser.zero_grad(set_to_none=True)
            loss = F.cross_entropy(probe(batch, GRID), targets[sel].to(device), weight=weight)
            loss.backward()
            nn.utils.clip_grad_norm_(probe.parameters(), PROBE_CLIP)
            optimiser.step()
            total += float(loss.detach()); batches += 1
        history.append({"epoch": epoch, "fit_loss": total / batches})
        print(f"  [probe] epoch {epoch:02d} fit loss {total / batches:.4f}", flush=True)
    probe.eval()   # FINAL epoch weights; no restoration of any earlier epoch

    # ---- qualification on the calibration split, TRUE latents ---------------
    @torch.no_grad()
    def predict(latent_block):
        out = []
        for start in range(0, latent_block.shape[0], 64):
            batch = latent_block[start:start + 64].to(device, torch.float32)
            out.append(probe(batch, GRID).argmax(1).cpu().numpy())
        return np.concatenate(out, 0)

    cal_pred = predict(cal_latents)
    qualification = S.P.metrics(cal_pred, cal_labels)
    iou = float(qualification.get("observable_occupied_iou", 0.0))
    qualified = iou >= QUALIFICATION_IOU

    package = {
        "status": STATUS, "claim_bearing": False,
        "assay": "post-confirmatory H=1 spatial-retention assay",
        "not_the_registered_co_outcome": True,
        "horizon_scope": "H=1 only; the corpus carries no usable occupancy label at H=2-4",
        "specification_digest": SPEC_DIGEST,
        "probe": {"architecture": "SharedTokenToBev(1024) -> 64x64 x 3",
                  "seed": PROBE_SEED, "epochs": PROBE_EPOCHS, "batch": PROBE_BATCH,
                  "lr": PROBE_LR, "weight_decay": PROBE_WD,
                  "epoch_taken": "final", "best_epoch_selection": False},
        "fit_rows": len(fit_rows), "calibration_rows": len(cal_rows),
        "selection_rows_inspected_during_fit_or_qualification": 0,
        "fit_history": history,
        "qualification": {"criterion": f"observable occupied IoU >= {QUALIFICATION_IOU}",
                          "true_target_latent_metrics": qualification,
                          "observed_iou": iou, "qualified": qualified},
    }
    weights_path = OUT / "probe_final_epoch.pt"
    torch.save(probe.state_dict(), weights_path)
    package["probe_weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    if not qualified:
        package["outcome"] = ("FAILED the frozen qualification criterion; the probe is "
                              "preserved and NO predictor checkpoint was loaded or scored")
        package["package_digest"] = hashlib.sha256(
            json.dumps(package, sort_keys=True).encode()).hexdigest()
        (OUT / "probe_package.json").write_text(json.dumps(package, indent=2))
        print(json.dumps({"qualified": False, "iou": iou,
                          "package_digest": package["package_digest"]}, indent=2))
        return 0

    package["package_digest"] = hashlib.sha256(
        json.dumps(package, sort_keys=True).encode()).hexdigest()
    (OUT / "probe_package.json").write_text(json.dumps(package, indent=2))
    print(f"probe QUALIFIED (IoU {iou:.4f}); package {package['package_digest'][:16]}", flush=True)

    # ================= only now may predictor checkpoints be loaded ==========
    del fit_latents, cal_latents
    map_record = json.loads((D.PROPRIO / "canonical_cache_map.json").read_text())
    rows = [json.loads(l) for l in
            (D.PROPRIO / "proprio_rows.jsonl").read_text().splitlines() if l.strip()]
    stats = json.loads((D.PROPRIO / "proprio_norm_stats.json").read_text())
    loader = D.CanonicalLoader(map_record, rows, stats, split="checkpoint_selection",
                              factorial=factorial)
    n = len(loader)
    batch = loader.batch(list(range(n)), device, stats)
    sel_labels = labels_for(selection_entries)
    clusters = [e["episode_cluster"] for e in loader.entries]
    families = [e["family"] for e in loader.entries]

    true_pred = predict(batch["y1"].cpu())
    true_metrics = S.P.metrics(true_pred, sel_labels)

    seeds = list(D.SEED_REGISTRY[:8])
    per_cell_seed = collections.defaultdict(dict)
    for seed in seeds:
        for cell in D.CELLS:
            spec_cell = D.CELL_SPEC[cell]
            path = D.OUT / f"seed_{seed}" / f"seed_{seed}_{cell}_epoch21.pt"
            model = P.build_paired(seed, use_proprio=spec_cell["use_proprio"]).to(device)
            model.load_state_dict(torch.load(path, map_location="cpu",
                                             weights_only=False)["model_state_dict"])
            model.eval()
            outs = []
            with torch.no_grad():
                for start in range(0, n, args.score_batch):
                    stop = min(start + args.score_batch, n)
                    sub = {k: (v[start:stop] if torch.is_tensor(v) else v)
                           for k, v in batch.items()}
                    step = P.unroll(model, sub["context"], [sub["a1"]],
                                    sub["proprio"] if spec_cell["use_proprio"] else None,
                                    sub["control"], max_h=1)[0]
                    outs.append(step.cpu())
            predicted = torch.cat(outs, 0)
            pred_labels = predict(predicted)
            # A row with no observable occupied cell has an undefined IoU; it is
            # recorded as NaN and skipped by the aggregator rather than counted as 0.
            row_iou = np.empty(n, dtype=np.float64)
            for i in range(n):
                value = S.P.metrics(pred_labels[i:i + 1], sel_labels[i:i + 1]).get(
                    "observable_occupied_iou")
                row_iou[i] = float(value) if value is not None else np.nan
            aggregate = E.episode_then_family(row_iou, clusters, families)
            per_cell_seed[cell][seed] = {
                "equal_family": aggregate["equal_family"],
                "per_family": aggregate["per_family"],
                "corpus_weighted": float(np.nanmean(row_iou)),
                "rows_with_defined_iou": int(np.isfinite(row_iou).sum()),
                "whole_split": S.P.metrics(pred_labels, sel_labels),
            }
            del model
            torch.cuda.empty_cache()
            print(f"  scored seed {seed} {cell}", flush=True)

    def stats_of(values):
        from scipy import stats as st
        m = float(np.mean(values)); sd = float(np.std(values, ddof=1))
        half = st.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))
        return {"mean": m, "sd": sd, "t_interval_95": [m - half, m + half],
                "per_seed": [float(v) for v in values]}

    eq = {cell: [per_cell_seed[cell][s]["equal_family"] for s in seeds] for cell in D.CELLS}
    cw = {cell: [per_cell_seed[cell][s]["corpus_weighted"] for s in seeds] for cell in D.CELLS}
    d_rgb = [eq["rgb_rollout"][i] - eq["rgb_one_step"][i] for i in range(8)]
    d_prop = [eq["proprio_rollout"][i] - eq["proprio_one_step"][i] for i in range(8)]
    inter = [d_prop[i] - d_rgb[i] for i in range(8)]

    family_names = sorted(per_cell_seed["rgb_one_step"][seeds[0]]["per_family"])
    per_family = {f: {cell: float(np.mean([per_cell_seed[cell][s]["per_family"][f]
                                           for s in seeds])) for cell in D.CELLS}
                  for f in family_names}
    for f in family_names:
        per_family[f]["interaction"] = float(np.mean(
            [((per_cell_seed["proprio_rollout"][sd]["per_family"][f]
               - per_cell_seed["proprio_one_step"][sd]["per_family"][f])
              - (per_cell_seed["rgb_rollout"][sd]["per_family"][f]
                 - per_cell_seed["rgb_one_step"][sd]["per_family"][f])) for sd in seeds]))

    report = {
        "status": STATUS, "claim_bearing": False,
        "assay": "post-confirmatory H=1 spatial-retention assay",
        "explicitly_not": ["a measurement of spatial retention at H=2-4",
                           "a retrospective repair of the registered occupied co-outcome"],
        "specification_digest": SPEC_DIGEST,
        "probe_package_digest": package["package_digest"],
        "probe_on_true_target_latents": {
            "calibration_split": qualification,
            "selection_split_true_targets": true_metrics},
        "selection_rows": n,
        "primary_equal_family": {cell: stats_of(eq[cell]) for cell in D.CELLS},
        "secondary_corpus_weighted": {cell: stats_of(cw[cell]) for cell in D.CELLS},
        "delta_rgb_equal_family": stats_of(d_rgb),
        "delta_prop_equal_family": stats_of(d_prop),
        "interaction_equal_family": stats_of(inter),
        "per_family_equal_family": per_family,
    }
    report["report_digest"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode()).hexdigest()
    (OUT / "spatial_retention_result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("per_family_equal_family",)}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
