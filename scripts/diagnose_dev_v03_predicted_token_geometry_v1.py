#!/usr/bin/env python3
"""Localisation diagnostic: is predicted-token geometry absent, or just rebased?

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only with respect to training: no arm
is retrained, no checkpoint is modified.

The completed evaluation showed that predicted future tokens decode to *worse*
occupancy than persistence, in both arms, under a probe trained on **true future
encoder tokens** and applied unchanged.  That probe remains the operational
acceptance test.

This diagnostic asks a different, narrower question: if a probe is allowed to fit
the predictor's own output distribution, does the geometry come back?  Four
identical-capacity probes, each trained on train tokens of one kind and evaluated
only on the matching checkpoint_selection tokens:

    1. frozen-arm predicted        3. frozen-arm persistence
    2. moving-arm predicted        4. moving-arm persistence

Reading:
  * still poor            -> geometry is genuinely discarded by the predictor
  * approaches true-future -> geometry is present but prediction leaves the
                              encoder's canonical feature basis
  * frozen ok, moving not  -> encoder movement destabilises predictor/target
                              compatibility

These fresh predicted-token probes are DIAGNOSTIC ONLY.  They must not replace
the fixed true-future probe as the acceptance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_frozen_dense_representation_screen_v1 as S  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import eval_dev_v03_temporal_action_jepa_v1 as V  # noqa: E402
from scripts import complete_dev_v03_temporal_action_jepa_evaluation_v1 as C  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
COMPLETION = CACHE / "temporal_action_jepa_v1" / "completion"
OUT = CACHE / "temporal_action_jepa_v1" / "predicted_token_diagnostic"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
EPOCH = 5


@torch.no_grad()
def encode_train_context(module, rows, offset_index, arm, device, cache: Path, batch=16):
    """Context frame `offset_index` (0 = t-480, 1 = t-240) for every train row."""
    shape = (len(rows), arm.token_grid[0] * arm.token_grid[1], arm.token_dim)
    if cache.is_file() and cache.stat().st_size == int(np.prod(shape) * 2):
        return torch.from_numpy(
            np.ascontiguousarray(np.memmap(cache, dtype=np.float16, mode="r", shape=shape))
        )
    memory = np.memmap(cache, dtype=np.float16, mode="w+", shape=shape)
    paths = [r["context_paths"][offset_index] for r in rows]
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        pixels = torch.stack([arm.preprocess(p) for p in chunk]).to(device, torch.float32)
        memory[start : start + len(chunk)] = module(pixels.unsqueeze(2)).half().cpu().numpy()
    memory.flush()
    return torch.from_numpy(
        np.ascontiguousarray(np.memmap(cache, dtype=np.float16, mode="r", shape=shape))
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows, sel_rows = T.load_rows()
    ordered = train_rows + sel_rows
    n_train, n_sel = len(train_rows), len(sel_rows)
    arm = E.VJepa21CroppedV03Arm()
    grid, dim = arm.token_grid, arm.token_dim
    tokens = grid[0] * grid[1]

    labels_future = np.concatenate(
        [C.future_labels(train_rows), C.future_labels(sel_rows)], 0
    )
    tr, se = np.arange(n_train), np.arange(n_train, n_train + n_sel)
    families = [r["family"] for r in sel_rows]

    record = {
        "status": STATUS,
        "claim_bearing": False,
        "read_only_with_respect_to_training": True,
        "diagnostic_only": (
            "these fresh predicted-token probes localise where the geometry goes; "
            "they do NOT replace the fixed true-future probe as the acceptance test"
        ),
        "acceptance_probe_provenance_confirmed": {
            "trained_on": "normalised FROZEN true-future encoder tokens, train role",
            "labels": "raster_labels of the t+240 endpoint",
            "applied_unchanged_to": ["true future", "persistence", "predicted"],
            "checkpoint_selected_on": "true-future checkpoint_selection tokens",
            "verified_in": "complete_dev_v03_temporal_action_jepa_evaluation_v1.py",
        },
        "true_future_reference": json.loads(
            (COMPLETION / "result.json").read_text()
        )["arms"]["frozen"]["spatial_predicted_future_tokens"]["true_future_upper_reference"],
        "probes": {},
    }

    for name in ("frozen", "moving"):
        checkpoint = torch.load(
            CACHE / "temporal_action_jepa_v1" / f"arm_{name}" / f"checkpoint_epoch{EPOCH}.pt",
            map_location="cpu",
        )
        predictor = T.Predictor().to(device)
        predictor.load_state_dict(checkpoint["predictor"])
        predictor.eval()
        module, _ = V.build_arm_encoder(arm, checkpoint, device)

        current = C.load_cache(f"{name}_current.f16", len(ordered), arm)
        ctx_train = [
            encode_train_context(module, train_rows, k, arm, device,
                                 OUT / f"{name}_train_ctx{k}.f16")
            for k in (0, 1)
        ]
        ctx_train.append(current[:n_train])
        ctx_sel = [C.load_cache(f"{name}_ctx{k}.f16", n_sel, arm) for k in range(3)]
        del module
        torch.cuda.empty_cache()

        # predicted tokens, correct action, for train and selection
        predicted = {}
        for tag, rows, ctx in (("train", train_rows, ctx_train), ("selection", sel_rows, ctx_sel)):
            blob = OUT / f"{name}_predicted_{tag}.f16"
            shape = (len(rows), tokens, dim)
            if blob.is_file() and blob.stat().st_size == int(np.prod(shape) * 2):
                predicted[tag] = torch.from_numpy(
                    np.ascontiguousarray(np.memmap(blob, dtype=np.float16, mode="r", shape=shape))
                )
                continue
            actions = T.action_tensor([r["primitive"] for r in rows], torch.device("cpu"))
            memory = np.memmap(blob, dtype=np.float16, mode="w+", shape=shape)
            for start in range(0, len(rows), args.batch):
                stop = min(start + args.batch, len(rows))
                block = torch.stack(
                    [c[start:stop].float() for c in ctx], dim=1
                )
                with torch.no_grad():
                    out = predictor(
                        T.normalise(block).to(device),
                        actions[start:stop].to(device),
                        torch.ones(stop - start, tokens, dtype=torch.bool, device=device),
                    )
                memory[start:stop] = out.half().cpu().numpy()
            memory.flush()
            predicted[tag] = torch.from_numpy(
                np.ascontiguousarray(np.memmap(blob, dtype=np.float16, mode="r", shape=shape))
            )

        persistence = torch.cat(
            [T.normalise(current[i : min(i + 256, len(ordered))].float()).half()
             for i in range(0, len(ordered), 256)], 0
        )

        for kind, stacked in (
            ("predicted", torch.cat([predicted["train"], predicted["selection"]], 0)),
            ("persistence", persistence),
        ):
            gpu = stacked.to(device)
            probe, _, epoch = S.train_probe(
                gpu, labels_future, tr, se, grid, dim, device, f"{name}/{kind}"
            )
            prediction = S.predict(probe, gpu, se, grid, device)
            summary = S.summarise(prediction, labels_future[se])
            per_family = S.grouped(prediction, labels_future[se], families)
            record["probes"][f"{name}_{kind}"] = {
                "trained_on": f"{name}-arm {kind} TRAIN tokens",
                "evaluated_on": f"{name}-arm {kind} checkpoint_selection tokens",
                "selected_epoch": epoch,
                "observable_occupied_precision": summary["observable_occupied_precision"],
                "observable_occupied_recall": summary["observable_occupied_recall"],
                "observable_occupied_iou": summary["observable_occupied_iou"],
                "predicted_occupied_fraction_all_cells":
                    summary["predicted_class_fraction_over_all_cells"]["occupied"],
                "target_occupied_fraction_all_cells":
                    summary["target_class_fraction_over_all_cells"]["occupied"],
                "open_obstacle_field": per_family.get("open_obstacle_field"),
                "per_family": per_family,
            }
            del gpu, probe
            torch.cuda.empty_cache()
            (OUT / "result.json").write_text(json.dumps(record, indent=2))

        del predictor, current, ctx_train, ctx_sel, predicted, persistence
        torch.cuda.empty_cache()

    reference = record["true_future_reference"]["observable_occupied_iou"]
    p = record["probes"]
    record["reading"] = {
        "true_future_reference_iou": reference,
        "fresh_predicted_iou": {k: p[f"{k}_predicted"]["observable_occupied_iou"]
                                for k in ("frozen", "moving")},
        "fresh_persistence_iou": {k: p[f"{k}_persistence"]["observable_occupied_iou"]
                                  for k in ("frozen", "moving")},
        "predicted_gap_to_reference": {
            k: reference - p[f"{k}_predicted"]["observable_occupied_iou"]
            for k in ("frozen", "moving")
        },
        "predicted_minus_persistence": {
            k: p[f"{k}_predicted"]["observable_occupied_iou"]
               - p[f"{k}_persistence"]["observable_occupied_iou"]
            for k in ("frozen", "moving")
        },
    }
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record["reading"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
