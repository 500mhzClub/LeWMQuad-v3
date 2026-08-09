#!/usr/bin/env python3
"""Minimal real-feature BF16 check across all four cells.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The completed FP32 integration exercise is preserved unchanged; this is a separate,
deliberately short check that the SAME production path behaves under the trainer's
bf16 autocast.  It reuses the fixture, warms up enough to defeat AdaLN-Zero, and
then verifies finiteness, objective separation, the applicable conditioning
gradients, future-slot inertness, checkpoint save/load, and resumed-versus-
uninterrupted continuation.  The long overfit diagnostic is deliberately NOT
repeated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import dev_checkpoint_v1 as CK  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402
from scripts import run_dev_proprio_integration_fixture_v1 as I  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT = D.CACHE / "factorial_v1" / "integration"
WARMUP = 24          # enough to leave the AdaLN-Zero identity regime
TOLERANCE = 5e-3     # bf16 round-trip tolerance for resume comparison


def run_cell(cell, seed, loader, positions, base, device, stats, out) -> dict:
    spec = D.CELL_SPEC[cell]
    model = D.make_cell_model(cell, seed, base, width=384, depth=6, heads=6).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=D.LR,
                                  weight_decay=D.WEIGHT_DECAY, foreach=False)
    batch = loader.batch(positions, device, stats)
    checks = {"precision": "bf16 autocast"}

    for _ in range(WARMUP):
        loss, _, _ = I.step(model, batch, spec, autocast=True)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    model.zero_grad(set_to_none=True)

    loss, e1, e2 = I.step(model, batch, spec, autocast=True)
    optimiser.zero_grad(); loss.backward()
    checks["loss_finite"] = bool(torch.isfinite(loss))
    checks["e1"] = float(e1.detach())
    checks["e2"] = float(e2.detach())
    checks["objective_separation"] = (
        checks["e2"] == 0.0 if not spec["rollout"] else checks["e2"] > 0.0)
    checks["all_gradients_finite"] = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    checks["action_grad_nonzero"] = I.grad_norm(model, "action") > 0
    checks["control_grad_nonzero"] = I.grad_norm(model, "control_") > 0
    if spec["use_proprio"]:
        checks["proprio_grad_nonzero"] = I.grad_norm(model, "proprio_") > 0
    else:
        checks["proprio_path_absent"] = not any(
            n.startswith("proprio_") for n, _ in model.named_parameters())
    model.zero_grad(set_to_none=True)

    if spec["use_proprio"]:
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            valid = torch.tensor([[True, True, False]] * len(positions), device=device)
            reference = model(batch["context"], batch["a1"], batch["mask"],
                              batch["proprio"], valid, batch["control"])
            poisoned = batch["proprio"].clone()
            poisoned[~valid] = float("nan")
            other = model(batch["context"], batch["a1"], batch["mask"],
                          poisoned, valid, batch["control"])
            checks["future_slot_inert"] = bool(torch.equal(reference, other))

    path = out / f"bf16_{cell}.pt"
    CK.save(path, model=model, optimizer=optimiser, epoch=D.CHECKPOINT_EPOCH,
            global_step=WARMUP, seed=seed,
            model_config={"cell": cell, **spec, "width": 384, "precision": "bf16"},
            scheduler=None, scheduler_absent_reason="fixed learning rate",
            data_order_generator=D.stream(seed, "data_order", 0),
            extra={"bf16_check": True})
    checks["checkpoint_saved"] = path.is_file()

    loss_continue, _, _ = I.step(model, batch, spec, autocast=True)
    optimiser.zero_grad(); loss_continue.backward(); optimiser.step()
    uninterrupted = float(I.step(model, batch, spec, autocast=True)[0].detach())

    restored = P.build_paired(seed, use_proprio=spec["use_proprio"],
                              width=384, depth=6, heads=6).to(device)
    restored_optimiser = torch.optim.AdamW(restored.parameters(), lr=D.LR,
                                           weight_decay=D.WEIGHT_DECAY, foreach=False)
    CK.load_for_resume(path, model=restored, optimizer=restored_optimiser,
                       data_order_generator=D.stream(seed, "data_order", 0))
    checks["checkpoint_loaded"] = True
    loss_resumed, _, _ = I.step(restored, batch, spec, autocast=True)
    restored_optimiser.zero_grad(); loss_resumed.backward(); restored_optimiser.step()
    resumed = float(I.step(restored, batch, spec, autocast=True)[0].detach())
    checks["resume_matches_uninterrupted"] = {
        "uninterrupted_next_loss": uninterrupted, "resumed_next_loss": resumed,
        "abs_difference": abs(uninterrupted - resumed),
        "within_tolerance": abs(uninterrupted - resumed) <= TOLERANCE,
        "tolerance": TOLERANCE,
    }
    return {"cell": cell, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-size", type=int, default=12)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    map_record = MAP.load()
    factorial = FM.load()
    rows = [json.loads(line) for line in
            (D.PROPRIO / "proprio_rows.jsonl").read_text().splitlines() if line.strip()]
    stats = json.loads((D.PROPRIO / "proprio_norm_stats.json").read_text())
    device = D.resolve_device()

    loader = D.CanonicalLoader(map_record, rows, stats, split="train",
                               expected_digest=map_record["digest"],
                               factorial=factorial,
                               expected_factorial_digest=factorial["digest"])
    fixture = I.pick_fixture(loader, args.fixture_size)
    seed = D.SEED_REGISTRY[0]
    base = D.build_base_weights(seed, out, 384, 6, 6)

    results = [run_cell(cell, seed, loader, fixture["positions"], base, device, stats, out)
               for cell in D.CELLS]
    record = {
        "status": STATUS, "claim_bearing": False, "scientific": False,
        "precision": "bf16 autocast (the production trainer path)",
        "scope": "training-split fixture only; the long overfit diagnostic is NOT repeated",
        "canonical_map_digest": map_record["digest"],
        "factorial_manifest_digest": factorial["digest"],
        "warmup_steps_discarded": WARMUP,
        "fixture": fixture,
        "cells": results,
    }
    (out / "bf16_check_result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({r["cell"]: r["checks"] for r in results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
