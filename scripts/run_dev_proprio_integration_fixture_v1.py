#!/usr/bin/env python3
"""Non-scientific integration exercise on real cached features.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

This runs all four cells through the ACTUAL production code path -- canonical map
lookup, action and control construction, proprioceptive construction and masking,
one-step and two-step objectives, forward, backward, optimiser step, checkpoint
save, checkpoint resume, evaluator load -- on a **small fixed subset of the
TRAINING split only**.

It never reads a selection row, never writes a selection metric, and never
produces a scientific outcome.  Its job is to prove the wiring, nothing else.
The overfit check at the end is an engineering diagnostic and must not be quoted
as performance.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import dev_checkpoint_v1 as CK  # noqa: E402
from scripts import run_dev_proprio_factorial_driver_v1 as D  # noqa: E402
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import dev_action_slew_reconstruction_v1 as SLEW  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
OUT = D.CACHE / "factorial_v1" / "integration"
WARMUP = 40          # discarded steps: AdaLN-Zero starts every block as identity


def pick_fixture(loader, size=16) -> dict:
    """A small training-only fixture covering the awkward cases."""
    chosen, reasons = [], {}

    def take(position, reason):
        if position not in chosen:
            chosen.append(position)
            reasons.setdefault(reason, position)

    families = {}
    sign_reversal = non_steady = near_reset = joint_signal = None
    for position, row in enumerate(loader.rows):
        block = row["action_blocks"][0]
        ticks = [block[i * 2:(i + 1) * 2] for i in range(SLEW.TICKS)]
        # sign reversal: a tick opposing the block's settled direction
        settled = ticks[-1]
        if sign_reversal is None and any(
                t[c] * settled[c] < -1e-9 for t in ticks for c in range(2)):
            sign_reversal = position
        control = row["control"]
        if non_steady is None and len({tuple(c) for c in control}) > 2:
            non_steady = position
        if near_reset is None and row["proprio_steps"][0] <= 20:
            near_reset = position
        sample = row["proprio"][-1]
        if joint_signal is None and max(abs(v) for v in sample[6:30]) > 1.0:
            joint_signal = position
        families.setdefault(row["family"], position)

    for reason, position in (("sign_reversal_action_block", sign_reversal),
                             ("non_steady_control_history", non_steady),
                             ("valid_row_close_to_a_reset", near_reset),
                             ("nontrivial_joint_and_gyro", joint_signal)):
        if position is not None:
            take(position, reason)
    for family, position in sorted(families.items())[:6]:
        take(position, f"family:{family}")
    for position in range(len(loader)):
        if len(chosen) >= size:
            break
        take(position, "padding")

    final = chosen[:size]
    return {"positions": final,
            "selection_reasons": {reason: position for reason, position in reasons.items()},
            "coverage": describe_coverage(loader, final),
            "families": sorted({loader.rows[p]["family"] for p in final}),
            "selection_rows_touched": 0}


def describe_coverage(loader, positions) -> dict:
    """Recompute every requirement over the FINAL deduplicated fixture, and assert it.

    The selection loop records a reason only the first time it adds a row, so a row
    satisfying two requirements suppressed the second reason and made the coverage
    report look incomplete.  Coverage is therefore measured here, on the rows that
    were actually chosen, independently of how they came to be chosen.
    """
    rows = [loader.rows[p] for p in positions]
    families = sorted({r["family"] for r in rows})

    def has_sign_reversal(row):
        block = row["action_blocks"][0]
        ticks = [block[i * 2:(i + 1) * 2] for i in range(SLEW.TICKS)]
        settled = ticks[-1]
        return any(t[c] * settled[c] < -1e-9 for t in ticks for c in range(2))

    coverage = {
        "families": families,
        "family_count": len(families),
        "more_than_one_family": len(families) > 1,
        "rows_with_sign_reversal_action_block": sum(1 for r in rows if has_sign_reversal(r)),
        "rows_with_non_steady_control_history": sum(
            1 for r in rows if len({tuple(c) for c in r["control"]}) > 2),
        "max_distinct_control_values_in_a_row": max(
            len({tuple(c) for c in r["control"]}) for r in rows),
        "earliest_proprio_start_step": min(r["proprio_steps"][0] for r in rows),
        "rows_close_to_a_reset_without_crossing": sum(
            1 for r in rows if r["proprio_steps"][0] <= 20),
        "max_abs_joint_position": max(max(abs(v) for v in s[6:18]) for r in rows
                                      for s in r["proprio"]),
        "max_abs_joint_velocity": max(max(abs(v) for v in s[18:30]) for r in rows
                                      for s in r["proprio"]),
        "max_abs_gyro": max(max(abs(v) for v in s[3:6]) for r in rows for s in r["proprio"]),
    }
    coverage["requirements_met"] = {
        "more_than_one_family": coverage["more_than_one_family"],
        "sign_reversal_action_block": coverage["rows_with_sign_reversal_action_block"] > 0,
        "valid_row_close_to_a_reset": coverage["rows_close_to_a_reset_without_crossing"] > 0,
        "non_steady_control_history": coverage["rows_with_non_steady_control_history"] > 0,
        "nontrivial_joint_and_gyro": (coverage["max_abs_joint_velocity"] > 1.0
                                      and coverage["max_abs_gyro"] > 0.5),
    }
    unmet = [name for name, ok in coverage["requirements_met"].items() if not ok]
    if unmet:
        raise AssertionError(f"fixture does not cover: {unmet}")
    return coverage


def step(model, batch, spec, autocast=False):
    """The production forward/loss.  ``autocast`` mirrors the trainer's bf16 path."""
    context = torch.autocast(device_type=batch["context"].device.type,
                             dtype=torch.bfloat16,
                             enabled=autocast and batch["context"].device.type == "cuda")
    with context:
        return _step_inner(model, batch, spec)


def _step_inner(model, batch, spec):
    p1 = T.normalise(model(batch["context"], batch["a1"], batch["mask"],
                           batch["proprio"] if spec["use_proprio"] else None,
                           batch["valid"] if spec["use_proprio"] else None,
                           batch["control"]))
    e1 = (p1 - batch["y1"]).abs().mean()
    e2 = torch.zeros((), device=p1.device)
    if spec["rollout"]:
        window = torch.stack([batch["context"][:, 1], batch["context"][:, 2], p1], 1)
        valid2 = torch.cat([batch["valid"][:, 1:],
                            torch.zeros_like(batch["valid"][:, :1])], 1)
        proprio2 = torch.cat([batch["proprio"][:, 1:],
                              torch.zeros_like(batch["proprio"][:, :1])], 1)
        control2 = torch.cat([batch["control"][:, 1:],
                              P.control_slot_from_action(batch["a1"])], 1)
        p2 = T.normalise(model(window, batch["a2"], batch["mask"],
                               proprio2 if spec["use_proprio"] else None,
                               valid2 if spec["use_proprio"] else None, control2))
        e2 = (p2 - batch["y2"]).abs().mean()
        jloss = e1
        sloss = torch.cat([p1, p2], 1).sub(
            torch.cat([batch["y1"], batch["y2"]], 1)).abs().mean()
        loss = jloss + sloss
    else:
        loss = e1
    return loss, e1, e2


def grad_norm(model, prefix) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            total += float(parameter.grad.abs().sum())
    return total


def run_cell(cell, seed, loader, fixture, base_path, device, stats, out) -> dict:
    spec = D.CELL_SPEC[cell]
    model = D.make_cell_model(cell, seed, base_path, width=384, depth=6, heads=6).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=D.LR,
                                  weight_decay=D.WEIGHT_DECAY, foreach=False)
    batch = loader.batch(fixture["positions"], device, stats)
    checks = {}

    # ---- discarded warm-up: AdaLN-Zero makes every path inert at step 0 ------
    for _ in range(WARMUP):
        loss, _, _ = step(model, batch, spec)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
    model.zero_grad(set_to_none=True)

    # ---- the audited step ----------------------------------------------------
    loss, e1, e2 = step(model, batch, spec)
    optimiser.zero_grad()
    loss.backward()
    checks["loss_finite"] = bool(torch.isfinite(loss))
    checks["e1"] = float(e1.detach())
    checks["e2"] = float(e2.detach())
    checks["one_step_receives_no_second_step_loss"] = (
        checks["e2"] == 0.0 if not spec["rollout"] else None)
    checks["all_gradients_finite"] = all(
        bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    checks["action_grad"] = grad_norm(model, "action")
    checks["control_grad"] = grad_norm(model, "control_")
    checks["proprio_grad"] = grad_norm(model, "proprio_")
    checks["action_grad_nonzero"] = checks["action_grad"] > 0
    checks["control_grad_nonzero"] = checks["control_grad"] > 0
    if spec["use_proprio"]:
        checks["proprio_grad_nonzero"] = checks["proprio_grad"] > 0
        checks["proprio_path_present"] = True
    else:
        checks["proprio_path_absent"] = not any(
            n.startswith("proprio_") for n, _ in model.named_parameters())
        checks["proprio_grad_nonzero"] = None

    # second-step gradient must be real for a rollout cell
    if spec["rollout"]:
        model.zero_grad(set_to_none=True)
        _, _, e2_only = step(model, batch, spec)
        e2_only.backward()
        total = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        checks["second_step_gradient"] = total
        checks["second_step_gradient_finite_and_nonzero"] = (
            total > 0 and all(bool(torch.isfinite(p.grad).all())
                              for p in model.parameters() if p.grad is not None))
    model.zero_grad(set_to_none=True)

    # ---- invalid proprio slots must stay exactly inert ------------------------
    if spec["use_proprio"]:
        with torch.no_grad():
            valid = torch.tensor([[True, True, False]] * len(fixture["positions"]),
                                 device=device)
            reference = model(batch["context"], batch["a1"], batch["mask"],
                              batch["proprio"], valid, batch["control"])
            poisoned = batch["proprio"].clone()
            poisoned[~valid] = float("nan")
            other = model(batch["context"], batch["a1"], batch["mask"],
                          poisoned, valid, batch["control"])
            checks["invalid_proprio_slots_exactly_inert"] = bool(torch.equal(reference, other))

    # ---- rollout control slots come from the planned action, not a log --------
    derived = P.control_slot_from_action(batch["a1"])
    checks["rollout_control_from_planned_action"] = bool(
        torch.equal(derived.reshape(len(fixture["positions"]), P.ACTION_DIM), batch["a1"]))

    # ---- determinism: repeat from the same state and keys ---------------------
    repeat_model = D.make_cell_model(cell, seed, base_path, width=384, depth=6,
                                     heads=6).to(device)
    repeat_optimiser = torch.optim.AdamW(repeat_model.parameters(), lr=D.LR,
                                         weight_decay=D.WEIGHT_DECAY, foreach=False)
    first_losses, second_losses = [], []
    for _ in range(3):
        loss_a, _, _ = step(model, batch, spec)
        first_losses.append(float(loss_a.detach()))
        optimiser.zero_grad(); loss_a.backward(); optimiser.step()
    for _ in range(WARMUP):
        loss_b, _, _ = step(repeat_model, batch, spec)
        repeat_optimiser.zero_grad(); loss_b.backward(); repeat_optimiser.step()
    for _ in range(3):
        loss_b, _, _ = step(repeat_model, batch, spec)
        second_losses.append(float(loss_b.detach()))
        repeat_optimiser.zero_grad(); loss_b.backward(); repeat_optimiser.step()
    checks["repeatable_losses"] = {
        "first": first_losses, "repeat": second_losses,
        "max_abs_difference": max(abs(a - b) for a, b in zip(first_losses, second_losses)),
    }

    # ---- checkpoint save, resume, and identical next update -------------------
    path = out / f"integration_{cell}.pt"
    CK.save(path, model=model, optimizer=optimiser, epoch=D.CHECKPOINT_EPOCH,
            global_step=WARMUP + 3, seed=seed,
            model_config={"cell": cell, **spec, "width": 384},
            scheduler=None, scheduler_absent_reason="fixed learning rate",
            data_order_generator=D.stream(seed, "data_order", 0),
            extra={"integration_fixture": True})
    uninterrupted = copy.deepcopy(model), copy.deepcopy(optimiser.state_dict())
    loss_continue, _, _ = step(model, batch, spec)
    optimiser.zero_grad(); loss_continue.backward(); optimiser.step()
    after_uninterrupted = float(step(model, batch, spec)[0].detach())

    restored = P.build_paired(seed, use_proprio=spec["use_proprio"],
                              width=384, depth=6, heads=6).to(device)
    restored_optimiser = torch.optim.AdamW(restored.parameters(), lr=D.LR,
                                           weight_decay=D.WEIGHT_DECAY, foreach=False)
    CK.load_for_resume(path, model=restored, optimizer=restored_optimiser,
                       data_order_generator=D.stream(seed, "data_order", 0))
    loss_resumed, _, _ = step(restored, batch, spec)
    restored_optimiser.zero_grad(); loss_resumed.backward(); restored_optimiser.step()
    after_resumed = float(step(restored, batch, spec)[0].detach())
    checks["resume_matches_uninterrupted"] = {
        "uninterrupted_next_loss": after_uninterrupted,
        "resumed_next_loss": after_resumed,
        "abs_difference": abs(after_uninterrupted - after_resumed),
    }
    del uninterrupted

    # ---- engineering-only overfit diagnostic ---------------------------------
    diag_model = D.make_cell_model(cell, seed, base_path, width=384, depth=6,
                                   heads=6).to(device)
    diag_optimiser = torch.optim.AdamW(diag_model.parameters(), lr=1e-3, foreach=False)
    first = last = None
    for index in range(120):
        loss_d, _, _ = step(diag_model, batch, spec)
        diag_optimiser.zero_grad(); loss_d.backward(); diag_optimiser.step()
        if index == 0:
            first = float(loss_d.detach())
        last = float(loss_d.detach())
    checks["overfit_diagnostic"] = {
        "first_loss": first, "last_loss": last, "reduced": last < first,
        "scope": "ENGINEERING DIAGNOSTIC ONLY -- not scientific performance",
    }
    return {"cell": cell, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-size", type=int, default=16)
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
    fixture = pick_fixture(loader, args.fixture_size)
    seed = D.SEED_REGISTRY[0]
    base = D.build_base_weights(seed, out, 384, 6, 6)

    results = [run_cell(cell, seed, loader, fixture, base, device, stats, out)
               for cell in D.CELLS]

    record = {
        "status": STATUS, "claim_bearing": False, "scientific": False,
        "scope": "training-split fixture only; no selection row read, no selection metric written",
        "canonical_map_digest": map_record["digest"],
        "device": D.environment_record(),
        "warmup_steps_discarded": WARMUP,
        "fixture": fixture,
        "fixture_rows": len(fixture["positions"]),
        "train_rows_available": len(loader),
        "cells": results,
    }
    (out / "integration_result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"fixture": fixture["coverage"], "families": fixture["families"],
                      "cells": {r["cell"]: r["checks"] for r in results}}, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
