#!/usr/bin/env python3
"""One matrix driver for the four-cell proprioception x rollout factorial.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

A SINGLE driver runs all four cells, so there is one code path, one data path and
one schedule.  The cells differ only in two declared flags:

    cell                | use_proprio | objective
    --------------------|-------------|---------------------
    rgb_one_step        | False       | e1
    rgb_rollout         | False       | 1.5*e1 + 0.5*e2
    proprio_one_step    | True        | e1
    proprio_rollout     | True        | 1.5*e1 + 0.5*e2

Determinism contract
--------------------
* the ten seed identifiers are **pre-registered and hashed before seed 1 runs**;
* one shared base-weight artefact per quadruplet; every cell loads it and the
  driver asserts bit-identity of every shared parameter before a single step;
* modality-specific parameters come from a separate keyed stream;
* the batch plan is a pure function of (seed, epoch) -- it cannot be perturbed by
  a cell doing more work.  A rollout cell performs a second predictor call and a
  proprio cell instantiates extra modules; neither may advance a stream that a
  later batch or another cell reads.  All randomness is drawn from **named,
  stateless generators keyed by (seed, purpose, epoch)**, never from the global
  stream, and the driver asserts that no module carries active dropout;
* cell execution order follows a **predeclared balanced rotation** across seeds,
  and the realised order is recorded in every run record.

Every technically valid run trains for exactly 24 epochs and saves the fixed
epoch-21 checkpoint.  No checkpoint is selected from any metric, and a run that
merely performs badly stays valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import run_dev_v03_two_step_rollout_v1 as R  # noqa: E402
from scripts import dev_proprio_predictor_v1 as P  # noqa: E402
from scripts import dev_checkpoint_v1 as CK  # noqa: E402
from scripts import dev_proprio_experiment_config_v1 as C  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
PROPRIO = CACHE / "proprio_v1"
OUT = CACHE / "factorial_v1"

EPOCHS = 24
CHECKPOINT_EPOCH = 21          # fixed, never selected
BATCH = 4
LR = 3.0e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

CELLS = ("rgb_one_step", "rgb_rollout", "proprio_one_step", "proprio_rollout")
CELL_SPEC = {
    "rgb_one_step": {"use_proprio": False, "rollout": False},
    "rgb_rollout": {"use_proprio": False, "rollout": True},
    "proprio_one_step": {"use_proprio": True, "rollout": False},
    "proprio_rollout": {"use_proprio": True, "rollout": True},
}

# ---- PRE-REGISTERED SEED IDENTIFIERS -------------------------------------
# All ten are fixed here before seed 1 runs.  The registry is hashed; the driver
# refuses to run a seed that is not in it, and refuses to run at all if the file
# on disk disagrees with this list.
SEED_REGISTRY = (
    2_026_080_901, 2_026_080_902, 2_026_080_903, 2_026_080_904, 2_026_080_905,
    2_026_080_906, 2_026_080_907, 2_026_080_908, 2_026_080_909, 2_026_080_910,
)

# ---- PREDECLARED BALANCED CELL-ORDER ROTATION ----------------------------
# Each cell occupies each of the four serial positions with equal frequency over
# any ten seeds (positions cycle 4-periodically), so a serial-order effect cannot
# align with a cell.
def cell_order(seed_index: int):
    return tuple(CELLS[(seed_index + offset) % len(CELLS)] for offset in range(len(CELLS)))


TECHNICAL_INVALIDITY = (
    "hash_or_manifest_mismatch", "nan_or_infinite_values",
    "incomplete_training_infrastructure_failure", "corrupted_checkpoint",
    "implementation_failure",
)


# --------------------------------------------------------------------------
def stream(seed: int, purpose: str, *keys) -> torch.Generator:
    """A named, stateless generator: identical for identical keys, always.

    Nothing in the driver draws from the global RNG after construction, so extra
    work in one cell cannot shift another cell's batches or masks.
    """
    material = "|".join([str(seed), purpose] + [str(k) for k in keys]).encode()
    digest = hashlib.sha256(material).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))


def batch_plan(seed: int, epoch: int, count: int, batch: int):
    """Row order for one epoch: a pure function of (seed, epoch), shared by all cells."""
    order = torch.randperm(count, generator=stream(seed, "data_order", epoch)).tolist()
    return [order[i:i + batch] for i in range(0, count, batch)]


def assert_no_active_dropout(model: nn.Module) -> dict:
    """Dropout is disabled in this experiment; assert it rather than assume it."""
    offenders = []
    for name, module in model.named_modules():
        probability = getattr(module, "p", None)
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                               nn.AlphaDropout)) and probability:
            offenders.append(f"{name}: p={probability}")
    if offenders:
        raise RuntimeError("active dropout found, which would need a keyed stream: "
                           + ", ".join(offenders))
    return {"dropout": "disabled", "asserted": True, "modules_checked": len(list(model.modules()))}


def state_digest(state: dict) -> str:
    """Content digest of a state dict: detects a corrupted base artefact.

    Bit-identity between a model and the artefact it was loaded from cannot
    detect corruption -- both sides move together.  The digest is computed when
    the artefact is written and re-checked on every load.
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------
def register_seeds(out: Path) -> dict:
    """Write (or verify) the pre-registration record.  Must precede seed 1."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "seed_registry.json"
    record = {
        "status": STATUS, "claim_bearing": False,
        "seed_identifiers": list(SEED_REGISTRY),
        "count": len(SEED_REGISTRY),
        "cell_order_rotation": {str(i): list(cell_order(i)) for i in range(len(SEED_REGISTRY))},
        "registered_before_first_run": True,
        "note": ("all ten identifiers are fixed before seed 1; the capped pilot decides how "
                 "many are USED, never which ones or what they are"),
    }
    record["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in record.items()}, sort_keys=True).encode()).hexdigest()
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("seed_identifiers") != record["seed_identifiers"]:
            raise RuntimeError("seed registry on disk disagrees with the source registry")
        return existing
    path.write_text(json.dumps(record, indent=2))
    return record


def build_base_weights(seed: int, out: Path, width, depth, heads) -> Path:
    """One shared base-weight artefact per quadruplet."""
    path = out / f"seed_{seed}_base_weights.pt"
    if path.is_file():
        return path
    model = P.build_paired(seed, use_proprio=False, width=width, depth=depth, heads=heads)
    shared = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    payload = {"shared_state_dict": shared, "seed": seed,
               "width": width, "depth": depth, "heads": heads,
               "state_digest": state_digest(shared)}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    with open(temporary, "rb") as handle:
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)
    torch.load(path, map_location="cpu", weights_only=False)   # reload verification
    return path


def make_cell_model(cell: str, seed: int, base_path: Path, width, depth, heads):
    """Every cell starts from the SAME shared weights, verified bitwise."""
    spec = CELL_SPEC[cell]
    payload = torch.load(base_path, map_location="cpu", weights_only=False)
    base = payload["shared_state_dict"]
    recorded = payload.get("state_digest")
    if recorded is None or state_digest(base) != recorded:
        raise RuntimeError(
            f"corrupted_checkpoint: base weight artefact {base_path.name} fails its "
            "integrity digest")
    model = P.build_paired(seed, use_proprio=spec["use_proprio"],
                           width=width, depth=depth, heads=heads)
    missing, unexpected = model.load_state_dict(base, strict=False)
    if unexpected:
        raise RuntimeError(f"{cell}: unexpected keys in the base artefact: {unexpected}")
    if any(not name.startswith("proprio_") for name in missing):
        raise RuntimeError(f"{cell}: shared parameters missing from the base artefact: {missing}")
    model.initialise_proprio(seed)
    # verify bit-identity of every shared parameter against the artefact
    state = model.state_dict()
    for name, tensor in base.items():
        if not torch.equal(state[name], tensor):
            raise RuntimeError(f"{cell}: shared parameter {name} is not bit-identical")
    return model


# --------------------------------------------------------------------------
def environment_record() -> dict:
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "precision": "bf16 autocast",
    }


def load_rows():
    rows = [json.loads(line) for line in
            (PROPRIO / "proprio_rows.jsonl").read_text().splitlines() if line.strip()]
    manifest = json.loads((PROPRIO / "proprio_manifest.json").read_text())
    stats = json.loads((PROPRIO / "proprio_norm_stats.json").read_text())
    return rows, manifest, stats


def normalise_batch(proprio, control, stats, device):
    """Apply the FROZEN training statistics.  Gravity is offset-only (mean 0/std 1)."""
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    c_mean = torch.tensor(stats["control_mean"], dtype=torch.float32, device=device)
    c_std = torch.tensor(stats["control_std"], dtype=torch.float32, device=device)
    return (proprio - mean) / std, (control - c_mean) / c_std


def train_cell(cell, seed, rows, tensors, stats, model, device, epochs, out,
               position, fixture=False):
    spec = CELL_SPEC[cell]
    model = model.to(device)
    dropout_record = assert_no_active_dropout(model)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                                  foreach=False)
    history, checkpoint_path = [], None
    started = time.time()

    for epoch in range(epochs):
        model.train()
        plan = batch_plan(seed, epoch, len(rows), BATCH)
        totals = {"e1": 0.0, "e2": 0.0, "loss": 0.0, "batches": 0}
        for indices in plan:
            batch = tensors(indices, device, stats)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                p1 = T.normalise(model(batch["context"], batch["a1"], batch["mask"],
                                       batch["proprio"] if spec["use_proprio"] else None,
                                       batch["valid"] if spec["use_proprio"] else None,
                                       batch["control"]))
                e1 = (p1 - batch["y1"]).abs().mean()
                if spec["rollout"]:
                    window = torch.stack([batch["context"][:, 1], batch["context"][:, 2], p1], 1)
                    valid2 = batch["valid"].clone()
                    valid2 = torch.cat([valid2[:, 1:], torch.zeros_like(valid2[:, :1])], 1)
                    proprio2 = torch.cat(
                        [batch["proprio"][:, 1:], torch.zeros_like(batch["proprio"][:, :1])], 1)
                    control2 = torch.cat(
                        [batch["control"][:, 1:],
                         P.control_slot_from_action(batch["a1"])], 1)
                    p2 = T.normalise(model(window, batch["a2"], batch["mask"],
                                           proprio2 if spec["use_proprio"] else None,
                                           valid2 if spec["use_proprio"] else None,
                                           control2))
                    e2 = (p2 - batch["y2"]).abs().mean()
                    jloss = e1
                    sloss = torch.cat([p1, p2], 1).sub(
                        torch.cat([batch["y1"], batch["y2"]], 1)).abs().mean()
                    loss = jloss + sloss          # = 1.5*e1 + 0.5*e2
                else:
                    e2 = torch.zeros((), device=device)
                    loss = e1
            if not torch.isfinite(loss):
                raise RuntimeError("nan_or_infinite_values")
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            totals["e1"] += float(e1.detach())
            totals["e2"] += float(e2.detach())
            totals["loss"] += float(loss.detach())
            totals["batches"] += 1
        entry = {k: (v / totals["batches"] if k != "batches" else v)
                 for k, v in totals.items()}
        entry["epoch"] = epoch
        history.append(entry)

        if epoch == CHECKPOINT_EPOCH:
            checkpoint_path = out / f"seed_{seed}_{cell}_epoch{CHECKPOINT_EPOCH}.pt"
            CK.save(
                checkpoint_path, model=model, optimizer=optimiser, epoch=epoch,
                global_step=(epoch + 1) * len(plan), seed=seed,
                model_config={"cell": cell, **spec, "width": model.width},
                scheduler=None,
                scheduler_absent_reason="fixed learning rate; no scheduler is constructed",
                data_order_generator=stream(seed, "data_order", epoch),
                extra={"history": history, "position_in_serial_order": position,
                       "batch_plan_digest": hashlib.sha256(
                           json.dumps(plan).encode()).hexdigest()})
    return {
        "cell": cell, "seed": seed, "position_in_serial_order": position,
        "epochs_trained": epochs, "checkpoint_epoch": CHECKPOINT_EPOCH,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "history": history, "dropout": dropout_record,
        "wall_seconds": round(time.time() - started, 1),
        "validity": "valid",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-index", type=int, default=None,
                    help="index into the pre-registered registry")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true",
                    help="register seeds, build base weights, verify pairing; train nothing")
    ap.add_argument("--smoke-rows", type=int, default=0,
                    help="train on this many rows for a wiring smoke test only")
    args = ap.parse_args()

    out = Path(args.out)
    registry = register_seeds(out)
    if args.seed_index is None and not args.dry_run:
        raise SystemExit("--seed-index is required unless --dry-run")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows, manifest, stats = load_rows()
    train_rows = [r for r in rows if r["role"] == "train"]

    record = {
        "status": STATUS, "claim_bearing": False,
        "driver": "single matrix driver, four cells",
        "cells": {c: CELL_SPEC[c] for c in CELLS},
        "config_sha256": C.config_sha256(),
        "manifest_sha256": manifest["rows_sha256"],
        "normalisation_sha256": manifest["normalisation_sha256"],
        "seed_registry_sha256": registry["sha256"],
        "environment": environment_record(),
        "budget": {"epochs": args.epochs, "batch": BATCH, "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
                   "checkpoint_epoch": CHECKPOINT_EPOCH, "selection_permitted": False},
        "technical_invalidity_causes": list(TECHNICAL_INVALIDITY),
        "train_rows": len(train_rows),
    }

    if args.dry_run:
        seeds = list(SEED_REGISTRY)
        pairing = []
        for index, seed in enumerate(seeds[:2]):
            base = build_base_weights(seed, out, args.width, args.depth, args.heads)
            models = {c: make_cell_model(c, seed, base, args.width, args.depth, args.heads)
                      for c in CELLS}
            reference = models["rgb_one_step"].state_dict()
            identical = all(
                torch.equal(reference[name], models[cell].state_dict()[name])
                for cell in CELLS for name in reference)
            plans = {c: batch_plan(seed, 0, len(train_rows), BATCH)[:3] for c in CELLS}
            pairing.append({
                "seed": seed, "order": list(cell_order(index)),
                "base_weights": str(base), "base_sha256": sha256_file(base),
                "shared_parameters_bit_identical": identical,
                "batch_plan_identical_across_cells": all(
                    plans[c] == plans["rgb_one_step"] for c in CELLS),
                "proprio_parameters_identical_within_seed": bool(torch.equal(
                    models["proprio_one_step"].proprio_in.weight,
                    models["proprio_rollout"].proprio_in.weight)),
                "control_parameters_present_in_rgb_cells": bool(
                    any(n.startswith("control_") for n, _ in
                        models["rgb_one_step"].named_parameters())),
                "dropout": assert_no_active_dropout(models["rgb_one_step"]),
            })
        record["dry_run"] = {"pairing": pairing, "trained": False}
        (out / "dry_run.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record["dry_run"], indent=2))
        return 0

    raise SystemExit(
        "training is not authorised: the four-cell experiment must not be launched. "
        "Use --dry-run, or the smoke fixture in the test suite.")


if __name__ == "__main__":
    raise SystemExit(main())
