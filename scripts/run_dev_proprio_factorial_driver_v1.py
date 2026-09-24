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
from scripts import build_dev_canonical_cache_map_v1 as MAP  # noqa: E402
from scripts import build_dev_factorial_manifest_v1 as FM  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
PROPRIO = CACHE / "proprio_v1"
OUT = CACHE / "factorial_v1"
EVAL_CACHE = CACHE / "temporal_action_jepa_v1" / "evaluation"
DIAG_CACHE = CACHE / "temporal_action_jepa_v1" / "predicted_token_diagnostic"
TWO_CACHE = CACHE / "two_step"

# ---- FROZEN DEVICE POLICY ------------------------------------------------
# Policy (a): ONE fixed physical GPU for every four-cell quadruplet and for all
# initial five quadruplets.  Device 1 on this host is the integrated Raphael APU
# (1 compute unit) -- not hardware-identical to the R9700 -- so there is no second
# eligible device and seed-to-device rotation is INAPPLICABLE, not merely untested.
DEVICE_POLICY = {
    "policy": "single fixed physical GPU for all quadruplets",
    "device_index": 0,
    "expected_device_name": "AMD Radeon AI PRO R9700",
    "rotation": "inapplicable -- no second hardware-identical device exists on this host",
    "ineligible_devices": {"1": "AMD Radeon Graphics (Raphael integrated APU, 1 CU)"},
    "cells_of_one_quadruplet_share_one_device": True,
    "cell_bound_permanently_to_a_device": False,
}

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

# ---- PREDECLARED CYCLIC LATIN-SQUARE CELL ORDER --------------------------
# CORRECTION.  An earlier note claimed each cell occupies each of the four serial
# positions "equally often" across ten seeds.  That is mathematically impossible:
# ten appearances cannot divide evenly into four positions.  The guarantee this
# schedule actually provides -- and the one that matters -- is that for EVERY
# stopping prefix n = 1..10 each cell's counts across the four positions differ by
# at most one.  ``prefix_balance`` verifies that for all ten prefixes.
#
# The order is a cyclic Latin square of order 4: seed index i runs the cells
# starting at offset i (mod 4).  Each row is a permutation of all four cells, and
# each cell occupies each position exactly once per four consecutive seeds.
def cell_order(seed_index: int):
    return tuple(CELLS[(seed_index + offset) % len(CELLS)] for offset in range(len(CELLS)))


def position_counts(prefix: int):
    """Per-cell counts across the four serial positions for the first ``prefix`` seeds."""
    import collections
    counts = {cell: [0] * len(CELLS) for cell in CELLS}
    for index in range(prefix):
        for position, cell in enumerate(cell_order(index)):
            counts[cell][position] += 1
    return counts


def prefix_balance(maximum: int = 10) -> dict:
    """The balance guarantee, checked for every possible stopping point."""
    table = {}
    for prefix in range(1, maximum + 1):
        counts = position_counts(prefix)
        spreads = {cell: max(values) - min(values) for cell, values in counts.items()}
        table[str(prefix)] = {
            "counts": counts,
            "max_minus_min_per_cell": spreads,
            "balanced_within_one": all(spread <= 1 for spread in spreads.values()),
        }
    return table


TECHNICAL_INVALIDITY = (
    "hash_or_manifest_mismatch", "canonical_map_digest_mismatch", "nan_or_infinite_values",
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
        "cell_order_schedule": {str(i): list(cell_order(i)) for i in range(len(SEED_REGISTRY))},
        "cell_order_type": "cyclic Latin square of order 4",
        "prefix_balance": prefix_balance(len(SEED_REGISTRY)),
        "balance_guarantee": ("for every stopping prefix n = 1..10, each cell's counts across "
                              "the four serial positions differ by at most one; exact equality "
                              "is impossible for n not a multiple of 4 and is NOT claimed"),
        "prefix_rule": ("the interim sample size selects a PREFIX of this frozen order; later "
                        "seeds are never reordered after n is calculated"),
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
        if existing.get("sha256") != record["sha256"]:
            # The identifiers match but the frozen schedule content differs.  Before
            # any seed has run this is a legitimate pre-launch correction; after one
            # has, it is tampering.  Refuse if any run artefact exists.
            ran = sorted(out.glob("seed_*_*_epoch*.pt"))
            if ran:
                raise RuntimeError(
                    "seed registry content changed after runs exist: " f"{[p.name for p in ran[:3]]}")
            record["superseded_registry_sha256"] = existing.get("sha256")
            record["regenerated_before_first_run"] = True
            path.write_text(json.dumps(record, indent=2))
            return record
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
def resolve_device() -> torch.device:
    """Pin the policy device and refuse anything else."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    index = DEVICE_POLICY["device_index"]
    name = torch.cuda.get_device_name(index)
    if name != DEVICE_POLICY["expected_device_name"]:
        raise RuntimeError(
            f"device policy violation: device {index} is {name!r}, expected "
            f"{DEVICE_POLICY['expected_device_name']!r}")
    return torch.device(f"cuda:{index}")


def environment_record() -> dict:
    record = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "precision": "bf16 autocast",
        "device_policy": DEVICE_POLICY,
        "determinism": {
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
            "global_rng_used_after_construction": False,
        },
    }
    if torch.cuda.is_available():
        index = DEVICE_POLICY["device_index"]
        properties = torch.cuda.get_device_properties(index)
        record.update({
            "device_index": index,
            "device_name": torch.cuda.get_device_name(index),
            "device_capability": f"{properties.major}.{properties.minor}",
            "device_memory_gib": round(properties.total_memory / 2**30, 1),
            "hip_version": getattr(torch.version, "hip", None),
            "cuda_version": torch.version.cuda,
        })
    return record


class CanonicalLoader:
    """The ONE cache path.  Every tensor is fetched through the canonical map.

    A filtered manifest position is never used as a cache position: the map's
    ``cache_index`` (base rows) and ``step2_cache_index`` (two-step rows) are the
    only indices this class will accept.
    """

    def __init__(self, map_record, rows, stats, split="train", expected_digest=None,
                 factorial=None, expected_factorial_digest=None):
        if expected_digest and map_record["digest"] != expected_digest:
            raise RuntimeError("canonical_map_digest_mismatch")
        # Both cells iterate the ONE ordered factorial artefact directly; neither
        # re-derives a row set from a filter.
        factorial = factorial if factorial is not None else FM.load()
        if expected_factorial_digest and factorial["digest"] != expected_factorial_digest:
            raise RuntimeError("factorial_manifest_digest_mismatch")
        if factorial["canonical_cache_map_digest"] != map_record["digest"]:
            raise RuntimeError("factorial manifest was built against a different cache map")
        self.digest = map_record["digest"]
        self.factorial_digest = factorial["digest"]
        self.stats = stats
        self.split = split
        self.entries = [row for row in factorial["rows"] if row["split"] == split]
        by_index = {r_index: row for r_index, row in enumerate(rows)}
        self.rows = [by_index[e["manifest_row_index"]] for e in self.entries]
        n_train = map_record["source_train"]
        n_sel = map_record["source_selection"]
        if split == "train":
            self.ctx0 = R.load_cache(DIAG_CACHE / "frozen_train_ctx0.f16", n_train)
            self.ctx1 = R.load_cache(DIAG_CACHE / "frozen_train_ctx1.f16", n_train)
            self.ctx2 = R.load_cache(EVAL_CACHE / "frozen_current.f16", n_train + n_sel)[:n_train]
            self.y1 = R.load_cache(EVAL_CACHE / "frozen_train_future.f16", n_train)
            self.y2 = R.load_cache(TWO_CACHE / "frozen_train_step2.f16",
                                   self._blob_rows(TWO_CACHE / "frozen_train_step2.f16"))
        else:
            self.ctx0 = R.load_cache(EVAL_CACHE / "frozen_ctx0.f16", n_sel)
            self.ctx1 = R.load_cache(EVAL_CACHE / "frozen_ctx1.f16", n_sel)
            self.ctx2 = R.load_cache(EVAL_CACHE / "frozen_current.f16", n_train + n_sel)[n_train:]
            self.y1 = R.load_cache(EVAL_CACHE / "frozen_sel_future.f16", n_sel)
            self.y2 = R.load_cache(TWO_CACHE / "frozen_sel_step2.f16",
                                   self._blob_rows(TWO_CACHE / "frozen_sel_step2.f16"))

    @staticmethod
    def _blob_rows(path: Path) -> int:
        return path.stat().st_size // (P.TOKENS * P.TOKEN_DIM * 2)

    def __len__(self):
        return len(self.entries)

    def batch(self, positions, device, stats=None):
        stats = stats or self.stats
        cache = [self.entries[i]["cache_index"] for i in positions]
        step2 = [self.entries[i]["step2_cache_index"] for i in positions]
        rows = [self.rows[i] for i in positions]

        context = torch.stack([
            T.normalise(self.ctx0[cache].float()),
            T.normalise(self.ctx1[cache].float()),
            T.normalise(self.ctx2[cache].float()),
        ], dim=1).to(device)
        y1 = T.normalise(self.y1[cache].float()).to(device)
        y2 = T.normalise(self.y2[step2].float()).to(device)

        a1 = torch.tensor([r["action_blocks"][0] for r in rows], dtype=torch.float32,
                          device=device)
        a2 = torch.tensor([r["action_blocks"][min(1, len(r["action_blocks"]) - 1)]
                           for r in rows], dtype=torch.float32, device=device)
        proprio = torch.tensor([r["proprio"] for r in rows], dtype=torch.float32,
                               device=device).reshape(len(rows), 3, P.SAMPLES_PER_SLOT,
                                                      P.PROPRIO_DIM)
        control = torch.tensor([r["control"] for r in rows], dtype=torch.float32,
                               device=device).reshape(len(rows), 3, P.SAMPLES_PER_SLOT,
                                                      P.CONTROL_DIM)
        proprio, control = normalise_batch(proprio, control, stats, device)
        return {
            "context": context, "y1": y1, "y2": y2, "a1": a1, "a2": a2,
            "proprio": proprio, "control": control,
            "valid": torch.ones(len(rows), 3, dtype=torch.bool, device=device),
            "mask": torch.ones(len(rows), P.TOKENS, dtype=torch.bool, device=device),
            "cache_index": cache, "step2_cache_index": step2,
            "stable_row_id": [self.entries[i]["stable_row_id"] for i in positions],
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


def terminal_window(history, key="loss", start=19, end=23) -> dict:
    """Diagnostics only.  Never used to select a checkpoint or exclude a run."""
    window = [e[key] for e in history if start <= e["epoch"] <= end]
    late = [(e["epoch"], e[key]) for e in history if 14 <= e["epoch"] <= 23]
    record = {"epochs": [start, end], "used_for_selection": False,
              "used_for_exclusion": False}
    if window:
        record["mean"] = float(np.mean(window))
        record["sd"] = float(np.std(window, ddof=0))
    if len(late) >= 2:
        x = np.array([e for e, _ in late], dtype=float)
        y = np.array([v for _, v in late], dtype=float)
        record["slope"] = float(np.polyfit(x, y, 1)[0])
    return record


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
        "terminal_window": terminal_window(history),
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
    ap.add_argument("--authorisation", default=None,
                    help="path to a scoped launch-authorisation receipt")
    args = ap.parse_args()

    out = Path(args.out)
    registry = register_seeds(out)
    if args.seed_index is None and not args.dry_run:
        raise SystemExit("--seed-index is required unless --dry-run")

    device = resolve_device()
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

    # ---- scientific launch requires a verified scoped authorisation ---------
    if not args.authorisation:
        raise SystemExit(
            "training is not authorised: the four-cell experiment must not be launched "
            "without a scoped launch-authorisation receipt (--authorisation).")
    from scripts import authorise_dev_proprio_launch_v1 as AUTH
    receipt = AUTH.verify(args.seed_index, Path(args.authorisation))

    seed = SEED_REGISTRY[args.seed_index]
    order = cell_order(args.seed_index)
    seed_out = out / f"seed_{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)

    factorial = FM.load()
    map_record = MAP.load()
    loader = CanonicalLoader(map_record, rows, stats, split="train",
                             expected_digest=map_record["digest"], factorial=factorial,
                             expected_factorial_digest=factorial["digest"])

    base = build_base_weights(seed, seed_out, args.width, args.depth, args.heads)
    models = {cell: make_cell_model(cell, seed, base, args.width, args.depth, args.heads)
              for cell in CELLS}
    reference = models["rgb_one_step"].state_dict()
    identical = all(torch.equal(reference[name], models[cell].state_dict()[name])
                    for cell in CELLS for name in reference)
    if not identical:
        raise RuntimeError("shared parameters are not bit-identical across cells")
    plans = {cell: batch_plan(seed, 0, len(loader), BATCH)[:3] for cell in CELLS}
    if any(plans[cell] != plans["rgb_one_step"] for cell in CELLS):
        raise RuntimeError("batch plans differ across cells")

    record.update({
        "stage": "initial five-seed scientific stage",
        "authorisation_receipt_digest": receipt["receipt_digest"],
        "factorial_manifest_digest": factorial["digest"],
        "canonical_map_digest": map_record["digest"],
        "seed": seed, "seed_index": args.seed_index,
        "execution_order": list(order),
        "base_weights": str(base), "base_weights_sha256": sha256_file(base),
        "shared_parameters_bit_identical": identical,
        "batch_plan_identical_across_cells": True,
        "rng_plan": "named stateless streams keyed by (seed, purpose, epoch)",
        "augmentation": "none in this experiment",
        "train_rows": len(loader),
        "cells_run": [],
    })

    def tensors(indices, dev, st):
        return loader.batch(indices, dev, st)

    for position, cell in enumerate(order):
        print(f"[seed {seed}] cell {position + 1}/4: {cell}", flush=True)
        result = train_cell(cell, seed, loader.rows, tensors, stats, models[cell],
                            device, args.epochs, seed_out, position)
        record["cells_run"].append(result)
        (seed_out / "run_record.json").write_text(json.dumps(record, indent=2))
        del models[cell]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    record["completed"] = True
    record["wall_seconds_total"] = sum(c["wall_seconds"] for c in record["cells_run"])
    (seed_out / "run_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"seed": seed, "cells": [c["cell"] for c in record["cells_run"]],
                      "epochs": [c["epochs_trained"] for c in record["cells_run"]],
                      "validity": [c["validity"] for c in record["cells_run"]],
                      "wall_seconds": record["wall_seconds_total"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
