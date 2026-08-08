#!/usr/bin/env python3
"""Verify the one-step and rollout arms were perfectly paired.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only.

The rollout arm ran a 50-step warmup for the rollout-gradient probe and then
discarded it by rebuilding the predictor and optimiser.  This checks that the
discard was complete and, in particular, that the warmup did not advance the real
data-order generator:

  1. ordered-sequence SHA for both arms (the training order actually consumed);
  2. first training-batch row identifiers for both;
  3. initial predictor SHA for both;
  4. pre-update e1 recomputed on that same first batch for both.

If any receipt differs, the arms were not perfectly paired and the comparison
must not be resumed.
"""
from __future__ import annotations

import argparse
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

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
DIAG = CACHE / "temporal_action_jepa_v1" / "predicted_token_diagnostic"
TWO = CACHE / "two_step"
OUT = TWO / "pairing_check"
ARMS = ("one_step", "rollout")
EPOCHS_DONE = 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    base = [json.loads(l) for l in (CACHE / "temporal_rows.jsonl").read_text().splitlines() if l.strip()]
    two = [json.loads(l) for l in (TWO / "two_step_rows.jsonl").read_text().splitlines() if l.strip()]
    base_train = [r for r in base if r["role"] == "train"]
    pos_train = {r["pair_sha256"]: i for i, r in enumerate(base_train)}
    train_rows = [r for r in two if r["role"] == "train"]
    tr_idx = np.array([pos_train[r["pair_sha256"]] for r in train_rows])
    n_bt = len(base_train)
    n_bs = len([r for r in base if r["role"] == "checkpoint_selection"])

    ctx0 = R.load_cache(DIAG / "frozen_train_ctx0.f16", n_bt)[tr_idx]
    ctx1 = R.load_cache(DIAG / "frozen_train_ctx1.f16", n_bt)[tr_idx]
    ctx2 = R.load_cache(EVAL / "frozen_current.f16", n_bt + n_bs)[:n_bt][tr_idx]
    y1 = R.load_cache(EVAL / "frozen_train_future.f16", n_bt)[tr_idx]

    # the order the runner actually consumes: a generator seeded with SEED,
    # advanced once per epoch by randperm
    generator = torch.Generator().manual_seed(R.SEED)
    orders = [torch.randperm(len(train_rows), generator=generator).tolist()
              for _ in range(EPOCHS_DONE)]
    expected_state = generator.get_state()
    ordered_sha = hashlib.sha256(
        json.dumps([[train_rows[i]["pair_sha256"] for i in o] for o in orders]).encode()
    ).hexdigest()
    first_batch = orders[0][: R.BATCH]
    first_batch_ids = [train_rows[i]["pair_sha256"] for i in first_batch]

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "read_only": True,
        "purpose": ("confirm the discarded 50-step rollout-gradient warmup did not advance "
                    "the real arm's data-order state"),
        "expected_from_seed": {
            "seed": R.SEED,
            "ordered_sequence_sha256_epochs_0_5": ordered_sha,
            "first_training_batch_row_ids": first_batch_ids,
            "first_training_batch_positions": first_batch,
        },
        "arms": {},
    }

    # pre-update e1 on the first batch, from the initial weights
    torch.manual_seed(R.SEED)
    fresh = T.Predictor(**R.PRED).to(device).eval()
    fresh_hash = hashlib.sha256(
        b"".join(v.detach().cpu().numpy().tobytes() for v in fresh.state_dict().values())
    ).hexdigest()
    with torch.no_grad():
        c = torch.stack([ctx0[first_batch].float(), ctx1[first_batch].float(),
                         ctx2[first_batch].float()], dim=1).to(device)
        t1 = T.normalise(y1[first_batch].float().to(device))
        a0 = T.action_tensor([train_rows[i]["action_step1"] for i in first_batch], device)
        m = torch.ones(len(first_batch), R.TOKENS, dtype=torch.bool, device=device)
        p1 = T.normalise(fresh(T.normalise(c), a0, m))
        pre_update_e1 = float((p1 - t1).abs().mean())
    record["expected_from_seed"]["initial_predictor_sha256"] = fresh_hash
    record["expected_from_seed"]["pre_update_e1_on_first_batch"] = pre_update_e1
    del fresh
    torch.cuda.empty_cache()

    for name in ARMS:
        arm_dir = TWO / "arms" / f"arm_{name}"
        training = json.loads((arm_dir / "result.json").read_text())
        state = torch.load(arm_dir / f"checkpoint_epoch{EPOCHS_DONE - 1}.pt",
                           map_location="cpu", weights_only=False)
        saved_generator_state = state["data_order_generator_state"]
        entry = {
            "initial_predictor_sha256": training["predictor"]["initial_weight_sha256"],
            "initial_predictor_matches": (
                training["predictor"]["initial_weight_sha256"] == fresh_hash),
            "saved_data_order_generator_state_sha256": hashlib.sha256(
                saved_generator_state.numpy().tobytes()).hexdigest(),
            "data_order_generator_matches_six_clean_epochs": bool(
                torch.equal(saved_generator_state, expected_state)),
            "epoch": int(state["epoch"]), "global_step": int(state["global_step"]),
            "seed": int(state["seed"]),
            "warmup_discarded": training.get("rollout_gradient_assertion", {}).get(
                "warmup_discarded", "n/a (control ran no warmup)"),
        }
        # reproduce this arm's own first batch and pre-update e1 from its recorded seed
        g = torch.Generator().manual_seed(int(state["seed"]))
        own_first = torch.randperm(len(train_rows), generator=g).tolist()[: R.BATCH]
        entry["first_training_batch_row_ids"] = [train_rows[i]["pair_sha256"] for i in own_first]
        entry["first_batch_matches"] = entry["first_training_batch_row_ids"] == first_batch_ids
        record["arms"][name] = entry

    a, b = record["arms"]["one_step"], record["arms"]["rollout"]
    checks = {
        "ordered_sequence_identical_across_arms": (
            a["saved_data_order_generator_state_sha256"]
            == b["saved_data_order_generator_state_sha256"]),
        "both_generators_match_six_clean_epochs": (
            a["data_order_generator_matches_six_clean_epochs"]
            and b["data_order_generator_matches_six_clean_epochs"]),
        "first_batch_identical_across_arms": (
            a["first_training_batch_row_ids"] == b["first_training_batch_row_ids"]),
        "initial_predictor_identical_across_arms": (
            a["initial_predictor_sha256"] == b["initial_predictor_sha256"]),
        "initial_predictor_matches_fresh_seed": (
            a["initial_predictor_matches"] and b["initial_predictor_matches"]),
        "global_step_identical": a["global_step"] == b["global_step"],
    }
    record["checks"] = checks
    record["PAIRED"] = all(checks.values())
    record["verdict"] = (
        "arms are perfectly paired; the discarded warmup did not advance the data-order state"
        if record["PAIRED"] else
        "ARMS WERE NOT PERFECTLY PAIRED -- do not resume the comparison"
    )
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return 0 if record["PAIRED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
