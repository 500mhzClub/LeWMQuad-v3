#!/usr/bin/env python3
"""Two-step autoregressive rollout objective vs a matched one-step control.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Rollout-objective intervention only.

Loss reduction, reproduced from the audited official source rather than
re-derived.  The official computes two elementwise means and sums them:

    jloss = mean(|z_tf - h|)          teacher-forced predictions
    sloss = mean(|z_ar - h|)          the auto_steps rollout predictions
    loss  = jloss + sloss             unweighted

With ``auto_steps=2`` the rollout tensor ``z_ar`` holds BOTH predicted frames --
the teacher-forced first frame and the autoregressive second -- so in the reduced
two-step setting the identical reduction is

    jloss = e1,   sloss = (e1 + e2) / 2,   L = 1.5*e1 + 0.5*e2

This is implemented by taking the two elementwise means, **not** by hardcoding
1.5 and 0.5, so the reduction follows the source rather than an algebraic
restatement of it.  ``e1 + e2`` would NOT be official-equivalent.

The audited source does **not** detach the fed-back prediction
(``torch.cat([..., z_tf[:, :tokens_per_frame]])`` keeps the graph), so p1 is fed
forward attached.  ``--assert-rollout-gradient`` checks on one batch that the
second-step term alone sends nonzero gradient into p1.

Step two uses the fixed sliding context ``[t-240, t, p1]``.  This is a
**sliding-three-frame adaptation**, not an exact reproduction of the official
growing-context architecture, which has eight temporal slots and appends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import dev_checkpoint_v1 as CK  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
DIAG = CACHE / "temporal_action_jepa_v1" / "predicted_token_diagnostic"
TWO = CACHE / "two_step"
OUT = CACHE / "two_step" / "arms"

TOKENS, DIM = T.TOKENS, T.TOKEN_DIM
SEED = T.SEED
EPOCHS, BATCH, LR, WD, CLIP = 6, 4, 3.0e-4, 0.01, 1.0
PRED = {"width": 384, "depth": 6, "heads": 6}


def load_cache(path: Path, count: int) -> torch.Tensor:
    expected = count * TOKENS * DIM * 2
    if not path.is_file() or path.stat().st_size != expected:
        raise RuntimeError(f"cache missing or wrong size: {path} ({path.stat().st_size if path.is_file() else 'absent'} vs {expected})")
    return torch.from_numpy(np.ascontiguousarray(
        np.memmap(path, dtype=np.float16, mode="r", shape=(count, TOKENS, DIM))))


@torch.no_grad()
def extract_step2(rows, device, blob: Path, batch=16):
    arm = E.VJepa21CroppedV03Arm()
    shape = (len(rows), TOKENS, DIM)
    if blob.is_file() and blob.stat().st_size == int(np.prod(shape) * 2):
        return
    module = arm.build(device, torch.float32)
    memory = np.memmap(blob, dtype=np.float16, mode="w+", shape=shape)
    paths = [r["step2_path"] for r in rows]
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        pixels = torch.stack([arm.preprocess(p) for p in chunk]).to(device, torch.float32)
        memory[start : start + len(chunk)] = module(pixels.unsqueeze(2)).half().cpu().numpy()
        if start % 800 == 0:
            print(f"  step2 encode {start}/{len(paths)}", flush=True)
    memory.flush()
    del module
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm", required=True, choices=("one_step", "rollout"))
    ap.add_argument("--assert-rollout-gradient", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--warmup-steps", type=int, default=50)
    args = ap.parse_args()
    device = torch.device(args.device)
    out = OUT / f"arm_{args.arm}"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    base = [json.loads(l) for l in (CACHE / "temporal_rows.jsonl").read_text().splitlines() if l.strip()]
    two = [json.loads(l) for l in (TWO / "two_step_rows.jsonl").read_text().splitlines() if l.strip()]
    base_train = [r for r in base if r["role"] == "train"]
    base_sel = [r for r in base if r["role"] == "checkpoint_selection"]
    pos_train = {r["pair_sha256"]: i for i, r in enumerate(base_train)}
    pos_sel = {r["pair_sha256"]: i for i, r in enumerate(base_sel)}
    n_base_train, n_base_sel = len(base_train), len(base_sel)

    train_rows = [r for r in two if r["role"] == "train"]
    sel_rows = [r for r in two if r["role"] == "checkpoint_selection"]
    train_idx = np.array([pos_train[r["pair_sha256"]] for r in train_rows])
    sel_idx = np.array([pos_sel[r["pair_sha256"]] for r in sel_rows])

    # frozen encoder caches, indexed onto the two-step-capable subset
    ctx0 = load_cache(DIAG / "frozen_train_ctx0.f16", n_base_train)[train_idx]
    ctx1 = load_cache(DIAG / "frozen_train_ctx1.f16", n_base_train)[train_idx]
    current_all = load_cache(EVAL / "frozen_current.f16", n_base_train + n_base_sel)
    ctx2 = current_all[:n_base_train][train_idx]
    y1 = load_cache(EVAL / "frozen_train_future.f16", n_base_train)[train_idx]

    step2_blob = TWO / "frozen_train_step2.f16"
    extract_step2(train_rows, device, step2_blob)
    if args.extract_only:
        print("step2 targets extracted"); return 0
    y2 = load_cache(step2_blob, len(train_rows))

    torch.manual_seed(SEED)                      # identical initial weights for both arms
    predictor = T.Predictor(**PRED).to(device)
    init_hash = hashlib.sha256(
        b"".join(v.detach().cpu().numpy().tobytes() for v in predictor.state_dict().values())
    ).hexdigest()
    optimiser = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WD, foreach=False)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    steps_per_epoch = (len(train_rows) + BATCH - 1) // BATCH

    def forward(sel, generator=None):
        """Returns (e1_terms, e2_terms, p1) with p1 attached for the rollout arm."""
        c0 = ctx0[sel].float().to(device)
        c1 = ctx1[sel].float().to(device)
        c2 = ctx2[sel].float().to(device)
        t1 = T.normalise(y1[sel].float().to(device))
        a0 = T.action_tensor([train_rows[i]["action_step1"] for i in sel], device)
        mask = torch.ones(len(sel), TOKENS, dtype=torch.bool, device=device)
        context1 = T.normalise(torch.stack([c0, c1, c2], dim=1))
        p1 = T.normalise(predictor(context1, a0, mask))
        if args.arm == "one_step":
            return p1, t1, None, None
        # sliding-three-frame adaptation: [t-240, t, p1]; p1 is NOT detached
        a1 = T.action_tensor([train_rows[i]["action_step2"] for i in sel], device)
        t2 = T.normalise(y2[sel].float().to(device))
        context2 = torch.stack([T.normalise(c1), T.normalise(c2), p1], dim=1)
        p2 = T.normalise(predictor(context2, a1, mask))
        return p1, t1, p2, t2

    record = {
        "status": "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING", "claim_bearing": False,
        "arm": args.arm,
        "intervention": "rollout objective only",
        "loss_reduction": {
            "source": "app/vjepa_droid/train.py, loss = jloss + sloss, both elementwise means",
            "jloss": "mean(|p1 - y1|) = e1",
            "sloss": "mean over BOTH rollout predictions = (e1 + e2)/2",
            "effective": "L = 1.5*e1 + 0.5*e2",
            "implemented_as": "two elementwise means summed, not hardcoded coefficients",
            "e1_plus_e2_would_not_be_official_equivalent": True,
        },
        "step2_context": {
            "form": "[t-240, t, p1]",
            "label": "sliding-three-frame adaptation",
            "not": "an exact reproduction of the official growing-context architecture (8 temporal slots, appends)",
            "p1_detached": False,
            "source_detaches": False,
        },
        "rows": {"train": len(train_rows), "checkpoint_selection": len(sel_rows),
                 "subset": "identical two-step-capable subset for both arms"},
        "predictor": {**PRED, "parameters": int(sum(p.numel() for p in predictor.parameters())),
                      "initial_weight_sha256": init_hash},
        "encoder": "official V-JEPA 2.1 ViT-L, frozen; never executed in training (cached tokens)",
        "schedule": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                     "weight_decay": WD, "grad_clip": CLIP, "optimiser": "AdamW, fresh state",
                     "amp": "bf16"},
        "epochs": [],
    }

    # ---- one-batch assertion: the rollout term alone reaches p1 --------------
    # NOTE: this predictor uses AdaLN-Zero initialisation -- every block's `ada`
    # projection is zeroed, so the gating terms g1,g2 are exactly 0 and the blocks
    # are the identity at step 0.  The network is therefore NECESSARILY
    # context-independent at initialisation and the gradient into p1 through p2 is
    # exactly zero.  That is a property of the init scheme, not of the rollout
    # wiring, so the assertion is evaluated at init (expected zero, recorded) and
    # again after a short warmup (must be nonzero).
    def rollout_gradient_probe(tag):
        generator_local = torch.Generator().manual_seed(SEED)
        probe = torch.randperm(len(train_rows), generator=generator_local).tolist()[:BATCH]
        p1, t1, p2, t2 = forward(probe)
        p1.retain_grad()
        (p2 - t2).abs().mean().backward()
        via_rollout = float(p1.grad.abs().sum()) if p1.grad is not None else 0.0
        predictor.zero_grad(set_to_none=True)
        p1b, t1b, p2b, t2b = forward(probe)
        p1b.retain_grad()
        torch.cat([p1b, p2b], 1).sub(torch.cat([t1b, t2b], 1)).abs().mean().backward()
        via_sloss = float(p1b.grad.abs().sum()) if p1b.grad is not None else 0.0
        predictor.zero_grad(set_to_none=True)
        gate = float(sum(b.ada.weight.abs().sum() for b in predictor.blocks))
        return {"when": tag,
                "grad_abs_sum_into_p1_from_second_step_term_alone": via_rollout,
                "grad_abs_sum_into_p1_from_sloss_alone": via_sloss,
                "adaln_gate_weight_abs_sum": gate,
                "second_step_term_reaches_p1": via_rollout > 0.0}

    if args.arm == "rollout" and args.assert_rollout_gradient:
        at_init = rollout_gradient_probe("initialisation")
        warmup_generator = torch.Generator().manual_seed(SEED)
        warm_order = torch.randperm(len(train_rows), generator=warmup_generator).tolist()
        predictor.train()
        for w in range(args.warmup_steps):
            sel = warm_order[w * BATCH : (w + 1) * BATCH]
            optimiser.zero_grad(set_to_none=True)
            with autocast:
                p1, t1, p2, t2 = forward(sel)
                loss = (p1 - t1).abs().mean() + torch.cat([p1, p2], 1).sub(
                    torch.cat([t1, t2], 1)).abs().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), CLIP)
            optimiser.step()
        after = rollout_gradient_probe(f"after {args.warmup_steps} warmup steps")
        record["rollout_gradient_assertion"] = {
            "adaln_zero_note": (
                "ada weight and bias are zeroed at init, so g1=g2=0, every block is "
                "the identity and the predictor is context-independent at step 0; a "
                "zero rollout gradient there is expected and is not a dead path"
            ),
            "at_initialisation": at_init,
            "after_warmup": after,
        }
        print(json.dumps(record["rollout_gradient_assertion"], indent=2), flush=True)
        if not after["second_step_term_reaches_p1"]:
            raise RuntimeError(
                "after warmup the second-step term still sends no gradient into p1; "
                "the rollout path is dead"
            )
        # discard the warmup: training starts from the recorded initial weights
        torch.manual_seed(SEED)
        predictor = T.Predictor(**PRED).to(device)
        rebuilt = hashlib.sha256(
            b"".join(v.detach().cpu().numpy().tobytes() for v in predictor.state_dict().values())
        ).hexdigest()
        if rebuilt != init_hash:
            raise RuntimeError("re-initialisation after the warmup probe did not reproduce the initial weights")
        optimiser = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WD, foreach=False)
        record["rollout_gradient_assertion"]["warmup_discarded"] = True

    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        predictor.train()
        totals = {"loss": 0.0, "jloss": 0.0, "sloss": 0.0, "e1": 0.0, "e2": 0.0}
        seen = 0
        for start in range(0, len(order), BATCH):
            sel = order[start : start + BATCH]
            optimiser.zero_grad(set_to_none=True)
            with autocast:
                p1, t1, p2, t2 = forward(sel)
                e1 = (p1 - t1).abs().mean()
                if args.arm == "rollout":
                    e2 = (p2 - t2).abs().mean()
                    jloss = e1
                    # sloss is ONE elementwise mean over both rollout predictions
                    sloss = torch.cat([p1, p2], 1).sub(torch.cat([t1, t2], 1)).abs().mean()
                    loss = jloss + sloss
                else:
                    e2 = torch.zeros((), device=device)
                    jloss, sloss, loss = e1, torch.zeros((), device=device), e1
            loss.backward()
            nn.utils.clip_grad_norm_(predictor.parameters(), CLIP)
            optimiser.step()
            for k, v in (("loss", loss), ("jloss", jloss), ("sloss", sloss), ("e1", e1), ("e2", e2)):
                totals[k] += float(v.detach()) * len(sel)
            seen += len(sel)
            if (start // BATCH) % 100 == 0:
                print(f"  [{args.arm}] epoch {epoch} step {start//BATCH} "
                      f"loss {totals['loss']/max(seen,1):.5f} e1 {totals['e1']/max(seen,1):.5f} "
                      f"e2 {totals['e2']/max(seen,1):.5f}", flush=True)
        entry = {"epoch": epoch, **{k: v / max(seen, 1) for k, v in totals.items()}}
        record["epochs"].append(entry)
        print(f"[{args.arm}] epoch {epoch} loss {entry['loss']:.5f} "
              f"e1 {entry['e1']:.5f} e2 {entry['e2']:.5f}", flush=True)
        receipt = CK.save(
            out / f"checkpoint_epoch{epoch}.pt",
            model=predictor, optimizer=optimiser, epoch=epoch,
            global_step=(epoch + 1) * steps_per_epoch, seed=SEED,
            model_config={**PRED, "token_dim": DIM, "tokens": TOKENS,
                          "class": "run_dev_v03_temporal_action_jepa_v1.Predictor",
                          "arm": args.arm, "initial_weight_sha256": init_hash},
            scheduler=None,
            scheduler_absent_reason="fixed learning rate; no scheduler is constructed",
            data_order_generator=generator,
            extra={"epochs": record["epochs"], "encoder_trainable": None,
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        )
        record.setdefault("checkpoint_receipts", []).append(receipt)
        record["wall_seconds"] = round(time.time() - started, 1)
        (out / "result.json").write_text(json.dumps(record, indent=2))

    (out / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"arm": args.arm, "epochs": record["epochs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
