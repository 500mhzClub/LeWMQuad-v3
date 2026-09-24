#!/usr/bin/env python3
"""DEVELOPMENT-TIER diagnostic B: bounded train-set capacity probe.

Fit only protocol-valid training groups with a within-state contrastive loss,
then evaluate train and scene-disjoint evaluation groups at coherent frozen
post-epoch snapshots.  Action-blind and action-shuffled arms are measured at
the same snapshots as negative controls.

Reaching the declared threshold demonstrates only capacity to fit this finite
training set under this intervention.  It neither establishes generalization
nor identifies why another objective did or did not learn action dependence.

Writes only under `.generated/dev/`. Not citable.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import statistics
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

probe = importlib.import_module("scripts.dev_probe_counterfactual_action_fidelity")
training = importlib.import_module(
    "scripts.run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
evaluation = importlib.import_module(
    "scripts.evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
metrics = importlib.import_module(
    "lewm.benchmarks.go2_rgb_recurrent_patch_memory_temporal_jepa_v1")


def json_safe_config(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_config(item) for item in value]
    return value


def group_tensors(group, device, *, pilot_bundle=None):
    if not group.get("context_frames") or not group.get("historical_actions"):
        raise probe.CounterfactualProtocolError(
            "capacity diagnostic requires verified H6 context/action provenance")
    context = torch.stack(
        [
            probe.decode(p, device, pilot_bundle=pilot_bundle)
            for p in group["context_frames"]
        ]).unsqueeze(0)
    targets = torch.stack([
        probe.decode(p, device, pilot_bundle=pilot_bundle)
        for p in group["targets"]
    ])
    actions = group["actions"]
    return context, targets, actions, group


def forward_group(
    model,
    context,
    targets,
    actions,
    historical_actions,
    mask,
    device,
    *,
    action_mode="factual",
):
    """Return the K x K energy matrix with gradients attached."""
    conditioned = probe.conditioned_candidate_actions(actions, action_mode)
    masks = mask if isinstance(mask, (list, tuple)) else [mask]
    if len(masks) != 4:
        raise ValueError("capacity diagnostic requires the fixed four-mask contract")
    energies = []
    for selected_mask in masks:
        with torch.no_grad():
            tgt_tokens = torch.stack([
                evaluation._target_tokens(
                    model, targets[i].unsqueeze(0), selected_mask
                )[0]
                for i in range(len(actions))
            ])
        preds = []
        for action in conditioned:
            seq = torch.tensor(
                [[historical_actions[0], historical_actions[1], action]],
                dtype=torch.long,
                device=device,
            )
            preds.append(
                evaluation._predict_future(
                    model, context, seq, selected_mask
                ).prediction[0]
            )
        predictions = torch.stack(preds)
        diff = predictions.unsqueeze(1) - tgt_tokens.unsqueeze(0)
        energies.append(0.5 * diff.square().sum(-1).mean(-1))
    return torch.stack(energies).mean(dim=0)


@torch.no_grad()
def evaluate_partition(
    model,
    cached,
    mask,
    device,
    *,
    action_mode,
    bootstrap_resamples=0,
):
    """Evaluate every group against one coherent frozen model snapshot."""
    was_training = model.training
    model.eval()
    results = []
    try:
        for context, targets, actions, group in cached:
            energy = forward_group(
                model,
                context,
                targets,
                actions,
                group["historical_actions"],
                mask,
                device,
                action_mode=action_mode,
            )
            result = probe._matrix_metrics(energy.detach().cpu().double(), group)
            result["action_mode"] = action_mode
            results.append(result)
    finally:
        model.train(was_training)
    return {
        "summary": probe.summarize(
            results,
            action_mode,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "group_results": results,
    }


def evaluate_snapshot(
    model,
    cached_by_role,
    mask,
    device,
    *,
    bootstrap_resamples=0,
):
    """Return split-preserving factual and negative-control snapshot metrics."""
    return {
        role: {
            action_mode: evaluate_partition(
                model,
                cached,
                mask,
                device,
                action_mode=action_mode,
                bootstrap_resamples=bootstrap_resamples,
            )
            for action_mode in ("factual", "action_blind", "action_shuffled")
        }
        for role, cached in cached_by_role.items()
    }


def train_epoch(model, cached, mask, device, optimizer, partition, args) -> float:
    """Train on the train partition only; evaluation happens after the epoch."""
    model.train()
    losses = []
    for context, targets, actions, group in cached:
        energy = forward_group(
            model,
            context,
            targets,
            actions,
            group["historical_actions"],
            mask,
            device,
            action_mode="factual",
        )
        labels = torch.arange(len(actions), device=device)
        logits = -energy / args.temperature
        contrastive = torch.nn.functional.cross_entropy(logits, labels)
        mse = energy.diagonal().mean()
        loss = contrastive + args.mse_weight * mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(partition.online, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return statistics.fmean(losses)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Exact immutable development checkpoint; no mutable latest alias.",
    )
    ap.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="Pinned lowercase SHA-256 of the immutable development checkpoint.",
    )
    ap.add_argument("--expected-update", type=int, required=True)
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument(
        "--expected-pilot-manifest-byte-count", type=int, required=True
    )
    ap.add_argument("--expected-pilot-manifest-sha256", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--capacity-threshold", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",
                    default=".generated/dev/counterfactual/overfit_capacity.json")
    args = ap.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.lr <= 0.0 or args.temperature <= 0.0:
        raise ValueError("lr and temperature must be positive")
    if args.mse_weight < 0.0:
        raise ValueError("mse-weight must be non-negative")
    if not 0.0 < args.capacity_threshold <= 1.0:
        raise ValueError("capacity threshold must lie in (0,1]")
    if args.expected_update < 0:
        raise ValueError("expected-update must be non-negative")
    if Path(args.checkpoint).name == "latest.pt":
        raise ValueError("mutable latest.pt checkpoints are forbidden")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_checkpoint_sha256):
        raise ValueError("expected checkpoint SHA-256 must be lowercase hex")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_pilot_manifest_sha256):
        raise ValueError("expected pilot manifest SHA-256 must be lowercase hex")
    checkpoint = probe.require_development_checkpoint(Path(args.checkpoint))
    out = probe.require_development_output(Path(args.out))
    code_bindings = [
        probe.file_binding(Path(path))
        for path in (
            __file__, probe.__file__, probe.model_module.__file__,
            training.__file__, evaluation.__file__, metrics.__file__, probe.h6.__file__,
            probe.pilot.__file__, probe.trainer.__file__,
        )
    ]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    loaded, pilot_bundle = probe.load_pilot_groups(
        args.pilot_root,
        expected_manifest_byte_count=args.expected_pilot_manifest_byte_count,
        expected_manifest_sha256=args.expected_pilot_manifest_sha256,
    )
    train_groups = list(loaded.groups_by_role["train"])
    eval_groups = list(loaded.groups_by_role["eval"])
    print(json.dumps(loaded.audit, sort_keys=True), flush=True)
    print(
        f"capacity groups: train={len(train_groups)}, eval={len(eval_groups)}",
        flush=True,
    )
    if not train_groups or not eval_groups:
        print(
            "both protocol-valid train and eval groups are required; refusing "
            "to open model or RGB",
            flush=True,
        )
        return 1
    mask = [
        metrics.batched_mask_indices("val", [row], device=device)[0]
        for row in (0, 1, 2, 3)
    ]

    model, label, model_identity = probe.build_model(
        checkpoint,
        device,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_update=args.expected_update,
    )
    model.train()
    partition = training.partition_parameters_v1(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=0.0)

    cached_by_role = {
        "train": [
            group_tensors(group, device, pilot_bundle=pilot_bundle)
            for group in train_groups
        ],
        "eval": [
            group_tensors(group, device, pilot_bundle=pilot_bundle)
            for group in eval_groups
        ],
    }
    records = []
    for epoch in range(args.epochs + 1):
        train_loss = None
        if epoch > 0:
            train_loss = train_epoch(
                model,
                cached_by_role["train"],
                mask,
                device,
                optimizer,
                partition,
                args,
            )
        snapshot = evaluate_snapshot(model, cached_by_role, mask, device)
        train_micro = snapshot["train"]["factual"]["summary"]["micro"]
        eval_micro = snapshot["eval"]["factual"]["summary"]["micro"]
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_micro_fidelity": train_micro["fidelity_rate"],
            "train_micro_chance": train_micro["chance"],
            "train_fidelity_over_chance": train_micro["fidelity_over_chance"],
            "eval_micro_fidelity": eval_micro["fidelity_rate"],
            "eval_micro_chance": eval_micro["chance"],
            "snapshot": snapshot,
        }
        records.append(rec)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_micro_fidelity": rec["train_micro_fidelity"],
                "train_micro_chance": rec["train_micro_chance"],
                "eval_micro_fidelity": rec["eval_micro_fidelity"],
                "eval_micro_chance": rec["eval_micro_chance"],
            }), flush=True)

    best_record = max(records, key=lambda record: record["train_micro_fidelity"])
    terminal_record = records[-1]
    terminal_record["snapshot"] = evaluate_snapshot(
        model,
        cached_by_role,
        mask,
        device,
        bootstrap_resamples=2000,
    )
    capacity_present = (
        terminal_record["train_micro_fidelity"] >= args.capacity_threshold
    )
    verdict = (
        "TRAIN_SET_CAPACITY_DEMONSTRATED_NO_GENERALIZATION_CLAIM"
        if capacity_present
        else "TRAIN_SET_CAPACITY_NOT_DEMONSTRATED"
    )
    reloaded, _ = probe.load_pilot_groups(
        args.pilot_root,
        expected_manifest_byte_count=args.expected_pilot_manifest_byte_count,
        expected_manifest_sha256=args.expected_pilot_manifest_sha256,
    )
    if reloaded.audit != loaded.audit:
        raise probe.CounterfactualProtocolError(
            "pilot receipts changed during capacity diagnostic"
        )
    probe.assert_file_bindings_unchanged(
        code_bindings, kind="capacity-diagnostic source"
    )
    probe.write_json_atomic(out, {
        "schema": "dev_counterfactual_overfit_capacity_v3",
        "status": "COMPLETE",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "note": (
            "overfitting is the intended outcome; train/eval are scene-role "
            "separated and every metric is a coherent frozen post-epoch snapshot"),
        "evidence_scope": "physics_executed",
        "claim_scope": "physical_pilot_train_capacity_only",
        "mask_protocol": {
            "role": "val",
            "row_indices": [0, 1, 2, 3],
            "fixed_four_mask_contract": True,
        },
        "protocol_audit": loaded.audit,
        "checkpoint": str(checkpoint),
        "source_bindings": code_bindings,
        "label": label,
        "model_identity": model_identity,
        "config": json_safe_config(vars(args)),
        "best_train_micro_fidelity": best_record["train_micro_fidelity"],
        "best_snapshot_epoch": best_record["epoch"],
        "terminal_train_micro_fidelity": terminal_record["train_micro_fidelity"],
        "terminal_snapshot_epoch": terminal_record["epoch"],
        "verdict": verdict, "records": records})
    print(
        f"terminal coherent train fidelity = "
        f"{terminal_record['train_micro_fidelity']:.3f} "
        f"(best={best_record['train_micro_fidelity']:.3f}) -> {verdict}",
        flush=True,
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
