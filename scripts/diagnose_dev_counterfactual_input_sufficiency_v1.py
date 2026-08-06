#!/usr/bin/env python3
"""Input-sufficiency diagnostic for counterfactual matching.

DEVELOPMENT ONLY.  NOT CLAIM BEARING.  Diagnostic only -- the encoder is frozen
throughout, which is acceptable here and is **not** the JEPA endpoint.

The basis-drift audit established a severe train-to-scene-selection
generalisation failure.  It did not establish that corpus size is the sole
cause.  This asks the prior question: does the predictor lack data diversity, or
does it lack sufficient state information?

Four input arms, identical predictor capacity, loss, temperatures, branch
groups, scene split and fitting budget:

``single_frame``         current patch tokens + action -- the present input
``temporal_context``     all three stored context frames + action, aggregated by
                         the smallest fixed operator that preserves the dense
                         256-patch layout (per-patch concatenation then one
                         linear projection back to 192)
``privileged_dynamics``  current tokens + recorded prebranch body-frame linear
                         and angular velocity and joint position/velocity +
                         action.  **Contact state is not recorded anywhere in
                         the V3 collection and is therefore omitted, not
                         manufactured.**
``action_only``          action without visual state -- isolates action and
                         branch priors

The prebranch dynamics come from each branch's ``trajectory_policy_step_samples``
at ``policy_step_index == 0``, which is verified bit-identical across all nine
branches of a group, so it is a genuine shared predecessor state rather than
post-action leakage.
"""
from __future__ import annotations

import argparse
import collections
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
import lewm.models.direct_egocentric_bev_state_jepa_v1 as _preload  # noqa: F401,E402

from lewm.datasets import go2_v3_counterfactual_branches_v1 as V3  # noqa: E402
from lewm.models.token_primary_action_conditioned_jepa_v1 import (  # noqa: E402
    TOKEN_DIM, initialize_token_predictor_v1,
)
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
import scripts.run_dev_token_counterfactual_matching_h1 as RUN  # noqa: E402
import scripts.run_dev_token_predictor_only_h1 as R  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
NINE = V3.BRANCHES_PER_GROUP
STEPS = 10_680          # same fitting budget as the paired run
GROUPS_PER_STEP = RUN.GROUPS_PER_STEP
LR = RUN.PRED_LR
DERANGEMENT_SEEDS = (11, 23, 37)
ARMS = ("single_frame", "temporal_context", "privileged_dynamics", "action_only")


def LN(t):
    return F.layer_norm(t, (TOKEN_DIM,))


def quat_to_body(vector: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame (conjugate rotation)."""
    w, x, y, z = quat_wxyz
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rotation.T @ vector


class InputAdapterV1(nn.Module):
    """Maps an arm's inputs to the predictor's (B, 256, 192) state."""

    def __init__(self, arm: str, proprio_dim: int):
        super().__init__()
        self.arm = arm
        if arm == "single_frame":
            width = TOKEN_DIM
        elif arm == "temporal_context":
            width = TOKEN_DIM * 3
        elif arm == "privileged_dynamics":
            width = TOKEN_DIM + proprio_dim
        elif arm == "action_only":
            width = TOKEN_DIM
            self.constant = nn.Parameter(torch.zeros(1, 256, TOKEN_DIM))
            nn.init.normal_(self.constant, std=0.02)
        else:
            raise ValueError(arm)
        self.project = nn.Linear(width, TOKEN_DIM)

    def forward(self, frames, proprio):
        if self.arm == "single_frame":
            hidden = frames[:, 2]
        elif self.arm == "temporal_context":
            hidden = torch.cat([frames[:, 0], frames[:, 1], frames[:, 2]], dim=-1)
        elif self.arm == "privileged_dynamics":
            hidden = torch.cat(
                [frames[:, 2], proprio[:, None, :].expand(-1, 256, -1)], dim=-1)
        else:
            hidden = self.constant.expand(frames.shape[0], -1, -1)
        return LN(self.project(hidden))


def normalised_ce(ce: float, entropy: float) -> float:
    return (ce - entropy) / (math.log(9) - entropy)


def matching_stats(pred, tbar, q_matrix, tau):
    pn = F.normalize(pred, dim=-1)
    logits = torch.einsum("gid,gjd->gij", pn, tbar) / tau
    log_probs = F.log_softmax(logits, dim=-1)
    eye = torch.eye(NINE, dtype=torch.bool, device=pred.device)
    own = (pred - tbar).pow(2).sum(-1)
    other = (pred[:, :, None, :] - tbar[:, None, :, :]).pow(2).sum(-1)
    other = other.masked_fill(eye[None], 0).sum(-1) / (NINE - 1)
    return {
        "matching_ce": float(-(q_matrix * log_probs).sum(-1).mean()),
        "top1": float((logits.argmax(-1)
                       == torch.arange(NINE, device=pred.device)[None]).float().mean()),
        "diagonal_probability": float(
            torch.diagonal(log_probs.exp(), dim1=1, dim2=2).mean()),
        "own_vs_other_l2_ratio": float(own.mean() / other.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    cache = torch.load(RUN.CACHE, map_location="cpu", weights_only=False)
    order = cache["state_ids"]
    train_ids = set(cache["train_state_ids"])
    selection_ids = set(cache["selection_state_ids"])
    idx = {"train": [i for i, s in enumerate(order) if s in train_ids],
           "selection": [i for i, s in enumerate(order) if s in selection_ids]}

    from scripts import run_go2_observability_ceiling_assay_v1 as ceiling
    ledger = ceiling.AccessLedgerV1()
    groups_all = V3.load_branch_groups_v1("train", ceiling, ledger)
    by_id = {g.state_id: g for g in groups_all}
    _g, records = ceiling.load_role_v1("train", ledger=ledger)

    encoder = P.load_stack(device)[0]
    encoder.eval().requires_grad_(False)

    # ---- precompute frozen inputs and targets ----
    data = {}
    for split, rows in idx.items():
        panel = [by_id[order[i]] for i in rows]
        frames = torch.empty(len(panel), 3, 256, TOKEN_DIM)
        proprio = torch.empty(len(panel), 30)
        for gi, group in enumerate(panel):
            record = records[group.state_id]
            artifacts = record["context"]["rgb_artifact_ids"]
            if len(artifacts) != 3:
                raise RuntimeError(
                    f"{group.state_id}: expected 3 context frames, got {len(artifacts)}")
            ctx = [ceiling.rgb_path_v1("train", a) for a in artifacts]
            batch = torch.stack([V3.preprocess_v3_frame_v1(p) for p in ctx]).to(device)
            with torch.no_grad():
                frames[gi] = LN(encoder.forward_tokens(batch)[:, 1:, :]).cpu()
            sample = sorted(record["branches"],
                            key=lambda b: int(b["action_id"]))[0]["trajectory_policy_step_samples"][0]
            quat = np.asarray(sample["base_quat_wxyz"], dtype=np.float64)
            proprio[gi] = torch.from_numpy(np.concatenate([
                quat_to_body(np.asarray(sample["base_lin_vel_world"]), quat),
                quat_to_body(np.asarray(sample["base_ang_vel_world"]), quat),
                np.asarray(sample["leg_joint_pos"]),
                np.asarray(sample["leg_joint_vel"]),
            ])).float()
        tbar_tokens = cache["successor_normalised_tokens"][rows].to(device)
        tbar_flat = F.normalize(cache["successor_flat_unit"][rows].to(device), dim=-1)
        q = torch.softmax(RUN.group_cosine(tbar_flat, tbar_flat) / RUN.TAU_T, dim=-1)
        data[split] = {
            "frames": frames.to(device),
            "proprio": proprio.to(device),
            "commands": cache["commands"][rows].to(device),
            "tbar_tokens": tbar_tokens,
            "tbar_flat": tbar_flat,
            "q": q,
            "entropy": float(-(q * q.clamp_min(1e-12).log()).sum(-1).mean()),
            "families": [by_id[order[i]].family for i in rows],
        }

    # proprio standardisation from TRAIN only
    mean, std = data["train"]["proprio"].mean(0), data["train"]["proprio"].std(0).clamp_min(1e-6)
    for split in data:
        data[split]["proprio"] = (data[split]["proprio"] - mean) / std

    one_hot_all = torch.eye(NINE, device=device)
    report = {
        "status": STATUS, "claim_bearing": False, "encoder": "frozen (diagnostic only)",
        "reference": {"ce_uniform": math.log(9), "top1_chance": 1.0 / 9.0,
                      "ce_floor_entropy_of_Q": {k: data[k]["entropy"] for k in data}},
        "fitting_budget": {"steps": STEPS, "groups_per_step": GROUPS_PER_STEP,
                           "lr": LR, "lambda_match": RUN.LAMBDA_MATCH,
                           "tau_t": RUN.TAU_T, "tau_p": RUN.TAU_P},
        "privileged_fields": {
            "recorded_and_used": ["base_lin_vel_world -> body", "base_ang_vel_world -> body",
                                  "leg_joint_pos", "leg_joint_vel"],
            "contact_state": "NOT RECORDED in the V3 collection; omitted, not manufactured",
            "source": "trajectory_policy_step_samples[policy_step_index == 0]",
            "verified_shared_across_all_nine_branches": True,
        },
        "arms": {},
    }

    for arm in ARMS:
        torch.manual_seed(RUN.SEED)
        adapter = InputAdapterV1(arm, proprio_dim=30).to(device)
        predictor = initialize_token_predictor_v1(RUN.SEED).to(device)
        optimiser = torch.optim.AdamW(
            list(adapter.parameters()) + list(predictor.parameters()),
            lr=LR, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(RUN.SEED)
        n_train = len(idx["train"])
        cursor, group_order = 0, torch.randperm(n_train, generator=generator).tolist()
        train_panel = data["train"]
        adapter.train(); predictor.train()
        for _step in range(STEPS):
            picked = []
            for _ in range(GROUPS_PER_STEP):
                if cursor >= n_train:
                    group_order = torch.randperm(n_train, generator=generator).tolist()
                    cursor = 0
                picked.append(group_order[cursor]); cursor += 1
            sel = torch.tensor(picked, device=device)
            state = adapter(train_panel["frames"][sel], train_panel["proprio"][sel])
            state = state.repeat_interleave(NINE, dim=0)
            actions = one_hot_all.repeat(len(picked), 1)
            commands = train_panel["commands"][sel].reshape(-1, 3)
            pred = LN(predictor(state, actions, commands))
            target = train_panel["tbar_tokens"][sel].reshape(-1, 256, TOKEN_DIM)
            branch_jepa = F.mse_loss(pred, target)
            groups_pred = pred.reshape(len(picked), NINE, -1)
            logits = torch.einsum("gid,gjd->gij", F.normalize(groups_pred, dim=-1),
                                  train_panel["tbar_flat"][sel]) / RUN.TAU_P
            match = -(train_panel["q"][sel] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = branch_jepa + RUN.LAMBDA_MATCH * match
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(adapter.parameters())
                                     + list(predictor.parameters()), 1.0)
            optimiser.step()

        adapter.eval(); predictor.eval()
        entry = {"adapter_parameters": sum(p.numel() for p in adapter.parameters()),
                 "predictor_parameters": sum(p.numel() for p in predictor.parameters())}
        for split, panel in data.items():
            n = panel["frames"].shape[0]
            with torch.no_grad():
                state = adapter(panel["frames"], panel["proprio"])
                state = state.repeat_interleave(NINE, dim=0)
                pred = LN(predictor(state, one_hot_all.repeat(n, 1),
                                    panel["commands"].reshape(-1, 3)))
            stats = matching_stats(pred.reshape(n, NINE, -1), panel["tbar_flat"],
                                   panel["q"], RUN.TAU_P)
            stats["normalised_ce_vs_floor_and_uniform"] = normalised_ce(
                stats["matching_ce"], panel["entropy"])

            # correct-versus-shuffled changed-token prediction
            target = panel["tbar_tokens"].reshape(n * NINE, 256, TOKEN_DIM)
            current = adapter(panel["frames"], panel["proprio"]).detach()
            current = current.repeat_interleave(NINE, dim=0)
            if split == "train":
                change = (target - current).pow(2).mean(-1)
                report["_threshold"] = float(
                    torch.quantile(change.flatten().float(), 0.75))
            mask = ((target - current).pow(2).mean(-1) > report["_threshold"])
            correct = R.metrics(pred, target, current, mask)
            shuffled = []
            for seed in DERANGEMENT_SEEDS:
                g = torch.Generator().manual_seed(seed)
                perm = torch.randperm(NINE, generator=g).to(device)
                sh = pred.reshape(n, NINE, 256, TOKEN_DIM)[:, perm].reshape(
                    n * NINE, 256, TOKEN_DIM)
                shuffled.append(R.metrics(sh, target, current, mask)["cosine"])
            stats["correct_changed_cosine"] = correct["cosine"]
            stats["shuffled_changed_cosine_mean"] = float(np.mean(shuffled))
            stats["correct_minus_shuffled"] = correct["cosine"] - float(np.mean(shuffled))
            entry[split] = stats

            if split == "selection":
                per_family = {}
                families = panel["families"]
                for family in sorted(set(families)):
                    rows = torch.tensor([i for i, f in enumerate(families) if f == family],
                                        device=device)
                    fs = matching_stats(pred.reshape(n, NINE, -1)[rows],
                                        panel["tbar_flat"][rows], panel["q"][rows],
                                        RUN.TAU_P)
                    q_f = panel["q"][rows]
                    entropy_f = float(-(q_f * q_f.clamp_min(1e-12).log()).sum(-1).mean())
                    per_family[family] = {
                        "matching_ce": round(fs["matching_ce"], 4),
                        "normalised_ce": round(normalised_ce(fs["matching_ce"], entropy_f), 4),
                        "top1": round(fs["top1"], 4),
                        "groups": int(len(rows)),
                    }
                entry["per_family_selection"] = per_family

        entry["train_to_selection_gap"] = {
            "matching_ce": entry["selection"]["matching_ce"] - entry["train"]["matching_ce"],
            "normalised_ce": (entry["selection"]["normalised_ce_vs_floor_and_uniform"]
                              - entry["train"]["normalised_ce_vs_floor_and_uniform"]),
            "top1": entry["selection"]["top1"] - entry["train"]["top1"],
            "correct_minus_shuffled": (entry["selection"]["correct_minus_shuffled"]
                                       - entry["train"]["correct_minus_shuffled"]),
        }
        report["arms"][arm] = entry
        del adapter, predictor
        torch.cuda.empty_cache()

    report.pop("_threshold", None)
    out = RUN.OUT / "input_sufficiency_diagnostic.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
