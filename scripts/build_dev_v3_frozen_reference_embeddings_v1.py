#!/usr/bin/env python3
"""Cache frozen-reference successor embeddings for the V3 counterfactual groups.

DEVELOPMENT ONLY.  NOT CLAIM BEARING.

The reference encoder is the frozen encoder of the accepted normalised-state
predictor baseline -- the same stack ``run_go2_representation_qualification_probe_v1``
loads, whose encoder never moved.  These embeddings are computed **once**, before
training, and are used for two things during the run:

* to build ``Q_ij = softmax_j(cos(t_bar_i, t_bar_j) / tau_t)``; and
* as the **target columns** of the matching logits
  ``S_ij = cos(p_i, t_bar_j) / tau_p``.

Both sides of the matching term are therefore anchored to a representation that
does not move.  If ``Q`` were recomputed from the moving EMA encoder, an encoder
drifting toward action-invariance would flatten ``Q`` toward uniform and lower
its own matching loss without improving action discrimination -- the objective
would supervise itself into the failure mode it exists to detect.

Every V3 frame is read through ``preprocess_v3_frame_v1``, which applies the
registered 224 -> 168 centre crop before the 112x112 resize.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import lewm.models.direct_egocentric_bev_state_jepa_v1 as _preload  # noqa: F401,E402

from lewm.datasets import go2_v3_counterfactual_branches_v1 as V3  # noqa: E402
from scripts import run_go2_observability_ceiling_assay_v1 as ceiling  # noqa: E402
from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402

OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_v3_frozen_reference_embeddings_v1"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
TOKEN_DIM = 192


def normalised_state(tokens: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(tokens, (TOKEN_DIM,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--role", default="train",
                    help="V3 role to cache; the development-selection split is "
                         "carved from the train role only")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    encoder, _dec, _head = P.load_stack(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()

    ledger = ceiling.AccessLedgerV1()
    groups = V3.load_branch_groups_v1(args.role, ceiling, ledger)
    train, selection = V3.split_branch_groups_v1(groups)

    successor = torch.empty(len(groups), V3.BRANCHES_PER_GROUP, 256, TOKEN_DIM)
    current = torch.empty(len(groups), 256, TOKEN_DIM)
    for index, group in enumerate(groups):
        batch = torch.stack(
            [V3.preprocess_v3_frame_v1(p) for p in group.successor_paths]
        ).to(device)
        with torch.no_grad():
            successor[index] = normalised_state(
                encoder.forward_tokens(batch)[:, 1:, :]
            ).cpu()
            current[index] = normalised_state(
                encoder.forward_tokens(
                    V3.preprocess_v3_frame_v1(group.current_path)[None].to(device)
                )[:, 1:, :]
            )[0].cpu()

    # Group-level flattened unit vectors: the cosine convention used by Q and S.
    flat = F.normalize(successor.reshape(len(groups), V3.BRANCHES_PER_GROUP, -1), dim=-1)

    payload = {
        "status": STATUS,
        "claim_bearing": False,
        "role": args.role,
        "reference_checkpoint": str(P.CHECKPOINT),
        "reference_checkpoint_sha256": hashlib.sha256(
            Path(P.CHECKPOINT).read_bytes()
        ).hexdigest(),
        "crop": {"ratio": V3.V3_CENTRE_CROP_RATIO,
                 "v3_native_height_px": V3.V3_NATIVE_HEIGHT_PX,
                 "cropped_height_px": V3.WPE_NATIVE_HEIGHT_PX,
                 "encoder_input_px": V3.ENCODER_INPUT_PX},
        "split_seed": V3.SPLIT_SEED_V1,
        "state_ids": [g.state_id for g in groups],
        "scene_ids": [g.scene_id for g in groups],
        "families": [g.family for g in groups],
        "train_state_ids": [g.state_id for g in train],
        "selection_state_ids": [g.state_id for g in selection],
        "successor_normalised_tokens": successor,
        "successor_flat_unit": flat,
        "current_normalised_tokens": current,
        "commands": torch.stack([g.commands for g in groups]),
    }
    path = OUT / f"frozen_reference_{args.role}.pt"
    torch.save(payload, path)

    summary = {
        "status": STATUS,
        "role": args.role,
        "groups": len(groups),
        "train_groups": len(train),
        "selection_groups": len(selection),
        "train_scenes": len({g.scene_id for g in train}),
        "selection_scenes": len({g.scene_id for g in selection}),
        "families_train": len({g.family for g in train}),
        "families_selection": len({g.family for g in selection}),
        "cache_path": str(path),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "successor_shape": list(successor.shape),
        "wall_seconds": round(time.time() - started, 2),
    }
    (OUT / f"summary_{args.role}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
