#!/usr/bin/env python3
"""Predictor with corrected action conditioning and optional proprioceptive tokens.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

This is a NEW module.  ``run_dev_v03_temporal_action_jepa_v1.Predictor`` is left
untouched so the frozen H=1-4 result stays exactly reproducible.

Two differences from the frozen predictor, and no others:

1. **Action** is the 15-D five-tick post-slew command trajectory instead of the
   12-D primitive one-hot plus nominal triple.  Conditioning is unchanged
   AdaLN-Zero over every block -- context structure, positional treatment and
   endpoint target construction are all preserved, as required.
2. **Control history** (efference copy) enters as ONE token per context slot in
   **every cell**, RGB included.  It is the robot's own past applied command, not
   sensed state, so leaving it inside the proprioceptive tensor would have
   confounded the proprioception factor with control history.  Because an applied
   command is a deterministic function of the action plan, it stays available at
   every rollout horizon and needs no validity mask.
3. **Proprioception** (the experimental factor, optional) enters as ONE token per
   context slot, formed from the five trailing 10 Hz samples of that slot.  It is
   SENSED PHYSICAL STATE ONLY: projected gravity, gyroscope, joint positions,
   joint velocities.  A slot whose frame is a *prediction* rather than an
   observation receives a learned ``absent`` token and is marked invalid.
   Proprioception is an input only: the target stays visual and there is no
   proprioceptive loss.

Seed pairing
------------
``build_paired`` constructs the shared parameters first, from the seed, in an
order that does not depend on ``use_proprio``; proprioception-specific
parameters are then initialised deterministically from a *separate* generator
derived from the same seed.  An RGB cell and a proprio cell built with the same
seed therefore hold bit-identical shared weights.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import dev_action_slew_reconstruction_v1 as SLEW  # noqa: E402
from scripts import build_dev_v03_proprio_action_manifest_v1 as M  # noqa: E402

STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

TOKENS = T.TOKENS
TOKEN_DIM = T.TOKEN_DIM
CONTEXT_POSITIONS = T.CONTEXT_POSITIONS
ACTION_DIM = SLEW.ACTION_DIM          # 15
PROPRIO_DIM = M.PROPRIO_DIM           # 30, sensed physical state only
CONTROL_DIM = M.CONTROL_DIM           # 2 per sample
CONTROL_SLOT_DIM = M.CONTROL_SLOT_DIM  # 10 per slot (5 samples x 2)
SAMPLES_PER_SLOT = M.SAMPLES_PER_SLOT  # 5

PROPRIO_SEED_OFFSET = 7_919            # a fixed, documented offset


class ProprioActionPredictor(nn.Module):
    """Frozen-architecture predictor + corrected action + optional proprio tokens."""

    def __init__(self, token_dim=TOKEN_DIM, width=384, depth=6, heads=6,
                 use_proprio: bool = False):
        super().__init__()
        self.width = width
        self.use_proprio = use_proprio

        # ---- shared parameters, in an order independent of use_proprio -------
        self.input = nn.Linear(token_dim, width)
        self.output = nn.Linear(width, token_dim)
        self.spatial = nn.Parameter(torch.zeros(1, TOKENS, width))
        self.temporal = nn.Parameter(torch.zeros(CONTEXT_POSITIONS + 1, 1, width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.spatial, std=0.02)
        nn.init.trunc_normal_(self.temporal, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.action = nn.Sequential(
            nn.Linear(ACTION_DIM, width), nn.SiLU(), nn.Linear(width, width)
        )
        # Control history is SHARED: present in every cell, so it cannot confound
        # the proprioception factor.  It is created among the shared parameters,
        # before any modality-specific parameter, so the shared RNG stream is
        # identical whatever ``use_proprio`` is.
        self.control_in = nn.Linear(CONTROL_SLOT_DIM, width)
        self.control_modality = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.control_modality, std=0.02)
        self.blocks = nn.ModuleList([T.PredictorBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

        # ---- proprioception-specific parameters, created last ---------------
        if use_proprio:
            self.proprio_in = nn.Linear(PROPRIO_DIM, width)
            self.proprio_modality = nn.Parameter(torch.zeros(1, 1, width))
            self.proprio_absent = nn.Parameter(torch.zeros(1, 1, width))

    def initialise_proprio(self, seed: int) -> None:
        """Deterministic init of the proprio parameters from a separate stream."""
        if not self.use_proprio:
            return
        generator = torch.Generator().manual_seed(int(seed) + PROPRIO_SEED_OFFSET)
        with torch.no_grad():
            weight = torch.empty_like(self.proprio_in.weight)
            nn.init.trunc_normal_(weight, std=0.02, generator=generator)
            self.proprio_in.weight.copy_(weight)
            self.proprio_in.bias.zero_()
            for parameter in (self.proprio_modality, self.proprio_absent):
                value = torch.empty_like(parameter)
                nn.init.trunc_normal_(value, std=0.02, generator=generator)
                parameter.copy_(value)

    # ----------------------------------------------------------------------
    def forward(self, context, action, mask, proprio=None, proprio_valid=None,
                control=None):
        """context (B,3,N,D); action (B,10); mask (B,N) True=predict.

        ``control``       (B, 3, 5, 2) efference copy per slot -- REQUIRED, all cells
        ``proprio``       (B, 3, 5, 30) trailing sensed samples per slot, or None
        ``proprio_valid`` (B, 3) bool, False -> the learned absence token
        """
        b, t, n, _ = context.shape
        x = self.input(context.reshape(b * t, n, -1)).reshape(b, t, n, self.width)
        x = x + self.spatial.unsqueeze(1) + self.temporal[:CONTEXT_POSITIONS].unsqueeze(0)
        x = x.reshape(b, t * n, self.width)

        query = self.mask_token.expand(b, n, -1) + self.spatial + self.temporal[CONTEXT_POSITIONS]

        parts = [x]

        if control is None:
            raise ValueError("control history is required in every cell")
        if control.shape[1] != t or control.shape[2] != SAMPLES_PER_SLOT:
            raise ValueError(
                f"control must be (B,{t},{SAMPLES_PER_SLOT},{CONTROL_DIM}); "
                f"got {tuple(control.shape)}")
        control_tokens = self.control_in(control.reshape(b, t, CONTROL_SLOT_DIM))
        parts.append(control_tokens + self.control_modality
                     + self.temporal[:CONTEXT_POSITIONS].squeeze(1))

        if self.use_proprio:
            if proprio is None:
                tokens = self.proprio_absent.expand(b, t, -1)
            else:
                if proprio.shape[1] != t or proprio.shape[2] != SAMPLES_PER_SLOT:
                    raise ValueError(
                        f"proprio must be (B,{t},{SAMPLES_PER_SLOT},{PROPRIO_DIM}); "
                        f"got {tuple(proprio.shape)}")
                if proprio_valid is None:
                    raise ValueError("proprio_valid is required whenever proprio is given")
                # Hard-gate the INPUT, not just the output: multiplying a masked
                # activation by zero would still propagate NaN/inf, so an invalid
                # slot's contents must never reach the projection at all.  This is
                # what makes "content of an invalid slot is inert" an exact
                # identity rather than an approximate one.
                gate = proprio_valid.unsqueeze(-1).unsqueeze(-1)
                safe = torch.where(gate, proprio, torch.zeros_like(proprio))
                pooled = self.proprio_in(safe).mean(dim=2)             # (B, t, W)
                tokens = torch.where(proprio_valid.unsqueeze(-1), pooled,
                                     self.proprio_absent.expand(b, t, -1))
            tokens = tokens + self.proprio_modality + self.temporal[:CONTEXT_POSITIONS].squeeze(1)
            parts.append(tokens)
        parts.append(query)

        sequence = torch.cat(parts, dim=1)
        conditioning = self.action(action)
        for block in self.blocks:
            sequence = block(sequence, conditioning)
        return self.output(self.norm(sequence[:, -n:]))


def build_paired(seed: int, use_proprio: bool, width=384, depth=6, heads=6):
    """Construct a cell's predictor so shared weights are seed-identical.

    The shared parameters are drawn from the global stream seeded by ``seed`` in
    an order that does not depend on ``use_proprio``; the proprio parameters come
    from their own generator afterwards.
    """
    torch.manual_seed(int(seed))
    model = ProprioActionPredictor(width=width, depth=depth, heads=heads,
                                   use_proprio=use_proprio)
    model.initialise_proprio(seed)
    return model


def shared_parameter_names(model: ProprioActionPredictor):
    return [name for name, _ in model.named_parameters()
            if not name.startswith("proprio_")]


def rollout_validity(horizon_step: int, slots: int = CONTEXT_POSITIONS):
    """Which context slots hold OBSERVED frames at rollout step ``horizon_step``.

    Step 1 consumes three observed frames.  Each further step slides the window
    by one and appends a prediction, so the count of observed slots falls
    3, 2, 1, 0 across steps 1..4.  This is the sole mechanism by which
    proprioception is withheld at horizon, and it needs no future tensor to exist.
    """
    if horizon_step < 1:
        raise ValueError("horizon_step is 1-based")
    observed = max(slots - (horizon_step - 1), 0)
    return [index < observed for index in range(slots)]


@torch.no_grad()
def control_slot_from_action(action_block):
    """The control slot of the frame a rollout step appends, from the action alone.

    The appended frame sits at step ``s + 5h``; its trailing five samples carry
    ``applied[k-1]`` for ``k`` in ``[s+5h-4 .. s+5h]``, i.e. the applied command at
    steps ``[s+5h-5 .. s+5h-1]`` -- exactly the five ticks of action block ``h-1``.
    So the efference copy is a deterministic function of the action plan and
    involves no measurement and no future observation.
    """
    b = action_block.shape[0]
    return action_block.reshape(b, 1, SAMPLES_PER_SLOT, CONTROL_DIM)


def unroll(model, context, action_blocks, proprio=None, control=None, max_h: int = 4,
           _future_fill: float | None = None):
    """Autoregressive rollout over the fixed sliding three-frame window.

    ``proprio`` holds ONLY the observed history, (B, 3, 5, 33).  At each rollout
    step the proprio window slides exactly as the image window does: the oldest
    slot is dropped and the newly appended slot -- which holds a *prediction* --
    is marked invalid, so it takes the learned absence token.  The tensor only
    ever loses entries, so no future proprioceptive value is required to exist
    and none can be read.

    ``_future_fill`` is a TEST HOOK.  It writes an arbitrary value into the slot
    appended for a predicted frame -- i.e. it simulates somebody supplying
    future proprioception at rollout.  Because that slot is marked invalid, the
    value must be inert; ``test_injected_future_proprioception_is_inert`` asserts
    exactly that.  Nothing in training or evaluation sets it.
    """
    outputs = []
    slots = context.shape[1]
    device = context.device
    mask = torch.ones(context.shape[0], TOKENS, dtype=torch.bool, device=device)
    if control is None:
        raise ValueError("control history is required in every cell")
    c_window = control

    window = context
    if proprio is not None:
        p_window = proprio
        valid = torch.ones(context.shape[0], slots, dtype=torch.bool, device=device)
    else:
        p_window, valid = None, None

    frame_a, frame_b = window[:, 1], window[:, 2]
    previous = T.normalise(model(window, action_blocks[0], mask, p_window, valid, c_window))
    outputs.append(previous)

    for step in range(1, max_h):
        window = torch.stack([frame_a, frame_b, previous], dim=1)
        if p_window is not None:
            # slide: drop the oldest slot, append an invalid (predicted) slot
            appended = torch.zeros_like(p_window[:, :1])
            if _future_fill is not None:
                appended = torch.full_like(appended, _future_fill)
            p_window = torch.cat([p_window[:, 1:], appended], dim=1)
            valid = torch.cat(
                [valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
            expected = rollout_validity(step + 1, slots)
            if [bool(v) for v in valid[0].tolist()] != expected:
                raise AssertionError(
                    f"rollout step {step + 1}: validity {valid[0].tolist()} != {expected}")
        c_window = torch.cat(
            [c_window[:, 1:], control_slot_from_action(action_blocks[step - 1])], dim=1)
        previous = T.normalise(
            model(window, action_blocks[step], mask, p_window, valid, c_window))
        outputs.append(previous)
        frame_a, frame_b = frame_b, outputs[-2]
    return outputs


def normalise_proprio(proprio: torch.Tensor, stats: dict) -> torch.Tensor:
    mean = torch.tensor(stats["mean"], dtype=proprio.dtype, device=proprio.device)
    std = torch.tensor(stats["std"], dtype=proprio.dtype, device=proprio.device)
    return (proprio - mean) / std


def parameter_report(width=384, depth=6, heads=6) -> dict:
    frozen = T.Predictor(width=width, depth=depth, heads=heads)
    rgb = build_paired(0, use_proprio=False, width=width, depth=depth, heads=heads)
    prop = build_paired(0, use_proprio=True, width=width, depth=depth, heads=heads)
    total = lambda m: sum(p.numel() for p in m.parameters())
    return {
        "frozen_predictor_12d_primitive_action": total(frozen),
        "rgb_cells": total(rgb),
        "proprio_cells": total(prop),
        "action_head_delta_vs_frozen": total(rgb) - total(frozen),
        "proprio_delta_vs_rgb": total(prop) - total(rgb),
        "control_parameters_present_in_all_cells": {
            name: p.numel() for name, p in rgb.named_parameters()
            if name.startswith("control_")},
        "proprio_parameters": {name: p.numel() for name, p in prop.named_parameters()
                               if name.startswith("proprio_")},
        "tensor_dims": {"action": ACTION_DIM, "control_per_slot": CONTROL_SLOT_DIM,
                        "proprio_per_sample": PROPRIO_DIM,
                        "samples_per_slot": SAMPLES_PER_SLOT, "slots": CONTEXT_POSITIONS},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(parameter_report(), indent=2))
