"""Primitive command bank construction — single source of truth.

Moved verbatim (behaviour-locked) from
``scripts/benchmark_lewm_closed_loop_mpc.py`` (``_primitive_active_blocks`` and
``_candidate_action_tensor``). ``candidate_action_tensor`` is pure
(numpy/torch/itertools only); ``active_blocks`` lazily imports the
``lewm_genesis`` primitive contract so this module stays importable without the
simulator for unit tests.
"""
from __future__ import annotations

import itertools
import random
from typing import Any

import numpy as np
import torch

from lewm.actions import encode_active_block


def active_blocks(registry: Any, primitive_names: list[str]) -> dict[str, np.ndarray]:
    """Encode each named primitive's first command block to an active-block vector."""
    from lewm_genesis.lewm_contract import expand_primitive_to_block  # lazy: avoids genesis at import

    encoded: dict[str, np.ndarray] = {}
    for name in primitive_names:
        matrix = expand_primitive_to_block(registry, name)
        encoded[name] = encode_active_block(matrix[:, 0], matrix[:, 1], matrix[:, 2])
    return encoded


def candidate_action_tensor(
    primitive_blocks: dict[str, np.ndarray],
    primitive_names: list[str],
    horizon: int,
    *,
    max_candidates: int | None,
    rng: random.Random,
    device: torch.device,
) -> tuple[list[tuple[str, ...]], torch.Tensor]:
    """Enumerate horizon-length primitive sequences and stack their action blocks.

    Deterministic given ``rng``: when ``max_candidates`` truncates the full
    product, the same seed selects the same subset (the behaviour-lock relies on
    this).
    """
    all_sequences = list(itertools.product(primitive_names, repeat=int(horizon)))
    if max_candidates is not None and len(all_sequences) > int(max_candidates):
        all_sequences = rng.sample(all_sequences, int(max_candidates))
    actions = np.stack(
        [
            np.stack([primitive_blocks[name] for name in seq], axis=0)
            for seq in all_sequences
        ],
        axis=0,
    )
    return all_sequences, torch.from_numpy(actions).float().to(device)
