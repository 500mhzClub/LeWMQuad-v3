"""Unit tests for ``_summarize_marginal_isotropy``.

Lives in its own module — no Genesis or ROS dependency — so the per-scene
marginal-isotropy audit described in
``docs/lejepa_identifiability_analysis.md`` can be CI-tested independent of
the heavy rollout fixture in ``test_rollout.py``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from lewm_genesis.rollout import _summarize_marginal_isotropy


def test_marginal_isotropy_distinguishes_concentrated_vs_uniform():
    """A single-cell single-yaw collector must rank strictly below a
    uniformly-spread one on both yaw entropy and primitive entropy. This
    is the load-bearing claim for the audit.
    """

    yaw_bins = 12
    concentrated_cells: dict[int, np.ndarray] = {}
    spread_cells: dict[int, np.ndarray] = {}
    for cell_id in range(4):
        c_hist = np.zeros(yaw_bins, dtype=np.int64)
        c_hist[0] = 8
        concentrated_cells[cell_id] = c_hist
        spread_cells[cell_id] = np.ones(yaw_bins, dtype=np.int64)

    summary = _summarize_marginal_isotropy(
        block_counts={"concentrated": 32, "spread": 32 * yaw_bins},
        cell_yaw_counts={
            "concentrated": concentrated_cells,
            "spread": spread_cells,
        },
        primitive_counts={
            "concentrated": Counter({"forward_medium": 32}),
            "spread": Counter(
                {
                    "forward_medium": 8,
                    "arc_left": 8,
                    "arc_right": 8,
                    "yaw_left": 8,
                    "yaw_right": 8,
                }
            ),
        },
        yaw_bins=yaw_bins,
    )

    conc = summary["concentrated"]
    spread = summary["spread"]

    assert conc["distinct_cells"] == 4
    assert conc["distinct_cell_yaw_bins"] == 4
    assert spread["distinct_cell_yaw_bins"] == 4 * yaw_bins
    assert conc["mean_per_cell_yaw_entropy_nats"] == 0.0
    assert (
        spread["mean_per_cell_yaw_entropy_nats"]
        > 0.99 * conc["max_per_cell_yaw_entropy_nats"]
    )
    assert conc["primitive_entropy_nats"] == 0.0
    assert spread["primitive_entropy_nats"] > 1.0


def test_marginal_isotropy_handles_empty_collector():
    """A collector that drew zero episodes yields well-defined zeros."""

    assert (
        _summarize_marginal_isotropy(
            block_counts={},
            cell_yaw_counts={},
            primitive_counts={},
            yaw_bins=12,
        )
        == {}
    )

    summary = _summarize_marginal_isotropy(
        block_counts={"ou_noise": 0},
        cell_yaw_counts={"ou_noise": {}},
        primitive_counts={"ou_noise": Counter()},
        yaw_bins=12,
    )
    iso = summary["ou_noise"]
    assert iso["blocks"] == 0
    assert iso["distinct_cells"] == 0
    assert iso["distinct_cell_yaw_bins"] == 0
    assert iso["mean_per_cell_yaw_entropy_nats"] == 0.0
    assert iso["primitive_entropy_nats"] == 0.0


def test_marginal_isotropy_max_entropy_matches_yaw_bin_count():
    """``max_per_cell_yaw_entropy_nats`` must be ``log(yaw_bins)`` — the
    aggregator in ``audit_jepa_corpus.py`` depends on this being a stable
    upper bound for normalization.
    """

    summary = _summarize_marginal_isotropy(
        block_counts={"x": 1},
        cell_yaw_counts={"x": {0: np.array([1] * 8, dtype=np.int64)}},
        primitive_counts={"x": Counter({"forward_medium": 1})},
        yaw_bins=8,
    )
    assert summary["x"]["max_per_cell_yaw_entropy_nats"] == float(np.log(8))
