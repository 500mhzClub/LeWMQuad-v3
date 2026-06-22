#!/usr/bin/env python3
"""Add an exact per-frame body egomotion field to Go2 memory dataset rows.

The controller's `integrated_body_motion_block/window` fields are fixed-window
integrals that under-capture the true inter-frame egomotion through the sharp
turn-away maneuvers that put landmarks out of frame; this corrupts vector-memory
propagation and was the sole cause of the out-of-frame steering failure (see
docs/lewm_go2_jepa_substrate_memory_update_2026-06-20.md, 2026-06-22).

This emits `exact_body_motion = [dx_m, dy_m, dyaw_rad]` for each row: the true
egomotion from the previous row's body frame into this row's, recovered as the 2D
rigid transform that aligns the static-landmark constellation (each landmark's
body position = range_m * [cos, sin](bearing_body_rad)) between consecutive rows.
This is the ground-truth proprioceptive egomotion the robot measures onboard at
deployment; it is recovered from landmark geometry only because this dataset lacks
a per-frame pose log. The convention matches
`train_go2_rgb_jepa_vector_memory_controller._propagate_vectors`:
    v_t = R(-dyaw) @ (v_{t-1} - [dx, dy])
and the units are meters (the controller divides translation by
--motion-translation-scale-m, like the existing motion fields).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_go2_rgb_jepa_vector_memory_controller import (  # noqa: E402
    _finite_float,
    _landmark_by_id,
    _seq_key,
)


def _body_positions(lbid: dict) -> dict[str, np.ndarray]:
    out = {}
    for oid, lm in lbid.items():
        b = _finite_float(lm.get("bearing_body_rad"), float("nan"))
        r = _finite_float(lm.get("range_m"), float("nan"))
        if math.isnan(b) or math.isnan(r):
            continue
        out[oid] = np.array([r * math.cos(b), r * math.sin(b)], dtype=np.float64)
    return out


def solve_egomotion(prev_lbid: dict, cur_lbid: dict) -> list[float] | None:
    """Returns [dx_m, dy_m, dyaw_rad] s.t. cur = R(-dyaw)(prev - [dx,dy]), or None."""
    pp = _body_positions(prev_lbid)
    cp = _body_positions(cur_lbid)
    common = sorted(set(pp) & set(cp))
    if len(common) < 2:
        return None
    P = np.stack([pp[o] for o in common])  # prev body positions
    Q = np.stack([cp[o] for o in common])  # cur body positions
    pb = P.mean(0)
    qb = Q.mean(0)
    Pc = P - pb
    Qc = Q - qb
    # Q ~= R(theta) P  (Kabsch, 2D)
    theta = math.atan2(
        float((Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0]).sum()),
        float((Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1]).sum()),
    )
    c, s = math.cos(theta), math.sin(theta)
    Rp = np.array([[c, -s], [s, c]])
    t_world = pb - Rp.T @ qb
    return [float(t_world[0]), float(t_world[1]), float(-theta)]


def process_file(src: Path, dst: Path) -> tuple[int, int]:
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    by_seq: dict = defaultdict(list)
    for i, row in enumerate(rows):
        by_seq[_seq_key(row)].append(i)
    n_solved = 0
    n_pairs = 0
    for _, idxs in by_seq.items():
        idxs.sort(key=lambda i: int(rows[i].get("episode_step", 0)))
        prev_lbid = None
        for pos, i in enumerate(idxs):
            cur_lbid = _landmark_by_id(rows[i])
            if pos == 0 or prev_lbid is None:
                rows[i]["exact_body_motion"] = [0.0, 0.0, 0.0]
            else:
                n_pairs += 1
                delta = solve_egomotion(prev_lbid, cur_lbid)
                if delta is None:
                    rows[i]["exact_body_motion"] = [0.0, 0.0, 0.0]
                else:
                    rows[i]["exact_body_motion"] = delta
                    n_solved += 1
            prev_lbid = cur_lbid
    dst.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return n_solved, n_pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_solved = total_pairs = 0
    for src in args.inputs:
        dst = args.output_dir / src.name
        solved, pairs = process_file(src, dst)
        total_solved += solved
        total_pairs += pairs
        print(f"{src.name}: {solved}/{pairs} pairs solved -> {dst}")
    print(f"TOTAL: {total_solved}/{total_pairs} consecutive pairs solved "
          f"({total_solved / max(1, total_pairs):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
