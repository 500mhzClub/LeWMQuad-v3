#!/usr/bin/env python
"""Append an interpreted findings section for one checkpoint to an ongoing report.

Reads whatever eval JSONs exist (decomposition, MPC proxy, bare-L2 nav, energy-head
nav, energy-head ckpt) and writes a markdown section + a machine-readable TSV row,
comparing against the previous checkpoint and the 2.8%-data e3 reference. Missing
evals are reported as n/a rather than failing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch

# 2.8%-data e3 model (prior best) reference, N=256 decomp + bare-L2 nav.
E3_REF = {
    "h10_free_pers": 0.59, "h10_zero_free": 0.106, "h10_free_tf": 4.13,
    "h8_zero_free": 0.095, "navL2_prog": -0.286, "navL2_succ": 0.0,
}


def _load(p):
    p = Path(p)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _byh(decomp):
    return {x["horizon"]: x for x in decomp["horizons"]} if decomp else {}


def _nav_prog(nav):
    if not nav:
        return None, None, None, None
    a = nav["aggregate"]
    l, r = a["lewm"], a["random"]
    return (l["mean_initial_distance_m"] - l["mean_final_distance_m"], l["success_rate"],
            r["mean_initial_distance_m"] - r["mean_final_distance_m"], r["success_rate"])


def arrow(cur, prev, better_lower=False, eps=1e-6):
    if cur is None or prev is None:
        return ""
    d = cur - prev
    if abs(d) < eps:
        return "≈"
    up = d > 0
    good = (not up) if better_lower else up
    return ("↑" if up else "↓") + ("✓" if good else "✗")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--decomp")
    ap.add_argument("--mpc")
    ap.add_argument("--nav-l2")
    ap.add_argument("--nav-head")
    ap.add_argument("--head-ckpt")
    ap.add_argument("--report", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--epoch-summary", default="", help="optional trainer eval-summary line")
    args = ap.parse_args()

    dec = _byh(_load(args.decomp))
    mpc = _byh(_load(args.mpc))
    navL2 = _load(args.nav_l2)
    navH = _load(args.nav_head)

    m = {}
    if dec:
        m["h10_free_pers"] = dec[10]["point_rollout_over_persistence"]
        m["h10_zero_free"] = dec[10]["point_zero_minus_rollout"]
        m["h10_free_tf"] = dec[10]["point_rollout_over_teacher_forced"]
        m["h8_zero_free"] = dec[8]["point_zero_minus_rollout"]
        m["h1_free_pers"] = dec[1]["point_rollout_over_persistence"]
    if mpc:
        hk = 5 if 5 in mpc else max(mpc)
        m["mpc_vs0"] = mpc[hk]["recorded_win_rate_vs_zero"]
        m["mpc_rec0"] = mpc[hk]["recorded_over_zero"]
        m["mpc_h"] = hk
    pL2, sL2, pR, sR = _nav_prog(navL2)
    m["navL2_prog"], m["navL2_succ"], m["rand_prog"], m["rand_succ"] = pL2, sL2, pR, sR
    pH, sH, _, _ = _nav_prog(navH)
    m["navH_prog"], m["navH_succ"] = pH, sH
    if args.head_ckpt and Path(args.head_ckpt).exists():
        try:
            m["head_acc"] = float(torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
                                  .get("best_eval_ranking_acc", float("nan")))
        except Exception:
            m["head_acc"] = None

    prev = _load(args.state) or {}
    ts = dt.datetime.now().strftime("%m-%d %H:%M")

    def g(k):
        v = m.get(k)
        return v if v is not None else None

    def fmt(v, spec="+.3f"):
        return "n/a" if v is None else format(v, spec)

    # ---- markdown findings section ----
    L = [f"\n## {args.name}  ·  {ts}"]
    if args.epoch_summary:
        L.append(f"<sub>{args.epoch_summary}</sub>\n")
    L.append("| metric | this | prev | e3(2.8%) | read |")
    L.append("|---|--:|--:|--:|---|")
    L.append(f"| action-cond `zero−free` @h10 | {fmt(g('h10_zero_free'))} "
             f"{arrow(g('h10_zero_free'), prev.get('h10_zero_free'))} | "
             f"{fmt(prev.get('h10_zero_free'))} | +0.106 | >0 = real action beats zero |")
    L.append(f"| `zero−free` @h8 | {fmt(g('h8_zero_free'))} | {fmt(prev.get('h8_zero_free'))} | +0.095 | |")
    L.append(f"| `free/persistence` @h10 | {fmt(g('h10_free_pers'),'.2f')} "
             f"{arrow(g('h10_free_pers'), prev.get('h10_free_pers'), better_lower=True)} | "
             f"{fmt(prev.get('h10_free_pers'),'.2f')} | 0.59 | <1 beats 'nothing changes' |")
    L.append(f"| compounding `free/TF` @h10 | {fmt(g('h10_free_tf'),'.2f')}× | "
             f"{fmt(prev.get('h10_free_tf'),'.2f')}× | 4.13× | target 1–3× |")
    L.append(f"| MPC win-rate vs-zero @h{m.get('mpc_h','?')} | {fmt(g('mpc_vs0'),'.2f')} | "
             f"{fmt(prev.get('mpc_vs0'),'.2f')} | 0.61 | >0.5 prefers true action |")
    L.append(f"| nav (bare L2) progress / succ | {fmt(g('navL2_prog'),'.3f')}m / {fmt(g('navL2_succ'),'.0%') if g('navL2_succ') is not None else 'n/a'} "
             f"{arrow(g('navL2_prog'), prev.get('navL2_prog'))} | {fmt(prev.get('navL2_prog'),'.3f')}m | −0.286m | vs random {fmt(g('rand_prog'),'.3f')}m |")
    L.append(f"| nav (energy head, pure-perc) | {fmt(g('navH_prog'),'.3f')}m / {fmt(g('navH_succ'),'.0%') if g('navH_succ') is not None else 'n/a'} | "
             f"{fmt(prev.get('navH_prog'),'.3f')}m | — | head rank-acc {fmt(g('head_acc'),'.2f')} |")

    # ---- auto interpretation bullets ----
    L.append("\n**Findings:**")
    zf = g("h10_zero_free")
    if zf is not None:
        d = "" if prev.get("h10_zero_free") is None else f" ({'up' if zf>prev['h10_zero_free'] else 'down'} from {prev['h10_zero_free']:+.3f})"
        L.append(f"- Action-conditioning {'holds (real action beats zero)' if zf>0 else 'SHORTCUT (zero ≥ real)'} at 5 s: `zero−free`={zf:+.3f}{d}; vs e3 ref +0.106.")
    fp = g("h10_free_pers")
    if fp is not None:
        L.append(f"- Long-horizon prediction: {'beats' if fp<1 else 'LOSES to'} persistence at 5 s by {(1-fp)*100:.0f}% (free/pers {fp:.2f}); compounding {fmt(g('h10_free_tf'),'.1f')}×.")
    if g("mpc_vs0") is not None:
        v = g("mpc_vs0")
        L.append(f"- Closed-loop: prefers the true action {v:.0%} of the time {'(>chance)' if v>0.5 else '(≈/below chance)'}, terminal-cost ratio {fmt(g('mpc_rec0'),'.2f')}.")
    if pL2 is not None:
        rel = "beats" if (pR is None or pL2 > pR) else "below"
        L.append(f"- Demo nav (bare clean planner): progress {pL2:+.3f}m, success {sL2:.0%} — {rel} random ({fmt(pR,'.3f')}m).")
    if pH is not None:
        L.append(f"- Demo nav (energy head, pure perception): progress {pH:+.3f}m, success {sH:.0%}.")
    # verdict
    verdict = "scaling signal: "
    parts = []
    if zf is not None and prev.get("h10_zero_free") is not None:
        parts.append("action-cond " + ("↑" if zf > prev["h10_zero_free"] else "↓/flat"))
    if fp is not None and prev.get("h10_free_pers") is not None:
        parts.append("prediction " + ("↑" if fp < prev["h10_free_pers"] else "↓/flat"))
    navbest = max([x for x in (pL2, pH) if x is not None], default=None)
    if navbest is not None:
        parts.append("nav " + ("beats random" if (pR is not None and navbest > pR) else "≤ random"))
    L.append(f"- **Verdict:** {verdict + ', '.join(parts) if parts else 'first datapoint logged.'}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "a") as f:
        f.write("\n".join(L) + "\n")

    # TSV row
    tsv = Path(args.tsv)
    if not tsv.exists():
        tsv.write_text("checkpoint\th10_zero_free\th10_free_pers\th10_free_tf\tmpc_vs0\tnavL2_prog\tnavL2_succ\tnavH_prog\tnavH_succ\thead_acc\ttime\n")
    with open(tsv, "a") as f:
        f.write("\t".join(str(x) for x in [
            args.name, fmt(g('h10_zero_free')), fmt(g('h10_free_pers'), '.2f'), fmt(g('h10_free_tf'), '.2f'),
            fmt(g('mpc_vs0'), '.2f'), fmt(g('navL2_prog'), '.3f'), g('navL2_succ'),
            fmt(g('navH_prog'), '.3f'), g('navH_succ'), fmt(g('head_acc'), '.2f'), ts]) + "\n")

    # save state for next comparison (only real numbers)
    Path(args.state).write_text(json.dumps({k: v for k, v in m.items() if v is not None}))
    print(f"appended findings for {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
