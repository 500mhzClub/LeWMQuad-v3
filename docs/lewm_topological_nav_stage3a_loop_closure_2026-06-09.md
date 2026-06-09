# Topological Nav — Stage 3a: LoopClosureHead consumer gate (FAILED — reassess before Stage 3)

Date: 2026-06-09
Base: frozen `seq4_e9`; Stage 2 v6 BeliefEncoder (3 seeds); bank cache
`belief_banks_seq4_e9_train32.pt` (254 train / 32 `test_id` scenes, H=8).
Code: `lewm/models/loop_closure.py`, `scripts/train_loop_closure_head.py`,
`lewm/tests/test_loop_closure.py` (5/5).
Artifact: `.generated/topo_nav/loop_closure_gate_seq4_e9_v6.json` (+ `.log`).

## Question (registered in the Stage 2 doc "v6 + decision" BEFORE running)

The Stage 2 v6 encoder missed the retrieval gate by 0.006 on an unsaturated
curve. Rather than optimizing the proxy further, the Stage-2→3 decision moved to
the operating point the topological memory actually consumes (spec §5.3):
**loop-closure recall at ≥99% precision**, with Platt calibration and the
deployment threshold chosen on scene-disjoint calibration data.

Registered criteria: (a) spec §5.3 — eval precision ≥0.99 at the deployed
threshold and ECE ≤5% for any adopted representation; (b) consumer margin —
every belief seed beats the single-frame head's recall by ≥+5 pp.

## Method

Pairs = within-scene window pairs under the banks' three-bucket masks (same-cell
positive, BFS≥2 negative, adjacent ambiguous-excluded); eval = ALL valid pairs
of the 32 held-out scenes (n=273,826, **positive rate 2.5%**). Head = symmetric
pair-MLP (`[a*b, |a−b|]` → 128×2 MLP) trained with BCE (all positives + 4×
sampled negatives per scene per epoch) on 203 train scenes; Platt + threshold on
51 calibration scenes; cosine scorer reported as the no-head ablation.

## Result — gate FAILED, on every branch

| representation | AP | recall @ deployed thr (P≥0.99) | ECE | oracle R@P0.99 |
|---|---:|---:|---:|---:|
| single-frame (3 head seeds) | 0.207–0.210 | 0.003 | ≤0.002 | ≤0.009 |
| mean-pool (3 head seeds) | 0.215–0.216 | 0.006–0.009 | ≤0.002 | ≤0.009 |
| **belief v6 (3 enc seeds)** | **0.286–0.292** | **0.015–0.028** | ≤0.008 | ≤0.031 |
| cosine (no head), best | 0.279 | 0.021 | — | 0.030 |

Precision–recall tradeoff on the eval curve (head scorer, seed 20260609):

| | P≥0.99 | P≥0.95 | P≥0.90 | P≥0.80 | P≥0.50 |
|---|---:|---:|---:|---:|---:|
| belief v6 | 0.002 | 0.057 | 0.082 | 0.126 | 0.230 |
| single-frame | 0.009 | 0.017 | 0.028 | 0.067 | 0.148 |
| mean-pool | 0.009 | 0.020 | 0.030 | 0.073 | 0.158 |

Sequence-aggregation probe (training-free, raw latents, full 8×8 cross-window
frame-pair cosine grid — does evidence aggregation rescue verification?):

| scorer | P≥0.95 | P≥0.90 | P≥0.80 | P≥0.50 |
|---|---:|---:|---:|---:|
| terminal-frame cosine | 0.000 | 0.044 | 0.067 | 0.131 |
| window mean-of-grid | 0.017 | 0.049 | 0.076 | 0.136 |
| window max-of-grid | 0.027 | 0.046 | 0.097 | 0.164 |

## Reading

1. **The BeliefEncoder is confirmed as the best place code** — AP +0.08 over
   single-frame (+38% relative), 2–3× recall at *every* precision target, 3/3
   seeds, calibration excellent. The consumer ranking validates Stage 2's H2:
   adopt v6 as the place representation *whatever else changes*.
2. **But absolute pairwise place verification is weak at every operating
   point** — best case 23% recall at even 50% precision. This is NOT a
   99%-bar artifact (the curve is shallow everywhere), and naive evidence
   aggregation does not rescue it (≤0.164 @P0.50 — the Stage-1 "pooling can't
   do it" pattern again).
3. **Consequence: the §5.4 memory design is not buildable as specified.** The
   global novelty check (max LoopClosure over all memory at τ_new=0.70) and
   merge decisions assume usable single-pair verification at very high
   precision. At 2.5% base rate that demands FPR ~1e-4·recall — biometric-grade
   discrimination this substrate does not have for **any-yaw** same-cell pairs.
4. **Prime suspect: yaw, again.** The latent is heading-dominated (yaw R² 0.81
   vs pos 0.16). Retrieval@5 (0.64) looks far better than verification because
   retrieval is rank-based — *some* similar-yaw same-cell frame usually exists
   in the database top-5 — while verification must accept same-cell pairs at
   arbitrary relative yaw, exactly what a view-dominated code cannot certify.
   This is the same root cause as the goal-image yaw collapse (0.92→0.00 at
   90°) and the v3 §5.1 weak-positive design tension.

## Registered next probes (the reassessment, in order)

1. **Yaw-conditioned verification probe.** Rebuild banks with per-window
   terminal `yaw_bin`; measure the same PR curves restricted to same-yaw-bin
   pairs (positives AND negatives). Hypothesis: a large lift. If confirmed,
   redesign memory nodes as **(cell × yaw-bin) keyframes** — loop closure
   becomes yaw-conditioned; cross-yaw same-place association comes from graph
   structure (a pivot edge connects the bins), not visual verification. This
   *converges* with the already-identified goal-facing-keyframe requirement:
   one node design satisfies both constraints.
2. **Action-token + body-motion-aux BeliefEncoder pass** (spec §5.1/§3.4
   default inputs, still untried). Motion is the natural disambiguator the
   visual-only window lacks, and verification is where it should show up first.
3. **Filter-level evaluation instead of single-pair.** The deployed Bayes
   filter restricts candidates by the transition prior and accumulates evidence
   over steps — a rank-among-few problem, not open-set verification at 2.5%
   base rate. If probes 1–2 lift verification into a workable band, proceed to
   Stage 3b and gate on **replay trajectory coherence** (§5.5) with the §5.3
   single-pair bar re-registered (with rationale) as a node-commit-only check.
   If they do not, that is a genuine substrate ceiling for memory-building and
   the DINOv2/patch-feature fork returns to the table — for the *memory key*
   only (LeWM stays the dynamics/servoing base).

## Probe 1 result — yaw-conditioned verification (run 2026-06-09, same day)

`scripts/probe_loop_closure_yaw.py`: rebuilt the 32 eval banks with per-window
terminal `yaw_bin` (8 bins, already in `labels.jsonl`, dropped by the Stage 1/2
window selector) and re-measured verification under three pair scopes —
positives AND negatives both restricted (under a (cell × yaw-bin) node design
the negatives are different cells seen at the same heading: the classic aliased
corridors, so this is not a giveaway). Training-free cosine scorers (the Stage
3a head added little over cosine: AP 0.286 vs 0.279).
Artifact: `.generated/topo_nav/loop_closure_yaw_probe_seq4_e9.json`; eval-bank
cache `belief_banks_yaw_eval.pt`.

| representation | scope | n pairs | base | AP | R@P95 | R@P90 | R@P80 | R@P50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| single-frame | all | 273,826 | .025 | 0.182 | 0.000 | 0.044 | 0.067 | 0.131 |
| single-frame | **same-yaw** | 35,506 | .048 | **0.501** | 0.001 | 0.174 | 0.272 | 0.500 |
| belief v6 (3 seeds) | all | 273,826 | .025 | 0.279–0.286 | 0.04–0.05 | 0.08 | 0.12 | 0.23 |
| belief v6 (3 seeds) | **same-yaw** | 35,506 | .048 | **0.620–0.637** | 0.12–0.16 | 0.25–0.29 | 0.39–0.40 | 0.64–0.67 |
| belief v6 (3 seeds) | adjacent-yaw (±45°) | 103,945 | .035 | 0.450–0.461 | 0.07–0.10 | 0.14–0.16 | 0.23 | 0.41–0.42 |

**Reading: yaw is confirmed as the dominant limiter.** Same-yaw conditioning
more than doubles AP (0.28→0.63 belief; 0.18→0.50 single-frame) and triples
recall at P≥0.90, consistently across all 3 encoder seeds, while the base rate
only doubles — not a base-rate artifact. The graceful adjacent-yaw degradation
says 45° bins are about right.

**But the registered usable band (R≥0.3 @ P≥0.95) is not yet reached**
(0.12–0.16) — with an important caveat: **v6 was trained yaw-INVARIANT** (its
supcon pulls any-yaw same-cell pairs together), which actively *fights* the
yaw-conditioned operating point. The encoder is being asked to verify a
distinction its objective explicitly erased.

**→ v7 (registered before running): yaw-conditioned BeliefEncoder.**
`scripts/train_belief_encoder_yaw.py` retrains the identical v4/v6 config with
the spec §5.1 yaw scheme at the registered knob λ_yaw_weak→0: strong positive =
same (cell, yaw_bin); same-cell-different-yaw masked out (ambiguous-ignore);
negatives = BFS≥2 any-yaw (same-heading aliased corridors arrive as the hard
negatives naturally). Evaluation = this same probe on the identical cached eval
banks (NOT retrieval R@5 — the proxy that misled Stage 2). **Bar: same-yaw
recall ≥ 0.3 at P≥0.95.** If v7 clears it → adopt **(cell × yaw-bin) keyframe
nodes** for the Stage 3 memory (converges with the goal-facing constraint;
cross-yaw association via pivot edges in the graph, not visual verification).
If v7 falls clearly short → yaw-selectivity alone is insufficient → probe #2
(action tokens + motion-aux) stacked on the yaw-conditioned objective.

## What this does NOT change

Level-3 local servoing (seq4 + `plan_cost`, visible goal-facing subgoals,
0.92/0.73/0.58) is unaffected — it never depended on loop closure. The
recognition-not-metric architecture decision stands; what is in question is the
*verification strength* of the place code, not the topological design.

## Reproduce

```bash
~/TinyQuadJEPA/bin/python lewm/tests/test_loop_closure.py
~/TinyQuadJEPA/bin/python scripts/train_loop_closure_head.py \
  --bank-cache .generated/topo_nav/belief_banks_seq4_e9_train32.pt \
  --belief-encoder-dir .generated/topo_nav/belief_encoder_seq4_e9_v6_train32_encoders \
  --output .generated/topo_nav/loop_closure_gate_seq4_e9_v6.json --device cuda
```
(~6 min on the R9700; banks cached, no genesis, no model load.)
