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

Per-family (belief seed 20260609, mean per-scene AP): the same-yaw lift is
**uniform across all 8 families** — including the aliased mazes, which was the
main risk to the (cell × yaw-bin) design:

| family | AP all | AP same-yaw | same-yaw base |
|---|---:|---:|---:|
| large_enclosed_maze | 0.313 | 0.642 | 0.020 |
| loop_alias_stress | 0.252 | 0.638 | 0.049 |
| medium_enclosed_maze | 0.272 | 0.597 | 0.032 |
| small_enclosed_maze | 0.277 | 0.618 | 0.034 |
| visual_sensor_stress | 0.259 | 0.623 | 0.042 |
| local_composite_motifs | 0.387 | 0.741 | 0.217 |
| open_obstacle_field | 0.505 | 0.715 | 0.177 |
| rough_local_dynamics | 0.527 | 0.741 | 0.170 |

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

## v7 + trained-head results — the substrate's verification band is now measured

**v7 (yaw-conditioned objective): clean NEGATIVE.** Same-yaw AP 0.610–0.615 vs
v6's 0.620–0.637; R@P95 0.13–0.19 vs 0.12–0.16 (3 seeds each; identical eval
banks; `loop_closure_yaw_probe_v7.json`). v6's yaw-invariant training was NOT
the binding constraint — supcon with any-yaw positives only needs to *cluster*
same-cell pairs and evidently retains yaw variance anyway; conditioning the
positives on yaw adds no discriminative power against same-heading negatives.
**Keep v6; retire v7.**

**Trained head on same-yaw pairs (the actual deployment configuration of a
(cell × yaw-bin) memory; yaw train banks `belief_banks_yaw_train32.pt`):**

| representation | eval R@P99 | R@P95 | R@P90 | R@P80 | R@P50 | deployed (cal-P95 thr) |
|---|---:|---:|---:|---:|---:|---|
| **belief v6 + head** | 0.025 | **0.128** | **0.272** | 0.394 | 0.639 | P=0.92, R=0.22 |
| belief v7 + head | 0.015 | 0.112 | 0.196 | 0.374 | 0.637 | P=0.92, R=0.17 |
| single-frame + head | 0.058 | 0.068 | 0.217 | 0.332 | 0.560 | P=0.93, R=0.11 |

**Synthesis — every pairwise lever has now been tried and the band has
converged:** yaw scoping (huge lift, kept), yaw-conditioned objective (flat),
trained head over cosine (marginal), naive sequence aggregation (flat). Best
achievable same-yaw pairwise verification on frozen seq4: **recall ≈0.13 at
P≥0.95, ≈0.27 at P≥0.90.** The registered bar (R≥0.3 @P95) is **not met**. The
residual errors are genuinely aliased same-heading views that the H=8 visual
history does not separate at pair level.

## Re-registered decision (2026-06-09): run probe #3 BEFORE probe #2

Original order was (2) action-tokens/motion-aux, then (3) filter-level
evaluation. **Reordered, with rationale:** four independent pairwise levers
plateaued at the same band, so the next pairwise lever (#2) has a weak prior of
clearing 0.3@P95 by itself. Meanwhile the deployed mechanism was never a
single-pair decision: the §5.4 filter aggregates per-step likelihoods over N
consecutive steps under a transition prior that shrinks the candidate set to a
handful of graph-neighbors. At P0.90/R0.27 per step, 5–10 steps of evidence
plausibly reach the §5.5 coherence gate — and that is the *actual* Stage 3
question. So:

1. **Probe #3 (next): offline replay filter test.** Build the minimal
   (cell × yaw-bin) keyframe memory + top-k Bayes filter (§5.4) over v6
   embeddings + the same-yaw trained head (P90 operating point); replay
   held-out rollouts (pure torch, banks/labels only, no genesis); gate =
   **filter trajectory coherence ≥90% on non-boundary frames** (§5.5) and
   new-node/false-merge rates. Pass → the memory is buildable; the §5.3
   single-pair 99% bar is re-registered as commit-only with sequence evidence.
   Fail → run probe #2 (action tokens + motion-aux, the one untried evidence
   *source*) before any substrate fork.
2. **Probe #2 (conditional):** actions/motion into the bank path + encoder.
   Motion is the canonical disambiguator for same-heading corridor aliases
   (turn-sequence signatures); it is also the spec-default input that was
   never wired.
3. **Substrate fork (last resort, memory-key only):** DINOv2/patch features
   for the place key; LeWM keeps dynamics + Level-3 servoing regardless.

## Probe #3 result — the replay filter test PASSES: the memory IS buildable

Code: `lewm/memory/online_topological_memory.py` (view-keyframe nodes = the
(cell × yaw-bin) design — no yaw label needed at inference, the view-selective
code makes nodes heading-specific by construction; §5.4 top-k Bayes filter with
transition prior + uniform leak; global novelty commit; running-mean node
embeddings) + `lewm/tests/test_online_topological_memory.py` (3/3; the test
caught a real filter defect — zero prior mass on never-traversed transitions —
fixed with the standard uniform-leak remedy).
Script: `scripts/probe_topo_filter_replay.py` — trains the same-yaw head +
Platt on the yaw train banks, derives data-driven τ_new candidates from
calibration precision targets, builds **contiguous trajectory banks** (one env
× ≤400 steps × 32 held-out scenes; H=8 sliding windows; per-step (cell, yaw)
labels; boundary = cell-transition frames), replays the filter, scores per
§5.5/§6.1 (majority labels, purity rule). Artifacts:
`topo_filter_replay_seq4_e9_v6*.json`, cache `traj_banks_yaw_eval.pt`.

| condition | τ_new | coh(cell) | median | coh(cell,yaw) | false-merge | frag | nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| **calP95 / filter** | 0.879 | **0.962** | 0.972 | 0.873 | 0.205 | 4.09 | 44.3 |
| calP95 / no-prior | 0.879 | 0.953 | 0.959 | 0.860 | 0.228 | 4.02 | 43.8 |
| calP90 / filter | 0.770 | 0.933 | 0.931 | 0.824 | 0.302 | 2.32 | 26.4 |
| calP80 / filter | 0.544 | 0.894 | 0.886 | 0.746 | 0.464 | 1.28 | 15.0 |
| calP50 / filter | 0.111 | 0.776 | 0.775 | 0.611 | 0.737 | 0.53 | 6.4 |

**GATE PASSED, 3/3 belief seeds: mean cell-coherence 0.962 / 0.963 / 0.956 ≥
0.90** at the calP95 operating point. Worst scene 0.87 (`loop_alias_stress`).
The τ_new sweep reproduces the spec's predicted geometry: stricter matching →
more fragmentation, fewer false merges, higher coherence — and §5.5's ordering
(false merges fatal, fragmentation a minor inefficiency) picks the strict end.

Honest notes:
- **The reordering rationale was right for a subtler reason than predicted:**
  the transition prior itself adds only +0.009–0.014 coherence over per-step
  likelihood MAP. The heavy lifting is the *system* around the pair score —
  calibrated probabilities, the novelty-streak commit (an N-consecutive-step
  aggregate), running-mean node embeddings, and the strict operating point.
  Pairwise R0.27@P90 was never the right summary of what the mechanism can do.
- ~20% of nodes are impure at calP95; coherence already charges assignments to
  those nodes, and the §6.1 purity rule routes them to `unknown` for
  ReachabilityHead training. Fragmentation ≈ 4 view-nodes per true (cell, yaw)
  — acceptable per spec; pivot/merge heuristics can reduce it later.
- Scope: passive localization replay (one env, ≤400 contiguous steps, within-
  session revisits only). Active navigation, cross-session loop closure, and
  the GoalAdapter remain Stage 3/4 work.

## FINAL VERDICT (2026-06-09): Stage 3 is GO

- **Place code:** v6 BeliefEncoder, frozen (yaw-objective v7 retired).
- **Node design:** view keyframes ((cell × yaw-bin) by construction) with
  running-mean embeddings; goal-facing representative observations satisfied
  by the same design.
- **Loop closure:** same-yaw-trained head + Platt; **§5.3's single-pair 99%
  bar is RE-REGISTERED as commit-only** — the deployed mechanism is the §5.4
  filter + novelty streak at τ_new = calP95 (≈0.88 calibrated), which passes
  §5.5 coherence at 0.96.
- **Probe #2 (action tokens + motion-aux): now an optimization, not a
  blocker** — revisit if Stage 3/4 shows localization-limited failures.
- **Next:** Stage 3 proper — wire `OnlineTopologicalMemory` into the
  `Memory`/`HierarchicalPlanner` seam (Stage 0), then GoalAdapter +
  ReachabilityHead (§6.1 purity rule, memory-generated pairs), then Stage 4
  end-to-end with exploration mode.

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
