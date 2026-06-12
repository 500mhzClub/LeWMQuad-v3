# Topological Nav — Stage 1: seq4 Recognition Re-validation + the Cheap H2 Test

Date: 2026-06-09
Checkpoint: `models/checkpoints_textured_v03_full_20260531/sweep_seq4/lewm_seq4_e9.pt` (frozen)
Env: GPU torch (ROCm, R9700) via `~/TinyQuadJEPA/bin/python`; data = textured v03 render,
corpus `minimum_tex_20260520T211541Z`, `test_id` split, 32 eval scenes.

This executes the first deliverables of
`docs/lewm_topological_nav_implementation_plan_2026-06-09.md`: (#1) re-validate
seq4 place-recognition R@1, and the **Stage 1 cheap H2 test** — can frozen
seq4 latents + a short history build a coherent place map *without* a trained
BeliefEncoder?

## 1. Deliverable #1 — seq4 recognition re-validated (the 0.42 was not a seq11 artifact)

`scripts/train_place_retrieval_head.py` `baseline_raw` (frozen single-frame
latents, 32 eval scenes):

| metric (seq4_e9, frozen, raw) | value | prior (pre seq4/seq11 split) |
|---|---:|---:|
| same-cell retrieval@1 | **0.431** | 0.42 |
| same-cell retrieval@5 | 0.640 | — |
| graph-distance Spearman ρ | **0.079** | 0.03 |

seq4 is a **good place-recognition code and a poor metric code** — exactly the
premise the topological stack rests on. The recognition figure (R@1 ≈ 0.42,
~21× same-cell chance) transfers to seq4; the metric ρ stays ~0. The
recognition-not-metric finding ([[project_lewm_aliasing_a2]],
[[project_lewm_nav_cost_phase0]]) is **confirmed on the adopted base**, not
inherited from the retired seq11 program.

### A trained single-frame projection adds nothing
The same script trains a supervised-contrastive `PlaceRetrievalHead` (3 seeds,
100 epochs) on the frozen latents and **fails the gate**:

| | raw baseline | learned head (3-seed mean) | Δ |
|---|---:|---:|---:|
| retrieval@1 | 0.431 | 0.423 | **−0.008** |
| retrieval@5 | 0.640 | 0.626 | **−0.014** |
| graph ρ | 0.079 | 0.09–0.10 | ~flat |

Gate (ΔR@5 ≥ 0.15) **not passed**. A learned projection over *single frames*
recovers neither extra recognition nor metric structure — confirming the
information limit is not "the readout head," and pointing the open lever at
**temporal history**, not a fancier per-frame encoder.

## 2. Stage 1 cheap H2 test — naive frozen history barely helps

`scripts/probe_lewm_history_retrieval.py` builds terminal/mean/concat
descriptors over the last H∈{4,8} frozen frames and measures same-place
retrieval against the single-frame terminal descriptor (32 eval scenes):

| descriptor | R@1 | R@5 | ΔR@5 vs single frame |
|---|---:|---:|---:|
| terminal_raw (1 frame) | 0.377 | 0.576 | — |
| h4_mean | 0.382 | 0.584 | +0.009 |
| h4_concat | 0.385 | 0.582 | +0.006 |
| h8_mean | 0.386 | 0.590 | +0.014 |
| h8_concat | 0.386 | 0.590 | **+0.014** |

The improvement is **real and monotone** (h8 > h4; both pooling modes agree) but
~4× below the +0.05 gate — **gate not passed**. Naive pooling of frozen latents
cannot exploit history for recognition.

**Plan consequence:** the cheap Stage 1 shortcut — skip the BeliefEncoder and key
the topological memory off a frozen mean/concat-of-history descriptor — is
**closed**. We cannot build the place memory on frozen-pooled history alone.

## 3. Does a *learned* history encoder stand a chance? (history-disambiguability)

The naive-pooling failure does not by itself decide whether to build the
BeliefEncoder (Stage 2). It only shows averaging/concatenation can't use
history. The §6/§10 contingency asks the prior question: among single-frame-
*aliased* pairs at *different* true cells, can a short history window separate
them at all? If yes → the information is present-but-not-trivially-poolable → a
trained BeliefEncoder is worth building. If no → the substrate destroys it →
fork to a substrate change (DINOv2 / patch retrieval), out of frozen-LeWM scope.

`probe_lewm_reachability_a3.py --skip-localization --skip-reachability`
(retrieval + history-disambiguability, AUC of history-window distance vs
single-frame distance separating same-cell from different-cell among
*single-frame-aliased, different-true-cell* pairs), seq4_e9, 8 history scenes:

| window | single-frame AUC | history-window AUC | Δ (history headroom) | median history AUC | n aliased pairs |
|---|---:|---:|---:|---:|---:|
| H=4 | 0.771 | 0.810 | **+0.039** | 0.852 | 1692 |
| H=8 | 0.802 | 0.857 | **+0.055** | 0.910 | 1585 |

History-window distance separates the hard aliased pairs with **AUC 0.81–0.86
(median 0.85–0.91), well above 0.5, and improves +0.04–0.055 over single-frame**,
monotonically with longer history (H8 > H4). And this is a *training-free*
distance readout — a trained encoder supervised on exactly these pairs should
match or beat it. So on the very pairs that matter for loop-closure false
positives, **the disambiguating information is present and history-separable.**

This reconciles with §2: history barely moves *global* recall under naive
mean/concat pooling (§2, +0.014), yet clearly separates the *aliased subset*
that drives map errors (§3, AUC 0.86). The §2 failure is a **pooling**
limitation, not an **information** limitation — which is precisely the case a
learned BeliefEncoder is for.

## 4. Decision — build Stage 2 (BeliefEncoder)

**Build the contrastive history BeliefEncoder.** The evidence forks cleanly:
- The cheap shortcut (frozen mean/concat-of-history as the memory key) is **out**
  (§2): naive pooling cannot exploit history for recognition.
- But the information needed to disambiguate aliased places **is present and
  history-separable** (§3, AUC 0.86 ≫ 0.5, +0.055 over single-frame). This is the
  "present-but-not-trivially-poolable" regime the §6/§10 contingency reserves for
  the BeliefEncoder, *not* the "destroyed-at-encode-time" regime that would force
  a substrate change.

**Concrete Stage 2 bar (registered):** the trained BeliefEncoder must beat the
naive-pooling baseline on same-place Recall@5 by the plan's registered margin,
and should approach/exceed the AUC-0.86 history-separation ceiling on aliased
pairs. If it cannot beat naive pooling despite the present signal, *then* H2 is
falsified and we fork to DINOv2 patch features
(`probe_dinov2_patch_retrieval.py`) per §6 — but the §3 AUC says that fork is
**not** indicated yet.

Substrate decision unchanged: base = `seq4_e9` frozen; recognition-not-metric
(ρ ≈ 0.08) holds; everything stays recognition/graph-based.

## Artifacts
- `.generated/topo_nav/place_retrieval_seq4_e9.json` (+ `.log`) — re-validation + learned-head gate.
- `.generated/topo_nav/history_retrieval_seq4_e9.json` (+ `.log`) — cheap H2 test.
- `.generated/topo_nav/reachability_a3_history_seq4_e9.json` (+ `.log`) — history-disambiguability AUC.
