# Bounded follow-up: seed audit, spatial-metric replacement, counterfactual corpus design

Date: 2026-08-10
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.**

Commit `443e591` and final report digest
`60b0bb2d0b13ba47eac5e306c33d97dcfdce31102870edfc50b01f7f9b247161` are treated as
**frozen**. Nothing here amends the confirmatory report, reruns predictor
training, launches seed indices 8–9, reselects checkpoints, or reinterprets the
invalid occupied metric as evidence.

---

# 1. Read-only seed and execution audit

Audit digest **`03ad9d3bd588b251385240d598cd1915344f97437d1f8e8f8b33705cc86760f0`**.
Implementation-integrity audit only. No first-four-versus-last-four test, no trend
against seed index, no seed excluded, no confirmatory result touched.

**Observation audited:** seeds …901–…904 negative, …905–…908 positive.

| check | result |
|---|---|
| registered integers distinct, untruncated | ✓ 8/8, all within int64; no 32-bit path is used |
| derived RNG stream keys | 192 data-order keys (8 seeds × 24 epochs), **192 distinct, 0 collisions** |
| proprio stream keys (`seed + 7919`) | 8 distinct, none collides with any seed |
| seed reuse | none |
| base-weight artefacts unique across seeds | ✓ 8 distinct state digests, file hashes match records, integrity digests valid |
| base artefact identical within each quadruplet | ✓ all 8 report shared parameters bit-identical |
| checkpoint collisions | **0** across all 32 |
| modality-specific init unique per seed | ✓ 8 distinct, shared parameters unaffected |
| batch/augmentation plans | distinct across seeds, cell-independent **by construction** (`batch_plan` takes only `(seed, epoch)`; the cell is not an argument). No train-time augmentation exists |
| manifest / normalisation / config / map / mask digests | constant across all eight |
| software environment, GPU identity, determinism settings | **constant across all eight** |
| any cell resumed, retried, or evaluated under a different executed source | **none** |

## The chronological question, answered directly

The executed-source boundary **does not coincide** with the sign split, and it is
off by exactly one seed:

| | boundary |
|---|---|
| sign split | at seed **…905** (first positive) |
| source/receipt change | at seed **…906** (first under continuation commit `043a343`) |

Seed …905 carries a **positive** estimate yet ran under the **same** initial launch
commit `99a6eea` and the **same** receipt `abe036ad…` as all four negative seeds.
A source or environment change therefore cannot explain the pattern. The
continuation machine-check separately proved all 15 scientific modules
byte-unchanged between the two commits.

> **Conclusion: no implementation explanation found.** The pattern is recorded as
> an **unexplained post hoc diagnostic consistent with ordinary seed variation**.
> It is not evidence of anything and was not tested statistically.

---

# 2. Spatial-retention replacement — specification frozen, assay NOT run

Specification digest **`646073a9b0a43d7a6c3230f55b3d68026d0632af70726c196603cb7ccf182478`**
(`factorial_v1/spatial_retention_spec.json`). This is a **post-confirmatory
spatial-retention follow-up assay**, explicitly not the originally registered
co-outcome, which remains invalid and unamended.

## The existing probes are disqualified — an important finding

`future_token_probe.pt` **cannot be used**. `train_probe` evaluates observable
occupied IoU on the **selection index** after every epoch and restores the
best-scoring epoch's weights. The probe's weights were therefore model-selected
using the split containing all 475 factorial selection rows. `fixed_probe.pt`
comes from the same procedure. Both fail the requirement that the probe be fitted
and calibrated without using those rows.

This is worth stating plainly: the disqualifying defect is the *same class* of
defect as the one that invalidated the original metric — a model-dependent choice
made on the evaluation split.

## Label availability — the blocker

| horizon | occupancy label source | coverage | decision |
|---|---|---|---|
| **H=1** | `raster_labels.u1` via `pairs.jsonl` → `endpoints.jsonl` | **4,566/4,566** | usable |
| H=2 | `native_step2_labels` flag on two-step rows | **81/475** selection, 853/3,922 train | **excluded** — 17 % is a different, family-imbalanced subset |
| H=3, H=4 | none; `horizon_frames` carry only `frame_index` and `path` | 0 | **blocked** |

> **The requested H=1–4 spatial reporting is not achievable from existing
> artefacts.** Only H=1 is defensible. I have frozen the specification at H=1 and
> recorded H≥2 as a blocker rather than inventing a label source or quietly
> reporting a 17 % subset alongside a full one.

## What the frozen specification fixes, in advance

Label source and spatial alignment (24×32 tokens → 64×64 raster, one contract for
all models); probe architecture `SharedTokenToBev(1024)`; fit on a disjoint subset
of the **3,922 factorial train** rows using **true target latents only**;
calibration on a further disjoint train subset, **never the selection rows**;
**fixed epoch budget with the final epoch taken — no best-epoch selection on any
split**, which is precisely the defect that disqualified the existing probe;
absolute threshold = argmax (occupancy probability ≥ 0.5), explicitly not a median
or percentile; the frozen changed-token mask `ce32489f…`; qualification criterion
**observable occupied IoU ≥ 0.35 on true target latents**, a conservative floor
taken from the frozen dense-representation screen's 0.510 and **not** from any
factorial result; equal-family primary and corpus-weighted secondary aggregation;
seed-quadruplet replication with 95 % *t*-intervals; and probe performance on true
target latents reported alongside every predicted score so probe error and
predictor error stay separable.

## Status

**The probe has not been fitted and the assay has not been run.** The
specification is frozen first, which is the required order. No spatial follow-up
results are reported, because the metric has not yet qualified. The remaining work
is: fit the probe on the train-fit split, check it against the ≥ 0.35
qualification criterion on true target latents, and — only if it passes — apply it
exactly once to all 32 frozen epoch-21 checkpoints at H=1.

---

# 3. Counterfactual planning-evaluation corpus — design only

No simulator branch was generated.

**Central question.** Does the rollout objective's higher future-state fidelity
translate into better ranking and selection of candidate actions, despite its
slightly weaker correct-versus-shuffled action margin?

That framing matters: the confirmatory study measured *fidelity*, and the
secondary margin result points the other way. Ranking is the quantity that decides
which of those matters for planning.

## Observation-contract compatibility (non-negotiable)

Identical rendering and image-preprocessing contract (v03 centre-crop, hashed);
observed visual and proprioceptive history only; the same 10-D five-tick post-slew
action representation; the same shared applied-command control history; **no
factual future proprioception supplied to any predictor**; the incompatible V3
branch corpus is **not** substituted.

## Branch generation

| item | specification |
|---|---|
| state snapshot/restore | Genesis full solver-state checkpoint at the branch state — rigid body pose/velocity, joint state, controller internal state **including the previous applied command**, and contact state. Restore must reproduce a zero-action step bit-identically; this is the gating test |
| randomness control | one fixed physics seed per branch state, reused across all alternatives from that state, so branches differ only in action; textures/lighting fixed per scene |
| slew limiting | applied from the state's **actual previous applied command**, via the same `reconstruct_block` used in training — never from zero |
| candidate bank | the 9 primitives × a 4-block (2.0 s) horizon, pruned to a **fixed bank of 12 sequences** per state covering forward/arc/yaw/backward/hold families |
| alternatives per state | **12** |
| horizon | 4 action blocks (H=4), matching the evaluated horizons |
| invalid branches | a branch that falls, leaves the scene, or fails restore verification is dropped with a reason code; a state retaining < 8 valid branches is dropped entirely |
| ties | exact utility ties broken by fixed candidate-bank index, recorded as ties |

## Sampling and balance

**200 branch states**, 25 per family across all eight families, drawn from
**held-out scenes only** (not the 80 scenes of the factorial), one state per
episode cluster so the episode-cluster bootstrap applies unchanged.

## Oracle utility, fixed before collection

`U = w_p · progress − w_s · safety_penalty + w_c · completion`, with
`w_p = 1.0`, `w_s = 2.0`, `w_c = 0.5`, hashed before any branch is rendered.
Progress = reduction in BFS distance-to-goal over the branch; safety penalty =
contacts and falls; completion = terminal goal attainment. Weights are frozen and
may not be tuned after seeing model scores.

## Metrics

Rank regret (primary), realised selected utility, Spearman rank correlation,
top-1 and top-3 recovery, pairwise ordering accuracy, and candidate-score spread
(to detect a model that ranks well only because it separates candidates at all).
Aggregation: equal-family primary, corpus-weighted secondary, **seed quadruplet as
the replication unit across all 32 frozen predictors — no retraining**. The
four-cell factorial structure is preserved, so the same
`I_s = (PropRoll − PropOne) − (RGBRoll − RGBOne)` contrast is computable on
ranking outcomes.

**Decision rule, declared now:** rollout is not to be called superior for planning
unless it improves realised candidate selection or rank regret.

## Cost

| item | estimate |
|---|---:|
| branch simulation | 200 states × 12 branches × 2 s ≈ 4,800 branch-seconds ≈ **12 h** wall at measured rollout rates |
| rendering | 200 × 12 × 4 blocks × 5 steps × 48 envs-equivalent ≈ 240 k frames ≈ **8 h**, ~14 GB |
| encoding + scoring 32 checkpoints | ≈ **6 h** |
| **total** | **≈ 26 h, ~15 GB** |

## Minimum pilot before committing to that

**20 states (2–3 per family) × 12 branches.** Its sole purpose is to check that
candidate utilities have enough spread to make ranking identifiable: require the
interquartile range of within-state oracle utility to exceed the measurement noise
floor, and ≥ 8 distinct utility values per state. **If candidate utilities are
near-degenerate, ranking is unidentifiable and the full corpus must not be built.**
Pilot cost ≈ 2.5 h.

---

# 4. Unresolved blockers

1. **Spatial assay is H=1-only.** H=2 labels cover 17 % of selection rows; H=3–4
   have no label source. Full H=1–4 spatial reporting requires regenerating
   occupancy rasters at those horizons.
2. **The replacement probe is specified but not fitted or qualified**, so no
   spatial follow-up result exists yet.
3. **Counterfactual corpus is design-only** and gated behind a pilot that could
   itself return "ranking unidentifiable".

---

## Stopping condition

Nothing is running. No branch generation, navigation evaluation, additional
predictor training or architecture change has been started.
