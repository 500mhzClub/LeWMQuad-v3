# Preregistration: representation-qualification probe V1

Date: 2026-08-05
Branch: `jepa-spatial-world-model-nav`
Attempt identity: `go2_representation_qualification_probe_v1_attempt_v1`
Supersedes an earlier draft of this file dated the same day, which was revised
before any run in response to three corrections recorded in §0.

Status: **development-tier, blind baseline probe — step 1 of a
baseline-and-intervention loop, not a standalone measurement programme.** It
trains probes, not a navigation candidate, and promotes nothing. It opens no
untouched, sealed, held-out, or V4 material and adds no custody cost.

---

## 0. Corrections this version incorporates

The first draft was wrong in three ways, and each is fixed here rather than
argued with.

1. **Measurement is not creation.** A probe cannot make an encoder preserve
   anything. This document is explicitly step 1 of: measure → localize the loss
   → revise the spatial/semantic objectives → retrain → rerun this same frozen
   probe as a **non-regression** test. §8 fixes that sequel.
2. **Occupancy is not semantics.** Two separate probe families are registered,
   §4 and §5.
3. **The target must be observable from the encoder's actual input.** The first
   draft's binary occupancy would have asked a single-frame latent to
   reconstruct occluded map content. `unknown` is now a **predicted class**, not
   a mask, and observability is derived from exactly the frame the encoder sees.

A fourth correction applies retroactively to an earlier claim of mine, and is
recorded in §9.

## 1. Question

**Which latent representations preserve, in a form that transfers to unseen
scenes, (a) local traversability geometry and (b) task-relevant visual
semantics — and where in the stack is any loss occurring?**

## 2. Data and observability

Immutable CPU-flat V3 collection, SHA-256
`711b8722c11dbae663ad1b004268b77c64ff3d2e818f2c895851c547240e3ed0`. All 3,072
frames were already opened; no new custody cost.

**Each frame is encoded independently.** The encoders under test consume one
224×224 RGB frame (resized per encoder), with no history. The target for a frame
is therefore constructed from **that frame's pose alone** — no history union, no
successor, no map, no privileged state is available to any probe. The probe sees
only the latent.

Poses come from `context_base_pose_world_sequence` for context frames and
`endpoint_state` for successor frames. World pose and manifest geometry are used
**solely to generate labels**, never as probe input.

## 3. Splits — scene clusters, three ways

Scene-level, fixed before any sample is generated:

| split | scenes | use |
|---|---:|---|
| fit | 24 | probe fitting only |
| validation | 8 | probe/rung selection only |
| final | 32 | reported result; opened once |

Fit and validation come from the collection's train role; final is the disjoint
eval role. Frames are **never** the unit of analysis: metrics aggregate per
scene, and all uncertainty is a whole-scene cluster bootstrap over the 32 final
scenes, family-balanced, 10,000 resamples, seed `2026080571`.

The rank-regret power figures are **not** reused; occupancy recall and IoU are
different endpoints with different variance, and their observed variance is
reported here for future designs.

## 4. Spatial qualification

Egocentric body-frame grid, 64×64, `x ∈ [0, 4]` m forward, `y ∈ [-2, +2]` m
lateral, 6.25 cm cells.

**Three classes, all predicted:**

- `occupied` — a wall footprint covers the cell and the cell is observable;
- `free` — observable and not covered;
- `unknown` — outside the horizontal frustum, or occluded.

Observability uses the render's own `genesis_yfov_deg = 78.323` on a square
frame, giving a `±39.1615°` horizontal half-angle, with occlusion resolved by a
radial sweep against wall footprints from the frame's own pose. `unknown` is a
class the probe must predict, so a probe cannot score well by hallucinating
structure behind walls, and is not penalised for failing to reconstruct it.

Reported: occupied recall, free recall, **occupied IoU, free IoU**, balanced
accuracy over the three classes, per-scene values, and cluster intervals.

## 5. Semantic qualification — scoped to what the labels actually support

**Registered limitation, established before designing the probe:** the corpus
manifests do not label beacons or distractors. Every landmark carries
`kind: "landmark"` and exactly two material classes, `landmark_red` and
`landmark_blue`, six per scene. A beacon/distractor/landmark tri-class target is
therefore **not constructible** from this data and is not attempted.

What the labels do support, all restricted to landmarks observable in the frame
under the same frustum and occlusion test as §4:

| target | form |
|---|---|
| any visible landmark | binary |
| visible landmark colour | red vs blue, conditioned on visibility |
| nearest visible landmark bearing | regression, radians |
| nearest visible landmark range | regression, metres |

Reported: balanced accuracy for the classification targets, median absolute
error for the regressions, per-scene and with cluster intervals. Conditioning on
visibility means a latent that has discarded the landmark cannot score by
guessing a scene-level prior.

## 6. Representations — the operational latent, and where pooling sits

Each encoder contributes at most two registered representations.

| encoder | pooled | spatial |
|---|---|---|
| `own_v4` — `go2_jepa_geometric_encoder_v4_medium41_crossfam_lat192_img128` | **192-d latent (operational)** | pre-flatten conv map |
| `dinov2` | CLS embedding | 16×16×384 patch tokens |
| `vjepa2_1` | mean-pooled tokens | 16×16×768 tokens |

`own_v4` is the representation the downstream stack actually consumes, at its
native 128×128 input. **Its operational output is a 192-d global vector — it has
no spatial tokens by construction.** Its pre-flatten convolutional map is
included as the fair spatial variant of the same backbone, and is labelled as
non-operational so the pooled result cannot be quietly replaced by the richer
intermediate.

This layout is what localizes the loss: pooled-vs-spatial within an encoder
separates a pooling bottleneck from a backbone bottleneck.

## 7. Probes, controls, and gate

**Probes — identical specification for every representation, no per-encoder
tuning and no architecture search:**

1. **linear** — one affine map to the target;
2. **small nonlinear** — one hidden layer, width 256, GELU.

The linear probe asks whether the information is readily accessible; the
nonlinear probe asks whether it is recoverable at all without the probe becoming
a new perception model. Same optimiser, schedule, and selection rule throughout;
selection only on the validation scenes.

**Controls:**

- `shuffled_latent` — latents deranged against labels within split; no frame
  keeps its own label;
- `class_frequency` — predicts the fit-split class frequencies per cell, with no
  input;
- `analytical_oracle` — regenerates the target from geometry and must reproduce
  it exactly. This validates label generation **only**; it is not a learned
  validity model and gates nothing else;
- `rgb_baseline` — the small nonlinear probe on downsampled raw RGB, establishing
  how much of the target is learnable from the raw input itself.

**Thresholds are not inherited.** The existing perception-gate numbers
(`0.80 / 0.68 / 0.88 / 0.25 / 0.42`) were calibrated on the direct-BEV line's own
raster definition. This probe differs in grid extent, resolution, visibility
convention, the presence of an `unknown` class, frame-history assumption, and
aggregation. Reusing those numbers would manufacture false continuity. They are
reported alongside for context and **are not a gate**. The registered result is
the comparative table plus control margins; any gate for this target is defined
separately, after the first baseline establishes its scale.

## 8. Predetermined interpretation, and the sequel

| result | implication and next action |
|---|---|
| `own_v4` decodes both families well on final scenes | representation preserves what is needed; investigate predictor, memory, planner |
| frozen encoders decode well, `own_v4` does not | our training objective or bottleneck discards available information |
| spatial decodes well, pooled does not | pooling/compression is the immediate failure point |
| fit high, final collapses | scene-specific shortcuts, not transferable structure |
| all fail including `rgb_baseline` | revisit target observability, label generation, diversity, split — before touching the encoder |
| geometry passes, semantics fail | strengthen the semantic objective independently |
| semantics pass, geometry fails | add metric/spatial supervision or equivariance constraints |

**The sequel is part of this registration.** Whatever the baseline says, the work
that promotes preservation is retraining with explicit spatial and semantic
objectives, with their contribution controlled against the JEPA prediction
objective — the 84-million-to-one starvation failure is the standing warning.
This probe then becomes a **non-regression qualification**: a checkpoint may not
be accepted because prediction loss improved if held-out geometry or semantics
deteriorated.

## 9. Retroactive correction to an earlier claim

I previously wrote that "frozen dense visual features don't carry
scene-transferable local physical geometry." That overstates the evidence. What
the observability-ceiling assay established is narrower:

> A particular learned readout failed to transfer the registered branch-ranking
> target across disjoint scenes, despite fitting the training data exactly and
> receiving actual successor-image features.

That failure is consistent with the representation, the chosen layer, global
pooling, the readout's inductive bias, the ranking target, scene-distribution
shift, or optimisation. This probe is what could support the stronger claim, and
the stronger claim is not to be written until it does.

## 10. What this does not authorize

No promotion, no perception qualification — passing this probe is **not** passing
the perception gate, which applies to a learned BEV state trained for the
purpose — no navigation or closed-loop claim, no threshold relaxation, no data
generation, and no access to untouched, sealed, held-out, or V4 material.

---

## Amendment 1 — semantic family deferred on measured target inadequacy

Date: 2026-08-05, **before any probe was fit and before any score was observed.**

### The measurement

Landmark observability was measured over all 1,536 train-role frames, under the
§5 frustum and occlusion test, at four range caps:

| semantic range cap | frames with a visible landmark | scenes with any visible landmark |
|---:|---:|---:|
| 4 m (as registered) | 147 / 1536 (9.6%) | **5 / 32** |
| 8 m | 234 / 1536 (15.2%) | **9 / 32** |
| 14 m | 234 / 1536 (15.2%) | **9 / 32** |
| 30 m | 234 / 1536 (15.2%) | **9 / 32** |

The count saturates at 9 scenes. The binding constraint is **not** the range cap
— it is frustum and occlusion. Most poses in these scenes never have a landmark
in view at all.

### Why this forces a deferral

§3 requires per-scene aggregation with whole-scene cluster intervals. With
landmarks visible in at most 9 of 32 scenes, **23 of 32 final-role scenes would
contribute no semantic observation**, so the semantic family cannot produce a
cluster-level interval at any range cap. Colour is conditioned on visibility and
is worse still.

Reporting a semantic number from 9 scene clusters, or from 234 correlated
frames treated as independent, would reproduce exactly the unit-of-analysis
error this preregistration was written to avoid.

### Decision

The **semantic qualification family (§5) is deferred**, not weakened. It is not
re-scoped to a target that happens to be visible, because no other task-relevant
semantic label exists in these manifests: walls carry only `material_id`
appearance classes, which are not task semantics.

The **spatial qualification family (§4) proceeds unchanged.** It is adequately
supported: occupied cells occur in every scene and every family, with an
occupied fraction of `0.032` overall and per-family values from `0.0097`
(`rough_local_dynamics`) to `0.0438` (`small_enclosed_maze`).

### What the deferral costs, stated plainly

This probe will therefore establish whether **geometry** is preserved and
transfers, and will **not** establish anything about semantic preservation. The
"geometry retained, semantics lost" and "semantics retained, metric arrangement
lost" branches of the §8 interpretation table become unreachable in this
attempt, and no claim about semantic preservation may be made from it.

### What would make the semantic family constructible

Any of: a collection whose spawn and trajectory policy places landmarks in view
for most frames; a corpus revision that annotates beacons and distractors
distinctly rather than a single `kind: "landmark"` with two colour materials; or
a semantic target drawn from something present in nearly every frame and still
task-relevant. That is a data-generation change and belongs to its own
preregistration.

### Class imbalance, recorded for the spatial family

Measured over 1,536 train frames: `free 0.455`, `occupied 0.032`,
`unknown 0.514`. Occupied is rare, so IoU and per-class recall — both already
registered in §4 — carry the result, and the probes use fit-split class
weighting. Balanced accuracy alone would be misleading at this imbalance and is
reported but not relied upon.
