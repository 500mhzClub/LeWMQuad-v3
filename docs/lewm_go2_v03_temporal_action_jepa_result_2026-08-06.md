# Matched temporal action-conditioned JEPA: frozen vs top-block encoder movement

Date: 2026-08-06
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** No manifest or authorization
status is inherited. `probability_calibration`, `evaluation`, `untouched` and
sealed data were never opened.

Artifacts: `/home/andrewknowles/.cache/lewm_go2_temporal_v03/` (root filesystem —
the workspace pool is full at 658 MB free)

---

## Verdict

> **REJECTED.** Encoder movement made the future *more predictable* and
> *less action-discriminative*, in all eight families, while spatial information
> was preserved.

The moving arm's JEPA loss was **11.5% lower** (0.10231 vs 0.11562) and its
correct-action cosine was **higher** (0.7861 vs 0.7534). Both are irrelevant
under the registered rule, because its shuffled-action and persistence baselines
rose further:

| changed-token cosine | frozen | moving |
|---|---:|---:|
| correct action | 0.7534 | **0.7861** |
| shuffled action (mean of 3 derangements) | 0.6949 | **0.7383** |
| persistence | 0.4800 | **0.5451** |
| **correct − shuffled** | **+0.0586** | **+0.0479** |
| correct − persistence | +0.2735 | +0.2411 |

**Action margin fell by 0.0107 (−18%), and fell in 8 of 8 families.** This is the
WP-E finding reproduced at 304M parameters, with genuine three-frame temporal
context, a masked context—target objective, an EMA target and a distinct future
temporal position. Every one of those was a candidate explanation for the WP-E
result. None of them was the cause.

## What is new: the failure is now isolated to action discrimination

WP-E could not separate "loses geometry" from "loses action sensitivity". This
run separates them, because the spatial representation **did not degrade**:

| | frozen | moving | delta |
|---|---:|---:|---:|
| fresh-probe occupied IoU | 0.5004 | **0.5054** | +0.0051 |
| fresh-probe occupied precision | 0.6530 | **0.6754** | +0.0224 |
| fresh-probe occupied recall | 0.6816 | 0.6676 | −0.0140 |
| fresh-probe `open_obstacle_field` IoU | 0.2133 | **0.2308** | +0.0175 |
| fixed-probe occupied IoU | 0.4986 | 0.4817 | −0.0170 |

Under a probe retrained on its own features the moving encoder is **as good or
slightly better** spatially, including on `open_obstacle_field`. Under the fixed
probe it drops 0.0170 — the representation *moved* relative to the frozen probe's
input space without becoming less informative. That is drift, not loss.

So the acceptance test failed on exactly one clause: the action margin.

## Raw token health

| | frozen | moving | delta |
|---|---:|---:|---:|
| raw token variance | 0.5685 | 0.5009 | **−11.9%** |
| raw effective rank | 85.81 | 97.71 | **+13.9%** |
| raw temporal delta | 0.4965 | 0.4430 | **−10.8%** |

No collapse: effective rank *rose*. But variance and temporal delta both fell
about 11%, and the persistence cosine rose from 0.4800 to 0.5451 — consecutive
frames became more similar in latent space. The encoder made the sequence
smoother. A smoother sequence is easier to predict under any action, which is
precisely why the margin shrank.

## Per-family — the effect is uniform

Action margin (correct − shuffled), and fresh-probe occupied IoU:

| family | margin frozen | margin moving | occ IoU frozen | occ IoU moving |
|---|---:|---:|---:|---:|
| `large_enclosed_maze` | +0.0701 | +0.0581 | 0.5671 | 0.5458 |
| `local_composite_motifs` | +0.0682 | +0.0553 | 0.5878 | 0.5772 |
| `visual_sensor_stress` | +0.0650 | +0.0529 | 0.5040 | 0.5123 |
| `medium_enclosed_maze` | +0.0607 | +0.0498 | 0.5264 | 0.5235 |
| `loop_alias_stress` | +0.0516 | +0.0411 | 0.3721 | 0.3674 |
| `rough_local_dynamics` | +0.0484 | +0.0402 | 0.5356 | 0.5580 |
| `small_enclosed_maze` | +0.0484 | +0.0399 | 0.7852 | 0.8142 |
| **`open_obstacle_field`** | **+0.0428** | **+0.0357** | **0.2133** | **0.2308** |

**8 of 8 families lose margin.** Four of eight *gain* occupied IoU. The
directions are independent, which is the point: geometry and action sensitivity
are separable, and only the latter regressed.

`open_obstacle_field` has both the weakest action margin and by far the weakest
occupancy — it remains the hardest family on every axis.

## Setup

**Data.** `development_raw_supervision_v1` designated roles over the dense
`render_textured_v03` render, centre-cropped 224×224 → 224×168 to the v04 field
of view (crop gate verified separately: zero analytic FOV error, empirical
crop-offset peak at row 28 and scale peak at 1.0000).

| role | sequences | scenes |
|---|---:|---:|
| `train` | 4,075 / 4,262 | 72 |
| `checkpoint_selection` | 491 / 495 | 8 |
| scene overlap | — | **0** |

Frozen causal contract: context `t−480, t−240, t`; target `t+240`; action = the
command block executed `t → t+240` (`block_size 5 × command_dt_s 0.1 = 0.5 s`).
Every frame verified against the rollout `frames.jsonl` for identical scene,
`env_index`, `episode_id` and `reset_count`. No duplicated frames, no inferred
filenames, no crossed resets.

**Arms.** Identical rows, ordering, seed (`2026080651`), optimiser, predictor,
action representation and 6-epoch schedule. Dense 24×32×1024 grids with explicit
learned temporal positions throughout; AdaLN action conditioning on the
9-primitive one-hot plus body-velocity command; masked context—target objective
at 50% mask; target normalisation. **No CLS, no global pooling, no BEV in the
predictive path.** Neither arm cached encoder features — both recomputed all four
frames every step.

* **A (frozen):** encoder frozen, predictor trainable.
* **B (moving):** identical initialisation; final encoder block + final norms
  trainable (**20 tensors, 12.6M of 304.7M**) at 1/20th the predictor LR, with a
  0.999 EMA target encoder.

The whole encoder is put in eval mode before the trainable blocks are opted back
in, with an assertion that no frozen module is left training and that the target
encoder takes no gradient — the WP-E retrospective defect, made non-repeatable.

**Comparator.** The matched frozen v03 spatial reference is **0.4994** observable
occupied IoU (precision 0.6526, macro 0.5105, shuffled margin +0.3703). The v04
figure `0.5103` is a cross-contract development reference only. Arm A's fixed
probe reproduces the reference exactly (0.4986 vs 0.4986), as it must — arm A's
encoder never moved.

## Reading

**This closes the "maybe it was the objective" hypothesis.** WP-E's
encoder-moving recipes were single-frame, unmasked, on a 2.7M task-trained
encoder, and both weakened action discrimination. The obvious rebuttals were:
too small an encoder, no temporal context, an objective that rewarded
contraction. This run removes all three — 304M pretrained initialisation, genuine
three-frame history, masked context—target with an EMA target and a distinct
future position — **and the same thing happens.**

The mechanism is now visible. The objective rewards the *future being
predictable*. It does not reward *different actions producing different futures*.
Given a lever on the encoder, gradient descent spends it making the sequence
smoother: persistence cosine +0.065, temporal delta −11%. That helps every arm of
the prediction equally, so the margin shrinks even as absolute accuracy improves.

**What is not the problem:** collapse (effective rank rose 14%), lost geometry
(fresh-probe IoU and `open_obstacle_field` both rose), encoder capacity,
temporal context, or masking.

## Next

The narrowest untested intervention is unchanged from WP-E §6, and this run
raises its priority rather than settling it: **an objective term that makes
action-conditioned futures differ**, not merely be predictable. Two candidates
from the literature already reviewed:

1. **Latent-difference action decoding (Delta-JEPA):** require the executed
   action to be recoverable from `z_{t+1} − z_t`. Directly penalises the
   smoothing this run measured, uses every ordinary transition, needs no matched
   branches.
2. **Action-contrastive term at the same current state:** penalise predicted-state
   similarity between the correct action and the other eight primitives.

Neither is authorised by this document. Also note the practical constraint: the
9 primitives are coarse and `hold` is in the action set, so some of the residual
margin ceiling may be corpus-side rather than model-side — worth measuring before
attributing further loss to the objective.

**Do not** add a geometry teacher on this evidence: geometry did not regress.

## Corrections made during this work

- The first evaluation run died with HIP OOM (8.63 GiB request against 23.27 GiB
  held). Cause was in the battery, not the experiment: the predictor was called
  on all 491 sequences at once, features were held in float32, and the
  changed-token threshold materialised the full train tensor. All three fixed;
  no measured quantity changed.
- An earlier reading of the temporal feasibility check reported a hard data
  blocker. That was wrong — it looked only at the v04 render and missed the dense
  v03 render on the 3.7 TB pool, which covers 100% of the corpus scenes.
- The v03 crop was first derived *analytically* as a 1.333× focal mismatch from
  the platform manifest's `native_resolution: [640,480]`. The pixels disproved
  it: the crop-offset and scale sweeps both peak sharply where a shared focal
  length predicts. The empirical result governs.
