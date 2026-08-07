# Two-step autoregressive rollout supervision vs a matched one-step control

Date: 2026-08-07
Status: **DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.** No manifest or authorization
status is inherited. `probability_calibration`, `evaluation`, `untouched` and
sealed data were never opened. The encoder never moved and was never executed in
training.

Artifacts: `/home/andrewknowles/.cache/lewm_go2_temporal_v03/two_step/`

---

# DECISION

> ## ROLLOUT TEST INCONCLUSIVE

**Step-one occupied IoU was still materially improving at the final epoch.** The
rollout arm's epoch-4→5 change was **+0.0414**, against a predeclared material
improvement threshold of **+0.005** — more than eight times the threshold. The
control behaves the same way (0.2328 → 0.2626, +0.0298).

Both arms are undertrained at six epochs. Under the registered rule an
undertrained model is not evidence about the rollout objective, so no acceptance
or rejection is returned, and **this is not a rejection of the rollout
formulation.**

Two operational gates also failed on the rollout arm at epoch 5 — step-one
occupied IoU 0.2550 against persistence 0.3128, and step-one margin +0.0487
against +0.0586 — but those numbers are read off a curve that has not converged
and carry no weight here.

---

## What was actually compared

**The rollout arm optimises `1.5·e1 + 0.5·e2`; the control optimises `e1`.** The
rollout arm therefore also carries **1.5× the weight on the one-step term**, so
this is an **official-inspired rollout-supervision bundle**, not a pure
rollout-only ablation.

Total losses are not comparable across arms and are **not compared anywhere** in
this document. The comparison is on `e1`, one-step operational geometry, action
sensitivity and step-two performance.

**If this bundle ever passes, a `1.5·e1` attribution control is required** before
claiming the benefit comes specifically from autoregressive feedback rather than
from the extra weight on the one-step term.

The loss reduction reproduces the audited source rather than restating it
algebraically: `jloss` and `sloss` are each computed as one elementwise mean and
summed, which *yields* 1.5/0.5 — the coefficients are never hardcoded. **`e1 + e2`
would not have been official-equivalent.**

## Official source correspondence

`app/vjepa_droid/train.py`, `configs/train/vitg16/droid-256px-8f.yaml`:

| question | official | ours |
|---|---|---|
| jloss/sloss combination | `loss = jloss + sloss`, unweighted | same |
| `auto_steps=2` | `for n in range(1, auto_steps)` → **one** extra AR step; `z_ar` holds both predicted frames | same reduction, two frames |
| first predicted frame fed back | `cat([z[:, :tpf], z_tf[:, :tpf]])` — context **grows** (8 temporal slots) | **sliding-three-frame adaptation** `[t−240, t, p1]` |
| actions per step | `actions[:, :n+1]`, states `[:, :n+1]` | a0 then a1; **no proprioception** |
| normalised before feedback | yes, `F.layer_norm` inside `_step_predictor` | same |
| detached before feedback | **no** | **no** |
| loss distance | L1 (`loss_exp: 1.0`) | same |
| target frames | jloss → frames 1..7, sloss → frames 1..2 | jloss → y1, sloss → [y1, y2] |

**The step-two context is a sliding-three-frame adaptation, not an exact
reproduction of the official growing-context architecture.** Our predictor has
exactly three learned temporal positions; appending a fourth would be an
architecture change.

## Rollout-gradient assertion

| | at initialisation | after 50 warmup steps |
|---|---:|---:|
| AdaLN gate weight abs sum | 0.0 | 9282.12 |
| grad into p1 from **second-step term alone** | 0.0 | **0.00787** |
| grad into p1 from sloss alone | 0.5000 | 0.5004 |

The zero at init is a property of **AdaLN-Zero** — `ada` weight and bias start at
exactly zero, so `g1 = g2 = 0`, every block is the identity and the predictor is
context-independent at step 0. The 0.5000 is exactly the direct term of an
elementwise mean over two frames, confirming the indirect contribution was
precisely zero. Confirmed independently on the trained 17.2M control: gate sum
11515, output changes by 11.02 for different context, second-step gradient into
p1 = 0.579. The warmup was **discarded** and the initial-weight SHA-256
re-verified before training.

Had this probe not run, a genuinely dead rollout wiring would have been
indistinguishable from a working one, because `e1` dominates the loss.

## Two-step sequence retention

Contract: context `t−480, t−240, t`; step 1 action a0 with target y1 at `t+240`;
step 2 action a1 with target y2 at `t+480`; **p2 consumes p1, never the true y1**.

`a1` is **directly recorded**, not inferred: every frame of the rollout
`frames.jsonl` carries `command_context.primitive_name` and a `sequence_id` that
changes exactly every 240 flat frames — one complete command block, the same
source and criterion the corpus used for a0. Verified per row that a0 read from
`frames.jsonl` equals the corpus pair's own primitive (zero disagreements) and
that `t+240` begins a distinct complete block of matching `block_size`.

| role | retained | of one-step base |
|---|---:|---:|
| train | **4,031** | 4,075 |
| checkpoint_selection | **488** | 491 |

Dropped: 40 frame indices absent from the rollout, 7 episode/reset crossings. All
five frames share scene, `env_index`, `episode_id`, `reset_count`; no duplicates;
no inferred filenames; 72/8 scenes, zero overlap.

By family (train / selection): `large_enclosed_maze` 496/64,
`local_composite_motifs` 408/64, `loop_alias_stress` 558/62,
`medium_enclosed_maze` 472/62, **`open_obstacle_field` 533/63**,
`rough_local_dynamics` 544/64, `small_enclosed_maze` 501/46,
`visual_sensor_stress` 519/63. All eight families retained.

Restricting to corpus-recorded successor *pairs* instead would have given
698/65 with **zero** selection `open_obstacle_field`.

## Training curves

Both arms: identical 4,031/488 subset, identical initial weights
(`830e2f05…`, hash-verified), fresh optimiser, same fp16 caches, same runner,
seed `2026080651`, 6 epochs, batch 4, lr 3e-4.

| epoch | control `e1` | rollout `e1` | rollout `e2` |
|---:|---:|---:|---:|
| 0 | 0.46056 | 0.46222 | 0.46861 |
| 1 | 0.38978 | 0.39338 | 0.41114 |
| 2 | 0.37417 | 0.37669 | 0.39724 |
| 3 | 0.36588 | 0.36806 | 0.38976 |
| 4 | 0.36027 | 0.36232 | 0.38447 |
| 5 | **0.35679** | **0.35877** | **0.38096** |

Despite weighting `e1` at 1.5×, the rollout arm ends marginally *worse* on `e1`
(0.35877 vs 0.35679).

## One-step selection metrics, every epoch

| epoch | ctrl s1 cos | roll s1 cos | ctrl margin | roll margin | ctrl occIoU | roll occIoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6991 | 0.6936 | +0.0201 | +0.0162 | 0.0647 | 0.0609 |
| 1 | 0.7221 | 0.7173 | +0.0362 | +0.0358 | 0.1595 | 0.1040 |
| 2 | 0.7273 | 0.7232 | +0.0404 | +0.0385 | 0.1671 | 0.1293 |
| 3 | 0.7318 | 0.7313 | +0.0432 | +0.0436 | 0.1876 | 0.1438 |
| 4 | 0.7348 | 0.7352 | +0.0459 | +0.0462 | 0.2328 | 0.2136 |
| 5 | **0.7333** | **0.7349** | **+0.0483** | **+0.0487** | **0.2626** | **0.2550** |

## Step-two metrics, every epoch

| epoch | ctrl s2 cos | roll s2 cos | ctrl s2 margin | roll s2 margin |
|---:|---:|---:|---:|---:|
| 0 | 0.6697 | 0.6698 | +0.0230 | +0.0181 |
| 3 | 0.6999 | 0.7024 | +0.0548 | +0.0512 |
| 5 | 0.6935 | **0.7057** | +0.0600 | +0.0599 |

Two-step persistence baseline latent cosine: **0.4267**. Both arms beat it by a
wide margin, and both beat their shuffled action *sequences*.

**The rollout arm's clearest effect is on step two: cosine 0.7057 vs the control's
0.6935 (+0.0122).** That is the one place the bundle does what it was meant to do.

**A caution about the step-two margin.** The control reaches +0.0600 at step two —
*above* the +0.0586 gate that step one fails, despite the control never seeing a
rollout term. Longer horizons make the action sequence easier to discriminate, so
step-two margin alone is a misleading success signal; a rollout arm must beat the
control, not merely beat zero. Here the two are tied (+0.0599 vs +0.0600).

## Fixed-probe one-step spatial, final epoch

| | occ IoU | precision | recall | occ fraction |
|---|---:|---:|---:|---:|
| true future (reference) | 0.4971 | — | — | — |
| **persistence (gate)** | **0.3128** | 0.5067 | 0.4498 | 0.01645 |
| control | 0.2626 | **0.6835** | 0.2989 | 0.01064 |
| rollout | 0.2550 | 0.6186 | 0.3026 | 0.00955 |
| *(target)* | | | | *0.00701* |

Neither beats persistence. Both predict *less* occupancy than persistence at
higher precision — the same erase-rather-than-smear failure seen at every
predictor size.

### `open_obstacle_field`

| | occ IoU | precision |
|---|---:|---:|
| control | 0.1223 | **0.3320** |
| **rollout** | **0.1368** | 0.3020 |
| persistence | 0.1329 | 0.1930 |

The rollout arm is the **first** frozen-encoder predictor in this line to beat
persistence on `open_obstacle_field` occupied IoU (0.1368 vs 0.1329). Gate 3
passes. On an unconverged curve this is suggestive, not conclusive.

### Step-two spatial — descriptive only

82 native-labelled selection rows, **one** `open_obstacle_field` row. No family
claims and no open-field gate are drawn from this.

| | occ IoU |
|---|---:|
| persistence | 0.2722 |
| control | 0.2083 |
| rollout | 0.2021 |

## Gate status at epoch 5 (recorded, not decisive)

| gate | rollout |
|---|---|
| 1. step-1 occ IoU > persistence | ✗ 0.2550 vs 0.3128 |
| 2. step-1 margin ≥ +0.0586 | ✗ +0.0487 |
| 3. o-field step-1 > persistence | ✓ 0.1368 vs 0.1329 |
| 4. not diffuse over-prediction | ✓ 0.00955 vs 0.01645 |
| 5. step-2 beats persistence and shuffled sequence | ✓ |
| 6. step-2 degradation bounded, in interface | ✓ |

**These are read off an unconverged curve and do not support a rejection.**

## Masks and denominators

Step-1 mask: the existing frozen definition, threshold `0.7618998289108276`,
94,168 of 374,784 tokens. Step-2 mask: 75th percentile of `|LN(y2) − LN(now)|`
computed on **train only** and frozen before selection was touched, threshold
`0.8970220685005188`, 92,913 tokens.

## Next

The registered next step is to establish convergence before interpreting this
comparison at all — both arms need a longer schedule, which is **not authorised
by this document**. Only then can the bundle be accepted or rejected, and
acceptance would still require the `1.5·e1` attribution control.

Any rejection, when it comes, will be scoped to **this sliding-three-frame
rollout formulation at 17.2M with the `1.5·e1 + 0.5·e2` bundle** and will not
generalise beyond it.

No longer context, proprioception, action-token conditioning or encoder movement
was launched. The operational gate is unchanged: a frozen-encoder predictor must
beat persistence (0.3128 here, 0.3133 on the full one-step set) on future
occupied geometry under the fixed true-future probe before encoder movement is
reintroduced.
