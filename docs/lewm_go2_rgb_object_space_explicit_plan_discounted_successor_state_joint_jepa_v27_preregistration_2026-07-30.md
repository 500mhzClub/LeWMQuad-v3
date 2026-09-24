# Go2 RGB object-space explicit-plan discounted-successor-state joint JEPA V27

Date: 2026-07-30

Status: exact scientific preregistration. This document authorizes source
implementation, source-only tests, and metadata-only preflight. It does not by
itself authorize training, GPU use, generated-output creation, checkpoint
access, navigation, G2, held-out, sealed, benchmark, promotion, production, or
deployment activity.

## Goal and single hypothesis

The repository goal remains a fully learned, RGB-only, perception-grounded
JEPA navigation stack, trained jointly and eventually validated on untouched
held-out mazes. V26 established that the object-space height-volume encoder is
active and physically informative, but its local one-step predictor never beat
copying the current latent. V27 tests one materially different hypothesis:
predicting the discounted learned state reached by an explicit ordered
four-action plan will produce a plan-sensitive temporal representation that
beats persistence and depends on the whole plan.

For one corrected H6 row, use only:

- current RGB `e2 = rgb[2]`;
- ordered future plan `a0:a3 = actions[2:6]`;
- factual future RGB `e3:e6 = rgb[3:7]` as EMA-target inputs.

Let `phi` be the online V18 object-space height-volume encoder and `phi_bar`
its stop-gradient EMA copy. With fixed `gamma = 0.9`, the sole new target is

`G = sum(h=1..4, gamma**(h-1) * phi_bar(e_(h+2))) / sum(h=1..4, gamma**(h-1))`.

A new absolute plan predictor `Psi(phi(e2), a0:a3)` predicts this one
continuous spatial state. The sole new JEPA loss is the mean existing
per-row channel-LayerNorm Smooth-L1 latent energy between `Psi` and `G`.
There is no TD or Bellman bootstrap, self-labelled successor, future RGB in
the predictor, navigation target, privileged pose, or separately trained
downstream predictor.

## Why this is not a retry of a closed mechanism

- Factual shared-transition H4 V2 recurrently produced four per-horizon fixed-
  teacher forecasts. It beat persistence by `+0.217711` but its action gap was
  only `+0.000616`, with negative history and hold controls. V27 instead makes
  one direct current-plus-plan prediction of one factual EMA object-space
  statistic, with no recurrence, teacher forcing, history encoder, particle
  support, or four horizon heads.
- Action-Query Spatial Successor V1 used one-step physical pairs and nine local
  action candidates. At update 100 it lost to persistence and failed action
  identity. V27 has one executed four-action string, no all-action query bank,
  local ranking NLL, residual identity path, or one-step target.
- WDPS/full-whitened predictive-state H4 used learned D8 compressors and
  covariance/whitening objectives. V27 retains the full spatial V18 state and
  direct per-sample correspondence; it has no learned target compressor or
  whitening loss.
- The Bellman/repeated-single-action successor proposal is explicitly
  rejected. V27 uses four factual RGB targets and can score arbitrary ordered
  plans rather than only `aaaa`.
- K-mode, codebook, posterior-expert, event-delta, increment, transport/warp,
  masked-tubelet, recurrent-history, and P25/J24 tuning variants remain closed.

V27 starts from a fresh invocation of the exact inherited V18 constructor. Its
sole input checkpoint is the accepted N320 Camera model
`.generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/checkpoint.pt`
(file SHA-256 `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
content SHA-256
`9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`).
The inherited constructor copies its encoder and migrates its Camera-evidence
state, then freshly initializes V18's point projection, volume block, semantic
decoder, and old survival/progress predictor exactly as before. No rejected
V19--V26 or H4 checkpoint, tensor, optimizer, EMA, RNG, trace, or mutable
runtime state may be opened or reused.

## Frozen model and joint update

- Base representation: exact V18 object-space height-volume model and EMA
  target, with the existing encoder, evidence lift, volume block, semantic
  decoder, one-step survival/progress machinery, and `0.996` EMA momentum.
- New head: `plan_predictor` only. It contains four separate
  `Embedding(9,16)` tables. Their four outputs are concatenated in position
  order, then passed through `Linear(64,128)`, exact GELU, and
  `Linear(128,128)`; split the result into `scale,bias` of width 64. The state
  path is `GELU(Conv2d(64,64,3,padding=1)(current))`. Fuse it as
  `state * (1 + tanh(scale)[:,:,None,None]) + bias[:,:,None,None]`, then apply
  `Conv2d(64,64,3,padding=1)`, exact GELU, and
  `Conv2d(64,64,1)` to obtain one absolute `64 x 64 x 64` state. Every weight,
  including embeddings, uses Xavier-uniform gain one; every bias is zero.
  Construction uses an isolated CPU generator seeded `20260730` and restores
  the caller RNG byte-for-byte. There is no normalization layer, dropout, skip
  from current to output, recurrence, rollout cell, horizon head, mixture,
  codebook, or stochastic branch.
- Every update uses four labelled-pair microbatches of four rows and four H6
  sequence microbatches of four rows: 16 physical pairs plus 16 H6 sequences.
- The labelled route retains V26's Camera, semantic, occupied-safety,
  survival/progress, and unchanged J24 losses. The failed P25 local
  persistence-contrastive term is absent from gradients rather than tuned.
- The H6 route contains only the new plan JEPA loss. Physical labels are never
  invented or joined onto H6 rows.
- For each physical microbatch, set the old executed-action EMA loss argument
  to a graph-connected exact zero. Thus `N27 = S + U + R + O`; P25 is not
  evaluated. Camera `C`, `N27`, and J24 are each averaged over the four
  physical microbatches and retain their inherited separate L2-to-one route
  clipping. The plan loss is averaged over the four H6 microbatches; its
  gradient over the online encoder, evidence lift, volume projection/block,
  and `plan_predictor` is separately clipped by
  `min(1, 1/max(global_L2, float32_tiny))`. Add all clipped gradients by
  parameter identity, then take exactly one optimizer step and one EMA update.
  The semantic head and old predictor are absent from the plan route.
- `plan_predictor` joins the existing predictor AdamW group at learning rate
  `3e-4`; the encoder remains at `1e-4`, all other online parameters remain at
  `3e-4`, and betas `(0.9,0.999)`, epsilon `1e-8`, and weight decay `1e-4`
  remain unchanged. The V27 partition registers `plan_predictor.*` as a
  separate plan role, then appends that role only to the optimizer's predictor
  group. The inherited physical/J24 view retains the exact old 15-tensor
  `predictor.*` inventory and excludes every `plan_predictor` parameter from
  J24's protected subset, counts, and gradients. There is no predictor-only,
  encoder-only, alternating, or downstream training step.

One pair or one H6 sequence is one training presentation. Therefore each
update is 32 presentations and the only probe ends at update 400 / 12,800
presentations, below the 16,000-presentation cap. Update 0 is structural and
update 100 is informational. There is no same-run update 1,000, retry, resume,
second seed, loss sweep, or threshold repair.

## Frozen inputs and camera rectification

The physical route reuses the exact V26 train schedule prefix through 6,400
pair presentations and the exact 495-row checkpoint-selection role. Its
labels, mappings, order, seed `20260713`, image transform, and physical
evaluator are unchanged.

The temporal route uses only corrected causal H6 V2:

| role | rows | bytes | SHA-256 |
|---|---:|---:|---|
| train | 16,000 | 10,328,000 | `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77` |
| validation | 2,048 | 1,317,888 | `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6` |

Use the first 6,400 train rows in their frozen order and all 2,048 validation
rows. Train and validation remain scene- and frame-disjoint. V2 supplies the
causal boundary-to-boundary action alignment; V1 is forbidden.

At update `u` (one-indexed), the four physical microbatches are consecutive
chunks of four from physical schedule positions `16*(u-1):16*u`; the four H6
microbatches are consecutive chunks of four from H6 row positions
`16*(u-1):16*u`. Compute the physical graphs in microbatch order first and the
H6 graphs in microbatch order second, without a random sampler or loader RNG.
For one H6 microbatch, flatten the four cropped future views from
`(B,4,3,112,112)` to `(4*B,3,112,112)`, call `encode_target` once, reshape to
`(B,4,64,64,64)`, and form `G` in float32. This ordering and the isolated new-
head initialization must leave inherited V18/V26 initial tensors unchanged.

The source `render_textured_v03` frames are square renders with actual
horizontal and vertical FOV `78.323` degrees. V18 expects horizontal
`78.323` and vertical approximately `62.837` degrees. Apply the existing
audited rectification to every H6 current and future image only: require exact
`224 x 224` RGB, center-crop `(left=0, top=28, right=224, bottom=196)` (100%
horizontal, 75% vertical), then bilinear-resize to `112 x 112` and apply the
same float32 normalization as V26: divide uint8 by 255, then channel means
`(0.485,0.456,0.406)` and standard deviations `(0.229,0.224,0.225)`. Cropping
is before resize. No other crop, augmentation, jitter, data rebuild, label
access, or image mutation is allowed.

## Mandatory metadata preflight and controls

Before source freeze, a deterministic index-only preflight must reproduce all
of these facts without opening RGB pixels:

- all 2,048 validation rows have a same-family donor with the same `a0`, a
  tail `a1:a3` differing in at least two positions, a different scene, and no
  shared RGB path;
- all 2,048 have a same-family, different-plan, different-scene, frame-disjoint
  full wrong-plan donor;
- at least 1,024 rows globally and at least 128 rows in six families have an
  exact-same-plan, different-scene, frame-disjoint donor.

The frozen deterministic donor rule selects the candidate minimizing
`((donor_index-row_index) mod 2048, donor_index)`, excluding zero offset. The observed panel
must be reproduced exactly: 2,048 tail donors; 2,048 full wrong-plan donors;
1,212 exact-plan wrong-scene rows; and exact-plan per-family counts
`137,144,141,159,184,170,127,150` in lexicographic family order. Donors supply
validation controls only, never a training target or label.

At every observation, compute plan metrics on all H6 validation rows, with the
1,212-row eligible panel used where exact-plan wrong-scene support is needed.
Let `E(x,y)` be the existing per-row channel-LayerNorm Smooth-L1 energy and
`d_i=max(E(phi_bar(e2_i),G_i),1e-4)`. The correct ratio is
`mean(E(Psi_i,G_i))/mean(E(phi_bar(e2_i),G_i))`. Every stated advantage is a
per-row energy difference divided by `d_i`: persistence minus correct, wrong-plan minus correct,
wrong-scene-target minus correct, or mean-prior minus correct.

The full wrong-plan and same-`a0` tail controls rerun `Psi` on the donor action
string while retaining the row's current RGB and factual `G`. The wrong-scene
control retains the correct prediction and compares it with the exact-plan
donor's separately EMA-encoded `G`. The mean prior is the float32 mean of all
validation `G` rows with the same family and `a0` after excluding every row
from the evaluated row's scene (leave-one-scene, never a training input); it is
accumulated by family/action and scene/action sums and counts without decoding
a semantic target.

For aggregation, first average rows within scene, then average scenes within
family, then average the eight family means equally. A family is positive only
when its equal-scene mean advantage is strictly above zero. For bootstrap, use
NumPy `Generator(PCG64(20260730))`; within each family independently resample
that family's complete scene-mean vector with replacement, average each
resampled family, then average the eight family values equally. Draw 2,000
replicates. The lower 95% bound is element at zero-based index 50 after sorting
the 2,000 finite replicate means ascending; no interpolation is used. For each
`(observation_update, metric_name)` instantiate a fresh generator with that
same seed, iterate families in lexicographic name order, and make exactly one
`integers(0, scene_count, size=(2000, scene_count))` call per family. No other
draw may occur from those generators.

## Hard update-400 gate

Every conjunct below must pass at update 400. Any failure is a valid terminal
scientific STOP.

1. All values are finite; target parameters have zero gradients; target EMA
   accounting is exact; target and online states pass the inherited V26
   noncollapse/integrity checks; aggregate mean unclamped persistence energy is
   strictly above `1e-6`; and every per-row denominator is finite.
2. The correct ratio defined above is strictly below `0.90`; normalized
   persistence advantage has scene-
   bootstrap lower 95% strictly above zero and is positive in at least six
   families.
3. A deterministic full wrong plan produces larger normalized target energy
   than the correct plan; its mean and scene-bootstrap lower 95% are strictly
   positive and at least six families are positive.
4. Keeping `a0` fixed while replacing `a1:a3` produces a normalized energy
   disadvantage of at least `0.05`; its scene-bootstrap lower 95% is strictly
   positive and at least six families are positive. This is the hard test that
   V27 used the plan tail rather than learning a reactive first-action shortcut.
5. On the eligible exact-plan panel, the correct factual target is closer to
   the prediction than the deterministic same-plan wrong-scene factual target;
   mean and bootstrap lower 95% are strictly positive and at least six
   families are positive.
6. The correct predictor beats a leave-one-scene validation target mean
   conditioned on family and `a0`; normalized mean advantage and bootstrap
   lower 95% are strictly positive and at least six families are positive.
   This rules out an action/family mean-prior shortcut.
7. The unchanged V26 physical update-400 continuation evidence also passes:
   all 12 causal checks, more than 72 of 189 physical margins, total physical
   shortfall strictly below `68.96954700805838`, and rough depth P95 strictly
   below `1.8582415819168085` metres.

No metric can compensate for another. Update-100 improvement is informative
but cannot waive the update-400 gate.

## Terminal authority

On STOP, every V27 checkpoint is inaccessible and cannot seed a retry or
nearby mechanism. The exact explicit-plan discounted-successor-state
formulation is closed.

On PASS, publish the update-400 checkpoint and complete receipts only as a
bounded development scale seed. PASS may justify a separately preregistered
scale phase that resumes that checkpoint, so the successful 400 updates need
not be repeated. PASS does not itself authorize that resume, navigation,
probability calibration, G2, held-out, sealed, benchmark opening, promotion,
production, or deployment. The existing sealed V4 30-scene benchmark remains
unopened.

`G` is an aggregate of states expressed in four different future ego frames.
It is therefore a learned trajectory-feature statistic, not a current-frame
metric map. This probe must not feed `G` or `Psi` through V18's instantaneous
semantic/survival physical decoder or claim navigation readiness from the
temporal gates. A later, separately reviewed planning interface is required
after—and only after—this representation passes.
