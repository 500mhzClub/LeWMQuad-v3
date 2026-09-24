# Go2 RGB object-space explicit-plan terminal-successor-state joint JEPA V28

Date: 2026-07-30

Status: exact scientific preregistration. This document authorizes source
implementation, source-only tests, and metadata-only preflight. It does not by
itself authorize training, GPU use, generated-output creation, checkpoint
access, navigation, G2, held-out, sealed, benchmark, promotion, production, or
deployment activity.

## Decision and one scientific change

V27 integrity replacement V2 is a valid terminal scientific failure, recorded
at commit `8dbb64caac05cca56b0edd71b5adab9b69534eff`. It completed exactly 400
joint updates and 12,800 presentations. It learned robust whole-plan, plan-tail,
wrong-scene, and leave-one-scene-prior discrimination, but its correct future
prediction remained `5.322598768213582` times the copy-current EMA baseline
instead of strictly below `0.90`. Its physical continuation gate also failed
rough depth (`2.2896822929382323 m`, required strictly below
`1.8582415819168085 m`). No checkpoint was published; V27 may not be retried,
resumed, extended, or used as initialization.

V28 tests exactly one materially different target. For one corrected causal H6
V2 row, the online path receives only current RGB `e2` and the ordered future
plan `a0:a3 = actions[2:6]`. The stop-gradient target path receives only terminal
future RGB `e6`. With the unchanged V18 online object-space encoder `phi`, its
EMA copy `phi_bar`, and the unchanged V27 plan predictor `Psi`, the sole new
target and temporal loss are

`T = stopgrad(phi_bar(e6))`

`P = Psi(phi(e2), a0:a3)`

`L28 = mean(existing_channel_LayerNorm_SmoothL1_energy(P, T))`.

The only scientific delta from V27 is `G -> T`: replace the normalized
gamma-0.9 average of `phi_bar(e3:e6)` with the single terminal state
`phi_bar(e6)`. V28 has no gamma, discounted sum, repeated-e6 surrogate, residual
current-state skip, new head, new loss, new coefficient, gradient surgery,
history input, rollout cell, extra horizon, target compressor, or checkpoint
reuse.

## Why one endpoint-only probe is warranted

- V27 proved that the direct predictor can use the ordered four-action string:
  wrong-plan, same-first-action tail, wrong-scene, and mean-prior controls all
  passed at update 400. Its correct, wrong-plan, and tail raw energies remained
  close, while persistence was much better.
- V27's target averaged four spatial volumes expressed in four different future
  robot-centred frames. That statistic was validly preregistered, but it is not
  one coherent terminal geometry state. `phi_bar(e6)` is one coherent terminal
  egocentric volume, although it is not a global or current-frame metric map.
- No tracked experiment implements the exact mapping
  `phi(e2), a0:a3 -> phi_bar(e6)` using the V18 object-space state and V27's
  direct whole-plan predictor. Recurrent H4 V1 predicted every EMA horizon and
  ended at `2.2802` times persistence. Factual shared-transition H4 V2 predicted
  every fixed-teacher horizon and reached `0.782289` at H4 but had action gap
  only `0.000616`. Dense H4 cross-attention predicted every horizon and ended at
  `1.447611` times persistence. Those recurrent, transition, and dense
  per-horizon families remain closed; V28 is not another variant of them.
- This one target-only synthesis tests whether V27's demonstrated plan
  sensitivity and prior evidence that an e6 endpoint can be predictable can
  coexist in one jointly learned object-space JEPA.

If V28 fails once, the V18 object-space plus V27 absolute-plan-predictor
deterministic endpoint branch is closed. There may be no endpoint V2, alternate
endpoint, gamma sweep, duration extension, seed repeat, residual repair, or
coefficient variant in that branch.

## Frozen model, initialization, and joint optimization

- Reuse the exact V27 model module and class, including the V18 object-space
  height-volume encoder/EMA target, four-position action embeddings, absolute
  plan predictor, parameter inventory, initialization draw order and seed
  `20260730`, and EMA momentum `0.996`. V28 adds no model parameter.
- Initialize fresh through the inherited V18 constructor from only the accepted
  N320 Camera checkpoint
  `.generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/checkpoint.pt`
  (file SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
  content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`).
  No V19--V27 or H4 checkpoint, tensor, optimizer, EMA, RNG, trace, or mutable
  runtime output may be opened or reused. The committed public V27 scientific
  result is chronology-only source evidence and grants no runtime access.
- Every update retains exactly four physical microbatches of four rows and four
  H6 microbatches of four rows: 16 physical and 16 terminal-plan presentations.
  The physical Camera, semantic, occupied-safety, survival/progress, and J24
  objectives are unchanged from V27. P25 remains absent. The only H6 objective
  is `L28`.
- Preserve every V27 optimizer group, learning rate, weight decay, beta,
  epsilon, route-specific L2-to-one clipping rule, parameter recipient, one
  combined optimizer step, and one EMA step. The online encoder,
  object-space representation, and plan predictor remain jointly trained in
  the same graph and optimizer process. There is no separately fitted
  predictor or downstream probe.
- Preserve the exact V27 physical and H6 row order, seed, initialization, and
  400-update cap. One row is one presentation, so terminal accounting is 12,800
  presentations: 6,400 physical plus 6,400 H6. Observe only updates 0, 100, and
  400. Update 100 is informational and cannot waive the terminal gate.

## Frozen data and endpoint-only access

The physical schedule, labels, 495-row checkpoint-selection role, mappings, and
evaluator are unchanged. The H6 V2 indexes remain exactly:

| role | rows | bytes | SHA-256 |
|---|---:|---:|---|
| train | 16,000 | 10,328,000 | `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77` |
| validation | 2,048 | 1,317,888 | `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6` |

Use the first 6,400 train rows and all validation rows in frozen order. Train
and validation remain scene- and frame-disjoint. For the temporal route, decode
only `e2` and `e6`. Metadata validation may inspect the registered path strings
for `e0:e6`, but RGB pixels for `e0,e1,e3,e4,e5` must never be requested or
opened. Apply the unchanged audited V27 crop, resize, float32 normalization, and
bounded validation cache to `e2` and `e6`.

The unchanged metadata-only donor preflight must reproduce 2,048 same-a0 tail
donors, 2,048 full wrong-plan donors, 1,212 exact-plan wrong-scene rows, and
per-family exact-plan counts `137,144,141,159,184,170,127,150`. It must open zero
RGB bytes and perform zero GPU work.

## Controls and hard update-400 gate

Use the exact V27 energy, normalization, scene/family aggregation, bootstrap
seed/draw order, donor rules, and thresholds after substituting `T` for `G`:

- persistence compares `phi_bar(e2)` with factual `T`;
- full wrong-plan and same-a0 tail controls change only the plan passed to
  `Psi` while retaining current RGB and factual `T`;
- wrong-scene compares the correct prediction with the exact-plan donor's
  separately EMA-encoded terminal target;
- the leave-one-scene family/a0 mean prior is formed only from terminal targets.

Every conjunct must pass at update 400:

1. Exact accounting, finiteness, no target gradients, noncollapsed online and
   target states, immutable observation state, and all custody checks.
2. Correct/persistence mean-energy ratio strictly below `0.90`; normalized
   persistence advantage mean and bootstrap lower 95% strictly positive and at
   least six positive families.
3. Wrong-plan advantage mean and lower 95% strictly positive and at least six
   positive families.
4. Same-a0 tail advantage at least `0.05`, lower 95% strictly positive, and at
   least six positive families.
5. Exact-plan wrong-scene advantage mean and lower 95% strictly positive and at
   least six positive families.
6. Leave-one-scene family/a0 mean-prior advantage mean and lower 95% strictly
   positive and at least six positive families.
7. The unchanged V26/V27 physical continuation gate: all 12 causal checks,
   more than 72 of 189 margins, total shortfall strictly below
   `68.96954700805838`, rough depth P95 strictly below
   `1.8582415819168085 m`, and structural integrity.

No conjunct compensates for another. The update-100-to-400 rough-direction
panel remains diagnostic only, matching the frozen V27 gate.

## Required source proofs and lifecycle

Focused source-only tests must prove that:

- each target-construction batch or microbatch makes exactly one no-grad
  `encode_target(e6)` call, with no four-frame flatten, repeat, or aggregate;
- changing synthetic forbidden `e0,e1,e3,e4,e5` cannot change the batch,
  target, loss, or access counters, while changing `e6` changes the target;
- the temporal batch contains exact current RGB `(B,3,112,112)`, terminal RGB
  `(B,3,112,112)`, and ordered plan `(B,4)` with no future-RGB tensor;
- for identical fresh inputs and RNG, V28's model parameters, optimizer groups,
  and predictor output are bit-identical to V27 before training;
- every V27 gradient recipient and accounting equation is unchanged except the
  route/mechanism identity and endpoint target; target gradients remain zero,
  with exactly one optimizer and EMA step;
- all validation controls use endpoint targets and retain the frozen gate; and
- a source-only composition test imports the actual runtime-selected V28
  training module and proves all 22 inherited V25 facade symbols, the complete
  V21/V23/V24/V25 schema chain, and physical-builder identity without invoking
  the builder or loading runtime data; and
- a separately authorized metadata-only preflight loads only the exact two H6
  indexes and donor panel with zero RGB, model, physical schedule, or physical
  builder access.

After implementation and focused tests, freeze an exact recursive source
closure, obtain an independent source review, and create only an explicitly
enumerated hash-bound clean export under a new AGENTS exception. A later exact
one-shot authority is required before GPU/data execution or output creation.

On scientific FAIL, publish complete immutable receipts, no checkpoint, and
close V28 and the V18 object-space plus V27 absolute-plan-predictor deterministic
endpoint branch. On PASS, publish only the update-400 bounded development
checkpoint and receipts.
PASS may justify a separately preregistered scale phase, but it does not
authorize resume, navigation, probability calibration, G2, held-out, sealed,
benchmark opening, promotion, production, or deployment. The V4 30-scene
development benchmark and all externally custodied held-out mazes remain
unopened.
