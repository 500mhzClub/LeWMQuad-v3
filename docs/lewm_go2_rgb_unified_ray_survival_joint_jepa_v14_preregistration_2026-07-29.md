# RGB Unified Ray-Survival Joint JEPA V14 preregistration

Date: 2026-07-29

Status: preregistered design only; no V14 training, checkpoint, qualification,
navigation, calibration, G2, or held-out access has occurred.

## Question and evidence

- V13 integrity replacement V3 completed its exact cap of 1,000 joint updates
  and 16,000 presentations. Its terminal scientific result is frozen in commit
  `db8502a` and file SHA-256
  `97977bd7186136fe5b1f7c5bef0b0910cb093bc4c841b336abdb0be2dc669dec`.
- V13 passed structural integrity, all inherited V12 checks `24/24`, rough
  pixel balanced accuracy, and rough ground balanced accuracy. It failed with
  `82/189` physical margins, shortfall `41.14716843735617`, zero complete
  scopes, and rough depth p95 `1.6219202995300286` m.
- At update 1,000, pixel depth median/p95 deficits contributed `56.3%` of total
  shortfall, ground-clear and distance-band deficits `22.2%`, and learned
  raster/class deficits `18.9%`. Wrong-RGB sensitivity and basic hit presence
  together contributed less than `3%`. The residual is therefore coherent
  metric-depth/free-space evidence, not missing RGB dependence or an inactive
  JEPA predictor.
- Exact Camera V4/V6 tail-depth and fixed derived-raster losses are closed.
  Camera V6 already used them for 8,000 Camera-only updates and 128,000
  presentations, reached `135/189` margins but zero complete scopes, and was
  terminally rejected. V14 does not restore those losses or compositions.

The falsifiable question is whether forcing occupied-depth and clear-ground
evidence to be two observations of one learned ordered ray distribution fixes
the inconsistency that survived V13, while retaining V13's joint JEPA learning.

## Sole material scientific change

- Start fresh from the same accepted N320 initialization used by V13. Keep the
  RGB encoder, dense decoder, 64 ordered first-hit hazard bins, within-bin
  offsets, nominal geometry, 40-plane FREE input shape, 64-plane OCCUPIED
  input shape, learned role projections, 64-channel sole JEPA bottleneck,
  semantic decoder, action-conditioned predictor, EMA target, labels, losses,
  optimizer, seeds, schedule, evaluation, and final thresholds.
- Remove the independent `40 -> 64 -> 1` ground-clear MLP and its 2,689
  parameters. Do not migrate, freeze, retain as a residual, or otherwise use
  its weights.
- Derive every nominal and calibration-bearing auxiliary ground-clear logit
  from the same learned 64-bin hazard tensor used for first-hit depth and
  OCCUPIED evidence. The existing balanced ground-clear BCE therefore sends
  its gradients into that shared hazard field.
- The existing hierarchical first-hit NLL, skew-balanced within-bin offset
  loss, and balanced ground-clear BCE remain equally weighted. No tail loss,
  raster loss, consistency coefficient, new label, data change, threshold
  change, encoder change, or separately trained predictor is introduced.

This is a joint JEPA perception mechanism, not a separately trained encoder or
post-hoc adapter. At inference it consumes normalized current RGB and the
registered fixed camera geometry only; it receives no map, collision state,
depth sensor, privileged raster, target label, or future observation.

## Registered survival transform

For each ground query, bilinearly sample all 64 learned hazard logits at the
query's calibrated image coordinate with `align_corners=False` and
`padding_mode="border"`. Border padding is fixed so a geometrically valid query
at an image boundary cannot acquire an artificial zero logit from padding. Let
sampled logit `h[j]` define
`ell[j] = log(sigmoid(-h[j]))`, the log probability of surviving depth bin
`j`. The registered near edge is `0.05` m, bin width is `0.10` m, and bin count
is 64.

For target distance `r`:

1. `u = clamp((r - 0.05) / 0.10, 0, 64)`.
2. `n = floor(u)` and `f = u - n`.
3. `log S(r) = sum(j < n, ell[j]) + I[n < 64] * f * ell[n]`.
4. Retain this exact `log S(r)` for the registered near/far identities. For
   BCE-logit conversion only, upper-clamp it to
   `log(1 - torch.finfo(dtype).eps)`, then return
   `log S_clamped - log(1 - exp(log S_clamped))` using stable `expm1`
   arithmetic. Invalid queries return neutral zero and remain excluded by the
   unchanged validity mask.

The fractional final bin is the constant-integrated-hazard interpretation.
At the near edge exact survival is one; at the far edge it exactly equals the
existing learned no-hit probability. The fixed conversion clamp resolves the
otherwise infinite near-edge logit without changing either identity. Query
sampling and the transform must be differentiable with respect to the hazard
field.

## Frozen optimization and accounting

- Preserve model/constructor seed `20260712`, schedule seed `20260713`,
  stochastic execution seed `20260728`, projection seed `20260729`, and
  bootstrap seed `20260728`.
- Preserve exact schedule-prefix SHA-256 values: update 100
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`,
  update 400
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`,
  and update 1,000
  `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.
- Preserve one float32 AdamW optimizer, encoder learning rate `1e-4`, every
  other online learning rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`,
  weight decay `1e-4`, the exact V13 route-wise norm-one clipping, four `B=4`
  microbatches per update, one optimizer step, and one EMA step per update.
- Expected parameter counts are V13 minus exactly the removed 2,689-parameter
  ground head: shared `3,102,824`, representation `22,020`, predictor
  `259,073`, total online `3,383,917`, target bottleneck `3,106,216`, and role
  projections `3,392`.
- There is one fresh attempt, no retry and no resume. Its root must initially
  be absent and is
  `.generated/go2_rgb_unified_ray_survival_joint_jepa_v14/attempt_v1`.
  Maximum accounting is 1,000 updates and 16,000 presentations. Immutable
  observations are update 0, 100, 400, and, only if earned, 1,000.

## Preflight and stopping rules

- Before reservation, require focused closed-form transform tests, near/far
  identities, monotonicity, finite gradients from ground BCE into the hazard
  head, exact absence of `ground_head` parameters, migration identity for the
  retained 11 state entries, V13 latent and nominal/auxiliary invariants, and
  one real-model synthetic joint update with exact gradient/accounting checks.
  Synthetic inputs confer no scientific presentations.
- At update 100, require structural integrity, finite state and metrics, every
  registered online gradient present and finite, active Camera and joint-JEPA
  routes, exact optimizer/EMA accounting, and zero target gradients. Otherwise
  stop terminally.
- Continue beyond update 400 only if all inherited V13 update-400 directional
  and twelve causal-control checks pass and V14 strictly beats the matched V13
  update-400 result on all three primary residual measures:
  passed margins greater than `71`, total shortfall less than
  `71.67935936391197`, and rough depth p95 less than
  `1.936374711990354` m. Equality fails. A stop at 400 consumes the attempt.
- At update 1,000 preserve the exact V13 final gate: structural integrity,
  inherited V12 `24/24`, at least `112/189` nonnegative physical margins,
  total shortfall strictly below `33.05143763708337`, at least one complete
  physical scope, rough pixel balanced accuracy strictly above
  `0.8198594673963917`, rough ground balanced accuracy strictly above
  `0.647134926562893`, and rough depth p95 strictly below
  `0.9777327477931971` m. Equality fails.
- Only an update-1,000 complete pass may publish a perception checkpoint. A
  fail publishes no checkpoint and closes V14. Improving but sub-threshold
  metrics do not authorize a seed, coefficient, loss, schedule, or architecture
  sweep.

## Authority boundary

Implementation, source review, clean export certification, and a one-shot
execution authority must be separately frozen before reservation. Until a
complete development pass, probability calibration, G2, navigation, held-out,
sealed, production, promotion, deployment, retry, and resume remain forbidden.
