# V23 action-prior-residualized wrong-scene survival-output joint-JEPA preregistration

Date: 2026-07-30

Status: preregistered fresh scientific successor only. No V23 source root,
output root, reservation, generated-input access, GPU work, training,
checkpoint, calibration, G2, navigation, held-out, or sealed access has
occurred or is authorized here.

## Frozen predecessor evidence

- The predecessor evidence is
  `docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_scientific_result_2026-07-30.json`.
  It is frozen in commit `f184a41ac99b1c66ea4db1e0b0a0845f23b48bbd`,
  with file SHA-256
  `1f4896e8f0ae8cadbf09e6f6f34417f3fa6362f9321cfd5abd0aeb09735453d0`,
  byte count `18445`, and canonical content SHA-256
  `d9c0376f381bb65c4246c9ff12611f4b563698a0539f81c63b95e8b083de18a2`.
- V22 is consumed and terminal at update 400. It passed structural integrity
  and the three physical thresholds, and it passed all persistence and
  shuffled-action checks. It passed none of the wrong-RGB or train-action
  prior checks: both controls were exact zero-delta ties in every scene and
  family. V22 therefore learned action-specific output behavior without
  measurable current-scene dependence or improvement over the action prior.
- V23 is the single evidence-led successor proposed by that failure. It is a
  fresh mechanism from exact initialization, not a V22 retry, resume,
  extension, replacement, checkpoint continuation, or coefficient variant.

## Sole scientific change

V23 removes V22's latent-space `I_two_axis` auxiliary completely and replaces
it with one direct survival-output objective,
`J23 = F + R`. The new objective asks the learned RGB-to-survival path to fit
scene-specific swept-progress labels and to beat both a wrong-scene output and
the frozen action-conditioned train prior. It adds no model component and
changes no inherited loss.

For each reviewed four-row train microbatch:

- `q[b,a]` is the expected progress in metres from the already-computed
  all-action survival logits. It is exactly the inherited
  `survival_scores_v1` value: apply sigmoid to the immediate and 15
  conditional logits, multiply the immediate probability by the cumulative
  product of conditional probabilities, and multiply the sum of the 15
  survival probabilities by `0.1` m.
- `t[b,a] = 0.1 * prefix_lengths[b,a]` metres is the already-loaded frozen
  train label. `D = 1.5` m is the fixed maximum predicted progress.
- `mu[a]` is the frozen train-action mean
  `0.1 * mean_train_rows(prefix_lengths[:,a])` metres. The V23 execution path
  derives it once from the already-loaded frozen train labels and threads the
  resulting detached `runtime.action_prior_m` tensor into training. It is
  authority-bound and is never recomputed from a microbatch or model output.
- In action-prior-residual coordinates, the target is
  `y[b,a] = (t[b,a] - mu[a]) / D`, the full prediction is
  `r_pos[b,a] = (q[b,a] - mu[a]) / D`, the wrong-scene prediction is
  `r_scene[b,a] = (q[n(b),a] - mu[a]) / D`, and the action prior is zero.
- The wrong-scene row `n(b)` is the inherited deterministic V21/V22 choice:
  scan cyclic offsets one through three and select the first row whose exact
  `scene_id` differs from row `b`. Every selected index must be in range,
  non-self, and different-scene. A microbatch with no valid row fails closed
  before tensor use.
- Let `h(x) = SmoothL1(x, 0, beta=1)`. For every non-HOLD action, define
  `E_pos[b,a] = h(r_pos[b,a] - y[b,a])`,
  `E_scene[b,a] = h(r_scene[b,a] - y[b,a])`, and
  `E_prior[b,a] = h(0 - y[b,a])`. Equivalently these are Smooth-L1 energies
  of `(q[b,a]-t[b,a])/D`, `(q[n(b),a]-t[b,a])/D`, and
  `(mu[a]-t[b,a])/D`.
- The action set is exactly indices `(0,1,2,3,4,5,7,8)` in the inherited
  nine-action order. HOLD, index `6`, contributes to none of `F`, `R`, the
  mechanism diagnostics, or their counts.
- Every non-HOLD row-action contributes to the fit term
  `F = mean(E_pos)`.
- A scene comparison is eligible exactly when
  `t[b,a] != t[n(b),a]`; equality is exact because both values derive from
  integer prefix lengths. A prior comparison is eligible exactly when
  `E_prior[b,a] > 0`. The full schedule must contain at least one comparison
  of each type, and every completed microbatch must retain its exact observed
  counts. An empty scene or prior comparison set fails closed.
- For every eligible negative comparison, the rank contribution is
  `softplus(E_pos[b,a] - E_negative[b,a]) / log(2)`. `R` is one arithmetic,
  count-weighted mean over the concatenation of every eligible scene
  contribution and every eligible prior contribution. It is not an average
  of two axis means, so an axis with few eligible pairs cannot receive half
  the objective weight silently.
- The registered auxiliary is exactly `J23 = F + R`. A tied negative has rank
  contribution one; `F` prevents success by merely making all errors large.
  No V21 or V22 auxiliary term remains active.

All prefix labels, action priors, scene identifiers, negative indices,
eligibility masks, and EMA targets are detached. The prior and wrong-scene
arms select and compare existing tensors; neither may introduce a new
forward.

## Gradient and accounting boundary

- Unlike V22's predictor-core-only auxiliary, `J23` is differentiated through
  the complete online path that produces survival output:
  `encoder.*`, `bev_lift.evidence_head.*`,
  `bev_lift.point_projection.*`, `bev_lift.volume_block.*`, and all of
  `predictor.*`, including the 65-parameter
  `predictor.swept_progress_head.*` that V22 excluded.
- `J23` excludes `semantic_head.*`, every `target_encoder.*` and
  `target_bev_lift.*` tensor, all labels and metadata, and evaluator-only
  tensors. Target-gradient counts must remain zero.
- The `J23` gradients are accumulated as one independent auxiliary route over
  the exact allowed online subset. After the four microbatches, one global L2
  unit-norm cap is applied to that route before its gradients are added to the
  independently accumulated inherited route gradients. It is not clipped
  independently per parameter group, and it does not alter inherited route
  clipping.
- The inherited joint prediction objective remains active on every
  microbatch, trains the online encoder, representation, and predictor
  together against the same stop-gradient EMA target, and is not replaced by
  output supervision. V23 therefore remains a jointly trained JEPA.
- Each completed update remains exactly four microbatch graphs, four
  all-action predictor forwards, four Camera-route gradient calls, four
  inherited joint-route gradient calls, four `J23` gradient calls, twelve
  total autograd calls, eight predictor objectives, 32 camera-frame
  objectives, 16 ordered presentations, one optimizer step, and one EMA
  step. The scene and prior rank comparisons are components of the single
  `J23` objective per microbatch.
- The loss receipt is `L = N + C + J23`. All inherited `S/P/U/R/O/C/N`
  definitions, coefficients, optimizer settings, EMA ordering, and gradient
  routes remain unchanged.

## Frozen identity

Except for removing V22's auxiliary, adding `J23`, routing it through the
specified online subset, and adding the corresponding V23 diagnostics and
lifecycle bindings, V23 preserves V18 exactly:

- the learned RGB encoder, eight-height object-space volume, semantic and
  survival heads, local action-conditioned predictor, architecture, parameter
  counts, and every initialization value;
- N320 initialization, constructor seed `20260712`, schedule seed `20260713`,
  experiment seed `20260728`, bootstrap seed `20260728`, projection seed
  `20260729`, float32 AdamW settings, learning rate, betas, epsilon, weight
  decay, EMA, inherited route clipping, parameter groups, and inherited
  joint-JEPA losses;
- the 4262-pair schedule from presentation zero, four microbatches of four,
  train and checkpoint-selection roles, source and data files, labels, camera
  metadata, observation updates `(0,100,400,1000)`, terminal updates
  `(400,1000)`, eight-family registry, physical metrics, causal controls, and
  all inherited thresholds; and
- the maximum of 1000 updates and 16000 ordered presentations.

V23 starts once, in a new process, from exact initialization. No V22 model,
optimizer, EMA, RNG, schedule state, trace, metric, receipt, output, or mutable
runtime state may be opened or reused. V22 published no checkpoint. V23 may
use predecessor documents only as exact source-review identity evidence.

## Focused source acceptance

- Pure tensor tests prove the expected-progress formula, fixed `D`, exact
  non-HOLD set, prior residualization, Smooth-L1 energies, eligibility masks,
  and the one concatenated count-weighted rank mean.
- Synthetic fixtures prove an action-prior solution cannot beat `E_prior`, a
  scene-independent action template cannot beat `E_scene`, and a correctly
  scene- and action-conditioned survival output can beat both while reducing
  `F`.
- Gather tests prove the positive and wrong-scene outputs come from the one
  inherited all-action prediction tensor, the wrong-scene action is unchanged,
  and V23 adds no RGB read, encoder pass, target-encoder pass, predictor
  forward, presentation, microbatch graph, or gradient call.
- Gradient tests prove a finite nonzero `J23` route through the encoder,
  evidence head, point projection, volume block, latent predictor, and both
  swept-progress output tensors; zero route through the semantic head and EMA
  target; one global auxiliary norm cap; and unchanged inherited routes.
- One real CPU synthetic update proves exact accounting, finite losses and
  gradients, one optimizer step, one EMA step, and no mutable-state leakage.
- Existing inherited model, evaluator, comparison, causal-control, custody,
  and source-closure tests must remain passing. Recursive closure, independent
  review, narrow clean-export certification, and separate one-shot authority
  must be committed before reservation or execution.

## Falsification and scale gates

- Update 0 is informational and must pass structural, source, access, custody,
  accounting, target-isolation, route-membership, finite-gradient, and
  single-forward integrity.
- Every train trace publishes separately for the scene and prior axes:
  eligible count, `sum(E_negative-E_pos)`, arithmetic mean advantage, and rank
  sum. It also publishes `F`, the single count-weighted `R`, and the exact
  online parameter/tensor counts reached by `J23`. These train-minibatch
  values are diagnostics only and never decide promotion.
- Update 100 is informational. It records all inherited metrics and controls
  plus the V23 mechanism diagnostics. There is no update-100 scientific
  threshold, branch, restart, or checkpoint.
- At update 400, the inherited gate is unchanged and conjunctive: structural
  integrity; all twelve causal-control checks; physical margin count strictly
  greater than `72`; total physical shortfall strictly below
  `68.96954700805838`; and rough depth p95 strictly below
  `1.8582415819168085` m. The twelve checks retain the registered three-check
  triplet for each of persistence, shuffled action, wrong RGB, and the frozen
  train-action prior: positive mean actual-utility delta, positive inherited
  bootstrap lower 95% bound, and at least six of eight qualifying families.
- V23 adds no checkpoint-selection energy gate and no new evaluation arm. Its
  train-batch scene/prior energies remain informational and cannot substitute
  for, repair, or be mixed into the inherited causal gate on actual navigation
  utility.
- Any update-400 gate failure is terminal. No checkpoint is written or
  retained. Only a complete update-400 pass may continue in the same process,
  without replaying or restarting its first 400 updates.
- Update 1000 is reached only after that pass and retains the inherited final
  gate exactly: V12 full arm `24/24`; at least `112/189` physical margins;
  total shortfall strictly below `33.05143763708337`; at least one complete
  physical scope; rough pixel balanced accuracy strictly above
  `0.8198594673963917`; rough ground balanced accuracy strictly above
  `0.647134926562893`; rough depth p95 strictly below
  `0.9777327477931971` m; and structural integrity. The V12 full arm already
  contains the registered causal checks. The V23 mechanism diagnostics are
  republished but do not replace or relax any inherited final threshold.
- Only a complete update-1000 pass publishes the development checkpoint. No
  checkpoint is written at update 100 or 400, and no incomplete or failed run
  may publish one.

## Family-stop rule

- If either inherited update-400 wrong-RGB or train-action-prior causal triplet
  fails its mean, bootstrap, or family check, the local
  latent-predictor/swept-progress-head auxiliary family is retired. There is
  no V23 retry, resume, extension, coefficient change, alternate eligibility
  rule, head variant, integrity replacement, or science-identical successor.
- Any later proposal after that failure must change the learned perception or
  world-model mechanism materially; it may not spend another attempt on a
  local output-ranking variant. This preregistration does not authorize such
  a successor.

## One-shot and protected-access boundary

- Schema/evidence prefix:
  `lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23`.
- Fresh output root:
  `.generated/go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23/attempt_v1`.
- Fresh clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-v23-survival-output-contrast-source`.
- Both roots must initially be absent. There is exactly one attempt and no
  retry, resume, recovery, extension, replacement, coefficient search,
  alternate-onset run, or second attempt.
- Authority covers only exact independently reviewed source plus frozen train
  and checkpoint-selection inputs on the reviewed runtime and hardware.
- Until a complete update-1000 pass, probability calibration, G2, navigation,
  held-out, sealed, promotion, production, and deployment remain forbidden.
  This preregistration grants none of those accesses and grants no source
  export, data, GPU, training, reservation, or execution authority.
