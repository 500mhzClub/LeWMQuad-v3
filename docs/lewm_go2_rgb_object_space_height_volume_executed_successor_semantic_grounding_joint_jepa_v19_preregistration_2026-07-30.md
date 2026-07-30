# V19 executed-successor semantic-grounding joint-JEPA preregistration

Date: 2026-07-30

## Status and authority

This document preregisters source work for one fresh RGB-only development
probe.  It authorizes implementation, CPU-only synthetic tests, recursive
source-closure work, and independent source review.  It does **not** authorize
dataset or generated-input access, checkpoint or tensor access, GPU work,
training, execution, calibration, G2, navigation, held-out, sealed,
production, promotion, or deployment work.  Runtime authority must be a
separate, content-bound document issued only after source freeze, independent
review, and narrow clean-export certification.

The controlling V18 result is commit
`f2e290ce42f7b0cd142131f3272d1119b7b5d3d1`, file
`docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v3_scientific_result_2026-07-30.json`,
file SHA-256
`48f1168b33b6bf8cc7c437940ed0dabef9b5a29802813e3a1351e8bba1e2875a`,
canonical content SHA-256
`b30742fb5f66a414c8789124e0eda0562b9efa2732f1f31db26333abb2018703`,
and byte count `11380`.

## Scientific finding being acted on

V18 update 400 was a valid scientific negative, not an infrastructure or
gradient-connectivity failure.

- The learned RGB encoder and object-space height volume improved from 39 to
  89 passed physical margins, reduced total shortfall from
  `191.5839970395566` to `53.14088230907582`, and reduced rough depth p95 from
  `4.869073963165282` m to `1.621789002418518` m.  It beat every registered
  matched update-400 physical comparator.
- Persistence and shuffled-action controls passed all six registered checks.
  Wrong-RGB and train-action-mean-prior controls failed all six registered
  checks at both updates 100 and 400.
- The encoder was demonstrably RGB-sensitive, every predictor and
  representation gradient route was live, and all optimizer/EMA accounting
  passed.  The registered causal-control failure is therefore downstream: the
  predictor did not turn useful scene-specific state into scene-specific
  action utility.  The representation itself remains promising rather than
  qualified because it completed zero physical scopes.
- The inherited normalized EMA-next term `P` rose from `1.0` at update 1 to
  `8.902461290359497` at update 400 and was 77.9% of the update-400 scalar
  navigation loss.  This does not by itself prove gradient domination because
  per-term gradients were not recorded.  V19 avoids that uncertainty by
  giving the new factual signal its own norm-capped predictor route rather
  than starting a coefficient search.

V18 is closed: no V18 retry, resume, alternate seed, learning-rate change,
coefficient search, threshold change, or update-1000 extension is authorized.

## One hypothesis

The V18 predictor can be made scene-dependent without changing its learned
perception state if its executed-action latent is required to decode into the
factual next semantic scene, and that requirement contributes a guaranteed
latent-transition-predictor-only gradient inside every existing joint JEPA
update.

This is called **executed-successor semantic grounding**.  It is deliberately
smaller than another predictor architecture or counterfactual-training line.
For each existing four-row microbatch:

1. Compute the unchanged V18 online current/next encodings, all-action
   prediction, EMA current/next targets, semantic loss, occupied-safety loss,
   normalized EMA-next loss `P`, survival loss, ranking loss, and Camera loss.
2. Select the predicted latent for each row's factual executed action.
3. Decode those four predicted latents through the existing V18 semantic head
   and compute
   `Q_succ = mean(final_class_macro_nll_per_row(predicted_executed_logits, next_labels)) / log(3)`.
4. Take `autograd.grad(Q_succ / 4, transition_predictor_parameters)` only,
   where the registered subset is every inherited predictor parameter except
   `predictor.swept_progress_head.*`.  Accumulate this as a fifth route,
   independently clip its global L2 norm to at most `1.0`, and add it to the
   matching tensors of the independently clipped inherited predictor route.
5. Perform the same sole AdamW step, then the same sole EMA update.  No
   representation, encoder, evidence-head, semantic-head, or target parameter
   receives a `Q_succ` gradient.

The semantic decoder remains normally trainable through the unchanged factual
current/next semantic objective.  Restricting the requested `Q_succ` gradient
to predictor parameters means only that this auxiliary route cannot directly
distort the V18 perception state.  The ordinary EMA-next JEPA route continues
to train encoder, representation, and predictor jointly in every microbatch;
V19 is not predictor pretraining, a frozen-encoder phase, or a separately
fitted predictor.

## Exact scientific delta

Relative to frozen V18/V3, V19 changes only:

- one parameter-free factual `Q_succ` objective;
- one independently norm-capped latent-transition-predictor-only gradient
  route inside the existing joint update;
- accounting and trace fields needed to expose that route and its factual
  diagnostics; and
- observation receipts that retain numeric causal comparisons already
  computed by the unchanged evaluator instead of discarding them after
  forming Boolean gates.

V19 preserves exactly:

- the V18 model class, parameter tensors, learned RGB encoder, eight-height
  object-space volume, semantic head, survival head, predictor architecture,
  action vocabulary, and initialization;
- the N320 encoder initialization source and its accepted gate;
- the training and checkpoint-selection roles, rows, labels, raw supervision,
  schedule order, microbatch size `4`, four microbatches per update, optimizer,
  learning rates, AdamW settings, clipping of all inherited routes, EMA
  coefficient/order, experiment seed, schedule seed, losses `S/P/U/R/O/C`,
  physical evaluator, and wrong-RGB mappings;
- one end-to-end online model, one frozen EMA target, one optimizer step and
  one EMA step per update; and
- the maximum `1000` updates and `16000` ordered presentations.

There are no new model parameters, RGB inputs, labels, negatives, priors,
corruptions, action mappings, histories, trajectory horizons, motion targets,
transport fields, or external teachers.  In particular, no wrong-RGB,
shuffled-action, train-action-mean-prior, cyclic-action, all-HOLD, or
checkpoint-selection mapping may enter training.

## Distinction from closed mechanisms

- Direct-BEV V5 added an all-nine state-probability-delta action-retrieval
  contrast and allowed it to affect representation learning; it degraded
  balanced perception and action discrimination by update 100.  V19 uses one
  absolute factual next-scene semantic target for only the executed prediction
  and isolates its new route to the predictor.
- Fixed-teacher local-innovation and latent-momentum probes trained explicit
  action/history corruptions and did not generalize their history dependence.
  V19 has no corruption, innovation state, H4 trajectory, history, momentum,
  recursion, or fixed teacher.
- Action-query and rigid-transport probes changed predictor architecture but
  did not learn reliable action identity.  V18 already passes the
  shuffled-action and persistence controls, so V19 does not reopen those
  architecture or transport families.
- V16/V17 temporal ray-consistency degraded physical learning and is closed.
  V19 adds no temporal consistency objective.

## Diagnostics, not extra tuned objectives

Every optimizer trace row must record finite values for:

- `Q_succ`;
- the same factual next-label NLL obtained by decoding detached current latent
  as a persistence diagnostic;
- `Q_succ - Q_persistence`;
- current-to-next changed-cell fraction;
- non-HOLD row count;
- the finite cosine between the matching 13 tensors of the accumulated
  inherited predictor gradient and accumulated `Q_succ` gradient before route
  clipping;
- the new route's preclip L2 norm, applied scale, parameter tensor count, and
  absent-gradient count; and
- the exact inherited loss and four inherited route receipts.

The new route must be finite, strictly nonzero, cover exactly the registered
13-tensor, 259008-parameter latent-transition predictor subset, and have zero
absent gradients on every completed update.  The two tensors and 65 parameters
under `predictor.swept_progress_head.*` are intentionally excluded because
`Q_succ` consumes predicted latents before the survival head; they remain
covered and trained by the unchanged inherited predictor route.  These are
structural requirements, not tunable success thresholds.  V19 does not
preregister a post-hoc `Q` coefficient, margin, temperature, changed-cell
weight, or early trend threshold.

Accounting must expose the added work exactly.  Per completed update there
remain four predictor forwards, one optimizer step, and one EMA step; there
are four new factual-successor objectives and four new gradient calls, making
12 total `autograd.grad`/backward calls rather than eight.  At update `u`,
`factual_successor_objectives = factual_successor_grad_calls = 4u`,
`predictor_objectives = 8u`, and `backward_calls = 12u`.

At every registered observation, the already-computed numeric comparison for
each of coordinate-matched persistence, shuffled action, wrong RGB, and
train-action mean prior must be retained in the immutable metric receipt.  At
minimum this includes equal-scene mean delta, bootstrap lower 95%, positive
family count, and per-family deltas.  The frozen Boolean decisions remain
unchanged and authoritative.

## One-shot schedule and gates

The sole scientific attempt is fresh initialization.  No V18 runtime tensor,
checkpoint, optimizer state, trace, metric, or output is reusable.

- Observations: updates `0`, `100`, `400`, and, only if earned, `1000`.
- Update 100: informational science observation plus mandatory structural,
  accounting, finite-gradient, target-isolation, access, and custody checks.
- Update 400 is the decisive falsification gate.  All of the following must
  pass:
  - structural and target integrity;
  - all twelve unchanged causal-control checks;
  - passed physical margin count strictly greater than `72`;
  - total physical shortfall strictly less than
    `68.96954700805838`;
  - rough depth p95 strictly less than `1.8582415819168085` m.
- Any update-400 failure is terminal with no retry or resume and no checkpoint.
- Passing update 400 continues **within the same process and attempt** to
  update 1000.  It does not restart from update 0.
- The update-1000 final gate remains the existing conjunction: inherited V12
  full arm `24/24`, at least `112/189` physical margins, total shortfall
  strictly below `33.05143763708337`, at least one complete physical scope,
  rough pixel balanced accuracy strictly above `0.8198594673963917`, rough
  ground balanced accuracy strictly above `0.647134926562893`, rough depth
  p95 strictly below `0.9777327477931971` m, and structural integrity.

Only a full update-1000 pass may publish a development checkpoint.  Such a
pass earns only a new, separately reviewed preregistration for the next
ordered development gate.  It does not directly authorize probability
calibration, G2, navigation, held-out, sealed, promotion, production, or
deployment access.

## Attempt and failure policy

- Maximum updates: `1000`.
- Maximum presentations: `16000`.
- Retry: false.  Resume: false.  Second seed: false.  Longer run: false.
- Reservation occurs before scientific payload access and consumes the sole
  attempt.
- A pre-reservation source, command, or authority rejection performs no
  scientific work and does not consume the attempt.
- Any post-reservation source, input, device, runtime, integrity, numerical,
  custody, or scientific failure is terminal for this root.  A zero-update
  science-identical integrity replacement would require its own committed
  failure receipt, preregistration, review, certification, and authority; it
  is not an implicit retry.

## Source, runtime, and custody requirements

Implementation must use new V19-named training, executor, launcher,
source-closure checker, and focused test paths while importing the exact V18
model source.  Before any runtime work:

1. freeze the implementation in a commit;
2. generate and commit a recursive path/SHA-256/byte-count source manifest;
3. obtain an independent source-only review;
4. add one exact AGENTS clean-export exception;
5. create a narrow clean source root containing only certified bindings;
6. independently rehash and certify that root; and
7. issue a separate one-shot authority bound to all preceding identities.

The execution command must use isolated Python (`-I -B`) and exactly one
visible GPU.  It must explicitly unset all competing device selectors before
setting `HIP_VISIBLE_DEVICES=0`.  Runtime must fail closed on a dirty or
uncertified source root, unexpected input, altered schedule, extra role,
wrong runtime fingerprint, extra visible device, symlink, nonregular file,
pre-existing output root, or source-binding mismatch.

The V4 30-scene sealed benchmark and every held-out, G2, navigation,
probability-calibration, rejected-checkpoint, prior-attempt tensor, production,
and deployment path remain unopened unless a later ordered gate explicitly
earns and separately authorizes access.
