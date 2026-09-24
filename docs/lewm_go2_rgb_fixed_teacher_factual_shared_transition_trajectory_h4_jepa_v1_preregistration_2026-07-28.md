# Go2 RGB fixed-teacher factual shared-transition trajectory-H4 JEPA V1 preregistration — 2026-07-28

## Decision and question

- Authorize source preparation and review for exactly one bounded development
  probe. Execution requires a separately frozen source identity, independent
  source review, clean preflight, and one explicit execution receipt.
- Scientific question: can one weight-shared spatial latent transition learn
  factual primitive dynamics on the two observed history edges and reuse that
  exact transition open-loop over four future edges, while a four-atom proper
  score preserves the prediction advantage already demonstrated by K4?
- This is not recurrent-H4 V4. The stopped V1--V3 branch used separate history
  and future mechanisms, including GRU recurrence, and the later K4 predictors
  decoded all future horizons in parallel. This probe is admissible only
  because the intervening K4 result established broad persistence improvement
  and the remaining failure localized to compositional action/history use.
- A simple tied-GRU retry, a separate history/future transition, a horizon
  embedding, an action-prefix decoder, or a prediction made after consuming
  its target observation would violate this preregistration.

## Frozen inputs and custody

- Use the same frozen 16,000-row train index, 2,048-row validation index,
  main-pool census receipt, RGB root, preprocessing, seed, scene split, and
  observation order as the dual-domain predecessor.
- The accepted N320 checkpoint may be opened exactly once by the bound runner.
  Only its reviewed `encoder.*` prefix initializes the online encoder and the
  permanently fixed target encoder. Every transition, hidden-state, action,
  mode, spatial, and output parameter is freshly initialized.
- The target encoder is fixed for the entire probe: no optimizer membership,
  gradient, EMA update, or state drift.
- No stopped predictor checkpoint, tensor, trace, or private artifact may be
  listed, statted, hashed, opened, copied, or reused. No test, held-out,
  sealed, label, navigation, pose, depth, flow, BEV, raster, or privileged
  state input is authorized.
- The legacy V4 sealed benchmark remains unopened, development-only, and
  permanently ineligible for final evaluation.

## Exact model mechanism

- Let `x_t = normalize(E_online(e_t))` for the three observed RGB frames
  `t=0,1,2`. Let `z_t = stop_gradient(normalize(E_fixed(e_t)))` denote fixed
  teacher targets. There are exactly `K=4` equal-mass coherent particles.
- A fresh initializer constructs four spatial hidden states `c_0^k` from
  `x_0`, one learned mode identity, and one learned 16-by-16 spatial identity.
- One spatial Transformer transition core `F` and one shared zero-initialized
  residual output head are called on every edge `p0:p5`. For particle `k`:

  ```text
  c_(t+1)^k, delta_t^k = F(x_t^k, c_t^k, action_embedding(p_t))
  xhat_(t+1)^k = normalize(x_t^k + delta_t^k)
  ```

- `F` contains exactly one spatial self-attention block. The same module
  object and parameters are reused at all six steps and for all four modes.
  There is no GRU, separate history cell, separate future cell, horizon query,
  direct parallel horizon decoder, or per-step parameter set.
- On observed edges `p0` and `p1`, `xhat_1` and `xhat_2` are scored before the
  next observation is available to the transition. After scoring, the visual
  carrier is replaced by the factual online `x_1` or `x_2`, broadcast across
  particles, while the causal hidden state `c_1^k` or `c_2^k` is retained.
- On future edges `p2:p5`, no observation is inserted. Each predicted visual
  carrier and hidden state feed the next invocation of the same transition.
  The four particle identities remain coherent through H1--H4.
- The residual output Linear has exact zero weight and bias. Update zero is
  therefore exact persistence for every particle, action, and history, while
  the first backward opens the residual head from the scored `p0/p1` priors.
  Upstream transition, hidden, action, mode, and spatial paths receive zero
  gradient through that zero head on the first backward and open only after a
  nonzero residual head has been learned. This staged gradient is required and
  must be covered by a source test.
- Future RGB never enters the online transition graph. It is visible only to
  the fixed teacher under `no_grad` after all predictions have been formed.

## Factual joint JEPA objective

- One backward pass jointly trains the online RGB encoder, hidden-state
  initializer, action/mode/spatial embeddings, shared transition, and residual
  output head. The predictor is not trained as a separate stage.
- Form six predicted local innovations from the two scored pre-observation
  priors plus the four open-loop future transitions. Their targets are the six
  adjacent fixed-teacher innovations `z_(t+1)-z_t` across `e0:e6`.
- Form the cumulative H4 particle trajectory from the four open-loop outputs
  and compare it with fixed-teacher `e3:e6`.
- In each domain, `ES_K4` is the existing proper equal-mass energy score: 50%
  joint-trajectory score plus 50% mean marginal-step score. The prediction
  objective is fixed exactly as:

  ```text
  L_prediction = 0.5 * ES_K4(local innovations over all six edges)
               + 0.5 * ES_K4(cumulative open-loop H4 states)
  L_total = L_prediction
          + L_online_history_to_fixed_teacher_alignment
  ```

- The history alignment has weight `1.0` and covers all three factual online
  history frames. There is no teacher-forced `p2:p5` predictor auxiliary, no
  future online encoding, and no new tunable loss coefficient.
- Cyclic wrong-action, all-hold, reordered-history, reset-history,
  persistence, centroid, and particle-spread branches are evaluation-only.
  There is no synthetic action/history ranking loss, best-of-K loss,
  diversity bonus, learned mixture weight, learned variance, whitening,
  reconstruction, or navigation loss.

## Data, optimizer, and cap

- Preserve the exact deterministic train order: 1,000 updates, batch 16,
  16,000 sequence presentations, seed `20260727`, and observations at updates
  `0,250,500,750,1000`. Validation is the same fixed 2,048 rows at all five
  observations, for 10,240 validation presentations.
- Preserve AdamW, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`,
  the inherited separate norm-`1.0` clip for each trainable parameter group,
  encoder LR `1e-4`, and all fresh transition, hidden, action, mode, spatial,
  and output parameters at LR `3e-4`.
- GPU-active cap is 5,400 seconds. There is exactly one fresh attempt, no
  retry, resume, alternate seed, longer run, nearby depth change, or same-root
  repair. A complete operational failure receipt also consumes the attempt.
- The physical-pool audit records 2.896 TB of allowlisted train/validation data
  and only 0.222618% RGB exposure in this schedule. That unused capacity is
  not authority to scale this probe. Scaling is considered only after a PASS.

## Required source and update-zero checks

- The model inventory must prove one transition module object serves all six
  edges, with no recurrent PyTorch RNN/GRU/LSTM module, horizon embedding,
  action-prefix MLP, or separate step-specific transition parameters.
- Predictions must be invariant to future-RGB replacement, fixed-target
  tensors must be detached, the target must remain frozen, and the optimizer
  must cover every and only trainable online parameter exactly once.
- Causal no-lookahead tests are controlling: the scored `p0` prior must be
  invariant to replacing `e1:e6`, and the scored `p1` prior must be invariant
  to replacing `e2:e6`. Later beliefs and forecasts may change after the
  corresponding factual `e1/e2` carrier is inserted.
- A source test must show a later future output depends on an earlier predicted
  carrier, not only on an action prefix. Another must show changing `e0/e1`
  while keeping `e2` fixed can change the final belief and forecast after a
  nonzero controlled parameter perturbation.
- Mode permutation must only permute the particle set, and the proper score
  must be invariant to that permutation.
- At update zero all four particles equal fixed online-`e2` persistence within
  `1e-5`; action, hold, persistence, history, distribution-value, and spread
  gaps are zero within `1e-5`; target rank drift, target near-zero drift, and
  teacher state drift are exactly zero.
- Validation must report the `p0/p1` pre-observation local-prior proper score
  separately from the `p2:p5` open-loop local score. The `p0/p1` score is
  normalized by the exact zero-innovation persistence score on the same fixed
  teacher targets, clamped at the inherited normalization epsilon `1e-6` with
  no row filtering, macro-averaged by family through scenes, and accompanied
  by a scene-bootstrap lower bound for its persistence gap. At update zero the
  normalized `p0/p1` score is `1.0` and its persistence gap is `0.0`, each
  within `1e-5`.

## Selection and PASS gate

- Select the trained, noncollapsed observation with minimum validation
  cumulative combined normalized energy score. Update zero is not selectable.
- Reuse the full cumulative-K4 and dual-domain gate surface, including:
  - exact completion at 1,000 updates / 16,000 presentations;
  - finite observations, fixed teacher, target and online noncollapse;
  - combined and joint scores below persistence, H1--H3 below persistence,
    and H4 score at most `0.90`;
  - `p0/p1` factual local-prior score below persistence, positive persistence-
    gap bootstrap lower bound, positive gap in at least six families, and no
    family gap below `-0.02`;
  - positive H4 persistence bootstrap lower bound, persistence-positive in at
    least six families, and no family below `-0.02`;
  - combined distribution value at least `0.05`, positive bootstrap lower
    bound, positive in at least six families, and H4 spread at least `0.05`;
  - validation-only cyclic H4 action gap at least `0.05`, positive bootstrap
    lower bound, nonnegative H1--H3 gaps, positive in at least six families,
    and no family below `-0.02`;
  - H4 ordered-history gap at least `0.03`, positive bootstrap lower bound,
    and positive in at least six families;
  - positive H4 all-hold gap, all-hold positive in at least six families, and
    no family below `-0.02`.
- Frozen-train support for the evaluation control's future-four all-hold
  string is exactly `large=1`, `local_composite=122`, `loop=20`, `medium=0`,
  `open=35`, `rough=19`, `small=52`, and `visual=12` (261 rows total). The
  already-recorded all-six-hold counts are not used as a proxy. Factual hold
  transitions occur at every position in every family; the six-family breadth
  gate remains controlling and the `-0.02` floor protects against a severe
  family-specific failure while the sparse large/medium support is reported.
- PASS is all-or-nothing. It establishes bounded development feasibility for
  this exact factual shared-transition K4 mechanism only. It does not itself
  authorize checkpoint access, scaling, navigation, held-out evaluation,
  promotion, production, or deployment.
- STOP closes this exact shared spatial transition, one-block, K4, factual
  50/50 objective category. Do not answer a STOP with another block depth,
  coefficient, seed, margin, longer run, or checkpoint reuse.

## Ordered consequence

- On PASS, independently audit receipts and then preregister a larger fresh
  main-pool training schedule drawn from unused row-disjoint windows. Only a
  scaled perception model that preserves the action/history/persistence gates
  can enter the repository's ordered navigation-development gates.
- On STOP, document whether failure was factual action learning, open-loop
  prediction, particle value, or ordered belief. Choose a genuinely different
  target/state mechanism or stop; do not reopen data-format work or the
  deterministic recurrent and parallel-decoder branches.
