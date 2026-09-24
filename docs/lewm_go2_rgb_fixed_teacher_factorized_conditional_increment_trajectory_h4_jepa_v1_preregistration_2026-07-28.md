# Go2 RGB fixed-teacher factorized conditional-increment trajectory H4 JEPA V1 preregistration — 2026-07-28

## Decision question

- The causal schedule-integrity V2 probe learned a useful, noncollapsed generic
  successor: selected combined score `0.742206799120` and H4 persistence gap
  `+0.217711077696`. It did not learn useful control: H4 cyclic-action gap was
  only `+0.000615848711`, ordered-history gap was `-0.028848747242`, and
  all-HOLD gap was `-0.000962185627`.
- V2's additive transition could send hidden state, current visual state, and
  action embedding through one shared block and then emit a delta. It could
  therefore fit common visual evolution while shrinking the action route and
  ignoring ordered history.
- This one-shot experiment asks whether replacing only that additive
  transition parameterization with an explicit incoming-increment baseline
  plus a mandatory centered state/history/action correction yields a useful
  jointly learned RGB JEPA world model.
- A STOP closes this exact mechanism. It does not authorize coefficient
  tuning, a second seed, a retry, a resume, a longer run, or another
  categorical conditional-increment variant.

## Exact learned mechanism

- Inputs remain three ordered online RGB frames `e0,e1,e2`, past requested
  primitive IDs `p0,p1`, future requested primitive IDs `p2:p5`, and fixed-
  teacher future RGB targets `e3:e6`.
- Let `z_t` be one normalized online or recursively predicted spatial latent,
  `h_t` one of four causal particle states, and `d_t` the incoming realized
  normalized-latent increment.
- The incoming increment is exact and causal:
  - `p0`: `d_0 = 0`; no missing predecessor is fabricated;
  - `p1`: `d_1 = z(e1) - z(e0)`;
  - `p2`: `d_2 = z(e2) - z(e1)`;
  - `p3:p5`: `d_t = z_t - z_(t-1)` from the preceding post-renormalization
    predicted carrier.
- One shared one-block spatial transition produces an action-independent
  belief factor `B(z_t,h_t,d_t)`. The current requested action is absent from
  this computation.
- One zero-preserving incoming-increment tower is
  `D(d_t) = Linear_no_bias(LayerNorm_no_affine(d_t))`.
- A learned categorical action table is passed through one action tower `T`.
  Its complete outputs are centered across the exact nine-action vocabulary:
  `c_a = T(E[a]) - mean_j T(E[j])`. Centering occurs after the complete action
  tower, so `mean_a c_a = 0` by construction rather than by a penalty.
- The sole raw conditional-increment input is

  `r_t = d_t + B(z_t,h_t,d_t) * (1 + tanh(D(d_t))) * c_(p_t)`.

- The predicted increment is

  `v_hat_(t+1) = W0(r_t)`,

  where `W0` is one shared bias-free linear map initialized to exact zero.
  The next carrier is
  `z_(t+1) = renormalize(z_t + v_hat_(t+1))`.
- At initialization every observed prior and future atom is exact persistence,
  preserving the V2 update-zero contract. Once `W0` opens, the action-average
  prediction is exactly the learned inertial baseline `W0(d_t)`. Current
  visual/belief content can affect a learned correction only through the
  exactly centered requested-action factor.
- The fixed `1` in `1+tanh(D)` permits different requested actions to produce
  different motion when `d_t=0`; there is no zero-motion lock. HOLD is not
  special-cased.
- The same transition object and `W0` are reused on all six `p0:p5` edges.
  The two observed priors are formed before their destination RGB is inserted.
  Factual online carriers then replace the visual prior while the causal
  particle state is retained. The four future edges are open-loop.
- Four equal-mass coherent particles retain the existing learned mode and
  spatial initialization. No horizon-specific head, action-specific operator,
  separately trained predictor, or post-hoc decoder is introduced.

## Exact objective and joint training

- The fixed target remains an unchanged copy of the accepted N320 RGB
  encoder. It receives no gradient and zero EMA updates.
- The online encoder, particle/history state, action table, factorized shared
  transition, and zero-initialized increment map train together in one model,
  one summed loss, one backward call, and one optimizer step per update.
- The objective remains exactly V2:
  - weight `1.0` three-frame online-to-fixed-teacher alignment;
  - weight `0.5` proper all-six factual local-innovation energy score;
  - weight `0.5` proper open-loop `p2:p5` cumulative-trajectory energy score.
- Each energy score remains the inherited equal mixture of joint-trajectory
  and mean marginal-horizon energy score over four equal-mass particles.
- There is no action/history/HOLD/cyclic hinge, all-action CE or NLL, inverse
  classifier, action-gain scalar, loss reweighting, whitening, reconstruction,
  semantic, geometry, navigation, variance, covariance, or best-of-K loss.
- Counterfactual actions, all-HOLD, reordered/reset history, persistence, and
  collapsed-centroid comparisons remain validation-only.

## Why this is a new category

- Patch-whitened action-residual V1-V6 already closed scalar action gain,
  trained hinges, explicit all-action energy identification, state-dependent
  flow/warp, and inverse dynamics. This model uses none of those mechanisms
  and consumes none of their checkpoints or traces.
- Geometry-anchored Action-Query V1 already closed a jointly trained token-
  local all-nine query plus action-CE successor. This probe performs one
  factual requested-action transition per edge, has no `(B,9,...)` prediction
  objective, and uses no geometry, raster, BEV, or semantic target.
- The scientific change is the recursive second-order latent state and its
  algebraic routing constraint: the only action-independent prediction route
  is the explicit incoming increment, while every state-dependent learned
  correction is action-centered.
- Numeric command values are deliberately not introduced. The action input
  remains the exact categorical interface used by V2, avoiding a simultaneous
  linear-speed/arc/yaw superposition assumption.

## Frozen data, schedule, optimizer, and cap

- Reuse the frozen development-only causal V2 schedules without rebuilding or
  reading metadata beyond the bound runner:
  - train: 16,000 rows, 10,328,000 bytes, SHA-256
    `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77`;
  - validation: 2,048 rows, 1,317,888 bytes, SHA-256
    `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6`;
  - manifest: 26,926 bytes, SHA-256
    `d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e`.
- Every edge remains same-episode `F(i-1,5) -> F(i,5)` under requested
  primitive `p_i`; no destination-action tick enters the edge.
- Seed remains `20260727`, effective batch size `16`, and observations remain
  updates `0,250,500,750,1000`.
- AdamW groups and rates remain encoder `1e-4`, history `3e-4`, predictor
  `3e-4`; weight decay `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, and
  independent group norm clipping at `1.0` remain unchanged.
- Hard cap: 1,000 optimizer updates, 16,000 training presentations, 10,240
  validation presentations, and 5,400 active GPU seconds. Expected RGB opens
  remain `183,680 = 7 * (16,000 + 5 * 2,048)`.
- This is one fresh attempt from accepted N320 `encoder.*` tensors only. No
  stopped predictor checkpoint, optimizer, RNG state, trace, or runtime output
  may be opened or reused. There is no retry or resume.

## Selection and unchanged complete gate

- Update zero must retain exact four-atom persistence, zero innovations, and
  zero action/history/HOLD/support gaps within the existing tolerance.
- Among trained observations passing the inherited noncollapse screen, select
  the minimum validation 50/50 joint-plus-marginal combined normalized energy
  score. No training loss or action/history diagnostic may select an update.
- All 32 V2 conjuncts and thresholds remain controlling, including:
  - completion, finiteness, fixed-target identity, rank, and update-zero gates;
  - combined/joint and H1-H4 prediction improvements over persistence;
  - positive and broad persistence and `p0:p1` factual-prior evidence;
  - distribution-value and particle-spread gates;
  - H4 cyclic-action gap at least `0.05`, positive bootstrap lower bound,
    nonnegative H1-H3 action gaps, at least six positive families, and the
    inherited family floor;
  - H4 ordered-history gap at least `0.03`, positive bootstrap lower bound,
    and at least six positive families;
  - positive H4 all-HOLD gap, at least six positive families, and the inherited
    family floor.
- There is no partial PASS and no threshold relaxation. PASS establishes only
  bounded development evidence that this RGB-only conditional JEPA learned a
  useful controllable latent state. It grants no checkpoint access,
  navigation, held-out/sealed access, scale promotion, or deployment.

## Source tests required before execution

- Prove one transition object is called exactly six times and all K4 output,
  belief, innovation, and receipt shapes are exact.
- Prove `c_a` is centered after the complete action tower and that, for fixed
  `z,h,d`, the mean action correction is zero at the frozen numerical
  tolerance.
- Prove `B` and `D` receive no current-action or target tensor, `W0` is
  bias-free and the only post-sum map, and no state-only delta bypass exists.
- Prove `d=0` does not annihilate the action correction, while collapsing all
  action-table rows removes only the action correction and leaves the explicit
  incoming-increment baseline.
- Prove future `d` is the post-renormalization realized `next_z-z`, not the raw
  pre-normalization head output.
- Prove exact update-zero observed/future persistence, causal pre-observation
  timing, no future-RGB leakage, recursive carrier dependence, K4 mode
  permutation invariance, and unchanged three-term loss arithmetic.
- On the first synthetic backward, require finite nonzero `W0` gradient and
  zero/absent upstream factor gradients implied by zero initialization. After
  opening `W0`, require finite nonzero encoder, history, incoming, action, and
  spatial-transition gradients; the target remains gradient-free.
- Prove the inherited optimizer groups are disjoint and cover every and only
  trainable online parameter, and prove a single summed backward/optimizer
  stage with no separately trained predictor.
- Runner tests must bind the exact V2 schedules, reject override/resume/seed/
  cap surfaces, preserve the exact evaluator and 32-gate decision, produce
  complete failure receipts, and adapt only the new model/identity receipts.

## Custody and execution order

- Exclusive output root:
  `.generated/go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_jepa_v1/probe_v1`,
  created mode `0700` only after source closure and a clean bound preflight.
- Reservation precedes index, RGB, accepted-checkpoint, or torch runtime work.
  Reservation or any operational/scientific terminal consumes the attempt.
- Runtime checkpoints and traces are write-only. Terminal review reads only
  canonical JSON receipts; it must not list, stat, hash, or open checkpoint
  files.
- Test, held-out, sealed, navigation, label, arbitrary-checkpoint, predecessor
  checkpoint, retry, and resume counters must remain exactly zero.
- Ordered work is: commit this preregistration; implement model and focused
  tests; implement the thin bound runner and tests; freeze hashes; obtain an
  independent source/science/custody review; run one exact preflight; execute
  once only if preflight passes; audit JSON receipts and record PASS or STOP.
