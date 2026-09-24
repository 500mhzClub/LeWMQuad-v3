# Go2 RGB fixed-teacher latent-momentum causal innovation-filter trajectory H4 JEPA V1 preregistration — 2026-07-28

## Decision and scientific question

- Preregister exactly one fresh bounded RGB-only JEPA falsification of a
  **latent-momentum causal innovation filter**. Source implementation, tests,
  review, and a non-training bound preflight must complete before execution.
- Factorized conditional-increment V1 is a valid terminal STOP. It proved that
  mean-centered categorical action conditioning can be broad and useful: H4
  action gap `0.108100`, bootstrap lower bound `0.094631`, and eight of eight
  positive families. It simultaneously showed that its ordered-history path
  was harmful in all eight families and that HOLD breadth was only three of
  eight.
- The question is now narrow: can a shared predict-before-observe filter turn
  ordered RGB/action context into a useful recursive predictive state while
  retaining the learned requested-action signal?
- This is not a retry, continuation, or checkpoint successor. It must start
  from the accepted N320 encoder tensors only. No factorized-V1 runtime tensor,
  checkpoint, trace, optimizer state, RNG state, or non-JSON artifact may be
  listed, statted, hashed, opened, or reused.

## Novelty boundary

- A generic causal RNN/GRU/Transformer bottleneck is already closed by
  recurrent H4 V1. Persistence-residual/gated repairs and history hinges are
  closed by recurrent V2; direct fixed-teacher delta readout is closed by V3.
- Direct factual-carrier replacement is closed by factual shared-transition
  V1/V2. Raw dense history/direct horizon queries, compact learned targets,
  whitening/CCA, local fixed-lag RGB differences, action-query CE/NLL, scalar
  action gains, per-action operator banks, inverse dynamics, flow/warp, and
  geometry/BEV transport are also closed.
- The new mechanism is defined by the conjunction of:
  1. one shared second-order latent prior with content and momentum;
  2. a learned posterior update driven by the newly observed innovation only
     after its prior has been emitted and scored;
  3. state-only recursive future rollout with the successful mean-centered
     categorical action correction.
- Removing any of those three commitments would collapse this proposal into a
  previously rejected family and is forbidden.

## Inputs, target, and causal schedule

- Preserve the exact causal V2 schedules without regeneration, filtering,
  reordering, resampling, or outcome mining:
  - train: 16,000 rows, 10,328,000 bytes, SHA-256
    `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77`;
  - validation: 2,048 rows, 1,317,888 bytes, SHA-256
    `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6`;
  - manifest: 26,926 bytes, SHA-256
    `d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e`.
- Each row remains seven RGB endpoints `e0:e6` and six requested categorical
  primitives `p0:p5` with the exact boundary
  `F(i-1,5) --p_i--> F(i,5)`.
- Online normalized history features are
  `z_i = normalize(E_online(e_i))`. The accepted N320 encoder is the only
  initialization input and fixed target. Only its reviewed `encoder.*` prefix
  may be copied into fresh online and target encoders; the target remains
  no-grad and byte-identical with zero EMA updates.
- Future RGB `e3:e6` is visible only to the fixed target encoder under
  `no_grad`, after all predictions have been formed. No target, future RGB,
  pose, odometry, executed command, depth, flow, label, map, or geometry value
  may enter the predictive state.

## Frozen K4 latent-momentum filter

- There are exactly four coherent equal-mass state atoms. Each state is only:

  ```text
  s_t^k = (q_t^k, v_t^k)
  ```

  where `q` is a learned feature-lattice content state and `v` is learned
  feature-lattice momentum. These have no physical units and are not pose,
  metric velocity, or hand-coded geometry.
- Initialize once from `e0`:

  ```text
  q_0^k = z_0
  v_0^k = 0
  ```

  Four learned mode embeddings are centered across modes and may enter the
  learned state context and observer. They never add directly to `q` or `v`,
  preserving update-zero persistence within the inherited `1e-5` tolerance.
- Transform the complete nine-row categorical action table, then center it:

  ```text
  c_j = A(E_action[j]) - (1/9) * sum_l A(E_action[l])
  ```

  Centering occurs after the complete action tower. There are no numeric
  command values, physical primitive parameters, per-action operators, or HOLD
  special cases.
- One action-free learned spatial context `B`, one shared bias-free
  zero-initialized acceleration head `W0`, and the exact same prior transition
  are reused on all six edges:

  ```text
  b_t^k             = B(q_t^k, v_t^k, centered_mode_k)
  acceleration_t^k  = W0(b_t^k * c_(p_t))
  v^-_(t+1)^k       = v_t^k + acceleration_t^k
  q^-_(t+1)^k       = renorm(q_t^k + v^-_(t+1)^k)
  v^-_(t+1)^k       = tangent(q^-_(t+1)^k, v^-_(t+1)^k)
  readout(s^-_(t+1)^k) = q^-_(t+1)^k
  ```

  `renorm` is the inherited radius-preserving local step. `tangent` removes
  the per-token radial component after every normalized `q` update. Thus the
  uniform-action mean acceleration is exactly zero, while learned momentum is
  the state-internal inertial route.
- For observed edges only, emit and score the prior before assimilation. The
  same learned observer `U` is used exactly twice:

  ```text
  r_(t+1)^k       = z_(t+1) - q^-_(t+1)^k
  u_(t+1)^k       = U(q^-_(t+1)^k, v^-_(t+1)^k, r_(t+1)^k,
                      centered_mode_k)
  gain_(t+1)^k    = 1 + tanh(Wg(u_(t+1)^k))
  q_(t+1)^k       = renorm(q^-_(t+1)^k + gain_(t+1)^k * r_(t+1)^k)
  v_(t+1)^k       = tangent(q_(t+1)^k,
                            v^-_(t+1)^k + Wv(u_(t+1)^k))
  ```

  `Wg` and `Wv` are exactly zero-initialized. The observer is a residual
  prior/innovation update, not a reinitializer: it must consume the prior and
  preserve its momentum path. The sole permitted factual assimilation is this
  post-prior observer update; no predictor-side or parallel assignment, copy,
  or encoding of `z` is allowed. Unit initial gain makes update zero assimilate
  factual content within the inherited tolerance while leaving momentum zero.
- Exact event order is:

  ```text
  initialize(e0)
  prior/score(p0) -> observe innovation(e1)
  prior/score(p1) -> observe innovation(e2)
  prior/score(p2) -> prior/score(p3) -> prior/score(p4) -> prior/score(p5)
  ```

- After the second observation, the belief contains only packed `(q_2,v_2)`.
  Future prediction receives only that state, the current categorical action,
  centered learned modes, and model parameters. There is no raw `z2`, explicit
  `z2-z1`/incoming-`d`, anchor slot/addition, predictor-side or parallel
  factual overwrite, dense-history memory, horizon query, or other bypass.
- On the two observed edges, the scored local innovations are
  `q^-_(t+1)-z_t`, using the registered online factual source exactly as the
  frozen evaluator does. On future edges they are the recursively realized
  state changes `q^-_(t+1)-q_t`. Raw acceleration and momentum are never used
  as prediction targets. This distinction is required because learned
  posterior content gain permits internal `q_t` to differ from factual `z_t`;
  it does not create a predictor-side `z` route.

## Unchanged JEPA objective and optimization

- Retain exactly four equal-mass coherent trajectory atoms and the proper
  energy score: 50% coherent joint trajectory plus 50% mean marginal-step
  score, including all ordered atom pairs and the zero diagonal.
- Retain the exact three-term objective and weights:

  ```text
  L_total = 1.0 * online_e0:e2_to_fixed_teacher_alignment
          + 0.5 * ES_K4(all_six_realized_local_innovations)
          + 0.5 * ES_K4(open_loop_p2:p5_cumulative_states)
  ```

- Cyclic wrong action, all HOLD, reversed history, reset history, persistence,
  centroid, and particle spread remain evaluation-only. No history/action
  hinge, ranking loss, inverse head, auxiliary classifier, reconstruction,
  navigation loss, whitening, covariance, or learned-target loss is allowed.
- The online encoder, learned modes, observer, state context, action tower, and
  acceleration head form one model and one graph. Sum the three losses,
  call backward once, clip the inherited disjoint parameter groups, and step
  one AdamW once. There is no phase, frozen predictor, second optimizer, or
  separately trained/checkpointed predictor.
- Preserve seed `20260727`, float32, batch 16, AdamW encoder LR `1e-4`, all
  fresh state/predictor parameters LR `3e-4`, weight decay `1e-4`, betas
  `(0.9,0.999)`, epsilon `1e-8`, and groupwise gradient clipping at `1.0`.

## Selection, gates, and cap

- Preserve the exact V2 evaluator, validation rows, bootstrap procedure,
  noncollapse checks, selection rule, thresholds, and all 32 gates without
  addition, deletion, relaxation, or relabeling.
- Update zero must be exact persistence with zero action, history, HOLD,
  persistence, and spread gaps within the existing tolerance.
- Observe updates `0,250,500,750,1000`. Execute exactly 1,000 updates and
  16,000 training presentations, with five complete 2,048-row validation
  passes totaling 10,240 presentations and a 5,400-second active-GPU cap.
- Select the eligible noncollapsed trained observation with minimum validation
  combined normalized energy. No gate-aware, tiered, early, or post-hoc
  checkpoint choice is allowed.
- PASS requires all 32 inherited gates. In particular, it must retain H4
  action gap at least `0.05` with positive bootstrap/family breadth, and it
  must newly earn H4 ordered-history gap at least `0.03`, positive bootstrap,
  and at least six positive families, plus broad HOLD support.
- PASS is perception/world-model development evidence only. It grants no
  checkpoint read, navigation, G2, held-out/sealed access, promotion,
  production, or deployment authority. STOP closes the exact mechanism.

## Required source proofs

- Prove exact K4 `(q,v)` state shapes, centered modes/action codes, tangent
  momentum, update-zero persistence, and realized-increment arithmetic.
- Instrument calls to prove one initializer on `e0`, the same prior transition
  exactly six times, the same observer exactly twice, and the registered event
  order. Priors must exist before `e1/e2` assimilation.
- Prove transition arguments contain only `(q,v,current_action)` plus learned
  parameters; observer arguments contain only prior state and the newly
  available online observation innovation; belief packs only `(q,v)`.
- Use hooks/sentinels and perturbations to prove there is no external
  `+z`, predictor-side `+d`, anchor, predictor-side or parallel factual
  overwrite, dense-history, direct-horizon, or future-target bypass and no
  later/future RGB leakage. The observer innovation `r` is the sole permitted
  observation-difference input.
- With `e2` held fixed and heads opened deterministically, changing ordered
  `e0,e1,p0,p1` must change the packed state and future forecast. Changing the
  requested future action must change the forecast; HOLD uses the same path.
- On an isolated prediction-score backward, require finite nonzero gradients
  on the zero heads and zero/absent staged `B/U`/action/mode gradients behind
  zero `W0/Wg/Wv`. The complete three-term loss must separately give the
  online encoder a finite nonzero alignment gradient. After deterministic head
  opening, require finite nonzero gradients in the online encoder, learned
  modes, observer, state context, action tower/embedding, and acceleration
  head; target gradients remain absent.
- Prove exact three-term loss arithmetic, causal recursive rollout, K4 mode
  permutation invariance, optimizer-group disjointness/full coverage, one
  summed backward/step, and absence of separately trained predictor state.
- Runner tests must bind the exact V2 schedules/evaluator/32 gates/caps, reject
  every binding/retry/resume/seed/cap/checkpoint override, adapt all inherited
  mechanism receipts truthfully, and produce complete normal and caught-
  failure receipt chains.

## Custody and one-shot execution order

- Exclusive output root:
  `.generated/go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_trajectory_h4_jepa_v1/probe_v1`.
- The root must be absent and is created mode `0700` only after exact source
  closure and a clean bound preflight. In execution mode, reservation precedes
  schedule, accepted N320, RGB, or torch runtime work; preflight is explicitly
  non-training, zero-RGB, and zero-reservation.
- One fresh attempt only. Reservation, operational terminal, scientific STOP,
  or PASS consumes it. There is no retry, resume, repair, replacement attempt,
  second seed, observer rerun, extension, or checkpoint reuse.
- Runtime checkpoints/traces are write-only. Terminal audit may read only the
  six canonical JSON receipts by exact path and must never list, stat, hash, or
  open a runtime checkpoint.
- Ordered work is: commit this preregistration; implement the model and focused
  tests; implement the thin inherited runner and tests; freeze exact hashes;
  obtain independent source/science/custody review; run one non-training bound
  preflight; execute once only if preflight passes; audit JSON receipts and
  record PASS or STOP.
