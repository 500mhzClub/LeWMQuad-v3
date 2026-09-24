# Go2 RGB fixed-teacher causal posterior-reweighted transition-expert trajectory H4 JEPA V1 preregistration — 2026-07-28

## Decision and scientific question

- Preregister exactly one fresh bounded RGB-only JEPA falsification of a
  **causal posterior-reweighted transition-expert state**. Source, focused
  proofs, independent review, and a zero-training bound preflight must finish
  before execution.
- Action-attributed causal system-identification V1 is a valid terminal STOP,
  recorded by commits `c86ba74d5601264143f028d0a4a74400c044ff27` and
  `12be3a679ec26f522ac64a38ce6cdc613c30ffaa`. Its corrected result document is
  13,862 bytes, SHA-256
  `27ee3aa51439b9ed2fa5e7973f38a54bcc54811d861135c2676d0a9120d6d7d2`.
- That probe produced the strongest current combination of generic and
  action-conditioned point prediction: combined score `0.735384`, H4 score
  `0.772121`, H4 action gap `+0.087621` with positive lower bound and eight-
  family breadth, and best-atom H4 error `1.151557`. It still made correct
  history harmful in every family: H4 gap `-0.013267`, bootstrap lower
  `-0.015510`, and zero of eight positive families. HOLD was positive in only
  four families and reached `-0.023626` in the worst family.
- Continuous recurrent state, dense history, incoming increments, latent
  momentum, and a writable error-by-action matrix have all failed to make
  ordered history useful. The narrow remaining hypothesis is different: the
  two observed transitions may identify which of several persistent learned
  action-response regimes applies, without needing another continuous history
  vector.
- This is not a retry, continuation, or checkpoint successor. It starts only
  from the accepted N320 encoder tensors. No predecessor predictor tensor,
  checkpoint, trace, optimizer state, RNG state, or non-JSON runtime artifact
  may be listed, statted, hashed, opened, copied, or reused.

## Novelty and closure boundary

- Four coherent equal-mass trajectory atoms already showed broad persistence
  and distribution value. Mean-centered categorical action conditioning then
  showed broad requested-action value. Both useful mechanisms are retained.
- Recurrent H4 V1--V3, dense spatiotemporal attention, factual shared
  transitions, factorized incoming increments, latent momentum, and system-ID
  matrices already tested learned continuous temporal carriers. The successor
  has no learned history updater and no continuous history statistic.
- Patch correspondence, transport, flow/warp, cost volume, retrieval, inverse
  dynamics, and per-action operator branches are closed. The successor performs
  no matching, spatial transport, inverse classification, or action bank.
- Compact learned targets, whitening/CCA, moving targets, history/action
  hinges, cyclic ranking, and reconstruction/navigation objectives are closed
  or deliberately excluded. The accepted fixed N320 patch lattice remains the
  target, and all corruptions remain evaluation-only.
- The new mechanism is exactly the conjunction of:
  1. four shared-parameter, mode-conditioned categorical-action transition
     experts;
  2. uniform initial expert mass and exactly two fixed normalized-evidence
     posterior updates after the two observed priors;
  3. history persists only as four probabilities, which affect prediction
     only by weighting the action-conditioned future trajectory distribution.
- Removing the fixed posterior update, adding a learned likelihood/gate/
  temperature, writing a continuous memory, letting probabilities move latent
  content directly, changing expert count, or adding per-action operators
  creates an unreviewed or already closed mechanism and is forbidden.

## Frozen inputs, target, and causal schedule

- Preserve the exact causal V2 schedules without regeneration, filtering,
  reordering, resampling, or outcome mining:
  - train: 16,000 rows, 10,328,000 bytes, SHA-256
    `aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77`;
  - validation: 2,048 rows, 1,317,888 bytes, SHA-256
    `83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6`;
  - manifest: 26,926 bytes, SHA-256
    `d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e`.
- Every row remains seven RGB endpoints `e0:e6` and six requested categorical
  primitives `p0:p5`, with exact reset-safe boundary
  `F(i-1,5) --p_i--> F(i,5)`.
- Online normalized history features are
  `z_i=normalize(E_online(e_i))`. The accepted N320 encoder is the sole
  initialization input and fixed target. Only its reviewed `encoder.*` prefix
  may initialize fresh online and target encoders; the target stays no-grad,
  byte-identical, and at zero EMA updates.
- Future RGB `e3:e6` is visible only to the fixed target encoder under
  `no_grad`, after prediction. No target, future RGB, executed/clipped command,
  pose, odometry, depth, flow, label, map, geometry, reward, or navigation
  value may enter the predictive state.

## Frozen K4 expert state

- There are exactly four coherent transition experts. At every causal time the
  complete state is:

  ```text
  s_t = ({q_t^k}_{k=1..4}, w_t)
  ```

  Each `q` is one normalized feature lattice. `w` is exactly four strictly
  positive probabilities summing to one. There is no other hidden state.
- Initialize from `e0` only:

  ```text
  q_0^k = z_0
  w_0^k = 1/4
  ```

- Four learned mode embeddings are centered across experts. Together with one
  learned spatial embedding table, they enter the exact inherited one-layer
  `_ActionFreeBeliefContext` through its centered context input. They never add
  directly to `q` or `w`.
- The complete nine-row learned categorical action table is transformed and
  centered exactly as in the successful factorized mechanism:

  ```text
  c_j = A(E_action[j]) - (1/9) * sum_l A(E_action[l])
  ```

  `A` remains non-affine LayerNorm, bias-free `192 -> 192` Linear, Tanh, and
  bias-free `192 -> 192` Linear. There are no numeric command parameters,
  one-hot history slots, per-action experts, or HOLD special cases.
- For the shared tensor belief API, the four `q` lattices are followed by one
  serialization-only carrier. Its first four row-major scalars contain `w`;
  every other scalar is exact zero. Packing/unpacking must reject nonfinite,
  nonpositive, nonsimplex, or nonzero-padding states. The carrier has no
  spatial meaning and cannot store other history.

## Shared action-conditioned expert prior

- Execution uses inherited width `192`, 256 patch tokens, one exact inherited
  action-free spatial context `B`, one exact inherited action tower `A`, and
  one shared bias-free `192 -> 192` output head `W0` initialized to exact zero.
- The same prior is called once on every `p0:p5` edge for every expert:

  ```text
  b_t^k       = B(q_t^k, centered_mode_k + spatial_context, 0)
  delta_t^k   = W0(b_t^k * c_(p_t))
  q^-_(t+1)^k = renorm(q_t^k + delta_t^k)
  ```

- `B` cannot see `w` or the current action. Probabilities cannot enter `B`,
  `W0`, `q`, or `delta`; their sole predictive use is distribution mass in the
  proper future score and readout. There is no generic state-only successor.
- Complete-table centering makes the mean pre-renormalization increment across
  all nine current actions exactly zero. `W0=0` makes every expert and control
  exact persistence at update zero. HOLD uses its ordinary centered code.

## Fixed causal normalized-evidence update

- On `p0` and `p1` only, emit the complete expert prior before seeing its
  destination. Once online `z_(t+1)` is available, compute the full-lattice
  squared prior error:

  ```text
  d_(t+1)^k = mean_token(sum_feature((q^-_(t+1)^k-z_(t+1))^2))
  dbar       = (1/4) * sum_j d_(t+1)^j
  L_k        = exp(-d_(t+1)^k / (dbar + 1e-6))
  w_(t+1)^k  = w_t^k * L_k / sum_j(w_t^j * L_j)
  q_(t+1)^k  = z_(t+1)
  ```

- `1e-6` is the inherited normalization epsilon. The coefficient and
  likelihood form are exact. There is no learned or configurable temperature,
  variance, gain, prior, decay, gate, floor, top-k, straight-through choice,
  entropy term, resampling, or expert-count parameter.
- Equal expert errors leave probabilities unchanged exactly after
  normalization. A lower error strictly increases that expert's posterior odds
  relative to a higher-error expert. The update is differentiable and remains
  inside the one joint graph; it has no separate loss or optimizer.
- Exact causal event order is:

  ```text
  initialize(e0,w=uniform)
  prior/score(p0) -> evidence_update(z1) -> assimilate(q=z1)
  prior/score(p1) -> evidence_update(z2) -> assimilate(q=z2)
  prior/score(p2) -> prior/score(p3) -> prior/score(p4) -> prior/score(p5)
  ```

- After `e2`, the belief contains only `{q2^k},w2`. `w2` remains bitwise fixed
  over `p2:p5`; there is no future evidence update. Future prediction receives
  only that state, the current categorical action, centered modes, and shared
  model parameters.

## Proper weighted trajectory score

- Let `Y_kh` be expert `k` at future horizon `h`, `T_h` the fixed target, and
  `D` the inherited token-lattice Euclidean distance divided by square root of
  token count. The posterior-weighted marginal energy score is:

  ```text
  ES_h = sum_k w2_k D(Y_kh,T_h)
         - 0.5 * sum_k sum_l w2_k w2_l D(Y_kh,Y_lh)
  ```

- The joint score uses the same formula after flattening all four horizons
  into one coherent trajectory. The combined future score remains exactly
  `0.5*ES_joint + 0.5*mean_h(ES_h)`. At uniform `w`, this reduces algebraically
  to the inherited equal-mass K4 energy score, including all ordered pairs and
  zero diagonal.
- Preserve the exact three-term objective except for the necessary weighted
  conditional future distribution:

  ```text
  L_total = 1.0 * online_e0:e2_to_fixed_teacher_alignment
          + 0.5 * equal_mass_ES_K4(all_six_realized_local_innovations)
          + 0.5 * posterior_weighted_ES_K4(open_loop_p2:p5_states, w2)
  ```

- The equal-mass all-six local term trains every expert and may not use a
  destination-derived posterior to rescore the prior that produced it. The
  `p0:p1` local-prior diagnostic likewise remains the exact equal-mass causal
  score. Future local and cumulative metrics use `w2`.
- Wrong-action and all-HOLD controls reuse the real branch's `w2`. Reversed and
  reset-history controls causally recompute their own posterior. Persistence
  is weight-invariant because its atoms are identical.
- Weighted evaluation uses the exact posterior mean for the spherical
  centroid and `sum_k,l w_k w_l D(Y_k,Y_l)` for pairwise spread. Best-atom
  squared error remains the minimum over the four support locations. All
  normalized denominators, family macros, bootstraps, selection logic, and
  thresholds otherwise remain unchanged.

## Joint optimization and forbidden objectives

- The online encoder, centered modes, spatial context, action embedding/tower,
  and shared transition head form one model and graph. Sum the three losses,
  call backward once, clip the inherited disjoint groups, and step one AdamW
  once per update. There is no separately trained predictor, inference model,
  system identifier, or expert checkpoint.
- Preserve seed `20260727`, float32, batch 16, encoder LR `1e-4`, fresh
  history/predictor LR `3e-4`, weight decay `1e-4`, betas `(0.9,0.999)`,
  epsilon `1e-8`, and groupwise clipping at `1.0`.
- There is no posterior supervision, action/history/HOLD ranking, hinge,
  inverse classification, expert label, best-of-K objective, likelihood loss,
  entropy/diversity bonus, learned mixture mass, reconstruction, target
  compressor, whitening/covariance term, navigation loss, or geometry target.
- Cyclic wrong action, all HOLD, reversed history, reset history, persistence,
  collapsed centroid, expert spread, and posterior statistics are evaluation-
  only.

## Selection, gates, and cap

- Observe updates `0,250,500,750,1000`. Execute exactly 1,000 updates and
  16,000 training presentations, with five complete 2,048-row validation
  passes totaling 10,240 presentations and a 5,400-second active-GPU cap.
- Select the eligible noncollapsed trained observation with minimum validation
  posterior-weighted combined normalized energy. No gate-aware, tiered, early,
  post-hoc, or entropy-aware checkpoint choice is allowed.
- Preserve the existing 32 gate names, thresholds, family rules, bootstrap
  seeds/replicates, noncollapse checks, fixed-target checks, and exact-cap
  checks. Weighted-score substitution is the sole required evaluator change.
- PASS still requires, among all other gates, H4 action gap at least `0.05`,
  H4 history gap at least `0.03`, positive action/history bootstrap bounds, at
  least six positive action/history/HOLD families, no family action/HOLD floor
  violation, persistence, distribution value, and update-zero exactness.
- PASS is bounded perception/world-model development evidence only. It grants
  no checkpoint read, navigation, G2, held-out/sealed access, promotion,
  production, or deployment authority. STOP closes this exact regime-
  selection mechanism without expert-count, likelihood, epsilon, seed,
  schedule, duration, or score-coefficient variants.

## Required source proofs

- Prove exact K4 q shapes, uniform initialization, strict posterior simplex,
  q-plus-probability-only pack/unpack, exact-zero padding, centered modes and
  actions, and update-zero persistence.
- Instrument one initializer, the same prior exactly six times, exactly two
  evidence updates after the correct observed priors, and zero future updates.
  Sentinels must prove `p0/z1` and `p1/z2` association without off-by-one use.
- Prove the exact relative-error, exponential likelihood, multiplication, and
  normalization arithmetic. Equal errors must preserve prior odds; lower error
  must increase posterior odds; probabilities must stay finite, positive, and
  normalized without learned state or configuration.
- Prove `w` cannot alter any expert latent or increment and remains bitwise
  fixed during future rollout. Zeroing or permuting probabilities may change
  only weighted distribution readout; expert locations remain identical.
- Prove the weighted marginal/joint/combined energy formulas, exact uniform-
  weight reduction to the inherited score, expert/weight joint-permutation
  invariance, finite gradients, and correct weighted centroid/spread controls.
- With `e2` and future actions fixed and heads opened deterministically,
  changing ordered `(e0,p0,e1,p1)` evidence must change `w2` and the weighted
  future score while leaving expert locations from the same packed q/action
  state unchanged. Reversed/reset histories must compute independent weights.
- Prove changing requested future action changes expert trajectories while
  weights stay fixed; HOLD uses the same route. Perturbing future RGB may
  change only fixed-target outputs and scores, never predictions or weights.
- Prove exact loss arithmetic: equal-mass all-six local, weighted cumulative
  future, and teacher alignment. Require disjoint/full optimizer coverage, one
  summed backward/step, correct zero-head gradient staging, opened gradients
  through online encoder/modes/context/action/head/posterior evidence paths,
  and absent target gradients.
- Runner tests must bind the exact V2 schedules, weighted evaluator, all 32
  unchanged gates, seed, caps, complete source closure, truthful receipts,
  normal/caught-failure JSON chains, and rejection of every binding, retry,
  resume, seed, cap, update, presentation, batch, likelihood, expert-count, or
  arbitrary-checkpoint override.

## Custody and one-shot execution order

- Exclusive output root:
  `.generated/go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_trajectory_h4_jepa_v1/probe_v1`.
- The root must be absent and is created mode `0700` only after exact source
  closure and a clean bound preflight. Preflight is zero-RGB, zero-training,
  and zero-reservation. Execution reservation precedes schedule, N320, RGB, or
  Torch runtime access.
- One fresh attempt only. Reservation, operational terminal, scientific STOP,
  or PASS consumes it. There is no retry, resume, repair, replacement attempt,
  second seed, extension, or predecessor-checkpoint reuse.
- Runtime checkpoints/traces are write-only. Terminal audit may read only the
  six canonical JSON receipts by exact path and must never discover, list,
  stat, hash, or open a runtime checkpoint.
- Ordered work is: commit this preregistration; implement focused model,
  weighted evaluator, runner, and proofs; freeze hashes; obtain independent
  science/custody review; run one bound preflight; execute once only if it
  passes; audit exact JSON receipts and record PASS or STOP.

## Source-only scope incident

- During read-only successor implementation mapping, one broad `rg` command
  used an ineffective `!*navigation*` glob and emitted five source-code lines
  from `scripts/benchmark_topo_nav_e2e.py`. The lines concerned an unrelated
  local posterior/reset/replan implementation.
- No navigation command, generated input, dataset, checkpoint, RGB, held-out,
  sealed, or runtime artifact was opened or run, and no file was edited. The
  search was stopped and remaining work was restricted to explicitly named
  JEPA model/evaluator/runner files. This source-only exposure grants no
  navigation or downstream authority and contributes no scientific input to
  this preregistration.
