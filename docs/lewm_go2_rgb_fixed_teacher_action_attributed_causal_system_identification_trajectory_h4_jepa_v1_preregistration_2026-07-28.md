# Go2 RGB fixed-teacher action-attributed causal system-identification trajectory H4 JEPA V1 preregistration — 2026-07-28

## Decision and scientific question

- Preregister exactly one fresh bounded RGB-only JEPA falsification of an
  **action-attributed causal system-identification memory**. Source,
  proof tests, independent review, and a non-training bound preflight must
  complete before execution.
- Latent-momentum causal innovation-filter V1 is a valid terminal STOP. It
  passed 29 of 32 gates, improved combined prediction to `0.758407`, retained
  action value in eight of eight families, and made HOLD value broad in eight
  of eight. It failed only ordered history: H4 gap `-0.039453`, bootstrap lower
  `-0.057520`, and zero of eight positive families.
- History was harmful at every trained observation, so neither more updates
  nor a momentum gain/damping repair is justified. The narrow question is:
  can observed prediction errors, attributed to their known requested past
  actions, identify a compact sequence-specific response code that improves
  future action-conditioned prediction?
- This is not a retry, continuation, or checkpoint successor. It starts only
  from the accepted N320 encoder tensors. No latent-momentum, factorized, or
  other predecessor runtime tensor, checkpoint, trace, optimizer state, RNG
  state, or non-JSON artifact may be listed, statted, hashed, opened, or reused.

## Novelty boundary

- Recurrent H4 V1--V3, dense spatiotemporal history, trajectory-distribution,
  local-innovation ranking, factual shared-transition V1/V2, and factorized
  conditional-increment V1 already tested generic history encoders, direct
  dense history, K4 uncertainty, trained history/action controls, factual
  carrier recurrence, and raw incoming-increment modulation.
- Dual-domain trajectory H4 V1 already tested coefficient-level refinement of
  the 50/50 local-plus-cumulative proper score. This successor inherits that
  objective unchanged and makes no score novelty claim.
- Patch-residual/correspondence V1--V7 tested local correspondence,
  action-conditioned transport, whitening, inverse dynamics, and related
  Camera-V6-style mechanisms. V8's all-candidate correspondence
  identification remained near chance and closed the local-transport branch;
  V9's dense pairwise spatial cost-volume inverse remained at `1/9` macro
  balanced accuracy and closed dense inverse/cost-volume identification. None
  is reopened here.
- The older June episodic/hidden-target recurrent-memory line served a
  navigation/query pipeline with geometry supervision and steering/claim
  losses. It neither preempts this fixed-teacher H4 statistic nor contributes
  any reusable artifact; all of its runtime state remains forbidden.
- Latent-momentum V1 tested additive second-order inertial carry. The successor
  has no momentum, velocity, incoming increment, damping, or additive inertial
  route. Its history code is a calibration of how requested actions produced
  observed prediction errors; it is held fixed during open-loop rollout.
- A generic GRU/Transformer/MLP update `m'=U(m,error,action)` is already closed
  by recurrent, dense, factual-state, and filter results. The new mechanism is
  defined by the conjunction of:
  1. current feature-lattice content `q` plus a compact nonspatial per-atom
     action-response matrix `M` with no patch axis and no nine-action axis;
  2. exactly two fixed bilinear writes, each the outer product of a learned
     compact prior-error response and the centered key of the known requested
     action that caused it;
  3. `M` may affect prediction only as a bounded multiplier inside the
     successful mean-centered categorical action interaction, never through a
     generic state-only successor.
- Removing the fixed bilinear write, adding a learned memory gate/decay,
  allowing `M` to move content without the current centered action, carrying
  raw increments into the future, using nine action slots, or turning `M`
  into token memory or momentum collapses the proposal into a closed family
  and is forbidden.

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
  primitives `p0:p5` with exact reset-safe boundary
  `F(i-1,5) --p_i--> F(i,5)`.
- Online normalized history features are
  `z_i = normalize(E_online(e_i))`. The accepted N320 encoder is the only
  initialization input and fixed target. Only its reviewed `encoder.*` prefix
  may be copied into fresh online and target encoders; the target stays no-grad,
  byte-identical, and at zero EMA updates.
- Future RGB `e3:e6` is visible only to the fixed target encoder under
  `no_grad`, after predictions exist. No target, future RGB, executed/clipped
  command, pose, odometry, depth, flow, label, map, geometry, or navigation
  value may enter the predictive state.

## Frozen K4 system-identification state

- There are exactly four coherent equal-mass atoms. Each causal state is:

  ```text
  s_t^k = (q_t^k, M_t^k)
  ```

  `q` is a normalized feature lattice. `M` is exactly a `16 x 16` nonspatial
  action-response matrix per atom: 256 scalars, no patch/token axis, and no
  nine-action slot axis. It has no physical units and is not velocity, pose,
  terrain metadata, an action operator bank, or a hand-coded controller model.
  Because exactly two rank-one writes are permitted, `rank(M2) <= 2` by
  construction.
- Initialize from `e0` only:

  ```text
  q_0^k = z_0
  M_0^k = 0
  ```

  Four learned mode embeddings are centered across modes and, together with
  one learned spatial embedding table, enter `B` through the exact inherited
  centered-mode context route. They never add directly to `q` or `M`. For the
  tensor belief API, row-major flattened `M` is stored in a fixed zero-padded
  carrier; that carrier is serialization only. Its padding must remain exact
  zero and may not contain spatial or dense-history information.
- Transform the complete nine-row learned categorical action table, then
  center it:

  ```text
  c_j = A(E_action[j]) - (1/9) * sum_l A(E_action[l])
  ```

  Centering is after the complete tower. A shared bias-free projection creates
  a 16-value attribution key for every complete-table row; the key table is
  centered again after its bounded nonlinearity:

  ```text
  k_j_raw = tanh(P_c(c_j))
  k_j     = k_j_raw - (1/9) * sum_l k_l_raw
  ```

  The same learned `c_j` table supplies future action conditioning. There are
  no numeric command parameters, per-action operators, one-hot action memory,
  or HOLD special cases.

### Exact execution-size modules and initialization

- Execution uses inherited feature width `192` and 256 spatial patch tokens.
  `B` is the exact inherited one-layer `_ActionFreeBeliefContext`: its visual
  argument is `q`, its hidden argument is the inherited centered
  mode-plus-spatial context, and its increment-modulation argument is exact
  zero. Thus it retains the inherited visual path and one `ViTBlock` while
  seeing neither `M` nor the current action.
- `A` is the exact inherited `_CenteredCategoricalActionTower`: non-affine
  LayerNorm, bias-free `192 -> 192` Linear, Tanh, bias-free `192 -> 192`
  Linear, then complete-nine-row mean subtraction.
- Writer normalization is `LayerNorm(192, elementwise_affine=False,
  eps=1e-5)` independently on every innovation token. Exact bias-free shapes
  are `P_r: 192 -> 16`, `P_c: 192 -> 16`, `P_M: 256 -> 192`, and
  `W0: 192 -> 192`.
- `P_r`, `P_c`, and `P_M` use fresh PyTorch default Linear weight
  initialization under the frozen seed. `W0` is exact zero. Mode and spatial
  embedding tables use the inherited normal initialization with mean `0` and
  standard deviation `0.02`; inherited `B` and action-tower parameters retain
  their exact constructor initialization. No unregistered initialization or
  scalar hyperparameter is permitted.

## Shared prior and modulation-only memory read

- One shared action-free spatial context `B`, one shared bias-free memory
  projection `P_M`, one shared bias-free zero-initialized output head `W0`, and
  the exact same prior are called on all six edges:

  ```text
  b_t^k             = B(q_t^k, centered_mode_k)
  mu_t^k            = 1 + tanh(P_M(vec(M_t^k)))
  delta_t^k         = W0(b_t^k * mu_t^k * c_(p_t))
  q^-_(t+1)^k       = renorm(q_t^k + delta_t^k)
  M^-_(t+1)^k       = M_t^k
  readout(s^-_(t+1)^k) = q^-_(t+1)^k
  ```

  `vec` is the frozen row-major flattening order. `mu` is broadcast over
  spatial tokens only at the read. `M` has no other
  consumer. There is no direct or action-independent `B -> delta`,
  `M -> delta`, `q -> delta`, raw-increment, or other learned prediction
  route; all three state/context terms reach `W0` only through multiplication
  by the current centered action. Since `b` and `mu` cannot see that action,
  the complete-table mean pre-renormalization `delta` is exactly zero. No
  zero-mean claim is made for the nonlinear realized post-renormalization
  content changes.
- `M=0` gives `mu=1`, preserving the successful ordinary centered-action path.
  `W0` is exactly zero at initialization, so every action/history/HOLD control
  is persistence within the inherited `1e-5` update-zero tolerance. HOLD later
  uses its ordinary centered code and the same shared prior.

## Fixed bilinear post-prior action attribution

- On the two observed edges only, emit and score the prior before seeing its
  destination. Then form the newly available online innovation:

  ```text
  r_(t+1)^k = z_(t+1) - q^-_(t+1)^k
  ```

- The spatial innovation becomes one bounded 16-value response through fixed
  token-mean pooling, non-affine normalization, and one shared bias-free
  projection:

  ```text
  rho_(t+1)^k = tanh(P_r(mean_token(LN(r_(t+1)^k))))
  ```

  Exact zero innovation maps to exact zero response. The only memory write is
  the fixed outer-product rule:

  ```text
  M_(t+1)^k = M_t^k
              + (1/sqrt(16)) * outer(rho_(t+1)^k, k_(p_t))
  q_(t+1)^k = z_(t+1)
  ```

  `P_r` and `P_c` are learned jointly, but the outer-product form, coefficient,
  additive write, and absence of decay/gating are fixed. There is no learned
  recurrent updater, attention memory, scalar gain, or per-action slot. The
  write assigns each observed prior error to the exact requested past action
  by construction rather than asking an inverse classifier to rediscover it.
  Post-prior insertion of `z_(t+1)` is the sole factual content assimilation.
- Exact event order is:

  ```text
  initialize(e0)
  prior/score(p0) -> write(outer(rho1,k_p0)) -> assimilate(e1)
  prior/score(p1) -> write(outer(rho2,k_p1)) -> assimilate(e2)
  prior/score(p2) -> prior/score(p3) -> prior/score(p4) -> prior/score(p5)
  ```

- After `e2`, the belief contains only `(q2,M2)`. `M2` remains bitwise fixed
  over future rollout; there is no unobserved write. Future prediction receives
  only this state, current categorical action, centered modes, and model
  parameters. There is no raw `z2`, explicit `z2-z1`, anchor, dense history,
  horizon query, future observation, or separately evolving inertial state.
- On observed edges, scored local innovations are `q^-_(t+1)-z_t`. On future
  edges they are recursively realized `q^-_(t+1)-q_t`. `rho`, keys, and `M`
  are never targets or direct predicted increments.

## Unchanged JEPA objective and optimization

- Retain exactly four equal-mass coherent trajectory atoms and the proper
  energy score: 50% coherent joint trajectory plus 50% mean marginal-step
  score, including all ordered atom pairs and zero diagonal.
- Retain the exact three-term objective and weights:

  ```text
  L_total = 1.0 * online_e0:e2_to_fixed_teacher_alignment
          + 0.5 * ES_K4(all_six_realized_local_innovations)
          + 0.5 * ES_K4(open_loop_p2:p5_cumulative_states)
  ```

- Cyclic wrong action, all HOLD, reversed history, reset history, persistence,
  centroid, and particle spread remain evaluation-only. No system-ID label,
  action/history ranking or hinge, inverse classifier, reconstruction,
  navigation loss, whitening, covariance, learned-target, correspondence, or
  transport loss is allowed.
- The online encoder, learned modes, response/key/memory projections, state
  context, action tower, and prediction head form one model and graph. Sum the
  three losses, call backward once, clip the inherited disjoint groups, and
  step one AdamW once. There is no phase, frozen predictor, second optimizer,
  separately trained system identifier, or separately checkpointed predictor.
- Preserve seed `20260727`, float32, batch 16, AdamW encoder LR `1e-4`, all
  fresh history/predictor parameters LR `3e-4`, weight decay `1e-4`, betas
  `(0.9,0.999)`, epsilon `1e-8`, and groupwise clipping at `1.0`.

## Selection, gates, and cap

- Preserve the exact V2 evaluator, validation rows, bootstrap, noncollapse
  checks, selection rule, thresholds, and all 32 gates without addition,
  deletion, relaxation, or relabeling.
- Observe updates `0,250,500,750,1000`. Execute exactly 1,000 updates and
  16,000 training presentations, with five complete 2,048-row validation
  passes totaling 10,240 presentations and a 5,400-second active-GPU cap.
- Select the eligible noncollapsed trained observation with minimum validation
  combined normalized energy. No gate-aware, tiered, early, or post-hoc
  checkpoint choice is allowed.
- PASS requires all 32 gates, including H4 action gap at least `0.05`, H4
  history gap at least `0.03`, positive action/history bootstrap bounds, at
  least six positive action/history/HOLD families, persistence, distribution
  value, noncollapse, and the exact-cap/update-zero conditions.
- PASS is perception/world-model development evidence only. It grants no
  checkpoint read, navigation, G2, held-out/sealed access, promotion,
  production, or deployment authority. STOP closes the exact mechanism.

## Required source proofs

- Prove exact K4 `(q,M)` shapes, `16 x 16` nonspatial matrices, fixed
  zero-padded pack/unpack round trip, centered modes/action codes/action keys,
  update-zero persistence, realized-increment arithmetic, and K4
  mode-permutation invariance.
- Instrument calls to prove one initializer, the same prior exactly six times,
  exactly two outer-product writes after observed priors, and zero writes on
  future edges. Sentinels must prove `p0` keys `r1` and `p1` keys `r2`, with no
  off-by-one action attribution.
- Prove prior arguments contain only `(q,M,current_action)` plus learned
  parameters; each writer receives only the new online prior error and its
  corresponding requested past action; belief packs only `(q,M)` and all
  serialized padding remains exact zero.
- Prove `P_r(0)=0`, the exact fixed `1/sqrt(16)` outer-product arithmetic, and
  `rank(M2) <= 2`, plus the absence of learned write gain, decay, gate,
  recurrent updater, token memory, or nine-action slot.
- Prove algebraically and by all-nine-action evaluation that `M` cannot change
  a prediction without the current centered action interaction and the
  uniform-action mean learned increment is zero. Zeroing `M` must remove only
  the history modulation while leaving the ordinary centered-action path.
  There must be no generic state-only successor, raw increment, momentum,
  anchor, or future-target path.
- With `e2` held fixed and heads opened deterministically, changing the ordered
  `(e0,p0,e1,p1)` action/error pairs must change `M2` and the future forecast.
  Holding errors fixed while swapping their past action keys must change `M2`.
  Changing the requested future action must change the forecast; HOLD must use
  the same route.
- Prove future rollout leaves `M2` bitwise fixed. Perturbing future RGB may
  change only fixed-target outputs, never predictions.
- On an isolated prediction-score backward, require finite nonzero gradients
  on `W0` and correct zero/absent staging behind it. After deterministic head
  opening, require finite nonzero gradients through online encoder, learned
  modes, response/key/memory projections, state context, action tower/
  embedding, and `W0`; fixed-target gradients remain absent.
- Prove exact three-loss arithmetic, causal recursion, optimizer-group
  disjointness/full coverage, one summed backward/step, and absence of separate
  predictor/system-ID training state.
- Runner tests must bind the exact V2 schedules/evaluator/32 gates/caps, reject
  every binding/retry/resume/seed/cap/checkpoint override, truthfully replace
  latent-momentum mechanism receipts, and preserve complete normal and caught-
  failure JSON chains.

## Custody and one-shot execution order

- Exclusive output root:
  `.generated/go2_rgb_fixed_teacher_action_attributed_causal_system_identification_trajectory_h4_jepa_v1/probe_v1`.
- The root must be absent and is created mode `0700` only after exact source
  closure and a clean bound preflight. Execution reservation precedes schedule,
  accepted-N320, RGB, or Torch runtime work; preflight is zero-RGB,
  zero-training, and zero-reservation.
- One fresh attempt only. Reservation, operational terminal, scientific STOP,
  or PASS consumes it. There is no retry, resume, repair, replacement attempt,
  second seed, extension, or predecessor-checkpoint reuse.
- Runtime checkpoints/traces are write-only. Terminal audit may read only the
  six canonical JSON receipts by exact path and must never list, stat, hash,
  or open a runtime checkpoint.
- Ordered work is: commit this preregistration; implement focused model and
  runner proofs; freeze hashes; obtain independent science/custody review; run
  one non-training bound preflight; execute once only if preflight passes;
  audit JSON receipts and record PASS or STOP.
