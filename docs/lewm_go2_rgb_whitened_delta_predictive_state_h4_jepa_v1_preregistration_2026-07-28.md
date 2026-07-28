# Go2 RGB whitened-delta predictive-state H4 JEPA V1 preregistration — 2026-07-28

## Question and category change

- This preregisters one development-only falsification. It grants no execution,
  checkpoint, navigation, held-out, sealed, promotion, or deployment authority
  by itself.
- The terminal K=4 trajectory-distribution result at commit `05bba5f` beat
  fixed-teacher persistence at all horizons and in every maze family, but H4
  action gap was only `+0.010815`, ordered-history gap was `-0.013974` in
  aggregate and negative in all eight families, and hold gap was `-0.006572`.
  H4 best-atom and centroid point errors remained `1.480741` and `1.377818`
  times persistence. The uncertainty/full-future-lattice category is closed.
- Question: does the fixed N320 representation contain a small, learnable
  future-change state that is genuinely predictable from RGB history and
  proposed actions, even though its full patch lattice is not?
- This mechanism changes the predictive target, not the seed, duration, data,
  or number of deterministic/full-lattice hypotheses. A STOP closes this exact
  D8 whitened-delta predictive-state formulation.

## Immutable data and custody

- Input rows remain `e0,p0,e1,p1,e2,p2,e3,p3,e4,p4,e5,p5,e6`. Online inputs
  are RGB `e0:e2`, past actions `p0:p1`, and proposed actions `p2:p5` only.
  Future RGB `e3:e6` is visible only in the fixed-teacher target branch during
  training/evaluation.
- No pose, odometry, depth, flow, occupancy, map, reward, collision, semantic,
  waypoint, simulator-state, or navigation label is an input or target.
- Training is the exact 16,000-row schedule SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`,
  10,024,000 bytes. Validation is the exact 2,048-row schedule SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`,
  1,278,976 bytes. Regeneration, reordering, filtering, or substitution is
  forbidden. RGB resolves only beneath the existing main-pool root.
- Held-out, test, sealed, legacy V4 sealed material, G2--G8 navigation, and all
  rejected predictor checkpoints/traces/tensors remain inaccessible. The only
  checkpoint input is the accepted N320 encoder initialization.

## Encoders and dense causal context

- `E_t` is the accepted N320 encoder prefix, checkpoint SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
  content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.
  It is permanently fixed, stopped-gradient, outside the optimizer, and has no
  EMA. It supplies normalized `e2:e6` patch features for the compact target and
  normalized `e0:e2` features for online-history alignment.
- `E_o` starts from the same N320 tensors and is trainable from update zero.
  It supplies the online `e0:e2` patch tokens. Target-state losses cannot reach
  `E_o`; its gradients come from prediction and alignment only.
- The online dense context is freshly initialized and retains all 256 spatial
  tokens from each of `e0:e2`, interleaved explicit `p0,p1` tokens, learned
  spatial/time/transition embeddings, and exactly two pre-norm width-192,
  six-head Transformer encoder blocks.
- For H1--H4, ordered action prefixes occupy fixed `p2:p5` slots with zero
  suffixes. Two shared pre-norm Transformer decoder blocks cross-attend future
  queries to the complete dense context. Horizons are direct and nonrecursive.

## Learned zero-preserving D8 target

- For horizon `h`, define fixed-teacher patch deltas
  `Delta_h[j] = normalize(E_t(e_(2+h))[j]) - normalize(E_t(e2)[j])`.
- One shared target pool produces `q_h in R^8`. Each state dimension has a
  learned nonzero-initialized content query, learned spatial logits over the
  256 ordered patch positions, and a learned bias-free orthonormal-initialized
  value vector. The pooled value has a fixed `sqrt(256)=16` scale compensating
  the variance reduction from spatial averaging. Attention weights may depend
  on delta content and learned spatial position, but the pooled values are only
  linear functions of `Delta_h`.
- There is no affine normalization, bias, residual constant, horizon ID,
  action, or history in the target value path. Therefore an exact-zero teacher
  delta maps algebraically to exact `q_h=0` while the nonzero initialization
  permits variance gradients.
- The predictor averages each horizon's decoded spatial tokens and applies one
  `LayerNorm-Linear(192,8)` state head. Only that final linear weight and bias
  are exact zero initialized. Its output is `p_h`; update-zero `p_h=0` is exact
  compact-state persistence for every action/history control.
- Four horizons form a 32-dimensional trajectory state, but variance and
  covariance are always computed separately across the batch for each D8
  horizon. Batch and horizon are never flattened together.

## Joint JEPA objective

- Similarity is `S = mean_(b,h,d) (p_bhd-q_bhd)^2`.
- For state `z` and each horizon independently across the 16 batch rows,
  `V(z) = mean_d relu(1-sqrt(var_b(z_d)+1e-4))^2`, then average horizons.
- `M(z)` is the mean squared per-horizon, per-dimension batch mean, averaged
  over horizons and dimensions. It prevents a shared DC offset from supplying
  artificial persistence-normalization energy.
- For each horizon, form unbiased batch covariance `C_h(z)`. Define
  `C(z) = mean_h sum_(i!=j) C_h(z)[i,j]^2 / 8`.
- `A` is mean squared tokenwise distance between normalized online and fixed
  N320 `e0:e2` features.
- The exact loss is
  `25*S + 25*(V(p)+V(q))/2 + 25*(M(p)+M(q))/2 +`
  `(C(p)+C(q))/2 + 1*A`.
- `E_o`, dense history, action path, predictor, and compact target pool share
  one optimizer and one backward. `E_t` is fixed. There is no separately
  trained predictor or compressor phase.
- Distribution atoms, learned variance/scale, best-of-K, diversity bonuses,
  contrastive negatives, codebooks, wrong-action/history/persistence/hold
  training losses, reconstruction, semantic, and navigation losses are absent.

## Optimizer and cap

- Seed `20260727`; float32 without autocast; cuDNN benchmarking disabled.
- AdamW groups: online encoder LR `1e-4`; dense history LR `3e-4`; action path
  and predictor LR `3e-4`; compact target pool LR `3e-4`. Weight decay `1e-4`,
  betas `(0.9,0.999)`, epsilon `1e-8`, and independent group gradient clipping
  at norm 1.0.
- Exactly 1,000 updates, batch 16, and 16,000 ordered training presentations.
  All 2,048 validation rows are evaluated at updates `0,250,500,750,1000`.
  Active GPU time is capped at 90 minutes.
- Fresh output root:
  `.generated/go2_rgb_whitened_delta_predictive_state_h4_jepa_v1/probe_v1`.
  Once reserved, any termination consumes the attempt. No retry, resume,
  extension, second seed, state-width change, loss change, or nearby V2.

## Evaluation and selection

- For each scene/horizon, raw squared-error numerators are summed across its
  rows before division by summed target energy `sum mean_d(q_h^2)`. Family and
  global metrics are then scene-then-family macro averages. Thus `p=0`
  persistence has normalized error 1 without per-row low-energy weighting.
- Wrong-action, all-hold, reordered-history, and reset-history predictions are
  evaluation-only. Every control is scored against the same real target `q`.
- Record per horizon for `p` and `q`: participation-rank ratio, minimum and
  maximum dimension standard deviation, maximum absolute dimension mean, RMS,
  mean-energy fraction, target energy, covariance through the rank calculation,
  and count of near-zero scene denominators.
- A trained checkpoint is eligible only if all values are finite; fixed and
  online encoder rank/variance satisfy the existing floors; every `p` and `q`
  horizon has participation-rank ratio at least 0.75, minimum dimension std at
  least 0.50, maximum dimension std at most 2.0, maximum absolute dimension
  mean at most 0.25, mean-energy fraction at most 0.25, RMS at most 3.0; and
  every scene denominator is nonzero above `1e-8`.
- Select the eligible checkpoint with minimum mean H1--H4 real normalized
  compact-state error. Controls never affect selection.

## All-conjunctive PASS gate

PASS requires all of the following; otherwise STOP:

- exact cap, all observations finite, fixed N320 teacher metric geometry
  unchanged within `1e-6`, and an eligible compact state exists;
- update-zero H1--H4 real errors equal 1 and every action, hold, persistence,
  and history gap equals zero within `1e-5`;
- every selected real error is below 1, selected mean error is at most 0.90,
  and selected H4 error is at most 0.90;
- H4 persistence bootstrap lower bound is positive, at least six of eight
  families are positive, and no family is below -0.02;
- H4 wrong-action gap is at least 0.03 with positive bootstrap lower bound,
  H1--H3 action gaps are nonnegative, at least six families are positive, and
  no family is below -0.02;
- H4 history gap, defined using the better of reset/reordered controls, is at
  least 0.03 with positive bootstrap lower bound and at least six positive
  families;
- H4 all-hold gap is positive;
- fixed teacher state is byte-identical before/after, outside the optimizer,
  receives no gradient, and records zero EMA updates.

PASS establishes bounded compact predictive-state JEPA feasibility only. It
does not authorize checkpoint access, navigation, held-out evaluation,
promotion, or deployment. STOP closes WDPS-D8 and leaves all written
checkpoints inaccessible.
