# Go2 RGB fixed-teacher trajectory-distribution H4 JEPA V1 preregistration — 2026-07-28

## Question and motivation

- This preregisters exactly one development-only falsification. It grants no
  execution, checkpoint reuse, navigation, held-out, sealed, promotion, or
  deployment authority by itself.
- The fixed-teacher recurrent V3 and dense deterministic V1 probes retained
  healthy registered encoder rank and learned action sensitivity, but both
  lost to fixed-teacher `e2` persistence and both failed the ordered-history
  gate. Dense V1 selected H4 normalized error `1.447611` with action gap
  `+0.111635`, history gap `-0.077081`, and persistence gap `-0.447611`.
- Hypothesis: a single squared-error future is averaging incompatible futures
  in visually aliased or only partly observed scenes. The clean falsification
  is a finite predictive distribution, not another deterministic layer, seed,
  schedule, or encoder/data refinement.
- If this exact mechanism fails, the uncertainty formulation closes. The next
  category must reformulate the learned predictive target/state rather than
  retrying K, seed, duration, or a nearby distributional V2.

## Immutable input and custody contract

- Each row remains the seven-RGB/six-action sequence
  `e0,p0,e1,p1,e2,p2,e3,p3,e4,p4,e5,p5,e6`.
- Online inputs are RGB `e0:e2`, past actions `p0:p1`, and proposed future
  actions `p2:p5`. RGB `e3:e6` is visible only to the stopped-gradient target
  encoder. No pose, depth, flow, occupancy, map, reward, collision, semantic,
  waypoint, simulator-state, or navigation label is an input or target.
- The nine-action vocabulary and ordering remain unchanged.
- Training is the exact existing 16,000-row schedule at
  `.generated/go2_recurrent_h4_rgb_sequence_index_v1/train.jsonl`, SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`,
  10,024,000 bytes. Validation is the exact existing 2,048-row schedule at
  `.generated/go2_recurrent_h4_rgb_sequence_index_v1/val.jsonl`, SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`,
  1,278,976 bytes. Their train/validation scenes and RGB leaves remain
  disjoint. Regeneration, filtering, reordering, or substitution is forbidden.
- RGB may resolve only beneath `.generated/datagen_full/render_textured_v03`.
  The complete main-pool census remains bound by SHA-256
  `aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408`.
- Held-out, test, sealed, legacy V4 sealed material, and G2--G8 navigation are
  not opened or run. No rejected predictor checkpoint, trace, or tensor is
  opened. The only checkpoint input is the accepted N320 encoder source.

## Fixed teacher and jointly learned online path

- The target encoder is an exact fixed copy of the accepted N320 encoder
  prefix from checkpoint SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
  content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`,
  13,777,100 bytes. It stays in eval mode, outside the optimizer, with no
  gradient and no EMA for the complete probe.
- The online encoder starts from the same accepted N320 encoder tensors and is
  trainable from update zero together with every history, action, mode,
  decoder, and projection parameter. There is one optimizer and one backward;
  the predictor is never trained separately.
- Both encoders produce a 16-by-16 lattice of 192-dimensional patch tokens.
  Target and predicted patch tokens are normalized along feature dimension.
- The complete dense history substrate from deterministic dense V1 is freshly
  initialized: all normalized `e0:e2` patch tokens plus explicit interleaved
  `p0,p1` tokens, learned spatial/time/transition embeddings, and exactly two
  pre-norm width-192, six-head Transformer encoder blocks. No predecessor
  state or predictor tensor is reused.

## Four coherent equal-mass trajectories

- The model predicts exactly `K=4` equal-mass trajectory atoms. A learned mode
  embedding is shared across H1--H4 within each atom, so one atom is one
  coherent four-horizon trajectory; modes are not independently permuted at
  each horizon.
- For each horizon, the ordered future-action prefix occupies fixed `p2:p5`
  slots and its unused suffix is exact zero. The prefix, horizon embedding,
  mode embedding, spatial embedding, and current online `e2` lattice form the
  future queries.
- Exactly two shared pre-norm width-192, six-head Transformer decoder blocks
  cross-attend each query lattice to the complete dense causal history. There
  is no recurrence, BEV, warp, flow surrogate, offset field, cost volume,
  retrieval, transport, reconstruction, inverse dynamics, or classifier.
- One shared `LayerNorm-Linear(192,192)` head emits direct, nonrecursive,
  `e2`-relative deltas. Its final linear weight and bias are exact zero. For
  atom `k` and horizon `h`:
  `Y_kh = normalize(raw_O(e2) + ||raw_O(e2)|| * delta_kh)` tokenwise.
  Thus every atom/control is exact persistence at update zero.
- Mixture masses are fixed at one quarter. There is no learned weight, scale,
  variance, temperature, entropy bonus, diversity bonus, or post-hoc fitting.
  Duplicate atoms are allowed and can represent masses in quarter increments.

## Exact proper-score objective

- For normalized patch lattices `A,B` with `P=256`, define
  `d_h(A,B) = ||A_h-B_h||_F / sqrt(P)`. For a coherent trajectory, flatten its
  four lattices and equivalently define
  `d_J(A,B) = ||A-B||_F / sqrt(4P)`.
- For either distance, the exact uniform empirical-distribution energy score
  is
  `ES_d(Y,T) = (1/K) sum_k d(Y_k,T) - (1/(2K^2)) sum_{k,l} d(Y_k,Y_l)`.
  The pair sum includes all 16 ordered pairs and the zero diagonal. It is not
  an off-diagonal mean with coefficient one half.
- `L_dist = 0.5*ES_J + 0.5*mean_h(ES_h)`.
- `L_align` is the mean squared tokenwise distance from normalized online
  `e0:e2` latents to their stopped-gradient fixed-teacher latents.
- The total loss is exactly `L_dist + L_align`, weights 1.0 and 1.0. Absolute
  centroid error is API-compatible but has training weight zero. Variance,
  wrong-action ranking, persistence/history hinges, best-of-K, reconstruction,
  semantic, and navigation losses are absent.

## Optimizer, schedule, and cap

- Fresh initialization and schedule seed: `20260727`. Float32, no autocast;
  cuDNN benchmarking disabled.
- AdamW groups remain exact and disjoint: online encoder LR `1e-4`; dense
  history LR `3e-4`; action/mode/horizon embeddings, action path, decoder, and
  shared delta head LR `3e-4`. Weight decay `1e-4`, betas `(0.9,0.999)`, epsilon
  `1e-8`; each group gradient norm is clipped to 1.0.
- Exactly 1,000 updates at batch size 16 and exactly 16,000 ordered training
  presentations. Validation uses all 2,048 fixed rows at updates
  `0,250,500,750,1000`. Active GPU time is capped at 90 minutes.
- Fresh output root:
  `.generated/go2_rgb_fixed_teacher_trajectory_distribution_h4_jepa_v1/probe_v1`.
  It must be absent before launch. Once reserved, any termination consumes the
  attempt. There is no retry, resume, second seed, extension, K change, or
  science replacement. Registered checkpoints are write-only during the run.

## Evaluation and checkpoint selection

- All scores are scene-then-family macro averages over the eight fixed maze
  families. H4 action, persistence, history, and distribution-value confidence
  bounds use the existing deterministic 1,000-replicate scene bootstrap.
- Persistence is the degenerate four-atom distribution at fixed-teacher `e2`.
  Marginal, joint, and combined real energy scores are divided by their
  corresponding persistence energy score.
- The combined normalized score is the raw
  `0.5*joint + 0.5*mean-marginal` real score divided by the same combined
  persistence score. This exactly matches the training proper score.
- Evaluation-only controls are cyclic-plus-one wrong future actions, all-hold
  future actions, reordered `e1,e0,e2` history with reordered actions, reset
  `e2,e2,e2` history with hold actions, and fixed-teacher persistence.
- The spherical centroid is `normalize(mean_k Y_k)`. Combined distribution
  value is `(combined_centroid_score-combined_ensemble_score) /`
  `combined_persistence_score`. Pairwise spread is the all-16-pair mean
  distance divided by marginal persistence distance. Best-atom and centroid
  squared errors are diagnostics only.
- A trained checkpoint is eligible only when all registered values are finite,
  target/online effective-rank ratios are at least 0.10, and target/online
  near-zero variance fractions are at most 0.05. Select the eligible checkpoint
  with the lowest combined normalized score. There is no per-gate selection.

## All-conjunctive PASS gate

PASS requires all of the following; any failure is terminal STOP:

- exact 1,000-update/16,000-presentation completion; all observations finite;
- fixed-teacher rank and near-zero fraction drift no more than `1e-6`, target
  rank at least 0.10, and target near-zero fraction at most 0.05 throughout;
- update-zero marginal, joint, and combined scores equal persistence within
  `1e-5`, and action/hold/persistence/history/distribution/spread gaps are zero
  within `1e-5`;
- an eligible trained checkpoint exists;
- selected combined and joint normalized scores are each strictly below 1.0;
- selected marginal H1--H3 scores are each below 1.0 and H4 is at most 0.90;
- H4 persistence bootstrap lower bound is positive, at least six families are
  positive, and no family is below -0.02;
- combined distribution value is at least 0.05, its bootstrap lower bound is
  positive, and it is positive in at least six families;
- H4 normalized pairwise spread is at least 0.05;
- H4 wrong-action gap is at least 0.05 with positive bootstrap lower bound,
  H1--H3 wrong-action gaps are nonnegative, at least six families are positive,
  and no family is below -0.02;
- H4 ordered-history gap (better of reset/reordered control minus real) is at
  least 0.03 with positive bootstrap lower bound and at least six positive
  families; H4 all-hold gap is positive;
- fixed-teacher state is byte-identical before/after, has zero EMA updates, is
  outside every optimizer group, and receives no gradient.

PASS establishes only a bounded development RGB/action JEPA substrate. It does
not authorize checkpoint use, navigation, held-out evaluation, promotion, or
deployment. STOP closes this exact K=4 fixed-teacher full-latent distribution
mechanism and leaves every written checkpoint inaccessible.
