# Go2 RGB fixed-teacher dual-domain trajectory H4 JEPA V1 preregistration — 2026-07-28

## Question and category boundary

- The cumulative-target K=4 trajectory JEPA achieved strong, broad future
  prediction at its selected observation: combined normalized energy score
  `0.725748`, H4 score `0.742969`, and H4 persistence gap `+0.257031`. It did
  not condition adequately on action or history: H4 gaps were `+0.010815` and
  `-0.013974`.
- The local-innovation/counterfactual successor produced a large trained
  cyclic-action gap (`+0.187365`) but weaker H4 prediction (`0.953251`), a
  statistically non-robust persistence gap, and effectively zero ordered-
  history value (`+0.000948`). Its all-hold gap was only `+0.001280`.
- Because the local target and ranking terms changed together, those two runs
  do not isolate which change caused either outcome.
- This experiment asks one narrow question: while keeping the complete
  local-innovation architecture fixed, does replacing its local-only training
  and counterfactual score with one equal-weight local-plus-integrated score
  recover enough cumulative quality to pass the inherited gate, retain action
  sensitivity, and establish ordered-history dependence?
- This is one objective change, not a new architecture, data refinement,
  schedule sweep, longer run, or stopped-checkpoint continuation.

## Exact model and initialization

- History RGB is `e0,e1,e2`; past actions are `p0,p1`; future actions and
  fixed-teacher targets are `p2:p5` and `e3:e6`.
- The online path is unchanged: accepted N320 RGB encoder, dense three-frame
  history/action transformer, four equal-mass coherent trajectory modes,
  ordered future-action prefixes, and one shared zero-initialized local-
  increment head.
- Each predicted local increment is integrated recursively from normalized
  online `e2`, with the realized state normalized after every step. The same
  four realized atoms therefore define both local innovations and cumulative
  H1--H4 trajectories.
- The target is a permanently fixed copy of the accepted N320 encoder. Future
  RGB reaches only that no-grad target. There is no EMA update.
- The online and fixed-target encoders both start from the accepted N320
  encoder prefix, opened exactly once. Dense history, action path, mode
  embeddings, and prediction head are freshly random/zero-head initialized.
  No tensor from either stopped trajectory branch may be opened or reused.
- The online encoder, dense history, action path, mode embeddings, and
  trajectory predictor train jointly in one backward pass. There is no
  separately trained decoder or downstream probe.

## Exact dual-domain objective

For `K=4` atoms, let `Z` be the recursively integrated predicted normalized
trajectory, `Y` the fixed-teacher normalized future trajectory, `D` the
successive realized changes along `Z`, and `Q` the fixed-teacher successive
changes `e3-e2,e4-e3,e5-e4,e6-e5`.

For either atom/target pair `A,T`, the horizon energy score is

`ES_h(A,T) = mean_k d(A[k,h],T[h]) - 0.5 mean_k,k' d(A[k,h],A[k',h])`,

where `d` is the Euclidean lattice norm divided by the square root of the
spatial-token count. The joint score uses the same formula after flattening
the four horizons and spatial tokens. The registered domain score is

`S(A,T) = 0.5 ES_joint(A,T) + 0.5 mean_h ES_h(A,T)`.

The registered dual-domain score is

`L_prediction = 0.5 mean S(D,Q) + 0.5 mean S(Z,Y)`.

- Both domain terms are raw proper energy scores in the same fixed-teacher
  latent geometry. They are not divided by per-sample motion, whitened,
  variance-scaled, or best-of-K selected.
- The four joint/marginal/domain contributions each have coefficient `0.25`.
- `L_prediction` is a proper score. The complete objective below also contains
  ranking hinges and is not itself claimed to be a proper scoring rule.
- Same-RGB online-to-fixed-teacher alignment remains weight `1.0`.
- For each real, cyclic wrong-action, reverse-history, and reset-history branch,
  its counterfactual score is the same mixture
  `M = 0.5 S(local innovations,Q) + 0.5 S(cumulative trajectory,Y)`.
- Ranking-only normalization uses the detached target-only mixed persistence
  energy: one half scores zero innovations against `Q`; one half scores the
  fixed-teacher `e2` repeated across all atoms/horizons against `Y`. It is
  clamped below at `1e-6`. The online `e2` state is never used in this
  denominator, and the proper prediction loss itself is never divided by it.
- Every scheduled row contributes after the `1e-6` clamp. There is no
  low-motion mask, threshold, filter, or conditional loss omission.
- The cyclic wrong-action hinge remains weight `1.0` and margin `0.05`, now
  comparing mixed real and cyclic scores.
- The reverse/reset ordered-history hinge remains weight `1.0` and margin
  `0.03`, comparing the mixed real score to the lower of the two complete
  mixed control scores. Domain-wise minima are not taken separately.
- The complete training objective is

  `L_prediction + L_history_alignment + L_cyclic_action + L_history`.

## Explicitly absent science

- no navigation, pose, depth, flow, BEV, reconstruction, or semantic label;
- no centroid squared-error term in the objective;
- no learned target compressor, variance, covariance, whitening, mixture
  weight, diversity bonus, or best-of-K objective;
- no data augmentation, masking curriculum, frame dropout, teacher update,
  recurrent-architecture change, optimizer change, or new parameter;
- no retry, resume, second seed, coefficient sweep, extension, or selection on
  training loss.

## Data, schedule, optimizer, and cap

- Train index remains the frozen 16,000-row / 1,000-scene main-pool schedule,
  SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`.
- Validation remains the frozen 2,048-row / 150-scene development schedule,
  SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`.
- The schedules cover all eight registered scene families and all nine action
  primitives. No held-out, test, sealed, or navigation split is eligible.
- Seed remains `20260727`; effective batch size remains `16`.
- AdamW parameter groups and learning rates remain unchanged: online encoder
  `1e-4`, dense history `3e-4`, predictor/action/modes/head `3e-4`; the same
  weight decay, gradient clipping, and update logic apply.
- Hard cap: exactly 1,000 optimizer updates and 16,000 training sequence
  presentations. Observations remain at updates `0,250,500,750,1000`.
- GPU-active cap remains 5,400 seconds. Validation contributes five times
  2,048 presentations but no optimizer presentation.
- Expected RGB opens remain `183680 = 7 * (16000 + 5 * 2048)`. The cumulative
  score reuses tensors and adds no RGB access. Training controls remain 16,000
  cyclic wrong-action and 32,000 history-control sequence evaluations.
- This is one fresh attempt. Operational failure produces complete failure
  receipts and no retry or resume.
- The only eligible output root is
  `.generated/go2_rgb_fixed_teacher_dual_domain_trajectory_h4_jepa_v1/probe_v1`.

## Evaluation and selection

- The development evaluator is unchanged and scores the integrated cumulative
  trajectory against fixed-teacher future latents. It reports marginal, joint,
  and combined normalized energy; persistence, cyclic-action, all-hold,
  reverse/reset-history, distribution-value, support-spread, point-error,
  family, bootstrap, and representation-collapse diagnostics.
- Update zero must remain exact four-atom `e2` persistence with zero realized
  innovations and zero action/history/support gaps within `1e-5`.
- Among registered trained observations that pass the existing noncollapse
  screen, selection is the minimum validation cumulative combined normalized
  energy score. Training loss cannot select an observation.
- The full predecessor trajectory gate remains controlling, and the weak
  aggregate `H4 all-hold gap > 0` leaf is supplemented by two breadth/floor
  uses of the already registered unseen all-hold metric:
  - positive H4 all-hold gap in at least six of eight families;
  - no family H4 all-hold gap below `-0.02`.
- This strengthening is evaluation-only and adds no new model call or data
  access. It prevents the exact trained cyclic corruption from being treated
  as general action semantics when the reserved all-hold control is null.

## Complete PASS gate

A PASS requires every registered gate:

- exact cap, finite observations, fixed-target geometry unchanged, target rank
  floor, exact update-zero persistence, and an eligible noncollapsed trained
  observation;
- combined and joint cumulative scores below persistence;
- H1--H3 marginal scores below persistence and H4 score at most `0.90`;
- positive H4 persistence bootstrap lower bound, positive persistence in at
  least six families, and no family persistence gap below `-0.02`;
- combined distribution-value gap at least `0.05`, positive bootstrap lower
  bound, value in at least six families, and H4 pairwise spread at least
  `0.05`;
- cyclic H4 action gap at least `0.05`, positive bootstrap lower bound,
  nonnegative H1--H3 action gaps, positive H4 gap in at least six families,
  and no family action gap below `-0.02`;
- H4 ordered-history gap at least `0.03`, positive bootstrap lower bound, and
  positive H4 history gap in at least six families;
- positive aggregate H4 all-hold gap and the two strengthened all-hold
  breadth/floor gates above.

There is no partial PASS. A STOP closes this exact dual-domain objective and
does not authorize its checkpoint.

## Interpretation and authority

- PASS would establish only bounded development feasibility: the same learned
  JEPA jointly retained local change, integrated future quality, broad action
  controls, and ordered-history value on the frozen development schedule.
- PASS would not establish maze navigation, general action counterfactuals,
  long-horizon memory, held-out generalization, or deployment. Before any
  data-scale promotion it would still require a separately authorized control-
  generalization check not trained as a margin.
- STOP distinguishes a failed synthesis from an infrastructure failure. It
  would rule out adding this equal cumulative score to the otherwise unchanged
  local mechanism; it would not rule out genuinely different temporal-state
  architectures.
- This preregistration grants no held-out, test, sealed-benchmark, navigation,
  promotion, deployment, stopped-checkpoint, training, GPU, data, or execution
  authority. The V4 sealed asset remains unopened, legacy development-only,
  and permanently ineligible for final evaluation.
