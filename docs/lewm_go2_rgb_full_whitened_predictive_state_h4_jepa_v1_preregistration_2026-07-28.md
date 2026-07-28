# Go2 RGB full-whitened predictive-state H4 JEPA V1 preregistration — 2026-07-28

## Question and category boundary

- This preregisters exactly one development-only falsification. It grants no
  execution, checkpoint, navigation, held-out, sealed, promotion, or deployment
  authority by itself.
- WDPS-D8 completed at the exact cap with a broad action signal at update 750,
  including H4 wrong-action gap `+0.232192`, positive bootstrap lower bound
  `+0.159182`, and positive effects in all eight families. It nevertheless
  produced an approximately one-dimensional predicted state, an increasingly
  rank-collapsed learned target, H4 error `1.276373` times persistence, and
  negative history evidence in all eight families.
- The diagnosed defect is coordinate redundancy: each D8 coordinate had
  nonzero marginal variance and tiny mean energy, but the predicted H4
  participation-rank ratio was `0.126746`, almost exactly the one-direction
  floor `1/8`. The old weight-1 raw covariance term contributed too little to
  prevent eight correlated copies.
- Question: if both learned branches must have identity covariance and their
  cross-covariance must also be identity at each horizon, can the compact JEPA
  retain the discovered action signal while learning a full-rank,
  history-conditioned predictive state that beats persistence?
- This is one structural objective replacement, not a weight sweep. A STOP
  closes full-whitened D8 learned targets. There is no covariance-weight series,
  D4/D16 variant, second seed, longer run, or nearby V2.

## Immutable data and custody

- Input rows remain `e0,p0,e1,p1,e2,p2,e3,p3,e4,p4,e5,p5,e6`. Online inputs
  are RGB `e0:e2`, past actions `p0:p1`, and proposed actions `p2:p5` only.
  Future RGB `e3:e6` is visible only to the fixed-teacher target branch during
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
  stopped/rejected predictor checkpoints, traces, and tensors remain
  inaccessible. The only checkpoint input is the accepted N320 encoder
  initialization.

## Unchanged joint JEPA architecture

- `E_t` is the accepted N320 encoder prefix, checkpoint SHA-256
  `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`,
  content SHA-256
  `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`.
  It is permanently fixed, stopped-gradient, outside the optimizer, and has no
  EMA. It supplies normalized future/current target tokens and normalized
  history features for alignment.
- `E_o` starts fresh from the same accepted N320 tensors and trains jointly.
  The dense history retains all 256 spatial tokens from each of `e0:e2`,
  interleaved past-action tokens, and two width-192 six-head Transformer
  encoder blocks. Two shared Transformer decoder blocks predict four direct,
  nonrecursive horizons from ordered future-action prefixes.
- The learned target pool is unchanged from WDPS-D8: eight nonzero content
  queries, spatial logits over 256 ordered positions, orthonormal-initialized
  bias-free value rows, and fixed `sqrt(256)=16` spatial-averaging
  compensation. Values are functions only of fixed-teacher future-minus-e2
  patch deltas. Exact-zero teacher delta maps algebraically to exact D8 zero.
- The predictor's final `Linear(192,8)` remains exact zero initialized, so
  update-zero prediction is exact persistence for every action/history control.
- Online encoder, dense history, action path, predictor, and target compressor
  share one optimizer and one backward. There is no separately trained
  predictor, compressor, probe, or downstream navigation head.

## Full-whitened CCA objective

- Shapes are batch `B=16`, horizons `H=4`, and state width `D=8`. For each
  branch `z` and horizon independently, define batch mean `mu_h`, centered
  state `Z_h`, and unbiased covariance `C_h(z)=Z_h^T Z_h/(B-1)`.
- Define within-branch whitening
  `W(z)=mean_h ||C_h(z)-I_8||_F^2/8`. All 64 matrix entries are included.
- Define unbiased predicted-target cross-covariance
  `C_h(p,q)=P_h^T Q_h/(B-1)`, using separately centered `P_h,Q_h`, and CCA
  alignment `X(p,q)=mean_h ||C_h(p,q)-I_8||_F^2/8`.
- Define zero-mean loss `M(z)=mean_(h,d) mu_h(z)[d]^2`. `A` remains mean
  squared tokenwise distance between normalized online and fixed N320 history
  features.
- The exact loss is
  `25*X(p,q) + 25*(W(p)+W(q))/2 + 25*(M(p)+M(q))/2 + 1*A`.
- Raw D8 prediction MSE is absent from training and remains evaluation-only.
  This matters at update zero: cross-covariance identity gives the predictor a
  nonzero opening gradient but does not reward the learned target for shrinking
  toward a zero predictor. At the optimum, within covariances and cross
  covariance are identity, implying centered branch agreement without inverse
  covariance or batch ZCA.
- The predecessor marginal variance hinge and weak raw off-diagonal covariance
  term are absent. Batch and horizon are never flattened together. There is no
  validation-fitted whitening, eigendecomposition, inverse square root,
  contrastive negative, codebook, control-ranking loss, reconstruction,
  semantic loss, or navigation loss.
- The published training receipt relabels the inherited internal fields as
  `predicted_target_cross_covariance_identity`,
  `predicted_within_covariance_identity`, and
  `target_within_covariance_identity`. Removed raw-MSE and hinge-variance terms
  are listed as disabled and are not published as measured losses.
- B16 covariance estimates are noisy, but D8 identity covariance is feasible
  because centered sample covariance has rank at most 15. No loss/weight/LR
  adjustment is permitted in response to minibatch noise; full 2,048-row
  validation geometry is authoritative.

## Optimizer and cap

- Seed `20260727`; float32 without autocast; cuDNN benchmarking disabled.
- AdamW groups are unchanged: online encoder LR `1e-4`; dense history LR
  `3e-4`; action path/predictor LR `3e-4`; target pool LR `3e-4`; weight decay
  `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, independent group gradient
  clipping at norm 1.0.
- Exactly 1,000 updates, batch 16, and 16,000 ordered training presentations.
  All 2,048 validation rows are evaluated at updates `0,250,500,750,1000`.
  Active GPU time is capped at 90 minutes.
- There is no early scientific stop. The prior full cap took about 10.5 active
  GPU minutes; adding a second terminal receipt path would add false-negative
  and implementation risk for little saved time.
- Fresh output root:
  `.generated/go2_rgb_full_whitened_predictive_state_h4_jepa_v1/probe_v1`.
  Once reserved, any termination consumes the attempt. No retry or resume.

## Evaluation and selection

- Per scene/horizon, raw D8 squared-error numerators are summed before division
  by summed target energy. Family and global values are scene-then-family macro
  averages. Therefore exact-zero persistence has normalized error 1.
- Wrong-action, all-hold, reordered-history, and reset-history predictions are
  evaluation-only and use the same real target.
- Record per horizon for both branches: participation-rank ratio; minimum and
  maximum dimension std; maximum absolute mean; RMS; mean-energy fraction;
  target energy; covariance-identity error; minimum/maximum covariance
  eigenvalue; maximum variance error; maximum off-diagonal covariance; and
  count of near-zero scene denominators. Evaluator-fitted whitening is absent.
- Eligibility retains the old comparability guards and adds direct whitening
  evidence: all values finite; inherited
  encoder rank/variance floors pass; every branch/horizon has participation
  rank ratio at least 0.75, minimum std at least 0.50, maximum std at most 2.0,
  maximum absolute mean at most 0.25, mean-energy fraction at most 0.25, and
  RMS at most 3.0; every within-branch covariance-identity error is at most
  0.50; every predicted-target cross-covariance-identity error is at most 0.50;
  and every scene denominator is nonzero above `1e-8`. Zero covariance has
  identity error exactly 1, so each new gate requires at least halving that
  collapsed baseline.
- Select the eligible trained observation with minimum mean H1--H4 raw
  normalized prediction error. Controls never affect selection.

## All-conjunctive PASS gate

PASS requires all of the following; otherwise STOP:

- exact cap, all observations finite, fixed N320 teacher metric geometry
  unchanged within `1e-6`, and an eligible full-rank compact state exists;
- update-zero H1--H4 real errors equal 1 and every action, hold, persistence,
  and history gap equals zero within `1e-5`;
- every selected real error below 1, mean error at most 0.90, H4 at most 0.90;
- positive H4 persistence bootstrap lower bound, at least six positive families,
  and no family below -0.02;
- H4 wrong-action gap at least 0.03 with positive bootstrap lower bound,
  nonnegative H1--H3 gaps, at least six positive families, and no family below
  -0.02;
- H4 history gap at least 0.03 with positive bootstrap lower bound and at least
  six positive families;
- positive H4 all-hold gap;
- fixed teacher byte-identical before/after, outside the optimizer, no gradient,
  and zero EMA updates.

PASS establishes bounded full-whitened predictive-state JEPA feasibility only.
It does not authorize checkpoint access, navigation, held-out evaluation,
promotion, or deployment. STOP closes the full-whitened D8 learned-target
category and leaves every written checkpoint inaccessible.
