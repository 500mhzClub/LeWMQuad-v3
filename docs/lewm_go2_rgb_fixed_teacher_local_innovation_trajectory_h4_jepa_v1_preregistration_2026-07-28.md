# Go2 RGB fixed-teacher local-innovation trajectory H4 JEPA V1 preregistration — 2026-07-28

## Question and category boundary

- This preregisters one development-only perception-learning falsification. It
  grants no checkpoint, navigation, held-out, sealed, promotion, deployment,
  or later-run authority by itself.
- The four-atom trajectory-distribution predecessor was the first bounded H4
  branch to beat persistence at every horizon and in all eight families. Its
  selected H4 normalized energy score was `0.742969`, but its H4 wrong-action
  gap was only `0.010815`, ordered-history gap was `-0.013974`, and all-hold
  gap was `-0.006572`. It learned plausible futures more than controllable,
  history-dependent dynamics.
- Full-whitened D8 separately repaired much of the compact-state rank collapse
  but did not align prediction with target samples: update-1,000 H4 error was
  `2.474828` times persistence and action/history gaps were negative. That
  category is closed and no whitening mechanism is carried forward.
- The main pool is not exhausted. The current schedule covers every train
  scene, family, and action but uses about 1% of packed H6 candidates. More
  exposure is reserved for a mechanism that first improves the relevant
  dynamics metrics; it is not a reason to extend either stopped objective.
- Question: if the successful four-atom trajectory model predicts the four
  **local fixed-teacher transition innovations** that telescope into a future
  trajectory, while self-supervised counterfactual energy margins require the
  correct actions and ordered history, can it preserve its persistence win and
  learn controllable dynamics?
- This is a target and training-signal change. A STOP closes this exact local-
  innovation trajectory/counterfactual formulation.

## Immutable data and custody

- Each row remains `e0,p0,e1,p1,e2,p2,e3,p3,e4,p4,e5,p5,e6`. Online inputs
  are RGB `e0:e2`, past actions `p0:p1`, and proposed actions `p2:p5` only.
  Future RGB `e3:e6` is visible only to the frozen target branch during
  training and evaluation.
- No pose, odometry, depth, optical flow, occupancy, map, reward, collision,
  semantic, waypoint, simulator-state, or navigation label is an input,
  target, or loss.
- Training is the exact 16,000-row schedule SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`.
  Validation is the exact 2,048-row schedule SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`.
  There is no regeneration, reordering, filtering, substitution, or extra
  sample. RGB resolves only beneath the existing main-pool render root.
- Protected roles and stopped/rejected predictor checkpoints remain
  inaccessible. The accepted N320 encoder prefix is the only checkpoint input.

## Joint local-innovation trajectory JEPA

- Frozen target encoder `T` and trainable online encoder `E` both start from
  the accepted N320 encoder prefix. `T` is stop-gradient, outside the optimizer,
  has no EMA, and must remain byte-identical. `E` trains jointly with history,
  action, mode, and prediction modules in one backward pass.
- Reuse the reviewed dense trajectory backbone unchanged: all 256 spatial
  tokens from each of `e0:e2`, explicit `p0:p1` tokens, two history Transformer
  blocks, four equal-mass coherent trajectory modes, ordered action-prefix
  queries, and two shared cross-attention decoder blocks.
- Normalize fixed-teacher patch tokens at each frame. For horizon index
  `h=0..3`, the local target is exactly
  `q_h = T(e_(h+3)) - T(e_(h+2))`, paired respectively with actions
  `p2,p3,p4,p5`. Thus `sum_(j=0..h) q_j = T(e_(h+3)) - T(e2)`.
- The shared zero-initialized head emits one local increment for every
  atom/horizon/patch. Starting from normalized `E(e2)`, increments are applied
  recursively and each successor is normalized. The training innovation is
  the realized difference between successive normalized predicted states, not
  the raw pre-normalization head value. Those realized changes telescope into
  each cumulative trajectory atom.
- Update zero is exact `e2` persistence for all four atoms and every action or
  history control. There is no learned target compressor, compact D8 state,
  decoder recurrence, BEV, warp, flow, reconstruction head, navigation head,
  or separately trained probe/predictor.

## Objective

- `S(z,q)` is the predecessor's proper uniform empirical-distribution energy
  score: equal weight on the joint four-step trajectory score and the mean of
  four marginal-horizon scores. Here it is evaluated on predicted and target
  **innovation sequences**, not absolute `e2`-relative futures.
- `L_innovation = mean S(real_innovations, q)`.
- `L_history_align` is the mean squared tokenwise distance between normalized
  online and fixed N320 features for all three observed frames.
- `L_action` uses the same history and target but replaces each proposed action
  `a` by `(a+1) mod 9`. Scores are divided by the greater of detached zero-
  innovation energy and the fixed `1e-6` normalization epsilon. The hinge
  requires the real score to beat the cyclic wrong-action score by `0.05`.
- `L_history` uses the same future actions and target with two history
  ablations: `[e1,e0,e2]` plus `[p1,p0]`, and `[e2,e2,e2]` plus hold/hold.
  Normalized-score hinge margin `0.03` is applied against the better of the two
  ablations, requiring both to be worse than the real ordered history.
- Every sample contributes to both margin means. Clamping only the detached
  denominator at `1e-6` prevents a nearly zero target change from causing a
  division by zero; there is no motion-based sample filtering.
- Exact objective:
  `L_innovation + L_history_align + L_action + L_history`. Every term has
  weight 1.0. Correct-target energy anchors both margins; control-specific
  offsets cannot pass unless the real innovation distribution also fits the
  frozen target.
- There is no absolute-future training loss, best-of-K loss, learned scale or
  mixture weight, diversity bonus, variance/covariance/whitening term,
  codebook, reconstruction, semantic, or navigation loss.

## Optimizer and cap

- Seed `20260727`; float32 without autocast; cuDNN benchmarking disabled.
- AdamW groups remain: online encoder LR `1e-4`; dense history LR `3e-4`;
  action/mode/predictor LR `3e-4`; weight decay `1e-4`, betas `(0.9,0.999)`,
  epsilon `1e-8`, with independent group gradient clipping at norm 1.0.
- Exactly 1,000 updates, batch 16, and 16,000 ordered training presentations.
  All 2,048 validation rows are evaluated at updates `0,250,500,750,1000`.
  Active GPU time is capped at 90 minutes.
- Counterfactuals reuse loaded training RGB; they add no scheduled row and no
  physical RGB-file open. One wrong-action predictor evaluation and two
  history-control predictor evaluations per training row are accounted
  separately in the access receipt.
- Fresh output root:
  `.generated/go2_rgb_fixed_teacher_local_innovation_trajectory_h4_jepa_v1/probe_v1`.
  Once reserved, termination consumes the attempt. There is no retry or resume.

## Evaluation and selection

- Validation intentionally remains on cumulative absolute trajectory atoms in
  the predecessor's fixed-teacher metric. Per-scene/horizon energy is divided
  by the same target persistence energy; exact persistence has normalized
  score 1. Joint and 50/50 combined trajectory scores remain registered.
- Registered validation diagnostics remain unchanged and gradient-free:
  wrong-action, all-hold, reordered-history, reset-history, persistence,
  centroid, best-atom, distribution-value, and spread. Wrong-action and the
  two history forms also have distinct training-time counterparts described
  above; hold, persistence, and distribution diagnostics remain evaluation-
  only.
- Record fixed-target and online-encoder rank/near-zero variance, all aggregate
  and family metrics, scene-bootstrap H4 lower bounds, objective components,
  exact access counts, and fixed-teacher identity.
- Eligibility requires all values finite, fixed-target effective-rank ratio at
  least `0.10` with near-zero variance fraction at most `0.05`, and online-
  encoder effective-rank ratio at least `0.10` with near-zero variance fraction
  at most `0.05`. No compact-state or full-whitening gate is carried forward.
- Select the eligible trained observation with minimum validation 50/50 joint-
  plus-marginal cumulative normalized energy score. Control margins never
  determine selection.

## All-conjunctive PASS gate

PASS retains the trajectory predecessor's exact gate:

- exact cap, all observations finite, fixed-teacher metric geometry unchanged
  within `1e-6`, target/online rank guards pass, and an eligible trained
  observation exists;
- update-zero marginal, joint, and combined scores equal 1 and all gaps/spread
  equal zero within `1e-5`;
- selected combined and joint scores beat persistence, H1--H3 marginal scores
  beat persistence, and H4 marginal score is at most `0.90`;
- positive H4 persistence bootstrap lower bound, positive persistence in at
  least six families, and no family below `-0.02`;
- combined distribution-value gap at least `0.05`, positive bootstrap lower
  bound, positive in at least six families, and H4 normalized pairwise spread
  at least `0.05`;
- H4 wrong-action gap at least `0.05`, positive bootstrap lower bound,
  nonnegative H1--H3 gaps, positive in at least six families, and no family
  below `-0.02`;
- H4 ordered-history gap at least `0.03`, positive bootstrap lower bound, and
  positive in at least six families;
- positive H4 all-hold gap; and
- fixed teacher byte-identical before/after, outside the optimizer, without
  gradient, and with zero EMA updates.

PASS establishes bounded local-innovation trajectory JEPA feasibility only. A
passing checkpoint would still require a separate scale decision using fresh
main-pool windows before navigation or protected benchmark evaluation. STOP
grants no checkpoint access and closes this exact mechanism.
