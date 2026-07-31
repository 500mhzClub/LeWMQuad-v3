# Go2 V18 spatial-token delay-line causal-convolution joint-JEPA V1 preregistration — 2026-07-31

## Decision and purpose

- Run at most one fresh development-only falsification of a jointly trained
  RGB perception, finite causal memory, and action-conditioned JEPA predictor.
- This is the required broader memory-integration successor to the valid V5
  scene-local place failure recorded in commit `8808d03`. It is not a V5
  retry, longer V5 run, isolated place-head V6, or rejected-checkpoint reuse.
- The question is now: **can ordered V18 object-space observations improve
  action-conditioned future-state prediction over current-frame persistence,
  reset memory, reversed history, and shuffled history?**
- A pass earns only a short-horizon causal-memory substrate. Longer episode
  memory, navigation, G2, and held-out mazes remain later stages.

## Why this mechanism

- V5 began with useful single-frame place structure and retained 2.13 times
  exact-chance R@5 after training, but same-place positive energy worsened and
  R@5 retained only 88.86% versus its frozen 90% guard. V5 nevertheless beat
  V4 materially, showing that the remaining problem is not absence of RGB
  signal.
- Earlier N320 recurrent-H4, dense-attention, whitened-state, momentum,
  system-identification, and posterior-expert branches learned some mixture
  of generic prediction, persistence, or action sensitivity, but correct
  ordered history remained harmful in every family.
- V27/V28 jointly trained V18 object-space whole-plan predictors, but they
  consumed only the current observation plus a proposed action plan. They did
  not carry an ordered observation memory and deterministic absolute
  whole-plan targets remained 3.46 to 5.32 times worse than copying the
  current state.
- The new mechanism changes both missing ingredients without reopening those
  exact mechanisms: it retains a lossless finite ordered tape and predicts
  recursive one-step V18 object-space targets with a local finite-impulse
  response reader.

## Frozen data and roles

- Reuse the corrected reset-safe H6 V2 indices exactly:
  - train: 16,000 scene-balanced sequences from 1,000 development scenes;
  - checkpoint selection: 2,048 sequences from 150 scene-disjoint development
    scenes;
  - each row contains seven same-stream RGB endpoints and six causal primitive
    actions;
  - all nine actions occur at every position in every family.
- Reuse the existing V18 physical train and checkpoint-selection roles and
  the V5 place checkpoint-selection panel only as specified below.
- No dataset/index build, corpus scan, probability calibration, navigation,
  G2, held-out, sealed, test, production, or deployment role is permitted.
- Scene, family, raw pose, cell, yaw, labels, and navigation outcomes may be
  used by the runner only for role integrity and evaluation aggregation. None
  may enter the model.

## Model and causal state

- Start fresh from the reviewed V18/N320 initialization. No V1–V5 private
  restart tensor, rejected temporal checkpoint, predictor checkpoint, or
  navigation checkpoint may be opened or reused.
- The online V18 encoder produces an object-space latent of shape
  `(B,64,64,64)`. Fixed 4×4 average pooling followed by per-spatial-token L2
  normalization produces `z_t` of shape `(B,64,16,16)`.
- Memory is one explicit four-slot FIFO delay line:
  - slot 0 is newest;
  - inserting a token map shifts slot `k-1` to slot `k`;
  - validity bits and the executed-action tape shift identically;
  - reset clears every older slot and action;
  - stored token maps are never blended, warped, refined, or mutated after
    insertion.
- At observed history endpoint `e2`, the valid tape is `[z2,z1,z0,empty]` and
  contains actions `p1,p0` in their exact age positions. During future rollout,
  each predicted token map and executed candidate action are inserted before
  the next step.
- This state is not an RNN/GRU/LSTM, transformer, dense cross-attention bank,
  momentum/innovation filter, writable system-identification matrix, learned
  warp/flow, codebook, posterior, or expert mixture.

## Learned predictor

- Add learned age embeddings to the four valid token planes and append their
  validity channel.
- One depthwise-separable causal `Conv3d` with kernel `(4,3,3)` collapses the
  four age planes while seeing only a local 3×3 spatial neighbourhood.
- A small MLP over the ordered four-position, nine-way action tape supplies
  channelwise FiLM scale and bias. There is one shared predictor for every
  horizon and every action; there is no per-action operator bank.
- The causal convolution is initialized as an exact newest-tap selector, the
  pointwise map is initialized to identity, and action FiLM is initialized to
  scale one/bias zero. Update-zero full-context rollout therefore equals
  current-token persistence at all four horizons without an external
  current-frame output bypass or a separately trained predictor.
- The predictor output is per-token L2 normalized and inserted back into the
  FIFO for recursive H1–H4 rollout.

## JEPA target and objective

- Future RGB is visible only to a stop-gradient EMA V18 target encoder. The
  target is hard-synced at initialization and updated once, after each
  successful optimizer step, with the existing EMA momentum `0.996`.
- The full-context memory loss is mean token cosine energy between four
  recursive predictions and the corresponding EMA V18 future token maps.
- A second context-mask branch deterministically masks half of the newest
  online 16×16 tokens in contiguous 4×4 blocks while retaining the older FIFO
  slots. It predicts the same full future targets with weight `0.5`.
- Reset, reverse, shuffle, wrong-action, HOLD, and persistence controls remain
  evaluation-only. No auxiliary ranking or history hinge trains against the
  answer key.
- Per optimizer update, accumulate exactly:
  - two inherited physical B4 microbatches: 8 physical presentations;
  - eight memory B2 microbatches: effective memory batch 16;
  - route losses are averaged within route before the shared optimizer step.
- The complete objective is the unchanged inherited physical JEPA objective
  plus `1.0 * full_memory_JEPA + 0.5 * masked_memory_JEPA`.
- One AdamW jointly updates the online RGB encoder, V18 representation,
  inherited physical predictor, and causal-memory predictor. Target modules
  are excluded and must receive zero gradients. Optimizer hyperparameters,
  float32 model state, mixed-precision execution policy, gradient clipping,
  seed, and EMA order must be frozen before execution.

## Schedule, checkpoints, and cap

- Effective memory batch: 16 sequences per update, implemented as eight B2
  microbatches for the 34 GB R9700 memory ceiling.
- Observations: updates `0`, `100`, `250`, `500`, `750`, and `1000`.
- Stage A is capped at update 500:
  - 8,000 memory-sequence presentations;
  - 4,000 physical presentations;
  - 12,000 total route presentations.
- Stage B is preauthorized only if the frozen update-500 continuation gate
  passes. It continues the same attempt to update 1,000:
  - 16,000 cumulative memory-sequence presentations;
  - 8,000 cumulative physical presentations;
  - 24,000 cumulative route presentations.
- Publish complete continuation snapshots at updates 250, 500, 750, and 1000
  containing model, optimizer, EMA, RNG, exact schedule cursor, and accounting.
  They permit exact same-attempt recovery after infrastructure interruption;
  they do not permit a scientific retry, alternate seed, post-failure resume,
  or reuse of an ineligible observation.
- No observation may mutate model or optimizer state. Selection uses only the
  registered checkpoint-selection roles.

## Metric definitions

- All primary temporal metrics are scene-then-family macro averages with a
  deterministic scene bootstrap lower 95% bound.
- For horizon `h`, with EMA target energy `E`:
  - real normalized score: `S_h = E(real,h) / E(persistence,h)`;
  - persistence lift: `P_h = 1 - S_h`;
  - action lift: `A_h = (E(wrong_action,h)-E(real,h))/E(persistence,h)`;
  - ordered-history lift:
    `H_h = (min(E(reset),E(reverse),E(shuffle))-E(real,h)) / E(persistence,h)`.
- Report all four horizons, family values, bootstrap bounds, HOLD controls,
  target/online/memory-state participation rank, scale, and near-zero feature
  fractions.
- Re-evaluate the V5 single-frame place panel and inherited physical panel at
  every registered observation. These protect the perception substrate but
  do not redefine the temporal task.

## Frozen gates

### Update zero and first update

- Exact source/split/schedule/access/target integrity; all eight families and
  valid persistence denominators present.
- Full-context prediction equals persistence at all horizons within `1e-5`;
  `P_h`, `A_h`, and `H_h` are correspondingly zero within numeric tolerance.
- Target and online states are finite and hard-sync identities hold. Target
  gradients and future-RGB online access are zero.
- The inherited substrate reproduces place R@5 at least 2× exact chance in at
  least six scenes and target place-key effective rank at least 2. Physical
  update zero is identity-bound; it is not required to clear the trained
  margin floor.
- Update one must show nonzero encoder, representation, physical-predictor,
  and memory-predictor gradient routes; exactly one optimizer and EMA step;
  and zero target gradients.

### Update 250 futility gate

- Stop immediately for integrity failure, target/online/memory collapse,
  place R@5 below 1.5× exact chance, or missing gradient/accounting routes.
- Collapse means memory participation-rank ratio below `0.10`, more than 5%
  near-zero dimensions, or nonfinite/zero-scale state.
- Otherwise stop for scientific futility only if **all** are true at both
  updates 100 and 250:
  - H4 persistence lift is nonpositive;
  - H4 action lift is nonpositive;
  - H4 ordered-history lift is nonpositive;
  - zero families have positive H4 ordered-history lift.

### Update 500 continuation gate

- Integrity and perception safeguards must pass.
- H4 persistence lift is positive, its bootstrap lower bound is positive, and
  at least six families are positive.
- H4 action lift is positive, its bootstrap lower bound is positive, and at
  least six families are positive.
- H4 ordered-history lift is positive in at least four families. Its lower
  bound may still cross zero by the terminal observation.
- Mean H1–H4 persistence lift is positive.
- Failure closes this exact mechanism at update 500. Passing automatically
  continues the same attempt to the exact terminal cap.

### Terminal qualification and selection

- Eligible observations are updates 500, 750, and 1000 that pass integrity,
  noncollapse, perception safeguards, and have positive H4 persistence,
  action, and ordered-history lifts.
- Select the eligible observation with minimum mean H1–H4 real normalized
  score. Do not mix selection metrics or choose an unregistered update.
- Terminal PASS additionally requires:
  - positive persistence lift at H1–H4;
  - H4 persistence lift at least `0.10`, positive lower 95% bound, and at
    least six positive families;
  - H4 action lift at least `0.05`, positive lower 95% bound, and at least six
    positive families;
  - H4 ordered-history lift at least `0.03`, positive lower 95% bound, at
    least six positive families, and nonnegative history lift at three of four
    horizons;
  - memory participation-rank ratio at least `0.10`, finite nonzero scale,
    and no more than 5% near-zero dimensions;
  - target/online perception noncollapse;
  - place R@5 at least 2× chance in at least six scenes, target place rank at
    least 2 and at least 80% of update zero;
  - physical passed-margin count at least 60/189 and all 12 inherited physical
    causal controls.

## Diagnostics that do not veto this memory stage

- V5's exact 90% single-frame R@5 retention and strict positive-energy-below-
  update-zero checks remain valid reasons V5 failed, but are not hard temporal
  memory gates.
- Immediate local prediction versus copy, absolute place energy, legacy 3×
  R@5/rank-4 targets, HOLD breadth, best-atom/centroid planner quality, and
  rough/tail/prior submetrics must be reported but cannot override demonstrated
  causal-memory value.
- Navigation success, collision rate, progress, timeout, and long-loop closure
  are not short-H6 gates. They require separate later authority and explicit
  memory-vs-reset/no-memory ablations.

## Failure and pass boundaries

- A scientific failure permits a terminal receipt and mechanism audit only:
  no retry, second seed, loss-weight tweak, mask variant, longer run, rejected
  checkpoint opening, navigation, G2, or held-out access.
- A pass permits a separately reviewed long-episode/revisit memory test. It
  does not itself qualify navigation or any held-out maze.
- No eligible sealed benchmark role currently exists. This preregistration
  does not create one.
- Execution remains denied until the implementation, focused tests, source
  closure, independent review, clean export, and one-shot authority are all
  complete and bound.
