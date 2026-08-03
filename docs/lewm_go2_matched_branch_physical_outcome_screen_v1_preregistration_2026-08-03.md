# Go2 matched-branch physical-outcome screen V1 preregistration

**Frozen:** 2026-08-03, before implementation and before any candidate score is
computed.  **Role:** development-only mechanism screen.  This is not held-out,
sealed, safety, rollout, planner, or navigation evidence.

## 1. Decision

This one-shot experiment asks the nearest unresolved question after the frozen
V-JEPA 2.1 and DINOv2 planning-interface stops:

> Do retained pre-action odometric history and current visual context permit a
> small action-conditioned physical dynamics model to rank the nine executed
> branches better than the fixed task/action prior?

It does not train a JEPA, RSSM, Dreamer agent, policy, critic, reward model, or
planner.  A positive visual result permits only a separately preregistered
dense-JEPA/conventional/Dreamer comparison.  A negative result stops this
retained-input visual route without tuning.

## 2. Input and claim boundary

The immutable development panel has 128 train states in 16 scenes and 128
evaluation states in 16 disjoint scenes, balanced over eight families.  Every
state has all nine physically executed candidate branches.  The effective
independent units are states and scenes, not the 1,152 branch rows per role.

The admitted joined JSONL omits numeric pre-action robot state.  The already
bound upstream state receipts retain exactly three context poses, the last of
which is the prebranch pose.  A pure metadata adapter may derive two local
odometry increments and branch physical outcomes from those receipts.  This
requires no collection, rendering, RGB decoding, or encoder execution.

The learned arm is therefore named **odometry-and-command-history**, not
proprioception.  The panel contains no pre-action numeric base velocity, IMU
stream, joint state, contact state, or stall label.  It has zero falls and zero
tips across 2,304 branches.  Safety is `NOT_TESTABLE_ZERO_EVENT_SUPPORT`, not
passed.

The evaluation panel has already informed earlier development work.  This run
is prospective with respect to this mechanism, but it cannot support a fresh
confirmation claim.

### Frozen direct inputs

| Input | SHA-256 | Bytes |
|---|---|---:|
| posthoc manifest | `87448995c905107453814a5e7e4cd9968d31cbc0e308513d17bc038c6585f15e` | 11,964 |
| posthoc terminal | `a1590fffc673f7676016bb70d4b4f5530f24b9a49bf05e84dcec6bc1756fbe56` | 1,250 |
| posthoc train JSONL | `edc6f88bb105c39575477fbfbb0224bf0312cf5ee3e90551f86a9c11c2ebb447` | 30,432,624 |
| posthoc eval JSONL | `531debbc431f2f8afc83a491b491b8822134c831b16ca4d283fe1e7f4ba07768` | 30,411,588 |
| posthoc terminal review | `bfd0250357d0f681c674db6c54ea4a8c4d5e617230332383beda3db3e0f38669` | 2,844 |
| upstream physics result | `25caf0a5d4c69e99559a663aa4cae96fb23ef191ccf34486804c3f2243553314` | 183,320 |
| physics receipt check | `faeb50293bc684e35b6d725b027983ad0110e739db2d7b1aca1926e89a547dc6` | 892 |
| consumed collection terminal | `f7d2796139645892d22ad6bb99d26caffc2b5c3dcac2a655b1883b299d22bff4` | 12,949 |
| authorized collection plan | `8fe34054bb9ae709b6a8ecfea5fdae55c742d1b2e22af3c289d27a77f11c66ef` | 343,973 |
| calibration receipt | `58d1291ede7ee03a93d68eb7cec80c9322c47cd0b1d5fd1c41bf8f4b49ad484e` | 72,475 |
| DINOv2 train cache | `164f1fef8c859976c93f7fc978e938c6c8f7f9963cf92bb154f51b23d158b34b` | 302,107,682 |
| DINOv2 train-cache receipt | `e94ec5d188811c44d4cc870e76d1888aa6f30ee6d423557ee9f3e2918a700994` | 1,770 |
| DINOv2 eval cache | `00a2e197d98effcd192392f50170648622a7210f954075002dc8b43110c636f8` | 302,106,281 |
| DINOv2 eval-cache receipt | `d3e928cc563beb4dd850f34ca41915b8e5974c6d0b1b182602f3e3f20828421c` | 1,770 |
| predecessor dense-DINO terminal review | `f6ed2d09a407a4cf70097eaa4b2dcffd223e598e4eb59cf8e751997459384020` | 27,120 |

The upstream physics result binds all 256 original state receipts.  The runner
must rehash every receipt through the existing strict derivation path before
using it.

## 3. Leakage-safe metadata projection

For each state, use only values at or before the prebranch time:

1. From the three context poses, derive pose-0 to pose-1 and pose-1 to pose-2
   increments as body-local `(dx, dy, wrapped_dyaw)`.
2. For each of the two past executed `5 x 3` command blocks, take mean
   `(vx, wz)`.
3. For the candidate, use only requested `(vx, wz)`.

This is the fixed 12-scalar physical input.  Absolute world position/yaw,
state/scene/family IDs, hashes, future executed commands, clipping, trajectory
samples, endpoint state, physical labels/ranks, target RGB, and future tokens
are forbidden as model inputs.  In particular, trajectory sample zero is 20 ms
post-action and is never a current-state input.

Targets are derived separately for each branch: prebranch-body-frame endpoint
`dx`, `dy`, wrapped `dyaw`, and physical path length.  Target progress remains
an evaluation label and is not a regression target.

## 4. Arms

### A — fixed task/action-only control

Reuse the exact 27-coefficient, nine-head ridge implementation and train-only
refit from the dense-DINO calibration.  Its required identity is
`69895316b19bc179e35fdd76905aadbd50b6ad3e22e965b662ba59672c52886a`
and its required evaluation regret is `0.17441406250000002`.

### B — odometry-and-command-history dynamics

The 12 physical inputs are followed by 16 zero-valued visual slots.

### C — odometry-and-command-history plus current visual context

Only the three context DINO grids may be indexed.  Each `(16,16,384)` grid is
mean-pooled over fixed non-overlapping `4 x 4` token blocks to `(4,4,384)`.
The three flattened grids are concatenated in time/row/column/channel order.
Fit a 16-component PCA using only the 128 train-state context vectors:
float64 column centering, thin SVD, descending singular order, and largest-
absolute-loading sign made positive.  Evaluation is projected with that frozen
train mean/components.  No target/successor token may enter PCA or either
learned arm.

## 5. Fixed learned mechanism

B and C have the identical `28 -> 16 -> 4` one-hidden-layer tanh MLP: 532
parameters per member.  Initialize both arms identically for each seed with
dedicated-CPU-generator Xavier-uniform weights and zero biases.

For each action, compute the train-only mean physical outcome.  The MLP predicts
the standardized residual from that action mean.  Input population mean/scale
and four residual-output population scales are train-only; a zero or sub-
`1e-8` scale is replaced by one.  Loss is the unweighted mean squared error of
the four standardized residuals.

- seeds: `2026080311`, `2026080312`, `2026080313`;
- complete nine-action state minibatches, 16 states per batch;
- exactly 1,024 AdamW updates per member;
- learning rate `3e-4`, weight decay `1e-4`, betas `(0.9,0.999)`, epsilon
  `1e-8`, gradient clip norm `1.0`;
- CPU float32, deterministic algorithms, one Torch thread;
- no early stopping, evaluation monitoring, checkpoint selection, coefficient
  search, retry, or resume.

Ensemble predictions are the arithmetic mean of the three decoded physical
outcomes.

## 6. Frozen action scoring

For predicted body displacement `(dx,dy)` and body-frame goal `g`, predicted
progress is `||g|| - ||g - (dx,dy)||`.  Predicted path length is clamped at
zero only for physical scoring.  Rank the nine predicted outcomes using the
existing 1 cm contract: progress quantized descending, then path length
quantized ascending, then action ID.  All observed fall/tip labels are zero, so
the predictor neither consumes nor fabricates safety predictions.

The privileged oracle uses true dense ranks and must first obtain zero regret
and 100% oracle-equivalent selection.  Random expectation is reported.

## 7. Primary analysis and gates

Primary metric: evaluation normalized physical rank regret, lower is better.
All intervals are paired candidate-minus-baseline, with equal family weight,
whole-scene resampling, 10,000 draws, and fixed seed `2026080314` over the 16
evaluation scenes.

1. **Infrastructure/custody:** exact source/input/output bindings; no
   RGB/encoder/protected access; no leakage; finite tensors.
2. **Oracle sensitivity:** regret `0` and oracle-equivalent rate `1`.
3. **Odometry headroom:** B minus A upper 95% endpoint `< 0`, and all three B
   seed point regrets are below A.
4. **Visual versus task:** C minus A upper 95% endpoint `< 0`, and all three C
   seed point regrets are below A.
5. **Incremental visual value:** C minus B upper 95% endpoint `< 0`, and every
   matched-seed C-minus-B point regret is negative.
6. **Random sanity:** any arm used for an advance decision has point regret
   below random expectation.
7. **Replay:** a fresh process independently rebuilds the projection/PCA,
   retrains all six members, and exactly reproduces identities, predictions,
   selected actions, summaries, intervals, gates, and verdict.

Per-output RMSE and joint standardized MSE versus zero motion and train-only
action means are diagnostic.  They cannot advance or stop the route.

## 8. Terminal decision

- Gates 1, 2, 4, 5, 6(C), and 7 pass:
  `PASS_VISUAL_PHYSICAL_DYNAMICS_HEADROOM`; permit only a new preregistration
  for the dense-JEPA/conventional/full-Dreamer comparison.
- Otherwise, if gates 1, 2, 3, 6(B), and 7 pass:
  `PASS_ODOMETRY_ONLY_PHYSICAL_DYNAMICS_HEADROOM`; retain B as a conventional
  baseline and stop visual escalation on this route.
- Otherwise:
  `STOP_RETAINED_INPUT_PHYSICAL_DYNAMICS_HEADROOM_NOT_ESTABLISHED`; no tuning,
  3 TB scaling, fresh 1,024-state campaign, or visual-JEPA escalation.
- Contract/infrastructure failure:
  `FAIL_INFRASTRUCTURE_NO_SCIENTIFIC_DECISION`; it grants no automatic retry.

No outcome authorizes a navigation claim, closed-loop use, protected access,
bulk data processing, or deployment.

## 9. One-shot execution and review

Implementation must be committed after this document, pass focused synthetic
tests, and receive an independent exact-source review.  A separate authority
must bind that reviewed commit, this preregistration, all direct inputs, and an
empty `attempt_v1` output root.  The attempt is consumed on reservation.  The
runner must fail closed, write the checkpoint before evaluation publication,
launch the fresh-process replay once, and publish an independently audited
terminal record.  Retry, resume, replacement, and result-dependent changes are
not preregistered.
