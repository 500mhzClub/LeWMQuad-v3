# RGB Patch-Whitened Action-Residual JEPA V1 preregistration

Date: 2026-07-25

## Decision and question

Authorize source preparation and independent source/science review for exactly
one RGB-only Patch-Whitened Action-Residual JEPA V1 probe.

The question is whether a current-RGB encoder can learn a non-collapsed,
requested-action-sensitive latent transition representation when the
objective:

1. predicts the residual change from current to next latent state instead of
   regressing the full next state directly;
2. whitens the exact position-centered patch features evaluated by the rank
   gate; and
3. separates the true requested primitive from every other real primitive,
   including the real `hold` primitive.

Execution is not authorized by this document. It requires a frozen source
manifest, one independent source/science review, and a distinct one-attempt
execution authorization.

## Prior evidence and mechanism decision

RGB JEPA Encoder Pretraining V3 is a valid terminal scientific failure. Its
tracked terminal audit is
`docs/lewm_go2_rgb_jepa_encoder_pretraining_v3_integrity_replacement_terminal_audit_2026-07-25.json`
at commit `2f96c3f436f0b43f84cb09c9f375542d537e35ec`, file SHA-256
`b6f74ff0a212ac4cf363e59c8fe2f07ed474b8ce4e6ee1ddbd5dafc0c19fe8de`,
content SHA-256
`ccc145e1cd7ed863ae97123b3f91c3a89b5833dbaa5483cff77ced8bcd33341c`,
11,594 bytes.

V3 completed 1,000 updates and 16,000 presentations. True-pair prediction
improved relative to shuffled current, shuffled next, and mean target, but:

- centered raw-patch effective rank fell from `27.7175` to `7.3357`;
- centered projected-target effective rank fell from `17.4267` to `1.4239`;
- true/wrong-action and non-hold true/zero-vector ratios ended at `0.9963`
  and `0.9977`, versus required maxima of `0.95`; and
- only `5/8` scene families had positive wrong-action and zero-vector margins.

This is not a license to extend, resume, reopen, or retune V3. The marginal
variance floor could be satisfied by strongly correlated low-rank features,
CLS-only SIGReg did not constrain patch rank, absolute-next regression
rewarded visual persistence, and the training hinge was much weaker than the
registered relative action gate.

V1 therefore changes one coherent objective mechanism: patch-whitened
action-residual JEPA. It does not change the encoder size, data, seed,
presentation schedule, physical evaluator, or downstream promotion order.

## Frozen data, roles, and initialization

Reuse exactly:

- audited Raw V13 train role: 4,262 pairs, 7,777 unique endpoints, 72 scenes;
- audited Raw V13 checkpoint-selection role: 495 pairs, 924 unique endpoints,
  8 scene-disjoint scenes/families;
- the qualified N320 encoder initialization only;
- base initialization seed `20260712`;
- schedule seed `20260713` and its exact first 16,000 presentations;
- RGB preprocessing: PIL RGB, bilinear `112x112`, `[0,1]`, ImageNet mean/std;
- the nine requested primitives in the frozen order:
  `arc_left`, `arc_right`, `backward`, `forward_fast`, `forward_medium`,
  `forward_slow`, `hold`, `yaw_left`, `yaw_right`.

Phase A may open only bound current RGB, bound next RGB, and the requested
primitive from train and checkpoint-selection roles. It may not open realized
motion, Camera/ray/depth/ground/raster supervision, calibration,
probability-calibration, G2, navigation, held-out, or sealed inputs.

The online path receives current RGB only. Frozen EMA target modules may
encode current and next RGB under `no_grad`. N320 copies initialize the online
and EMA raw encoders exactly. No prior-attempt checkpoint or runtime output is
an input.

## Frozen architecture and residual prediction

Retain the V3 Phase2D model budget:

- ViT encoder: `112x112`, patch `7`, 256 patch tokens, latent width 192,
  depth 6, 6 heads, MLP ratio 4, dropout 0;
- spatial predictor: 2 blocks, 6 heads, head width 32, MLP width 384,
  dropout 0;
- online/EMA target projectors and prediction projector: 192 to 192;
- EMA momentum `0.996`;
- float32, no autocast.

For online raw current patch state `s`, define:

- `z_current = normalize(P_ema(s_current_ema))`;
- `z_next = normalize(P_ema(s_next_ema))`;
- `u(a) = P_prediction(Predictor(s, a))`;
- `alpha = 0.1/sqrt(192) = 0.007216878364870322`;
- `p(a) = normalize(z_current + alpha*u(a))`.

Both EMA targets and the current-state skip are stop-gradient. `p(a)` is an
explicit current-state skip plus a fixed-scale learned action-conditioned
residual. The primary JEPA loss is mean patch-feature squared error between
`p(requested_action)` and stop-gradient `z_next`. The online encoder receives
gradient through the true-action residual predictor, not through the EMA
skip.

At initialization, keep every predictor AdaLN shift/scale row and bias zero.
Create one CPU float32 `torch.Generator`, seed it once with `20260712`, and
advance it across predictor blocks in module order. For each block, draw the
attention-gate weight rows `[2D:3D]` and then the MLP-gate weight rows
`[5D:6D]` with
`torch.randn(shape, generator=g)*0.01/sqrt(192)`, in that order, and copy
those values to the parameter device. Set only the matching bias rows to
`0.01`. All other modulation weights and biases remain zero. Reseeding per
block or using a device generator is forbidden. This opens a small
action-dependent path at update zero without changing global initialization
RNG state or adding parameters.

The inherited appearance projector remains present for state-shape
compatibility but is frozen, excluded from optimizer and gradient-clip lists,
and receives no gradient. It is not deleted or silently left as a
trainable-but-gradientless parameter.

These initialization edits are part of the new mechanism and must be bound in
the Phase-A initialization receipt.

## Exact patch-whitening objective

Whitening is applied separately to:

1. online raw current patch tokens; and
2. normalized online projected current patch tokens.

For tokens `Z` with shape `(B,N,D)`, form the exact rank-gate matrix:

- `Q = reshape(Z - mean_B(Z), (B*N,D))`;
- `A = Q / sqrt(stop_gradient(mean(Q^2)) + 1e-4)`;
- `C = A^T A / (B*N - 1)`;
- `V(A) = mean_d(relu(1 - sqrt(C_dd + 1e-4)))`;
- `K(A) = sum_{i != j}(C_ij^2) / D`.

The detached global RMS denominator preserves relative covariance directions
instead of allowing per-coordinate standardization to conceal rank collapse.
`V` maintains feature spread and `K` directly penalizes off-diagonal
covariance. The computation is deterministic float32, differentiates no
eigendecomposition, uses no running statistics, and never uses
checkpoint-selection data for gradients.

The V3 CLS-only appearance SIGReg and marginal projected spatial-variance
loss are disabled in this probe. Their replacements are exactly the two
patch-whitening branches above.

## Exact all-action separation objective

For each training row, evaluate `p(a)` for all nine real one-hot primitives.
The true requested primitive is positive and the other eight are negatives.
The all-zero vector is neither trained nor evaluated. For non-`hold` rows, the
real `hold` one-hot is the no-motion baseline.

Let `E_true` be the mean patch-feature MSE to `z_next`, and let `E_k` be the
corresponding error for action candidate `k`. Define:

- for a non-`hold` row,
  `L_wrong` over its seven non-true, non-`hold` alternatives;
- for a true-`hold` row, `L_wrong` over all eight alternatives;
- compute each row's candidate mean first, then the mean across rows:
  `L_wrong = mean_rows(mean_candidates(relu(`
  `stop_gradient(E_true)/0.95 - E_k)))`;
- on non-`hold` rows,
  `L_hold = mean(relu(stop_gradient(E_true)/0.95 - E_hold))`.

If a microbatch contains no non-`hold` row, define `L_hold` as an exact scalar
zero on the same device/dtype.

Compute `p(requested_action)` and `E_true` once from the live online state for
`L_jepa`. Compute only non-true and real-hold control candidates from the
detached online state, and use `stop_gradient(E_true_live)` in both hinges.
For an observed `hold` row, the live true branch is not also treated as a
detached hold control.

Zero loss is therefore the exact registered inequality
`E_true/E_alternative <= 0.95`, rather than V3's much weaker target-change
floor. Non-true candidate states are detached, so these losses train the
shared predictor/prediction-projector/action path rather than letting the
encoder manufacture action-specific current states. The ordinary true-action
JEPA loss and whitening losses retain their normal encoder gradients.

The total per-microbatch objective is exactly:

`L = L_jepa + L_wrong + L_hold`
`    + 0.50*(V_raw + V_projected)`
`    + 0.02*(K_raw + K_projected)`.

The whitening proportions are the VICReg `25:25:1`
invariance/variance/covariance proportions divided by 25 and split across the
two exact raw/projected patch branches. There is no weight sweep.

No consequence, realized-motion, Camera, action-utility, pixel, depth,
occupancy, reconstruction, or navigation loss is present.

## Optimizer and schedule

Retain:

- AdamW, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`;
- encoder learning rate `1e-4`;
- predictor and projector learning rate `3e-4`;
- global gradient clip norm `1.0`;
- microbatch 4, four microbatches per update, effective batch 16;
- checkpoint observations at updates 0, 100, 400, and 1,000;
- maximum 1,000 optimizer/EMA updates and 16,000 presentations;
- maximum 60 Phase-A GPU-active minutes;
- one fixed seed and schedule, one fresh attempt, no retry or resume.

## Diagnostics, early stop, and final Phase-A gate

Diagnostics use the complete 495-row checkpoint-selection population and
preserve RNG/model state. They evaluate all nine real primitives. The
registered acceptance control remains the frozen cyclic wrong primitive
`(requested_index + 1) mod 9`. For each row, `hardest_wrong` is additionally
reported as the lowest-MSE alternative among all eight wrong primitives, but
it is informational and never an acceptance conjunct. The non-`hold` baseline
is the actual `hold` primitive.

Informational hardest-wrong aggregation is fixed as:

- `E_hardest[row] = min_{k != requested}(E_k)`;
- ratio `= mean_rows(E_true) / mean_rows(E_hardest)`;
- per-family margin `= mean_family(E_hardest - E_true)`.

Receipts record 495 rows and 3,960 wrong candidates. Shuffled-current
diagnostics apply the same scene-local mapping to both the online raw current
state and its matching EMA-current skip. The original requested action and
original EMA-next target remain unshuffled.

At update 100, continue only if all are true:

- raw effective rank `>27.717458724975586`;
- projected effective rank `>17.426651000976562`;
- true/cyclic-wrong and non-`hold` true/hold ratios are both `<0.99`; and
- cyclic-wrong and real-hold margins are each positive in at least `6/8`
  scene families.

At update 400, continue only if all are true:

- raw effective rank `>=37.85872936248779`;
- projected effective rank `>=32.71332550048828`;
- true/cyclic-wrong and non-`hold` true/hold ratios are both `<=0.975`;
- cyclic-wrong and real-hold margins are each positive in at least `6/8`
  scene families; and
- the final shuffled-next, shuffled-current, mean-target, variance-retention,
  spatial-diversity-retention, finiteness, population, EMA-gradient, and
  mutation conjuncts already pass.

Each intermediate rank threshold is fixed from V3 update zero and the final
rank-48 gate; update 400 is the exact halfway value. Missing any continuation
conjunct is a terminal scientific fail, not a retry license.

At update 1,000, Phase A passes only if all are true:

- centered raw-patch effective rank `>=48`;
- centered projected-target effective rank `>=48`;
- raw cross-sample variance and content-residual spatial diversity each retain
  at least `25%` of update zero;
- true/shuffled-next ratio `<=0.90`;
- true/mean-target ratio `<=0.90`;
- true/shuffled-current ratio `<=0.95`;
- true/cyclic-wrong-action ratio `<=0.95`;
- non-`hold` true/real-hold ratio `<=0.95`;
- cyclic-wrong margin is positive in at least `6/8` scene families;
- real-hold margin is positive in at least `6/8` scene families;
- all values are finite, populations exact, EMA targets gradient-free, and
  diagnostics mutation-free.

No threshold may be relaxed after execution.

## Conditional Phase B and downstream boundary

Only an exact final Phase-A pass may enter the existing unchanged Phase-B
physical-readout probe:

- copy only the Phase-A terminal online raw encoder;
- hard-sync only the target encoder;
- train only the evidence head;
- keep the physical evaluator, loss, data roles, schedule, 1,000-update /
  16,000-presentation cap, and absolute physical thresholds unchanged from
  RGB JEPA Encoder Pretraining V3;
- keep the 120-minute cumulative GPU-active cap.

An early-stop or final Phase-A failure forbids Phase B. A Phase-B failure
terminates this mechanism. A Phase-B pass authorizes only a separately
preregistered perception qualification step; it does not authorize G2,
navigation, held-out, sealed, production, promotion, or deployment.

## Fresh custody

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_patch_whitened_action_residual_jepa_probe_v1`

It must be absent before reservation. V1/V2/V3 attempt roots and every
checkpoint beneath them are historical evidence only and may not be opened by
the launcher or runner.

Use exactly one new source manifest, one combined independent source/science
review, one one-attempt execution authorization, and one terminal audit. Do
not add a new audit framework, generic runner framework, resume machinery,
second seed, schedule extension, or observer rerun.

The exact schema prefix is
`lewm_go2_rgb_patch_whitened_action_residual_jepa_v1`. An update-100
continuation failure publishes
`FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL`; an update-400
continuation failure publishes
`FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL`. Both are sealed
scientific failures with no Phase B and no retry.
