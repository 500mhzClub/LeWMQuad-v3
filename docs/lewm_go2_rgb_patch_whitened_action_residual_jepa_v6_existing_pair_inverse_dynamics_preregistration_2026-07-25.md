# RGB Patch-Whitened Action-Residual JEPA V6: Existing-Pair Inverse Dynamics

Date: 2026-07-25

## Scientific question

Can a training-only inverse-dynamics path force the learned RGB representation
and the shared action embeddings to preserve the small action-specific change
that V5's forward latent-flow objective still confuses between nearest actions?

This is one bounded perception-only JEPA falsification. It is not a retry or
resume of V5.

V5 is a valid terminal scientific failure bound by:

- terminal-audit commit
  `c7bd138cf9a7a8195c968199fcbda5025564fe55`;
- terminal-audit file SHA-256
  `aed6af35922d55cfa292c243fee8cf0cd27b43b01d96caca8bf12562f042406d`;
- terminal-audit content SHA-256
  `f55b6854ca873e49598909ce2f238f098945a11d320bbe9a734f59479add9164`.

At update 100, V5 activated nonzero bounded flow for all eight movement
actions, kept hold exactly zero, and improved:

- true/cyclic-wrong from `0.9960063427985371` to
  `0.9844888263945084`;
- non-hold-true/hold from `0.9856515904619205` to
  `0.9047588710938321`;
- true-pair MSE from `0.0009916148846969008` to
  `0.0009312229230999947`.

It still failed:

- true/hardest-wrong at `1.020108815174314`, with the hardest wrong action
  better in all eight families;
- true/mean-target at `1.0616841452956087`.

The hardest-wrong ratio moved only
`1.0233191337607208 -> 1.0224249969916874 -> 1.020108815174314` across
V3, V4, and V5. This rejects another flow-width, flow-bank, displacement-bound,
temperature, scalar-loss, or threshold variant.

A causal two-frame successor was considered and rejected before runtime.
The frozen index has an in-index predecessor for at most `747/4262` train
rows and exactly `66/495` checkpoint-selection rows. Completing the proposed
all-row temporal population would require new metadata and RGB rendering.
That source-only conclusion is bound by commit
`6598b49e40be12da008de47590c129a611e8ae43`, file SHA-256
`2e34e4f4ef22abdc8d5ecd89c1c12acff8541e41b8440a307ba8cbeb4a432c3d`,
content SHA-256
`2d947ea6857ddcbed82d778692c18831b2c920da6b3aa7c5723833ecb495a929`,
and `3261` bytes at
`docs/lewm_go2_rgb_causal_two_frame_latent_motion_state_source_feasibility_audit_2026-07-25.json`.
V6 therefore uses the existing pairs without rebuilding, backfilling,
resampling, rebalancing, or rendering data.

## One scientific delta

Preserve the complete V5 forward state-dependent latent-flow JEPA. Add one
training-only inverse-dynamics projection that sees the learned online current
and next RGB states and scores the already-existing nine action embeddings.

Let `c_i` and `n_i` be the online-encoder geometry tokens for the current and
next RGB images for token `i`, each in `R^192`. Define:

`d_i = F.layer_norm(n_i - c_i, normalized_shape=(192,), weight=None,`
` bias=None, eps=1e-5)`;

`x_i = concat(c_i, n_i, d_i) in R^576`;

`q = mean_i(W_inv x_i) in R^192`;

`p_a = E(a) - mean_b(E(b))`;

`ell_a = dot(q, p_a) / sqrt(192)`.

`E` is the existing trainable predictor action embedder already used by V5
flow. `W_inv` is exactly the parameter
`prediction_projector.inverse_weight`, has exact shape `[192,576]`, no bias,
and exactly `110592` trainable scalars. It is initialized to exact zero
without an RNG draw.
There is no inverse MLP, attention block, token selector, class bias,
temperature, prototype bank, or new action embedding.

At initialization all inverse logits are exactly zero. A deterministic
source-only synthetic fixture must prove a finite nonzero gradient for
`W_inv` at zero using the fixed distinct action embeddings. Because
`W_inv = 0` blocks both backward paths, inverse-loss gradients to the online
current state, online next state, shared encoder, and action embeddings must
all be exactly zero on the first step. The same fixture must prove that those
paths have finite nonzero gradients after `W_inv` is made bitwise nonzero.

The inverse head is training-only. It adds no inference input and no pose,
depth, odometry, optical flow, occupancy, traversability, physical label, or
navigation supervision.

## Preserved V5 forward mechanism

For every candidate action `a`, retain:

`e_rel_a = E(a) - E(hold)`;

`u_i,a = h_i * e_rel_a`;

`delta_cell_i,a = tanh(W_flow u_i,a)`;

`delta_grid_i,a = (2/15) * delta_cell_i,a`;

`z_hat_i,a = normalize(grid_sample(z_current_ema, identity + delta_grid_a,
bilinear, border, align_corners=True)_i`
` + (0.1/sqrt(192)) * r_shared_i)`.

The action-independent predictor trunk, shared residual, exact-zero
bias-free `[2,192]` flow projection, one-cell closed displacement bound,
hold-relative exact-zero flow, V5 candidate ordering, and V5 detach topology
remain unchanged.

## Exact loss

Retain the V5 forward energies:

`E_i,a = mean_patch_feature_mse(z_hat_i,a, z_next_ema_i)`;

`m_i = stop_gradient(mean_a(E_i,a)).clamp_min(1e-8)`;

`L_ID = mean_i(m_i * cross_entropy(-E_i,all/m_i, executed_action_i))`.

Define the inverse term without a new scalar scale:

`L_INV = mean_i(m_i * cross_entropy(ell_i, executed_action_i))`.

The exact Phase-A objective is:

`L = mean_i(E_i,executed) + L_ID + L_INV`
`    + 0.50*(V_raw + V_projected)`
`    + 0.02*(K_raw + K_projected)`.

Both identification coefficients are exactly `1.0`. Reusing detached `m_i`
keeps the inverse term on the registered JEPA-energy scale and avoids an
arbitrary cross-entropy weight.

There is no wrong-action hinge, hold hinge, inverse margin, fixed or learned
temperature, target-derived saliency weight, flow-magnitude loss, or
sentinel-specific training term.

## Gradient and detach topology

The V5 forward topology is unchanged:

- executed forward candidate: online state, shared trunk/residual, flow
  projection, and action embedder are live;
- wrong forward candidates: shared state/trunk/residual are detached while
  `W_flow` and the action embedder remain live;
- EMA-current warp values and EMA-next targets are detached;
- target encoder, target geometry, and target projector remain gradient-free.

For `L_INV` only:

- the shared online encoder and online geometry process both current and next
  RGB and remain live;
- `W_inv` and the centered existing action embeddings remain live;
- no EMA tensor enters the inverse path;
- observation and derangement controls run under `no_grad` while preserving
  RNG and model state.

`prediction_projector.inverse_weight` belongs to the existing Phase-A
auxiliary AdamW optimizer group at learning rate `3e-4`, with the preserved
`1e-4` weight decay and global gradient clip `1.0`. It is the only new
parameter tensor and contributes exactly `110592` new trainable scalars.
It is frozen and excluded from every Phase-B optimizer and is not copied into
the Phase-B model. Phase B continues to consume only the qualified frozen
encoder state under the existing contract.

## Frozen inputs and schedule

Preserve V5 exactly:

- raw V13 train and checkpoint-selection roles: `4262` and `495` pairs,
  `72` and `8` scenes, and the exact existing endpoint bytes and hashes;
- the exact current RGB, next RGB, and executed action for every scheduled
  pair;
- the exact nine-action vocabulary and order;
- no previous RGB/action, new frame, data rebuild, refinement, backfill,
  rebalancing, filtering, resampling, or render;
- qualified N320 online and EMA encoder initialization only;
- base seed `20260712` and schedule seed `20260713`;
- the exact frozen first `16000` presentations;
- ViT, projector, predictor, optimizer groups and learning rates, AdamW
  settings, float32, EMA `0.996`, global clip `1.0`, and no autocast;
- V5 patch whitening, weights, residual alpha, flow mechanism, Energy-NLL,
  observation populations, and conditional Phase B.

The extra online-next encoder call consumes no new row or image. One scheduled
pair remains one presentation.

## Observation-only inverse controls

At each checkpoint, evaluate inverse logits on all `495` frozen
checkpoint-selection pairs.

Construct the existing deterministic within-scene next-endpoint derangement
exactly as follows. Group selection rows by `scene_id`. Within each scene,
sort row indices by the pair `content_sha256`. For each row, walk cyclically
forward from the next sorted position and select the first row whose
`next_endpoint_sha256` differs from the source row. Fail closed if no such row
exists or if any selected endpoint identity is unchanged.
For the deranged inverse control, keep current RGB and executed action fixed
and replace only the online next state. It is observation-only and must not
enter any gradient.

Record:

- correct-pair inverse cross-entropy: the unscaled arithmetic mean of standard
  per-row cross-entropy over the `495` correct logits and labels;
- scene-deranged-next inverse cross-entropy: the same unscaled arithmetic
  mean using the exact deranged online-next states;
- their ratio, defined as correct unscaled mean divided by deranged unscaled
  mean; the denominator must be finite and strictly positive;
- nine-class top-1 accuracy using `torch.argmax(dim=1)`, whose documented
  first-maximum rule selects the lowest action index on a tie;
- nine-class macro balanced accuracy, defined as the arithmetic mean of the
  nine per-action recalls, with every action required to be present in the
  `495`-row population;
- per-family deranged-minus-correct cross-entropy;
- maximum absolute inverse logit;
- finiteness and nonzero state of `W_inv`.

The reported inverse cross-entropies are deliberately unscaled diagnostics.
Only the training loss uses the detached per-row JEPA energy scale `m_i`.

At update zero:

- all inverse logits must be bitwise zero over all `495 x 9` values;
- correct and deranged inverse cross-entropies must be bitwise equal;
- every existing V5 update-zero symmetry, flow, rank, RNG, mutation, and
  gradient-health check remains exact.

At update 100, additionally require:

- `W_inv` has at least one bitwise-nonzero scalar;
- all inverse values are finite;
- correct inverse cross-entropy is strictly below `log(9)`;
- correct/scene-deranged inverse cross-entropy is strictly below `0.99`;
- macro balanced accuracy is strictly above `2/9`;
- deranged-minus-correct inverse cross-entropy is positive in at least `6/8`
  scene families.

These inverse gates must remain true at update `400` and final update `1000`.

Every V5 update-100 gate also remains mandatory:

- raw and projected effective rank strictly exceed the registered V3
  update-zero values;
- true/cyclic-wrong, true/hardest-wrong, and non-hold-true/hold are each
  strictly below `0.99`;
- true/mean-target is strictly below `1.0`;
- cyclic and hold margins are positive in at least `6/8` families;
- all eight non-hold flows are nonzero, hold flow is exactly zero, and every
  component is finite and within `[-1,1]`;
- exact populations, finiteness, EMA-gradient freedom, RNG preservation, and
  zero diagnostic model-state mutation.

Failure of any conjunct at update 100 publishes
`FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL` and stops.

At update `400`, all preserved V5 update-400 continuation conjuncts and every
inverse gate above must pass or the attempt publishes
`FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL` and stops.

At final update `1000`, all preserved V5 final Phase-A conjuncts and every
inverse gate above must pass. `QUALIFIED_FOR_CONDITIONAL_PHASE_B` requires
both conjunctions. Failure publishes the preserved terminal Phase-A failure
control and does not enter Phase B. The inverse metrics are not Phase-B
training targets and cannot relax any existing Phase-B threshold.

## Fresh custody

The exact schema prefix is:

`lewm_go2_rgb_patch_whitened_action_residual_jepa_v6_existing_pair_inverse_dynamics`.

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_patch_whitened_action_residual_jepa_probe_v6_existing_pair_inverse_dynamics`.

That root must be absent before reservation. V1 through V5 runtime roots,
checkpoints, and traces are historical evidence only and cannot be runtime
inputs.

During Phase A, runtime may open only the already-bound raw-V13 pair index,
endpoint index, train and checkpoint-selection current/next RGB leaves,
executed actions, the qualified N320 initialization checkpoint, its gate, and
the frozen schedule. During Phase A, the general raw-V13 frame loader and
camera-supervision arrays remain denied with zero calls/opens.

Conditional Phase B is the preserved V5 physical probe and is the sole
exception: only after an exact final Phase-A pass, it may use the existing
reviewed matched-training frame path and the exact already-bound train and
checkpoint-selection camera-supervision leaves required by that frozen
physical evaluator. Probability-calibration, prior runtime outputs, rejected
checkpoints, and every non-authorized role remain denied throughout.

## Cap and terminal rule

This candidate receives at most one fresh attempt:

- first decision: `100` updates / `1600` presentations;
- maximum Phase A: `1000` updates / `16000` presentations /
  `60` GPU-active minutes;
- conditional Phase B only after an exact final Phase-A pass;
- cumulative cap including conditional Phase B remains `2000` updates /
  `32000` presentations / `120` GPU-active minutes.

There is no retry, resume, second seed, schedule extension, observer rerun,
inverse width/depth/bias/prototype variant, loss-weight or temperature edit,
flow variant, threshold relaxation, rejected-checkpoint access, or automatic
successor.

G2, navigation, held-out, sealed, production, promotion, and deployment stay
denied unless the complete perception gate later qualifies and a separate
authority explicitly opens the next stage.

## Interpretation boundary

This is a direct inverse-action auxiliary attached to the JEPA encoder. A pass
would show that the learned current/next RGB representation contains
action-discriminative information while preserving the forward JEPA gates.
Because the inverse projection receives both absolute states as well as their
normalized difference, it would not by itself prove that only motion-specific
features were used. The later held-out perception and navigation gates remain
necessary for the repository's learned-navigation claim.
