# RGB Patch-Whitened Action-Residual JEPA V4 Action-Indexed Energy-NLL preregistration

Date: 2026-07-25

## Decision

Authorize source preparation and one independent combined source/science
review for exactly one V4 Action-Indexed Energy-NLL probe.

Execution is not authorized by this document. It requires a fresh exact source
manifest, a passing independent review, and a distinct one-attempt execution
authorization.

## Bound V3 evidence

V3 Live-Reference Hinge is a valid terminal update-100 scientific failure. Its
audit is:

- path:
  `docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v3_live_reference_hinge_terminal_audit_2026-07-25.json`;
- commit: `3202cbecf2b6042ca3b4e4b8b6485b4f06cfd574`;
- file SHA-256:
  `ed0a911f009cd1f7f7fb1849178b3478ad963f135fa41809411adf61f501553c`;
- content SHA-256:
  `80d840d8f012b6343f26691e08b47290f44c04fe1eefbae65e0f77b9514acd6a`;
- byte count: `14,731`;
- status:
  `PASS_EXACT_VALID_SCIENTIFIC_FAILURE_TERMINAL_FAMILY_TERMINATED_NO_RETRY`.

V3 stopped at 100 optimizer/EMA updates and 1,600 presentations. Phase B was
not entered. The auditor opened only the six JSON receipts and did not open,
hash, parse, copy, or deserialize the checkpoint or training trace.

The live true-energy gradient produced a material mechanistic advance without
damaging the representation:

- raw and projected effective rank remained strong at
  `53.528953552246094` and `41.31370544433594`;
- true/shuffled-next and true/shuffled-current improved to
  `0.5339265376851087` and `0.5332494478903137`;
- the real-hold ratio improved to `0.906619103277254`, with a positive margin
  in all `8/8` families;
- the cyclic ratio improved to `0.9958246024128456`, with a positive margin in
  `7/8` families, but missed the required strict `<0.99` boundary;
- most importantly, true/hardest-wrong was `1.0233191337607208`: at least one
  wrong action remained closer to the observed future than the executed
  action.

V3 and its live-reference hinge plus shared small-open AdaLN action-conditioning
family may not be retried, resumed, extended, threshold-relaxed, or
scalar-retuned.

## Structural diagnosis

V3 showed that a live action-comparison gradient works, but every action still
had to express its effect through the same small shared AdaLN conditioning
path. The cyclic near-miss and the worse-than-one hardest-wrong ratio therefore
identify insufficient separable all-action capacity, not an encoder-rank
failure and not another coefficient problem.

The successor must give each real action an explicit learned residual operator
and train the executed action against all eight alternatives uniformly. It
must not optimize a cyclic, hold, family, or other diagnostic sentinel.

## Exact V4 mechanism

For online current-state tokens `s`, compute one action-independent shared
trunk:

`h = H(s, zero_condition)`.

The existing predictor transformer, position embedding, normalized output, and
small-open block initialization are preserved. Its action embedder is bypassed:
the executed action and all candidate actions are never passed into its AdaLN
blocks, and every block receives the same exact all-zero conditioning tensor.
The trunk is therefore shared and action-independent.

Keep the existing shared prediction projector:

`r_shared = P(h)`.

Add exactly nine bias-free tokenwise residual operators:

`A_a in R^(192 x 192), a in {0,...,8}`.

They use the frozen action-vocabulary order. Every scalar in every `A_a` is
initialized to exact zero without an RNG draw. There is no action-head bias,
low-rank factorization, hidden layer, or additional normalization. Candidate
residuals and predictions are:

`r_a = r_shared + A_a h`;

`z_hat_a = normalize(z_current_ema + (0.1/sqrt(192))*r_a)`.

For the executed action, `h`, `r_shared`, and `A_a` remain live. For every
non-executed action, `h` and `r_shared` are detached while that action's
`A_a` remains trainable. Thus wrong-action repulsion cannot inflate or corrupt
the shared visual predictor. The EMA-current skip and EMA-next target remain
detached.

All nine predictions must be bitwise equal before update 1. The update-zero
checkpoint-selection observation must compare all 36 unordered candidate
pairs over the complete 495-row population and fail closed if any pair differs.

For row `i` and candidate action `a`, define:

`E_i,a = mean_patch_feature_mse(z_hat_i,a, z_next_ema_i)`;

`m_i = stop_gradient(mean_a(E_i,a)).clamp_min(1e-8)`;

`logit_i,a = -E_i,a / m_i`;

`L_ID = mean_i(m_i * cross_entropy(logit_i, executed_action_i))`.

The total Phase-A objective is:

`L = mean_i(E_i,executed) + L_ID`
`    + 0.50*(V_raw + V_projected)`
`    + 0.02*(K_raw + K_projected)`.

The coefficient of `L_ID` is exactly `1.0`. There is no wrong-action hinge,
real-hold hinge, `10.0` action coefficient, temperature sweep, margin, or
sentinel-specific term. At equal energies, the detached row scaling gives an
`8/9` attractive energy gradient to the executed action and a `1/9` repulsive
energy gradient to each wrong action. Zero initialization therefore does not
trap the nine operators in symmetry.

Training may receive only the executed action index and the uniformly ordered
nine energies. It must not construct, import, or consume cyclic mappings,
hold-specific masks or weights, scene-family identities, hardest-wrong
indices, or diagnostic sentinel identities. Those remain observation-only.

## Everything else remains exact

Preserve V3 exactly except where the mechanism above explicitly replaces its
action path and loss:

- Raw V13 train and checkpoint-selection roles, counts, bytes, and hashes;
- qualified N320 online and EMA encoder initialization only;
- base seed `20260712`, schedule seed `20260713`, and the same first 16,000
  presentations;
- RGB-only current/next inputs and the exact nine-action vocabulary/order;
- ViT, shared predictor trunk, shared projector, and evidence-head dimensions;
- float32, no autocast, EMA momentum `0.996`, AdamW groups and learning rates,
  weight decay, epsilon, and global clip `1.0`;
- residual alpha `0.1/sqrt(192)`;
- isolated small-open transformer-block initialization, standard deviation,
  bias, and preservation of global RNG state;
- frozen appearance projector and optimizer/clip exclusion;
- both exact patch-whitening branches and their formulas, epsilon, and weights;
- update observations at 100, 400, and 1,000;
- Phase-A cap of 1,000 updates, 16,000 presentations, and 60 active GPU
  minutes;
- cumulative cap of 2,000 updates, 32,000 presentations, and 120 active GPU
  minutes;
- every existing V3 rank, content, health, cyclic, hold, population,
  finite-value, EMA, RNG, mutation, final Phase-A, and conditional Phase-B
  gate;
- conditional Phase B, entered only after an exact final Phase-A pass;
- all denials for G2, navigation, held-out, sealed, production, promotion, and
  deployment.

Promote the existing rowwise-minimum hardest-wrong metric from informational to
a required all-action gate, without changing its definition:

`hardest_wrong_i = min_{a != executed_i}(E_i,a)`;

`hardest_wrong_mse = mean_i(hardest_wrong_i)`.

Require:

- at update 100, `true_pair_mse / hardest_wrong_mse < 0.99`;
- at update 400, `true_pair_mse / hardest_wrong_mse <= 0.975`;
- at final Phase A, `true_pair_mse / hardest_wrong_mse <= 0.95`.

The unchanged update-100 gates additionally require:

- raw effective rank strictly greater than `27.717458724975586`;
- projected effective rank strictly greater than `17.426651000976562`;
- true/cyclic-wrong ratio strictly less than `0.99`;
- non-hold-true/real-hold ratio strictly less than `0.99`;
- positive cyclic and hold margins in at least `6/8` families;
- exact populations, finiteness, EMA-gradient freedom, RNG preservation, and
  zero model-state mutation during observation.

Failure of any update-100 conjunct must publish
`FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL` and stop without Phase B,
retry, resume, or observer rerun. Passing update 100 continues the same attempt
through the fixed update-400 and final gates; it does not authorize a second
attempt.

## Fresh custody and terminal rule

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_patch_whitened_action_residual_jepa_probe_v4_action_indexed_energy_nll`

It must be absent before reservation. V1, V2, V3, and every earlier runtime
root, checkpoint, and trace are historical evidence only and may not be runtime
inputs.

The exact schema prefix is:

`lewm_go2_rgb_patch_whitened_action_residual_jepa_v4_action_indexed_energy_nll`.

Use one fresh source manifest, one independent combined source/science review,
one distinct one-attempt authorization, and one terminal audit. There is no
valid-science retry, resume, second seed, coefficient edit, alpha edit,
temperature edit, head-size or bias variant, gate relaxation, schedule
extension, observer rerun, or automatic V5.

If V4 fails any update-100 or update-400 continuation gate, the final Phase-A
gate, or conditional Phase B, terminate this action-indexed operator-bank
mechanism. Any later candidate must be materially different and separately
preregistered.
