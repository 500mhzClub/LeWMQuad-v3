# RGB Patch-Whitened Action-Residual JEPA V5 State-Dependent Latent-Flow preregistration

Date: 2026-07-25

## Decision

Authorize source preparation and one independent combined source/science
review for exactly one V5 State-Dependent Latent-Flow probe.

Execution is not authorized by this document. It requires a fresh exact source
manifest, a passing independent review, and a distinct one-attempt execution
authorization.

## Bound V4 evidence

V4 Action-Indexed Energy-NLL is a valid terminal update-100 scientific
failure. Its audit is:

- path:
  `docs/lewm_go2_rgb_patch_whitened_action_residual_jepa_v4_action_indexed_energy_nll_terminal_audit_2026-07-25.json`;
- commit: `20a5099f17a6da17bb2858d96724f9f8e88ae3f9`;
- file SHA-256:
  `ddb3c784382f92161b82d7321c8ad3c70901cb8d5a813c3ecc7153083480d809`;
- content SHA-256:
  `c3edbe1932c5647e576b25216cee38ad904f5b5fa581d39f70c1d8cef3e92f01`;
- byte count: `15,366`;
- status:
  `PASS_EXACT_VALID_SCIENTIFIC_FAILURE_TERMINAL_FAMILY_TERMINATED_NO_RETRY`.

V4 stopped at 100 optimizer/EMA updates and 1,600 presentations. Phase B was
not entered. The independent auditor opened only the six JSON receipts and
did not open, hash, parse, copy, or deserialize the checkpoint or training
trace. All 21 receipt-link, access, operation-count, inventory, and sealing
assertions passed.

The representation remained healthy, but the action mechanism did not learn
the intended ordering:

- raw and projected effective rank were `53.53171157836914` and
  `41.31428527832031`;
- true/shuffled-next and true/shuffled-current remained strong at
  `0.5441887981365259` and `0.5437932296556834`;
- true/cyclic-wrong was `0.9960063427985371`, failing strict `<0.99` despite
  a positive sign in all `8/8` scene families;
- true/hardest-wrong was `1.0224249969916874`: the nearest wrong action was
  better than the executed action in every `0/8` positive-margin family;
- non-hold-true/hold regressed from V3's `0.906619103277254` to
  `0.9856515904619205`;
- true/mean-target regressed above one to `1.0845515041536307`.

V3 and V4 produced essentially identical hardest-wrong ratios
(`1.0233191337607208` and `1.0224249969916874`) despite materially different
losses. V4's exact linear action-indexed operator plus detached-scale
Energy-NLL family may not be retried, resumed, extended, threshold-relaxed,
head-resized, deepened, biased, or scalar-retuned.

## Structural diagnosis

V4 gave each action a large learned `192 x 192` operator, but each operator
only remixed feature channels independently at the same patch location.
Changing a camera viewpoint requires information to move between locations on
the ordered `16 x 16` latent grid. V4 could express a weak action bias, but it
had no explicit spatial-transport operation.

The next test therefore changes the transition architecture, not the encoder,
data, loss coefficient, schedule, or gate. It asks whether a shared,
state-dependent, action-conditioned spatial flow can move the persistent
current-view latent content toward the observed future.

## Exact V5 mechanism

For online current-state tokens `s`, preserve the V4 action-independent shared
trunk and shared prediction projector:

`h = H(s, zero_condition)`;

`r_shared = P(h)`.

The existing predictor transformer, position embedding, normalized output,
and small-open block initialization remain unchanged. Every transformer block
receives the same exact all-zero conditioning tensor. Candidate actions do not
enter AdaLN.

Remove V4's nine `192 x 192` action-indexed residual operators. Reuse the
existing initialized predictor action embedder outside AdaLN. For the exact
frozen nine-action vocabulary, compute:

`e_a = action_embed(one_hot(a)[:, None, :])[:, 0, :]`;

`e_rel_a = e_a - e_hold`.

This use of hold is an architectural zero-motion reference only. It does not
create a hold loss, hold weight, hold training mask, or diagnostic sentinel
objective.

Add one shared bias-free flow projection:

`W_flow in R^(2 x 192)`.

Every scalar in `W_flow` is initialized to exact zero without an RNG draw.
There is no flow bias, per-action flow bank, hidden layer, correction network,
occlusion head, smoothness loss, or flow supervision.

For patch `i` and candidate action `a`, define the flow in patch-cell units:

`u_i,a = h_i * e_rel_a`;

`delta_cell_i,a = tanh(W_flow u_i,a)`.

Each horizontal and vertical component is therefore fixed to the closed range
`[-1, 1]` patch cells. This is a local-neighborhood architectural bound, not a
tuned loss gain. It prevents the repulsive all-action objective from obtaining
artificial discrimination by pushing wrong candidates arbitrarily outside the
grid.

Tokens use the existing row-major `16 x 16` layout. Construct the identity
sampling grid in `(x, y)` order over `[-1, 1]`, and convert patch-cell
displacement to normalized coordinates exactly as:

`delta_grid_i,a = (2 / 15) * delta_cell_i,a`.

Sample the detached EMA-current projected token grid with
`torch.nn.functional.grid_sample` using:

- `mode="bilinear"`;
- `padding_mode="border"`;
- `align_corners=True`;
- sampling coordinates `identity_grid + delta_grid_i,a`.

`W_flow` output component `0` is the grid `x`/column displacement and output
component `1` is the grid `y`/row displacement.

The candidate prediction is:

`z_hat_a = normalize(warp(z_current_ema, delta_grid_a)`
`                  + (0.1/sqrt(192))*r_shared_a)`.

The shared residual is added after the spatial warp. This preserves its V4
role as action-independent innovation at the output grid while the new flow
handles persistent spatial content.

For the executed candidate, `h` and `r_shared` remain live. For every wrong
candidate, `h` and `r_shared` are detached. `W_flow` and the action embedder
remain live for all nine candidates. EMA-current sampled values and the
EMA-next target remain detached. Wrong-action repulsion therefore trains the
shared action-transition rule without sculpting the visual encoder or shared
residual into a control discriminator.

Because `W_flow` is exactly zero, all nine predictions must be bitwise equal
before update 1. Because `e_rel_hold` is an exact self-subtraction, hold flow
must be exactly zero at initialization and throughout training. Before
authorization, a deterministic source-only synthetic fixture must prove a
finite nonzero `W_flow` gradient at `W_flow=0` through bilinear spatial
derivatives and the distinct deterministically initialized action embeddings.
This fixture is a source-mechanism check, not a runtime continuation gate. The
action embedder is permitted to receive zero gradient on the first step while
`W_flow` is zero; it becomes live once `W_flow` moves.

## Exact objective

Keep the V4 all-action objective unchanged. For row `i` and candidate `a`:

`E_i,a = mean_patch_feature_mse(z_hat_i,a, z_next_ema_i)`;

`m_i = stop_gradient(mean_a(E_i,a)).clamp_min(1e-8)`;

`logit_i,a = -E_i,a / m_i`;

`L_ID = mean_i(m_i * cross_entropy(logit_i, executed_action_i))`.

The total Phase-A objective remains:

`L = mean_i(E_i,executed) + L_ID`
`    + 0.50*(V_raw + V_projected)`
`    + 0.02*(K_raw + K_projected)`.

The coefficient of `L_ID` remains exactly `1.0`. There is no wrong-action
hinge, real-hold hinge, fixed temperature, temperature sweep, margin,
flow-magnitude loss, sentinel-specific term, realized-motion input, optical
flow, depth, pose, odometry, occupancy, traversability, or navigation
supervision.

Training may receive only current RGB, next RGB, the executed action index,
and the uniformly ordered nine candidate energies. Cyclic mappings, scene
families, hardest-wrong indices, and acceptance sentinels remain
observation-only.

## Everything else remains exact

Preserve V4 exactly except where the mechanism above replaces its action
operator:

- Raw V13 train and checkpoint-selection roles, counts, bytes, and hashes;
- no data rebuild, refinement, rebalancing, filtering, or resampling;
- qualified N320 online and EMA encoder initialization only;
- base seed `20260712`, schedule seed `20260713`, and the same first 16,000
  presentations;
- RGB-only current/next inputs and the exact nine-action vocabulary/order;
- ViT encoder, shared predictor trunk, shared projector, and evidence-head
  dimensions;
- float32, no autocast, EMA momentum `0.996`, AdamW groups and learning rates,
  weight decay, epsilon, and global clip `1.0`;
- residual alpha `0.1/sqrt(192)`;
- isolated small-open transformer-block initialization, standard deviation,
  bias, and preservation of global RNG state;
- frozen appearance projector and optimizer/clip exclusion;
- both exact patch-whitening branches and their formulas, epsilon, and weights;
- the V4 detached-scale Energy-NLL formula and coefficient;
- update observations at 100, 400, and 1,000;
- Phase-A cap of 1,000 updates, 16,000 presentations, and 60 active GPU
  minutes;
- cumulative cap of 2,000 updates, 32,000 presentations, and 120 active GPU
  minutes including conditional Phase B;
- every existing V4 rank, content, health, cyclic, hold, hardest-wrong,
  population, finite-value, EMA, RNG, mutation, final Phase-A, and conditional
  Phase-B gate;
- conditional Phase B, entered only after an exact final Phase-A pass;
- all denials for G2, navigation, held-out, sealed, production, promotion, and
  deployment.

The update-zero checkpoint-selection observation must compare all 36
unordered action pairs over all 495 rows and fail closed if any candidate
differs. At every observation, additionally require:

- all flow values finite;
- every `delta_cell` component within the closed interval `[-1, 1]`;
- hold flow exactly zero over every observed hold candidate.

At update 100, additionally require:

- each of the eight non-hold actions has at least one bitwise-nonzero flow
  component over the complete 495-row selection population;
- `true_pair_mse / mean_target_mse < 1.0`.

The unchanged update-100 continuation gates require:

- raw effective rank strictly greater than `27.717458724975586`;
- projected effective rank strictly greater than `17.426651000976562`;
- true/cyclic-wrong ratio strictly less than `0.99`;
- true/hardest-wrong ratio strictly less than `0.99`;
- non-hold-true/real-hold ratio strictly less than `0.99`;
- positive cyclic and hold margins in at least `6/8` families;
- exact populations, finiteness, EMA-gradient freedom, RNG preservation, and
  zero model-state mutation during observation.

Passing update 100 continues the same attempt through the unchanged
update-400 and final gates. It does not authorize a second attempt. Failure of
any update-100 conjunct must publish
`FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL` and stop without Phase B,
retry, resume, or observer rerun.

## Fresh custody and terminal rule

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_patch_whitened_action_residual_jepa_probe_v5_state_dependent_latent_flow`

It must be absent before reservation. V1 through V4 and every earlier runtime
root, checkpoint, and trace are historical evidence only and may not be
runtime inputs.

The exact schema prefix is:

`lewm_go2_rgb_patch_whitened_action_residual_jepa_v5_state_dependent_latent_flow`.

Use one fresh source manifest, one independent combined source/science review,
one distinct one-attempt authorization, and one terminal audit. There is no
valid-science retry, resume, second seed, action-embedding reset, displacement
bound edit, flow-bank variant, flow-width or bias variant, loss edit, alpha
edit, temperature edit, gate relaxation, schedule extension, observer rerun,
or automatic successor.

If V5 fails any update-100 or update-400 continuation gate, the final Phase-A
gate, or conditional Phase B, terminate this exact local shared latent-flow
mechanism. A later candidate must be materially different and separately
preregistered; the evidence-led next question would be whether the
single-frame requested-action target is identifiable, not whether this flow
head needs another scalar or width.
