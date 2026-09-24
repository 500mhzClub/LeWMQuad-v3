# RGB Action-Conditioned Local-Correspondence Transport JEPA V7

Date: 2026-07-25

## Scientific question

Can an RGB-only JEPA learn the small local patch motion that distinguishes
nearby maze actions when the correspondence is represented before spatial
pooling, rather than asking a global pooled inverse head or a continuous
two-coordinate flow head to recover it afterward?

This is one bounded perception-only falsification. It is not a V5 or V6 retry.
V6 is a valid terminal scientific failure bound by:

- terminal-audit commit
  `c3259adc3b48a3f1d5784a1ada0eaac8b12f7855`;
- terminal-audit file SHA-256
  `9ab15ff2100e46fd341d4266c534c289bd453f74517743886ceccea165e15d01`;
- terminal-audit content SHA-256
  `aed1a84c890ae5a7b5ad068cffe7cc1b3260bc4b39919366fcb9bee888666c6d`.

At update 100, V6's inverse projection was finite and nonzero, but it remained
at chance:

- correct inverse cross-entropy was `2.2920234203338623`, above
  `log(9)`;
- correct/deranged inverse cross-entropy was
  `0.9994120001792908`;
- macro balanced accuracy was exactly `1/9`;
- true/hardest-wrong remained `1.0191631948996198`;
- true/mean-target remained `1.075003676039105`.

The V5-to-V6 true-pair MSE improvement was only `0.3785592558403494%`.
The global inverse head therefore did not preserve usable local
action-conditioned motion. V6's inverse width, depth, class balance,
temperature, loss scale, threshold, seed, checkpoint, and runtime family are
closed.

V7 tests the evidence-led alternative named in the V6 terminal audit:
local token correspondence before pooling. It reuses the exact existing
current/next pairs and does not rebuild, backfill, rebalance, refine, resample,
or render data.

## One materially different mechanism

Remove both rejected transition tensors:

- V5/V6 `prediction_projector.flow_weight`;
- V6 `prediction_projector.inverse_weight`.

Remove V6's online-next inverse branch and `L_INV`.

Preserve the N320-initialized encoder, EMA encoder, online target projector,
action-independent predictor trunk, shared residual projector, existing
trainable action embedder, Energy-NLL, patch whitening, optimizer settings,
data, seed, schedule, observations, and conditional Phase B.

Replace the continuous `grid_sample` flow with a discrete, learned
action-conditioned distribution over a fixed local `3 x 3` neighborhood. The
distribution directly transports detached EMA-current tokens toward the
EMA-next target.

The only new trainable tensor is:

`prediction_projector.transport_weight`

with exact shape `[8,192]`, no bias, exactly `1536` trainable scalars, and
exact-zero initialization without an RNG draw. There is no transport MLP,
attention block, offset bank, temperature, class bias, occlusion head, new
action embedding, or learned target.

## Exact neighborhood

The geometry-token grid is the preserved row-major `16 x 16` grid with
`D=192`. The full offset order is:

`[(-1,-1), (-1,0), (-1,1), (0,-1), (0,0),`
` (0,1), (1,-1), (1,0), (1,1)]`.

For destination token `(y,x)` and offset `(dy,dx)`, define the source index:

`J(y,x,dy,dx) = 16*clamp(y+dy,0,15) + clamp(x+dx,0,15)`.

The exact `[256,9]` integer neighbor table is a nonpersistent buffer. Border
clamping preserves the V5 `padding_mode="border"` boundary convention.
No runtime geometry, pose, depth, optical flow, calibration, or navigation
input is used to construct it.

The eight learned-logit output positions, in order, are the full offset order
with `(0,0)` omitted:

`[(-1,-1), (-1,0), (-1,1), (0,-1),`
` (0,1), (1,-1), (1,0), (1,1)]`.

## Detached EMA correspondence target

Let `zc_i` and `zn_i` be detached EMA-current and EMA-next geometry tokens.
Define parameter-free token normalization:

`LN(v) = F.layer_norm(v, (192,), weight=None, bias=None, eps=1e-5)`.

For destination token `i` and local offset `o`, define:

`s_target_i,o = dot(LN(zn_i), LN(zc_J(i,o))) / sqrt(192)`;

`Q_i = softmax_o(s_target_i,o)`.

`Q` is detached. The fixed `1/sqrt(192)` dot-product scale is part of this
registration and is not a tunable or swept temperature. Duplicate clamped
border neighbors remain duplicate entries in the nine-way distribution.

The target must be finite, strictly positive, and normalized. Its mean
`KL(Q || U9)` over the exact checkpoint-selection population, where
`U9_o=1/9`, must be finite and strictly positive at update zero. This is a
pretraining viability check, not a tunable acceptance threshold.

## Action-conditioned student and residual transport

Let `h_i` be the preserved action-independent online-current predictor trunk
token, and let:

`e_rel_a = E(a) - E(hold)`

use the preserved existing action embedder.

For each candidate action:

`u_i,a = h_i * e_rel_a`;

`g_i,a,noncenter = F.linear(u_i,a, transport_weight, bias=None)`;

`g_i,a,center = -sum_noncenter(g_i,a,noncenter)`.

Insert the center logit at full offset index `4`. Thus every nine-logit row
sums to zero and contains no redundant learned common-logit direction.

Define:

`P_i,a = softmax_o(g_i,a,o)`;

`U_i,a = softmax_o(zeros_like(g_i,a,o))`;

`C_i,a,o = P_i,a,o - U_i,a,o`;

`v_i,a = zc_i + sum_o C_i,a,o * (zc_J(i,o) - zc_i)`.

Finally retain the shared V5 output-grid residual:

`z_hat_i,a = normalize(v_i,a + (0.1/sqrt(192))*r_shared_i)`.

Every `U` entry represents exact float32 `1/9`; constructing it through the
same softmax operation makes `C` bitwise zero at exact-zero logits without
assuming a device-specific reciprocal implementation.

This is a residual local-correspondence transport, not a claim that `v` is a
pure convex warp. The centered-softmax residual is deliberate: exact-zero
`transport_weight` gives `P=U9`, `C=0`, and exact identity transport while
retaining a nonzero first derivative for the learned transport tensor.

Because `e_rel_hold` is exact zero, hold remains exactly uniform and exact
identity transport throughout Phase A. At initialization all nine actions
are uniform and identity, so the preserved update-zero all-action prediction
symmetry remains exact.

The implementation must use indexed detached neighbor reads plus
`F.linear`, `softmax`, batched matrix multiplication, elementwise operations,
and normalization. It must not materialize a
`[B,9,256,9,192]` tensor and must not call `grid_sample`, `unfold`, or
differentiable padding.

## Exact loss

For every candidate action retain:

`E_i,a = mean_patch_feature_mse(z_hat_i,a, zn_i)`;

`m_i = stop_gradient(mean_a(E_i,a)).clamp_min(1e-8)`;

`L_ID = mean_i(m_i * cross_entropy(-E_i,all/m_i, executed_action_i))`.

For a target row `Q` and student log-probability row `logP`, define the exact
centered-log soft cross-entropy, using full offset index `4`:

`Hc(Q,logP) = -logP_4 - sum_o Q_o*(logP_o-logP_4)`.

This is the registered training and diagnostic evaluation form. It is
algebraically the standard `-sum_o Q_o*logP_o` when a probability row sums to
one, while making every uniform-student value bitwise independent of
float32 target-row reduction error.

For the executed action define:

`CE_corr_i = (1/256) * sum_token`
`             Hc(Q_i,token, log_softmax(g_i,executed,token))`;

`L_CORR = mean_i(m_i * CE_corr_i)`.

The exact Phase-A objective is:

`L = mean_i(E_i,executed) + L_ID + L_CORR`
`    + 0.50*(V_raw + V_projected)`
`    + 0.02*(K_raw + K_projected)`.

Both identification coefficients are exactly `1.0`. Reusing detached `m_i`
places correspondence cross-entropy on the registered JEPA-energy scale
without a new scalar loss weight.

There is no wrong-action hinge, hold hinge, offset-magnitude loss,
correspondence entropy loss, fixed or learned temperature, target saliency
weight, margin, or diagnostic-specific training term.

## Gradient and detach topology

- `zc`, `zn`, `Q`, and all target-encoder/projector states are detached and
  gradient-free.
- For the executed action, `h`, `r_shared`, `transport_weight`, the action
  embedder, online target projector, and online encoder remain live.
- For wrong Energy-NLL candidates, `h` and `r_shared` are detached while
  `transport_weight` and the action embedder remain live.
- Only the executed `P` enters `L_CORR`.
- The detached row scale `m` receives no gradient.
- Derangement and checkpoint diagnostics run under `no_grad` while preserving
  RNG and model state.

At exact-zero `transport_weight`, the source-only synthetic fixture must show:

- all student logits are exact zero;
- all `P` rows are bitwise equal to `U`;
- all centered coefficients are exact zero;
- all transports are bitwise equal to the EMA-current center token;
- hold is exact identity;
- the transport-weight gradient is finite and nonzero;
- the transport-path gradients to `h`, the online encoder, and action
  embeddings are exact zero.

After a deterministic bitwise-nonzero transport weight is installed, the same
fixture must prove finite nonzero gradients to the transport tensor, `h`,
online state, and action embeddings, with no EMA gradient.

`prediction_projector.transport_weight` belongs to the preserved Phase-A
auxiliary AdamW optimizer group at learning rate `3e-4`, weight decay `1e-4`,
and global gradient clip `1.0`. It is excluded from every Phase-B optimizer
and is not copied into Phase B.

## Frozen inputs and schedule

Preserve exactly:

- raw V13 train and checkpoint-selection roles: `4262` and `495` pairs,
  `72` and `8` scenes, and the exact existing endpoint bytes and hashes;
- exact current RGB, next RGB, and executed action per scheduled pair;
- exact nine-action vocabulary and ordering;
- qualified N320 online and EMA encoder initialization only;
- base seed `20260712` and schedule seed `20260713`;
- exact first `16000` frozen pair presentations and prefix hashes:
  - update `100`: `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`;
  - update `400`: `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`;
  - update `1000`: `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`;
- ViT/projector/predictor configuration, optimizer groups and learning rates,
  AdamW settings, float32, EMA `0.996`, global clip `1.0`, and no autocast;
- V5 patch whitening, shared residual alpha, observation populations, and
  conditional Phase B.

Phase A runs with strict deterministic algorithms and `warn_only=False`
throughout. It permits and expects exactly zero determinism warnings. The
preserved conditional Phase-B determinism procedure is unchanged.

The schedule adapter may normalize schema only. It must not mutate, reorder,
filter, regenerate, reseed, replace, or extend schedule indices. Multiple
tokens, neighbors, candidate actions, or encoder calls do not increase the
count: one scheduled current/next pair is exactly one presentation.

No prior RGB, third frame, new frame, pose, depth, odometry, optical-flow
label, occupancy, traversability, physical label, navigation label,
scene-family feature, held-out input, sealed input, refinement, backfill,
rebalancing, filtering, resampling, or render is authorized.

## Observation-only correspondence controls

Evaluate all forward V5 observations on all `495` exact frozen
checkpoint-selection pairs.

For correspondence observations, use the same `495` pairs. Construct the
deterministic within-scene next-endpoint derangement:

1. group rows by `scene_id`;
2. sort each group by `content_sha256`;
3. walk cyclically forward from the next sorted position;
4. choose the first row with a different `next_endpoint_sha256`;
5. fail closed if no distinct endpoint exists or any selected identity is
   unchanged.

Keep current RGB and executed action fixed. Replace only the detached
EMA-next state when constructing `Q_deranged`. This observation runs under
`no_grad` and does not enter training.

Record:

- correct target centered-log soft cross-entropy
  `Hc(Q_correct,log(P_executed))`;
- deranged target centered-log soft cross-entropy
  `Hc(Q_deranged,log(P_executed))`;
- correct/deranged ratio, with a finite strictly positive denominator;
- per-family deranged-minus-correct centered-log soft cross-entropy;
- its positive-family count;
- all nine correct-target candidate centered-log cross-entropies
  `Hc(Q_correct,log(P_candidate))`;
- the rowwise minimum wrong-candidate cross-entropy, executed/hardest-wrong
  ratio, per-family hardest-wrong-minus-executed margin, and its
  positive-family count;
- mean target `KL(Q_correct || U9)`;
- finiteness, maximum absolute student logit, and transport-weight nonzero
  state;
- all nine probability-row normalization and positivity;
- maximum absolute expected offset
  `mu_i,a=sum_o(P_i,a,o*(dy_o,dx_o))`, which must remain within `[-1,1]`;
- for each of eight non-hold actions, whether any valid student probability
  differs bitwise from hold over the selection population;
- exact uniform hold probabilities, zero hold expected offset, and exact hold
  identity transport.

The aggregation is exact. For selection pair `i`, let:

- `c_i` be the arithmetic token mean of
  `Hc(Q_correct_i,log(P_i,executed))`;
- `d_i` be the arithmetic token mean of
  `Hc(Q_deranged_i,log(P_i,executed))`;
- `h_i` be the minimum over the eight wrong candidate actions of that
  candidate's arithmetic token-mean
  `Hc(Q_correct_i,log(P_i,candidate))`.

Over all exact `495` rows, the correct/deranged ratio is
`mean_i(c_i)/mean_i(d_i)` and the executed/hardest-wrong ratio is
`mean_i(c_i)/mean_i(h_i)`. Both denominators must be finite and strictly
positive. For scene family `f`, the two margins are exactly
`mean_{i in f}(d_i-c_i)` and `mean_{i in f}(h_i-c_i)`. No mean of rowwise
ratios, ratio of family means, weighting, or row exclusion is permitted.

At update zero require:

- all preserved V5 update-zero prediction, rank, RNG, mutation, and
  gradient-health checks;
- exact-zero transport weight and logits;
- all action distributions bitwise equal to hold and to `U`;
- correct and deranged soft cross-entropies bitwise equal;
- exact identity transport for every action;
- finite positive normalized target and student distributions;
- finite strictly positive mean target `KL(Q_correct || U9)`.

At updates `100`, `400`, and `1000`, additionally require:

- transport weight is finite and bitwise nonzero;
- all correspondence observations are finite and normalized;
- correct soft cross-entropy is strictly below its frozen update-zero value;
- correct/deranged soft-cross-entropy ratio is strictly below `0.99`;
- deranged-minus-correct soft cross-entropy is positive in at least `6/8`
  scene families;
- executed/hardest-wrong correspondence cross-entropy is strictly below
  `0.99`;
- hardest-wrong-minus-executed correspondence cross-entropy is positive in
  at least `6/8` scene families;
- all eight non-hold candidate distributions differ bitwise from hold;
- hold remains uniform with exact-zero expected offset and exact identity
  transport;
- every expected-offset component remains within `[-1,1]`.

These are mechanism gates. They do not replace or relax the forward JEPA
ordering gates.

## Preserved forward gates

At update `100`, require:

- raw effective rank strictly above `27.717458724975586`;
- projected effective rank strictly above `17.426651000976562`;
- true/cyclic-wrong, true/hardest-wrong, and non-hold-true/hold ratios each
  strictly below `0.99`;
- true/mean-target strictly below `1.0`;
- cyclic and hold margins positive in at least `6/8` families;
- exact populations, finiteness, EMA-gradient freedom, RNG preservation, and
  zero diagnostic model-state mutation;
- every V7 correspondence gate above.

Failure publishes
`FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL` and stops.

At update `400`, require the exact preserved continuation conjunction:

- raw and projected ranks at least `37.85872936248779` and
  `32.71332550048828`;
- true/cyclic-wrong, true/hardest-wrong, and non-hold-true/hold at most
  `0.975`;
- true/shuffled-next and true/mean-target at most `0.90`;
- true/shuffled-current at most `0.95`;
- raw cross-sample variance and spatial diversity each at least one quarter
  of update zero;
- cyclic and hold margins positive in at least `6/8` families;
- every V7 correspondence gate above.

Failure publishes
`FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL` and stops.

At final update `1000`, require:

- raw and projected effective ranks at least `48.0`;
- raw variance and spatial diversity each at least one quarter of update
  zero;
- true/shuffled-next, true/mean-target, true/cyclic-wrong, and
  non-hold-true/hold at most `0.90`;
- true/hardest-wrong and true/shuffled-current at most `0.95`;
- cyclic and hold margins positive in at least `6/8` families;
- every V7 correspondence gate above.

Only this complete conjunction publishes
`PASS_PHASE_A_ENTER_FROZEN_PHYSICAL_PROBE`. Otherwise Phase A publishes
`FAIL_PHASE_A_TERMINAL_NO_PHASE_B_NO_RETRY`.

## Fresh custody

The exact schema prefix is:

`lewm_go2_rgb_action_conditioned_local_correspondence_transport_jepa_v7`.

The sole output root is:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_action_conditioned_local_correspondence_transport_jepa_probe_v7`.

It must be absent before reservation. The root is reserved mode `0700`
before importing Torch or opening runtime RGB, schedule, N320, gate, or
checkpoint bytes. Reservation begins `attempt_index=1` of
`maximum_attempts=1`; any post-reservation integration or runtime failure
consumes the attempt.

Phase A may consume only:

- exact frozen development current/next RGB endpoint pairs;
- exact executed-action identities and frozen nine-action vocabulary;
- exact frozen N320 checkpoint and gate, initialization-only;
- exact frozen schedule, seed, ordered indices, pair identities, and prefix
  hashes;
- committed manifests, audits, pair/endpoint indexes, authorization
  documents, and reviewed source bindings solely for validation and
  resolution.

Validation metadata is not a model feature. During Phase A, the general raw
V13 frame loader, camera-supervision arrays, probability-calibration data,
held-out data, sealed data, and every unauthorized role remain denied with
zero opens.

V1 through V6 committed documents are evidence-only. No V1 through V6
generated runtime root, receipt, metrics payload, checkpoint, tensor, or
trace may be opened, hashed, copied, loaded, resumed, or used for
initialization. Record:

- `prior_runtime_output_open_count=0`;
- `rejected_checkpoint_open_count=0`.

Any newly written V7 checkpoint or trace is a sealed output, not a reusable
input. It may be written once, receipt-bound, and sealed, but must not be
reopened or independently hashed by terminal auditors. Auditors may inspect
only its declared filename, byte count, and mode.

## Conditional Phase B boundary

Phase B is the sole conditional custody exception. It may begin only in the
same process after the exact final Phase-A conjunction passes.

Only the in-memory terminal online raw encoder state may be copied.
The V7 transport tensor, predictor, projectors, correspondence targets, and
optimizer state must not be copied or optimized in Phase B. A Phase-A
checkpoint must not be reopened.

Phase B otherwise remains the existing frozen evidence-head physical
procedure, inputs, schedule, seed, evaluator, thresholds, and separate
`1000`-update / `16000`-presentation / `60`-GPU-active-minute cap. Its
matched-training camera-supervision leaves may open only after the exact
Phase-A pass under that already-reviewed exception.

Neither a Phase-A nor a Phase-B pass authorizes G2, navigation, held-out,
sealed, promotion, production, or deployment. Those require a sealed terminal
audit, independent review, and separate authority.

## Cap and terminal rule

This mechanism receives exactly one fresh attempt:

- first decision: `100` updates / `1600` pair presentations;
- second decision: `400` updates / `6400` pair presentations;
- maximum Phase A: `1000` updates / `16000` pair presentations /
  `60` GPU-active minutes;
- conditional Phase B only after an exact final Phase-A pass;
- cumulative cap if Phase B is entered: `2000` updates /
  `32000` pair presentations / `120` GPU-active minutes.

There is no retry, resume, replacement, second seed, schedule extension,
observer rerun, neighborhood-radius variant, border-rule variant,
transport-width/depth variant, temperature, loss-weight edit, threshold
relaxation, rejected-checkpoint access, or automatic successor.

Terminal receipts must attest exact consumed-input rehashes and roles,
forbidden-access counts, schedule prefix identities, operation counts,
determinism restoration, Phase-B entry state, and the exact sealed inventory.
Terminal files become mode `0444` and directories mode `0555`.

## Interpretation boundary

A Phase-A pass would show that a learned RGB JEPA encoder and
action-conditioned local latent transport can preserve action-specific
short-range correspondence strongly enough to beat the nearest wrong action,
the action-free mean target, and a within-scene deranged next view on frozen
development mazes.

It would still not establish navigation or held-out-maze generalization.
Those claims require the unchanged physical evidence gate, then the
separately authorized G2-to-G8 navigation and sealed held-out sequence.
