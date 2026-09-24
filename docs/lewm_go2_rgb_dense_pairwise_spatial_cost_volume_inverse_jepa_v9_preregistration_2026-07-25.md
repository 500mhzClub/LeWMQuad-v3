# RGB Dense Pairwise Spatial Cost-Volume Inverse JEPA V9 preregistration

Date: 2026-07-25

## Decision

Authorize source work for exactly one fresh perception-only RGB JEPA
falsification named:

`RGB_DENSE_PAIRWISE_SPATIAL_COST_VOLUME_INVERSE_JEPA_V9`

This document does not itself authorize training. Execution requires frozen
source, focused source tests, an independent science/source review, and one
exact execution authorization.

V8 is closed as a valid scientific failure by commit
`9f3e2bc96a6e4ea419574f109c890299d0608659`. Its terminal audit is:

`docs/lewm_go2_rgb_action_conditioned_local_correspondence_all_candidate_identification_jepa_v8_terminal_audit_2026-07-25.json`

with file SHA-256
`3ea4a8cc4405b0880d2e05217e4b4acefc5b9df5fad9bcdd9a682db42e273173`
and content SHA-256
`ff8339aa6109933e85d60ad118dc912fd091dddf7dfd80b18d00453ce7c01367`.
V8 improved its new action NLL by only `0.2859%`; nine-action macro balanced
accuracy reached only `0.13847330833949592`, and every inherited
nearest-action and forward-ordering failure remained. No V8 checkpoint or
trace may be opened, resumed, or reused.

## Scientific question

Can the RGB encoder learn action-relevant local motion when the loss sees the
complete live current/next spatial match structure before nonlinear spatial
aggregation, instead of:

- V6 projecting same-position pairs and then globally mean-pooling them; or
- V7/V8 asking nine candidate action-conditioned `3x3` transport templates to
  explain a detached local target?

The hypothesis is that the earlier objectives erased or indirected the small
motion signal before it could train the encoder. V9 makes the observed image
pair label-blind input to a dense spatial inverse-action head. The executed
action is used only as the classification target in that head.

## Exact changed mechanism

For a microbatch of current and next RGB, define both states exactly as:

`z = online_geometry(encoder.forward_tokens(rgb)[:, 1:])`.

Run current and next RGB through that same live path:

`z_cur, z_next in R^(B x 256 x 192)`.

Both branches remain in the autograd graph. There is no stop-gradient on
either input to the V9 inverse head.

Apply parameter-free per-token layer normalization without affine parameters,
using epsilon `1e-5`:

`u_cur = LN(z_cur)`

`u_next = LN(z_next)`.

Construct two label-blind all-pairs cost volumes:

`C_cn = u_cur @ transpose(u_next) / sqrt(192)`

`C_cc = u_cur @ transpose(u_cur) / sqrt(192)`.

Both have shape `[B, 256, 256]`. There is no learned temperature, radius,
mask, top-k, candidate action, action embedding, or V7/V8-style
action-conditioned local transport distribution. Apply row softmax over the
final target-token dimension:

`P_cn = softmax(C_cn, dim=-1)`

`P_cc = softmax(C_cc, dim=-1)`.

Preserve the complete normalized pair structure for the learned head:

`diff = P_cn - P_cc`, with axes `[batch, source_token, target_token]`

`V = diff.transpose(1, 2).reshape(B, 256, 16, 16).contiguous()`.

Thus target-token index is the channel axis, the `16x16` axes are the
row-major source grid, and each current spatial position retains all `256`
target-position probability differences until the first learned nonlinear
spatial stack. Every value must be finite and lie in `[-1, 1]`. The exact
float32 conservation check is
`torch.allclose(V.sum(dim=1), torch.zeros_like(V.sum(dim=1)), rtol=0,
atol=1e-6)`; it must pass at every observed checkpoint.

For observation only, construct coordinates first on CPU in float32 exactly
as:

`axis_cpu = torch.linspace(-1, 1, 16, device="cpu", dtype=torch.float32)`

`rows_y, cols_x = torch.meshgrid(axis_cpu, axis_cpu, indexing="ij")`

`q_cpu = torch.stack([rows_y.flatten(), cols_x.flatten()], dim=-1)`.

Here the two coordinate columns are explicitly `[dy, dx]` in row-major token
order. Transfer `q_cpu` once to `P_cn.device` and `P_cn.dtype`—which must be
float32—giving `q`. Define the identity-referenced soft displacement
diagnostic:

`D = (P_cn - P_cc) @ q`.

Reshape and permute it exactly as
`D.reshape(B,16,16,2).permute(0,3,1,2).contiguous()` to
`[B, 2, 16, 16]`. Every component must be finite and remain in the closed
interval `[-2, 2]`. `D` is never an input to the learned head or loss.

The exact learned spatial head is:

1. bias-free `Conv2d(256, 16, kernel_size=1, stride=1, padding=0)`;
2. `GELU(approximate="none")`;
3. bias-free `Conv2d(16, 16, kernel_size=3, stride=1, padding=1)`;
4. `GELU(approximate="none")`;
5. exact `AvgPool2d(kernel_size=4, stride=4, padding=0)`, mapping `16x16`
   to `4x4`;
6. flatten to `256`;
7. `Linear(256, 9, bias=True)`.

The exact new parameter count is `8,713`:

- first convolution: `256 * 16 * 1 * 1 = 4,096`;
- second convolution: `16 * 16 * 3 * 3 = 2,304`;
- final weight: `256 * 9 = 2,304`;
- final bias: `9`.

Construct the complete head on CPU in float32 while snapshotting and restoring
the surrounding global CPU and every accelerator RNG state. Initialization
then uses one isolated CPU generator with seed `20260725` and the exact draw
order first-convolution weight, second-convolution weight, final linear
weight. Apply
`kaiming_normal_(a=0, mode="fan_in", nonlinearity="relu", generator=g)` to
each convolution, then
`normal_(mean=0, std=1/16, generator=g)` to the final weight. Set the final
bias to exact zero without an RNG draw, then transfer the initialized head to
the model device. Module construction, explicit initialization, and device
transfer together must leave the surrounding global CPU and accelerator RNG
states bitwise unchanged. Before update 1, `count_nonzero(weight) ==
numel(weight)` must hold separately for all three weights.

The final bias is intentional: an exact zero-motion field must be able to
learn the hold class. A bias-only action prior cannot pass the frozen macro
balanced-accuracy and pair-control gates.

## Preserved JEPA and objective

V9 starts fresh from the same qualified N320 encoder initialization used by
V5-V8. It does not load any V1-V8 runtime output. The exact reviewed forward
base is V5 source-and-review commit
`c93124b15387acf1fd440d281e9c4503a9e8355a`. V6 contributes evidence about
failed global pooling only; its inverse projection, parameters, objective,
diagnostics, and runtime state contribute nothing to V9.

Preserve that exact V5 scientific base while excluding the closed V7/V8 local
correspondence transport:

- the V5 state-dependent bounded latent-flow current-plus-action JEPA forward
  path;
- the exact action vocabulary and order;
- the executed JEPA energy;
- the detached-row-scale all-action Energy-NLL and coefficient `1.0`;
- raw and projected patch whitening, epsilon `1e-4`, variance weight `0.50`,
  and covariance weight `0.02`;
- residual alpha `0.1/sqrt(192)`;
- EMA target encoder/projector with momentum `0.996`;
- float32, no autocast, strict deterministic algorithms;
- encoder AdamW learning rate `1e-4`, auxiliary learning rate `3e-4`,
  betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, and one global
  clip at norm `1.0`.

Remove the V6 same-position global inverse projection and every V7/V8
action-conditioned local correspondence transport, local `Q/P`, centered
correspondence loss, and all-candidate correspondence-action loss. None may
be present in the V9 forward graph, parameter inventory, diagnostics, or
receipts. V9's label-blind all-pairs `P_cn` and `P_cc` above are not the
removed V7/V8 action-conditioned local `Q/P`.

Let `a_i` be the executed-action index and let `m_i` be the existing detached,
positive row energy scale from the all-action JEPA energy. For V9 head logits
`s_i`, define:

`L_dense_inverse = mean_i(m_i * CE(s_i, a_i))`.

Its coefficient is exactly `1.0`. Record the unscaled row CE and its arithmetic
mean separately. Do not add class weights, focal loss, margin, temperature,
confidence loss, flow supervision, pose, depth, odometry, optical-flow labels,
or a second head.

The Phase-A total is exactly:

`L_JEPA + L_EnergyNLL + L_dense_inverse`

`+ 0.50*(raw_whitening.variance + projected_whitening.variance)`

`+ 0.02*(raw_whitening.covariance + projected_whitening.covariance)`.

Only Phase-A parameters under the exact prefixes `encoder.`,
`online_target_projector.`, `prediction_projector.`, `predictor.`, and
`dense_pairwise_inverse_head.` are trainable. `encoder.` is the sole
`1e-4` optimizer group; the other four prefixes, including all `8,713` V9
head parameters, form the `3e-4` auxiliary group. Appearance, EMA, Camera
evidence, BEV, occupancy, and Phase-B state remain frozen.

## Frozen data and schedule

Preserve exactly:

- Raw V13 train and checkpoint-selection roles: `4,262` and `495` pairs from
  `72` and `8` scenes;
- current RGB, next RGB, and executed requested action only;
- base initialization seed `20260712`;
- schedule seed `20260713`;
- microbatch `4`, four microbatches per update, effective batch `16`;
- the exact first `16,000` scheduled pair presentations and prefix hashes:
  - update `100` / presentation `1,600`:
    `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`;
  - update `400` / presentation `6,400`:
    `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`;
  - update `1,000` / presentation `16,000`:
    `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.

One scheduled pair is one presentation despite the `65,536` pairwise token
comparisons. The schedule-schema adapter may normalize only the already
reviewed schema representation. It may not mutate, reorder, filter,
regenerate, reseed, replace, reopen, or extend any schedule index.

## Diagnostics and controls

Use the fixed `495` checkpoint-selection pairs at updates `0`, `100`, `400`,
and `1,000`. Preserve RNG and model state exactly across every observation.
No observer rerun is allowed.

Record:

- unscaled dense inverse NLL over all `495` rows; top-1 from `argmax` in the
  frozen action order, with the lowest action index winning exact ties;
  per-action row count/NLL/recall, where recall is the integer number of
  correct predictions divided by that action's fixed row count; and
  nine-action macro balanced accuracy as the unweighted arithmetic mean of
  the nine recalls;
- correct-pair NLL over all `495` rows;
- scene-local deranged-next NLL over those same `495` rows using the already
  frozen derangement mapping; this control changes only `z_next` and leaves
  current state, labels, row order, and all other inputs fixed;
- correct-pair NLL restricted to the exact `435` non-hold rows and
  current-current NLL over those same `435` rows, obtained by replacing
  `z_next` with the matching `z_cur`;
- correct/deranged ratio as the all-`495` correct mean divided by the
  all-`495` deranged mean, and non-hold correct/current-current ratio as the
  exact-`435` correct mean divided by the exact-`435` current-current mean;
- per-scene-family deranged-minus-correct NLL as the arithmetic mean of
  `deranged_row_CE - correct_row_CE` over that family's rows, always using
  the original executed-action labels; the positive-family count uses strict
  `family_mean > 0`;
- displacement finiteness, maximum absolute component, cross-pair displacement
  RMS defined as `sqrt(mean(D.float() ** 2))` over all
  `495 * 2 * 16 * 16` values, and exact-zero same-tensor
  identity-reference displacement;
- all existing rank, variance, spatial-diversity, shuffled-current,
  shuffled-next, mean-target, cyclic-wrong, hardest-wrong, hold, population,
  finiteness, EMA-gradient, RNG, and mutation diagnostics.

The same-device float32 `CE(zeros([495,9]), executed_actions)` arithmetic mean
is the frozen `log(9)` reference. All nine action populations must be nonempty.

Source acceptance must prove with synthetic tensors that:

- action-label changes leave `C_cn`, `C_cc`, `P_cn`, `P_cc`, `diff`, `V`,
  `D`, and logits bitwise unchanged and change only the CE target/gradient;
- in one routing fixture current is live and next is detached, and in a
  second current is detached and next is live; each fixture gives at least
  one actual shared online-encoder parameter a finite nonzero gradient from
  `L_dense_inverse`, rather than merely giving a nonzero leaf-`z` gradient;
- with both branches live, each of the first-convolution weight,
  second-convolution weight, final-linear weight, and final-linear bias
  receives a finite nonzero gradient;
- EMA tensors receive no gradient;
- an index-coded synthetic layout fixture proves for every index that
  `V[b, t, sy, sx] == diff[b, 16*sy + sx, t]`;
- on a fixed seeded asymmetric synthetic input,
  `torch.roll(V, shifts=(1, 3), dims=(2, 3))` changes the logits; changing or
  permuting diagnostic-only `D` cannot change logits;
- identical current/next tensors produce bitwise exact-zero `diff`, `V`, and
  identity-referenced `D`;
- component bounds, shapes, parameter count, initialization, and final-bias
  rules are exact;
- no pool or other spatial reduction occurs before the exact
  `Conv1x1 -> GELU -> Conv3x3 -> GELU` stack.

## Staged falsification gates

The earlier experiments required every final forward-ordering gate at update
100 and repeatedly terminated while the new mechanism was only weakly active.
V9 instead freezes one staged gate in advance. This is not a V5-V8 threshold
edit or retry; V9 is a fresh mechanism and still must pass the complete final
gate before Phase B.

At update `100` / `1,600` presentations, continue only if:

- unscaled dense inverse NLL is strictly below `0.98` times the frozen
  same-device `log(9)` reference;
- macro balanced accuracy is strictly above `2/9`;
- correct/deranged-next NLL ratio is strictly below `0.99`;
- non-hold correct/current-current NLL ratio is strictly below `0.99`;
- deranged-minus-correct NLL is positive in at least `6/8` scene families;
- raw effective rank is strictly above `27.717458724975586`;
- projected effective rank is strictly above `17.426651000976562`;
- raw variance and spatial diversity are each at least one quarter of their
  update-zero values;
- true/shuffled-next is at most `0.90` and true/shuffled-current is at most
  `0.95`;
- all eight non-hold latent-flow actions are active, hold flow is exact zero,
  every V5 flow value is finite and within `[-1,1]`, all `V` and `D` values
  are finite and within their registered bounds, the exact `V`
  channel-conservation and same-tensor `diff`/`V`/`D` identity checks pass,
  and all population, EMA, RNG, and mutation checks pass.

Failure stops immediately with no retry or Phase B.

At update `400` / `6,400` presentations, continue only if every update-100
mechanism/health gate still passes, unscaled NLL is strictly lower and macro
balanced accuracy is at least its update-100 value, and:

- true/cyclic-wrong is strictly below `0.99`;
- true/hardest-wrong is strictly below `0.99`;
- non-hold-true/hold is strictly below `0.99`;
- true/mean-target is strictly below `1.0`;
- cyclic and hold positive margins each occur in at least `6/8` families;
- raw rank is at least `37.85872936248779`;
- projected rank is at least `32.71332550048828`.

Failure stops immediately with no retry or Phase B.

At update `1,000` / `16,000` presentations, Phase A passes only if unscaled
NLL is strictly lower and macro balanced accuracy is at least its value at
update 400, every inverse pairing control still passes, and the complete exact
V5-lineage final JEPA gate passes unchanged:

- raw and projected effective rank each at least `48.0`;
- raw variance and spatial diversity each at least one quarter update zero;
- true/shuffled-next at most `0.90`;
- true/shuffled-current at most `0.95`;
- true/mean-target at most `0.90`;
- true/cyclic-wrong, true/hardest-wrong, and non-hold-true/hold each at most
  `0.95`;
- cyclic and hold positive margins each in at least `6/8` families;
- every update-400 mechanism and health invariant still passes, including all
  eight non-hold V5 latent-flow actions active and hold flow exact zero;
- exact populations, finiteness, EMA-gradient freedom, RNG preservation, and
  zero observation mutation.

Loss improvement without these representation gates is not a pass.

## Caps, Phase B, and terminal rules

V9 has exactly one attempt, one seed, and one fresh output root:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_probe_v9`

It must be absent before reservation. Reserve it mode `0700` before importing
Torch or opening runtime RGB, schedule, N320, gate, or checkpoint bytes.

The exact schema prefix is:

`lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9`.

The reservation must declare `attempt_index=1` and `maximum_attempts=1`. The
exact Phase-A controls are:

- update-100 failure:
  `FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL`;
- update-400 failure:
  `FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL`;
- update-1,000 failure:
  `FAIL_PHASE_A_TERMINAL_NO_PHASE_B_NO_RETRY`;
- complete Phase-A pass:
  `PASS_PHASE_A_ENTER_FROZEN_PHYSICAL_PROBE`.

Phase-A limits are:

- first decision: `100` updates / `1,600` presentations;
- second decision: `400` updates / `6,400` presentations;
- final limit: `1,000` updates / `16,000` presentations;
- maximum `60` GPU-active minutes.

Only a complete update-1,000 Phase-A pass may enter the already reviewed
physical-perception Phase B in the same process. Phase B may copy only the
terminal in-memory online encoder state into a fresh physical model, then
hard-sync that state into the fresh target encoder exactly once. Phase B
trains only parameters under `evidence_head.`. It must not copy the V9 inverse
head, target-BEV decoder, evidence head, latent flow, predictor, projectors,
optimizer, or any Phase-A checkpoint; no Phase-A checkpoint payload is read.
Phase B retains its existing `1,000`-update / `16,000`-presentation /
`60`-GPU-minute cap and the frozen physical development gate:

- at least `1/9` complete scopes;
- passed margins strictly above `97/189`;
- total shortfall strictly below `41.01776266878769`;
- rough pixel balanced accuracy above `0.8198594673963917`;
- rough ground balanced accuracy above `0.647134926562893`;
- rough depth p95 below `0.9777327477931971 m`.

The cumulative cap is `2,000` updates, `32,000` presentations, and `120`
GPU-active minutes only if Phase B is validly entered.

Any integration failure after reservation consumes the attempt. There is no
retry, resume, second seed, observer rerun, temperature, radius, coordinate,
head-width, head-depth, pooling, loss-weight, threshold, schedule, data, or
initialization sweep. An obvious pre-execution source defect may be corrected
only before the output root is reserved and requires re-review of the changed
source.

Every normal Phase-A scientific terminal publishes exactly the canonical
receipt chain `reservation.json` -> `phase_a/metrics.json` ->
`phase_a/artifact.json` -> `access.json` -> `result.json` ->
`completed.json`, plus its receipt-declared write-only trace/checkpoint
inventory. The update-100, update-400, and update-1,000 Phase-A failure
receipts use their exact registered failure control as the metrics, artifact,
result, and completion status. Valid Phase-B entry additionally publishes
`phase_b/metrics.json` and `phase_b/artifact.json`: their statuses are,
respectively, `PASS_PHASE_B` and
`PASS_FROZEN_ENCODER_PHYSICAL_PROBE`, or `FAIL_PHASE_B_TERMINAL` and
`FAIL_FROZEN_ENCODER_PHYSICAL_PROBE_TERMINAL`. The corresponding result and
completion statuses are
`PASS_BOUNDED_FALSIFICATION_SEPARATE_QUALIFICATION_ONLY` / `TERMINAL_PASS`, or
`FAIL_PHASE_B_MECHANISM_TERMINATED` / `TERMINAL_FAIL`.

A post-reservation integration or operational failure retains every partial
file and adds only canonical `failure.json` and `completed.json`. Its failure
receipt must make a best-effort complete custody attestation and record the
exact partial inventory, operation counts, determinism/access state, missing
normal receipts, and every available binding before sealing. Missing receipts
must be named explicitly and must never be synthesized or fabricated. If
publication of `reservation.json` itself fails after the `0700` root is
created, the terminal contains only the files that really succeeded plus
`failure.json` and `completed.json`; it records
`TERMINAL_RESERVATION_PUBLICATION_FAILURE` and must not claim that
`reservation.json` exists. Runtime checkpoints and traces are write-only
evidence: reviewers may inspect only their receipt-declared hashes and
filesystem metadata, never their payloads. Every terminal exact inventory is
sealed read only after completion.

## Authority boundary

V9 may open the exact bound source/authority metadata and the reviewed raw
manifest, audit, pairs, and endpoints under the `authority` and `index`
roles. Model-facing RGB/pair rows are restricted to the exact development
`train` and `checkpoint_selection` roles. Receipts must bind the
preregistration, source manifest, independent review, execution
authorization, every reviewed source, raw manifest and audit, pair/endpoints
index, exact schedule and prefixes, N320 gate and initialization checkpoint,
operation counts, Phase-B entry and one-time target-sync state, exact
inventory, and sealing result. Immediately before `access.json`, every opened
runtime input and reviewed source is rehashed and required to match its
binding; `result.json` binds the access and phase artifacts, and
`completed.json` binds the result and exact terminal inventory.

Receipt counters must be exactly zero for prior runtime outputs, rejected
checkpoints, probability calibration, Phase-A Camera-supervision array opens,
Phase-A general raw-loader calls, G2, navigation, held-out, and sealed access.
Explicit production-input and deployment-input open counters must also each
equal zero. Observer reruns must equal zero. V9 may not open V1-V8 runtime
checkpoints or traces, production inputs, or deployment inputs.

Neither a Phase-A pass nor a Phase-B pass authorizes G2, navigation, held-out,
sealed, promotion, production, or deployment. A Phase-B pass only creates a
separately reviewable pre-G2 perception candidate.
