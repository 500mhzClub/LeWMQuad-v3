# RGB Action-Conditioned Next-Target Retrieval JEPA V10 preregistration

Date: 2026-07-25

## Decision and evidence boundary

Authorize source preparation and one independent source/science review for
exactly one fresh perception-only falsification named:

`RGB_ACTION_CONDITIONED_NEXT_TARGET_RETRIEVAL_JEPA_V10`

This document does not authorize execution. Execution requires frozen source,
focused tests, a passing independent review, a fresh source manifest, and a
separate exact one-attempt authorization.

V9 is closed as a valid early scientific failure by commit
`f02fdb02db328b339df5ec897424a42fe45a258b`. Its terminal audit is:

`docs/lewm_go2_rgb_dense_pairwise_spatial_cost_volume_inverse_jepa_v9_terminal_audit_2026-07-25.json`

with file SHA-256
`a95b81c30e619c0fe5ef06c46e7cc60270ef27751c4588291482bdf9d0319ad8`
and content SHA-256
`82038aecd65d1d9b844903c768c7a0cee0750f981f4d824c05731fff95970120`.
V9 passed every preregistered update-100 generic representation-health gate,
but its inverse pathway stayed at chance and its diagnostic first-moment
displacement contracted. That result does not localize failure to the encoder,
inverse integration, head, or optimization. No V9 checkpoint, trace, tensor,
or other runtime payload may be opened, resumed, or reused.

## Scientific question

Can the V5 current-plus-action predictor both identify the executed action and
retrieve its actual next latent target among fixed same-scene alternatives
when both questions use the same direct, parameter-free compatibility score?

V10 factorizes one retrieval mechanism in two directions. Action retrieval
selects the executed prediction from all nine action-conditioned queries for
the true next target. Target retrieval uses that executed query to select the
correct future over a fixed same-action future from the same scene and, for
non-hold rows, the no-change current target. The score and losses add no
learned parameters.

This is a test of the joint RGB encoder plus the preserved V5 predictor/action
path, not an encoder-only assay. A failure cannot by itself distinguish an
invariant encoder from inadequate prediction dynamics or optimization. A pass
would show accessible action-specific temporal structure on development data;
it would not establish navigation utility.

## Frozen V5 base

Start fresh from the same qualified N320 initialization and exact reviewed V5
forward base at source-and-review commit
`c93124b15387acf1fd440d281e9c4503a9e8355a`. Do not load any V1-V9 runtime
output.

Preserve exactly:

- the V5 state-dependent bounded latent-flow current-plus-action predictor,
  nine-action vocabulary/order, hold index `6`, and residual alpha
  `0.1/sqrt(192)`;
- the exact executed-action V5 JEPA regression;
- raw and projected patch whitening, epsilon `1e-4`, variance weight `0.50`,
  and covariance weight `0.02`;
- the online and EMA encoder/projector paths, EMA momentum `0.996`, float32,
  no autocast, and strict deterministic algorithms;
- encoder AdamW learning rate `1e-4`, auxiliary learning rate `3e-4`, betas
  `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, and one global clip at
  norm `1.0`;
- the existing trainable-prefix and frozen-module boundaries.

All V5 initialization draw order, values, and RNG-preservation rules remain
exact. V10 introduces no initialization draw because it introduces no
parameter.

Remove the V5 detached-row-scaled all-action Energy-NLL. Delete the V9 dense
inverse head, cost volumes, displacement diagnostic, and inverse loss from the
V10 graph, parameter inventory, optimizer, observations, and receipts. The two
V10 retrieval CEs replace those classification losses. V10 has no inverse
head, retrieval head, temperature, margin, queue, memory bank, top-k miner, or
new trainable parameter.

## Fixed candidate construction

For each role separately, group rows by `(role, scene_id, primitive)`, sort
each group by the lowercase hexadecimal `content_sha256` in bytewise ascending
order, and freeze the following map before execution:

- if a group contains at least two rows, map each row to the next row in that
  exact cyclic order, wrapping once at the end;
- if a group is a singleton, map that row to the next row in its scene after
  sorting the complete scene by `content_sha256`, again wrapping once.

The mapped row supplies only its next RGB endpoint. The map must change the
`next_endpoint_sha256`, remain within the same role and scene, and be bound by
ordered-row and mapping SHA-256 values in the frozen V10 science contract and
execution authorization. The source manifest binds that contract source; it
need not duplicate runtime-map fields. There is no random draw or runtime
remapping.

Metadata-only feasibility establishes:

- train: `4,237/4,262` rows are mapped within one of `578/603` non-singleton
  `(scene, primitive)` groups;
- checkpoint selection: `494/495` rows are mapped within `71/72` such groups;
- the sole checkpoint-selection fallback is the small-scene `forward_slow`
  singleton.

Every mapped next endpoint in a non-singleton group is used exactly once as a
positive and exactly once as a negative in that group. This symmetric use is
what makes the same-action subset resistant to a constant or action-prior
query. The singleton fallback keeps the candidate count exact but is excluded
from any same-action or class-prior claim.

For row `i`, let `d(i)` be this frozen map and let `a_i` be the executed action.
The target list, in fixed order, is:

`T_i = [t_next_i, t_next_d(i)]` when `a_i == hold`;

`T_i = [t_next_i, t_next_d(i), t_current_i]` otherwise.

Thus every row has the correct and deranged next target; current is a negative
only for exact non-hold rows. Candidate identities are fixed, while their
latents are recomputed through the current detached EMA target path at that
training step. No cached target embedding, cross-role candidate, in-batch
mining, or candidate substitution is allowed.

One scheduled primary pair remains one presentation. Its one mapped negative
RGB read is an attached candidate read, not a second schedule presentation.
Training mapped-negative endpoint requests must be counted separately and equal
one per primary presentation; observation requests are counted separately.
Physical reads, cache hits, and cache misses are distinct stage-specific
receipt fields and are not required to equal request counts. Every unique
consumed file is rehashed before access publication. The map never changes
primary schedule order or optimizer/EMA counts.

## Exact factorized retrieval math and objective

For one primary row, use the preserved V5 all-action predictions and gather the
executed query:

`q_i = q_i,a_i in R^(256 x 192)`.

Let each `t` in `T_i` be the corresponding detached EMA target-projector
tokens. Queries and targets must be float32, finite, and per-token L2-normalized
with the frozen V5 `eps=1e-8` rule. Define the parameter-free symmetric metric:

`E(q, t) = (1/256) * sum_n=1^256 sum_d=1^192 (q[n,d] - t[n,d])^2`.

The metric is symmetric, and non-singleton candidate usage is bijectively
symmetric. Gradient routing is not bidirectional because every EMA target is
deliberately detached.

Because both token vectors are L2-normalized, every row energy must be finite
and lie in the closed interval `[0,4]`. First define action-retrieval logits
against the correct next target:

`action_logits_i[a] = -E(q_i,a, t_next_i)` for `a in {0,...,8}`;

`ell_action_i = cross_entropy(action_logits_i, target_index=a_i)`.

Then define target-retrieval logits over the ordered candidates:

`target_logits_i[k] = -E(q_i, T_i[k])`;

`ell_target_i = cross_entropy(target_logits_i, target_index=0)`.

There is no division by a learned or detached scale, no temperature, no
detached JEPA-energy multiplier outside either CE, and no class or stratum
weight. Each microbatch loss is the plain arithmetic mean of its row CE.
Full-role target-retrieval reporting recombines hold and non-hold stratum means
by their actual row counts, not equal stratum weights, and records the two
stratum means separately. The coefficients of `L_action_retrieval` and
`L_target_retrieval` are each exactly `1.0`.

The Phase-A objective is exactly:

`L_JEPA + L_action_retrieval + L_target_retrieval`

`+ 0.50*(raw_whitening.variance + projected_whitening.variance)`

`+ 0.02*(raw_whitening.covariance + projected_whitening.covariance)`.

Only the existing V5 trainable parameters receive retrieval gradients. All
EMA target tensors remain detached and gradient-free. Neither retrieval branch
uses the removed V5 detached row scale or any V9 quantity.

## Action-necessity controls

At each checkpoint-selection observation, compute all nine V5 predictions from
the same current RGB and score every one against the same correct next target:

`A_i,a = E(q_i,a, t_next_i)`.

Record the executed energy and these fixed controls:

- cyclic wrong: action index `(a_i + 1) mod 9`;
- hardest wrong: `min_a!=a_i A_i,a`;
- hold-for-nonhold: `A_i,6` on the exact non-hold rows. This is the required
  zero-action/zero-motion control: hold is in vocabulary, its hold-relative
  embedding is exact zero, and its V5 flow is exact zero at initialization;
- frozen within-scene action permutation: within each checkpoint-selection
  scene, sort rows by `(primitive, content_sha256)`, set `s` to that scene's
  maximum executed-action multiplicity, and assign row at position `j` the
  action from position `(j+s) mod n`. This gives `s=8` in each of seven
  `64`-row scenes and `s=13` in the `47`-row scene. The mapping must be a
  bijection and must change the action on every row.

For each control, record the ratio of the relevant arithmetic-mean executed
energy to the arithmetic-mean control energy. Also record per-scene-family mean
`control - executed` margins and strict-positive family counts. The frozen
selection role has exactly one scene in each of its eight scene families; the
contract must bind that one-to-one identity before aggregating. All
denominators must be finite and positive; otherwise the gate fails closed.

These controls change only action while holding current RGB, target, row, and
EMA snapshot fixed. An action-ignoring predictor gives ratio exactly `1` for
all four and cannot pass. On the exact `494`-row same-action derangement subset,
a constant or action-prior query sees the same target multiset as positive and
negative and cannot pass the strict correct-versus-deranged gate. Do not extend
that claim to the one fallback row.

## Observations and registered diagnostics

Observe the fixed `495` checkpoint-selection rows exactly once at updates `0`,
`100`, `400`, and `1,000`, preserving RNG and model state bitwise. There is no
observer rerun.

Record at minimum:

- retrieval CE over all `495` rows and over the exact `494` same-action rows;
  hold and non-hold CE separately; candidate counts; and top-1 counts from
  `argmin`, with the lowest candidate index winning exact ties for reporting;
- unscaled nine-way action-retrieval NLL, top-1 accuracy in frozen action
  order, per-action row count/NLL/recall, and nine-action macro balanced
  accuracy;
- a two-candidate correct-versus-deranged CE on the same-action `494` subset,
  with float32 equal-logit reference `log(2)`;
- correct, deranged-next, and non-hold current-target mean energies; the
  correct/deranged ratio on the same-action subset; the non-hold
  correct/current ratio; and per-family signed margins;
- all four same-current all-action control energies, ratios, populations, and
  per-family margins;
- every preserved V5 rank, variance, spatial-diversity, shuffled-current,
  shuffled-next, mean-target, latent-flow, population, finiteness, EMA-gradient,
  RNG, and mutation diagnostic.

The canonical analytical equal-logit references for the variable candidate
counts are computed once in host IEEE-754 binary64 as:

- all `495` rows (`60` hold with two candidates, `435` non-hold with three):
  `1.049465002836817`;
- same-action `494` subset (`60` hold, `434` non-hold):
  `1.0493655144039604`.

The shown decimals are the binary64 formula results, not claims about the last
ulps of a device float32 reduction. They are observation references, not
learned baselines, gate thresholds, or loss rescalings. Source acceptance
freezes the formulas, dtype, shapes, and populations for the gate references.
After reservation, the authorized execution computes them exactly once on the
execution device before update zero: float32
`CE(zeros([494,2]), zeros([494], int64))` for the two-target reference and
float32 `CE(zeros([495,9]), executed_actions)` for the action reference. The
resulting immutable scalars are recorded with update zero and every later
Phase-A observation and bound in terminal metrics. No later recomputation is
allowed.

## Staged falsification gates

At update `100` / presentation `1,600`, continue only if:

- action-retrieval NLL is strictly below the frozen float32 action equal-logit
  reference and nine-action macro balanced accuracy is strictly above `1/9`;
- exact-`494` same-action correct-versus-deranged CE is strictly below both
  the frozen float32 two-target equal-logit reference and its update-zero
  value;
- the strict same-action win rate
  `count(E_correct < E_deranged) / 494` is strictly above `1/2`; exact ties are
  not wins;
- frozen-permuted-action-minus-executed and non-hold
  hold-zero-motion-minus-executed margins are each positive in at least `6/8`
  scene families;
- raw effective rank is strictly above `27.717458724975586` and projected
  effective rank is strictly above `17.426651000976562`;
- raw variance and spatial diversity are each at least one quarter of update
  zero, true/shuffled-next is at most `0.90`, and true/shuffled-current is at
  most `0.95`;
- all eight non-hold V5 flows are active, hold flow is exact zero, all flow and
  retrieval values satisfy their registered finite bounds, all target-map and
  candidate populations are exact, EMA tensors remain gradient-free, and RNG
  and observation-mutation checks pass.

Failure stops immediately and closes this family; there is no Phase B, retry,
resume, V11 loss tweak, or observer rerun.

At update `400` / presentation `6,400`, continue only if every update-100 gate
still passes and:

- action-retrieval NLL is strictly lower than at update 100 and strictly below
  `0.99` times the frozen float32 action equal-logit reference, while macro
  balanced accuracy is at least its update-100 value and strictly above `2/9`;
- all-row and same-action target-retrieval CE, and same-action two-candidate
  CE, are each strictly lower than at update 100;
- same-action correct/deranged and non-hold correct/current-target energy
  ratios are each strictly below `0.99`;
- executed/cyclic, executed/hardest-wrong, non-hold executed/hold, and
  executed/frozen-permuted-action energy ratios are each strictly below
  `0.99`;
- cyclic, hold, and frozen-permutation margins are each positive in at least
  `6/8` families;
- true/mean-target is strictly below `1.0`;
- raw rank is at least `37.85872936248779` and projected rank is at least
  `32.71332550048828`.

Failure stops immediately with the same terminal rule.

At update `1,000` / presentation `16,000`, Phase A passes only if all-row and
same-action target-retrieval CE, same-action two-candidate CE, and
action-retrieval NLL are each strictly lower than at update 400; action macro
balanced accuracy is at least its update-400 value; all update-400 invariants
still pass; and:

- same-action correct/deranged and non-hold correct/current energy ratios are
  each at most `0.95`;
- the four same-current all-action ratios are each at most `0.95`;
- raw and projected effective rank are each at least `48.0`;
- raw variance and spatial diversity are each at least one quarter update
  zero; true/shuffled-next is at most `0.90`; true/shuffled-current is at most
  `0.95`; and true/mean-target is at most `0.90`;
- cyclic, hold, and frozen-permutation margins remain positive in at least
  `6/8` families;
- every flow, target-map, candidate-count, finiteness, EMA, RNG, and mutation
  invariant still passes.

Loss improvement without the retrieval, action-necessity, and representation
gates is not a pass.

## Frozen data, schedule, caps, and Phase B

Preserve the Raw V13 train and checkpoint-selection roles exactly: `4,262`
and `495` primary pairs from `72` and `8` scenes. Training receives only
current RGB, next RGB, executed requested action, and the one fixed same-scene
negative next RGB. There is no data rebuilding, filtering, rebalancing,
resampling, pose, depth, odometry, optical flow, occupancy, traversability,
navigation label, or calibration role.

Preserve base seed `20260712`, schedule seed `20260713`, microbatch `4`, four
microbatches per update, effective batch `16`, and the exact first `16,000`
primary presentations with prefix hashes:

- update `100` / `1,600`:
  `9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51`;
- update `400` / `6,400`:
  `6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92`;
- update `1,000` / `16,000`:
  `3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528`.

Phase A is capped at `1,000` updates, `16,000` primary presentations, and
`60` GPU-active minutes. V10 has one seed, one attempt, no retry or resume,
and one fresh output namespace:

`.generated/go2_shared_observable_camera_ray_jepa_v5/rgb_action_conditioned_next_target_retrieval_jepa_probe_v10`

That root must be absent before a mode-`0700` reservation declaring
`attempt_index=1` and `maximum_attempts=1`. Any integration or operational
failure after reservation consumes the attempt. An obvious source defect may
be corrected only before reservation and requires re-review of every changed
source binding.

Only a complete update-1,000 Phase-A pass may enter Phase B in the same
process. Phase B must instantiate a fresh physical-perception model, copy only
the terminal in-memory online encoder into it, hard-sync that encoder into its
fresh target once, and train only `evidence_head.`. It may not copy the V10
predictor, projectors, latent flow, optimizer, retrieval state, decoder, or any
Phase-A checkpoint. No checkpoint payload is read.

Phase B retains the frozen `1,000`-update / `16,000`-presentation / `60`-GPU
minute cap and must pass all of:

- at least `1/9` complete scopes;
- passed margins strictly above `97/189`;
- total shortfall strictly below `41.01776266878769`;
- rough pixel balanced accuracy above `0.8198594673963917`;
- rough ground balanced accuracy above `0.647134926562893`;
- rough depth p95 below `0.9777327477931971 m`.

The cumulative maximum is `2,000` updates, `32,000` presentations, and `120`
GPU-active minutes only after valid Phase-B entry. Neither phase authorizes G2,
navigation, held-out, sealed, promotion, production, or deployment access.

## Minimal source and acceptance tests

Keep the implementation local to the existing V5 model helper, Phase-A
contract, runner, launcher, and their focused tests. Add no generic retrieval
framework, data abstraction, trainer, configuration layer, or compatibility
path. The frozen source manifest must enumerate every touched file.

Source acceptance must prove with synthetic or metadata-only fixtures that:

- the target maps have exact counts, ordering, same-scene/role identity,
  non-self endpoints, and bound hashes; the `494/495` selection eligibility
  distinction is preserved;
- the selection action permutation has the exact `8`/`13` shifts, is
  bijective, changes every action, and preserves the bound one-scene-per-family
  aggregation;
- candidate order/count and non-hold-only current-negative semantics are exact;
- energy equals the registered formula, lies in `[0,4]`, both logit tensors are
  bitwise `-energy`, and neither CE has a scale, temperature, outer multiplier,
  class weight, or new parameter;
- the two gate-reference formulas have the exact float32 dtype, `494x2` and
  `495x9` populations, fixed targets, one-time post-reservation computation,
  observation reuse, and receipt bindings specified above;
- the row arithmetic mean and hold/non-hold recombination agree exactly;
- each retrieval CE separately gives a finite nonzero gradient to the intended
  existing V5 action path, their sum gives one to an actual online encoder
  parameter, and every EMA target parameter remains gradient-free;
- a constant-query fixture cannot beat the same-action correct/deranged
  reference, and an action-ignoring fixture produces all four action-control
  ratios exactly `1` and fails the gates;
- observation is RNG-preserving and mutation-free, schedule prefixes and caps
  are unchanged, V9 inverse symbols are not imported or called by the V10
  runner, no inverse state enters the V10 graph, parameter inventory,
  optimizer, observations, or receipts, and no prior runtime or protected role
  can be opened.

## Custody, terminal, and successor boundary

Reserve the fresh output root mode `0700` before importing Torch or opening
runtime inputs. Reuse the exact canonical V9 receipt-chain and fail-closed
publication semantics under the new V10 schema prefix
`lewm_go2_rgb_action_conditioned_next_target_retrieval_jepa_v10`: normal
terminals publish reservation, Phase-A metrics/artifact, access, result, and
completion receipts; valid Phase B additionally publishes its metrics and
artifact. The update-100, update-400, update-1,000, and complete-pass controls
are respectively
`FAIL_PHASE_A_UPDATE_100_CONTINUATION_GATE_TERMINAL`,
`FAIL_PHASE_A_UPDATE_400_CONTINUATION_GATE_TERMINAL`,
`FAIL_PHASE_A_TERMINAL_NO_PHASE_B_NO_RETRY`, and
`PASS_PHASE_A_ENTER_FROZEN_PHYSICAL_PROBE`.

Operational failures retain every real partial file and add only
`failure.json` and `completed.json`. A failure receipt must identify the exact
failed operation, partial inventory, missing normal receipts, schedule/update/
presentation counts, candidate-request/read/cache counts, determinism and
access state, and every available hash binding. Missing receipts are named,
never fabricated. Runtime checkpoints and traces remain write-only evidence
whose payloads may not be reviewed.

All prior-runtime, rejected-checkpoint, calibration, Phase-A camera-supervision,
general raw-loader, G2, navigation, held-out, sealed, production, deployment,
and observer-rerun counters must be exactly zero. Immediately before access
publication, rehash every opened source/authority/input binding and fail closed
on any mismatch.

V10 is the final bounded attempt for the present single-frame
current-plus-action V5 latent-flow predictor/objective family. Any update-100,
update-400, update-1,000, or conditional Phase-B failure terminates it. There
is no V11 temperature, loss-scale, negative-set, threshold, seed, schedule,
flow, or candidate tweak. A later experiment must instead be a materially
different early two-frame/tubelet encoder or direct physical-perception
architecture, separately preregistered and reviewed.
