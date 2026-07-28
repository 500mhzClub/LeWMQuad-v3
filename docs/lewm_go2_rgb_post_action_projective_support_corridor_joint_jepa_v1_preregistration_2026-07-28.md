# RGB Post-Action Projective-Support Corridor Joint JEPA V1 preregistration

Date: 2026-07-28

Experiment ID: `rgb_post_action_projective_support_corridor_joint_jepa_v1`

Status: preregistered for source-only implementation, synthetic tests, independent
source review, and one development-label preflight. Training remains fail-closed
until those steps pass and a separate execution binding freezes the reviewed
source and exact runtime inputs.

## Repository goal and bounded question

The repository goal remains a fully learned RGB-only, action-conditioned JEPA
navigation stack, ultimately validated once on fresh externally custodied held-out
mazes. The JEPA predictor or rollout must causally affect deployed action scores;
learned components must eventually choose viewpoints, routes, subgoals, and normal
motion primitives.

This experiment asks one narrower question:

> From current RGB and a candidate action, can a jointly trained JEPA predict a
> spatial next-view latent whose shared semantic decoder ranks the candidate's
> post-action remote corridor inside fixed camera-projective support better than
> persistence, shuffled-action, wrong-RGB, and RGB-independent action-prior
> controls?

A PASS is development evidence for a planner-useful action-conditioned rollout
candidate. It does not establish a JEPA treatment effect before the matched
no-JEPA arm, and it does not establish persistent mapping, immediate collision
avoidance, route selection,
closed-loop navigation, G2, held-out generalization, or deployment readiness.

## Why this is the next materially different test

Closed predecessors optimized latent reconstruction, nine-way action identity,
rigid transport, action queries, event modes, history discrimination, or posterior
expert assignment. Some learned action effects, but none required the decoded JEPA
rollout to improve a planner-facing spatial decision.

The existing geometry-anchored model has enough machinery already:

- an N320-initialized RGB encoder;
- a local geometry-anchored 64-channel BEV lift;
- a shared UNKNOWN/FREE/OCCUPIED semantic decoder;
- a nine-action local latent predictor; and
- a stop-gradient EMA target.

No new neural parameters are authorized. The material change is the training and
evaluation signal: all nine predicted next-view latents are decoded through the
same semantic head and scored against frozen development-only remote-corridor
geometry.

## Projective-support correction

The fixed camera projection makes only 1,964 of 4,096 BEV cells image-dependent.
The first learned ground cell is at +0.95 m. The longest primitive block translates
only 0.15 m and the canonical footprint radius is about 0.462 m, so an immediate
one-step swept-footprint mask has zero overlap with learned semantic support.

Therefore this experiment must not claim that this decoder predicts immediate
primitive collision. It scores a disconnected remote corridor whose first
footprint station is 1.45 m ahead of the predicted next pose. At that distance the
complete canonical footprint mask lies inside fixed projective support; this says
nothing about occlusion visibility. Near-field safety remains a later
responsibility of reversible persistent memory plus a deterministic veto over
learned evidence.

Any implementation that scores the immediate one-step footprint through the
forced-UNKNOWN region violates this preregistration.

## Fixed inputs and custody

Model-facing inputs are limited to:

- current RGB;
- next RGB only as a training target observation; and
- one of the nine frozen candidate action identities.

Pose, scene geometry, labels, maps, goal vectors, simulator state, future RGB, and
evaluator feedback are forbidden inference inputs.

The evaluation-only wrong-RGB diagnostic is the sole exception to using the
original row's current RGB: it substitutes one other role-local, same-scene
current endpoint as the model's only RGB input. It is never a future observation
or a side input, and its complete derangement is hash-frozen before any RGB or
model access.

The only approved learned initialization is the 78-tensor `encoder.*` state from:

- `.generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/checkpoint.pt`
- bytes: `13,777,100`
- file SHA-256: `ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0`
- tensor-content SHA-256: `9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b`

The lift, semantic head, predictor, EMA target, optimizer, RNG, observations, and
all runtime state are fresh. No rejected checkpoint, head, predictor, optimizer,
trace, tensor, or runtime output may be opened or reused.

Development raw-supervision bindings are:

- manifest: `e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360`
- pairs: `5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d`
- endpoints: `34e47ddcc40ad8c1f092c73193d16773cf4dedae05e7f4f684abb385cc2c0d01`
- audit V13: `0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76`
- presentation schedule: `08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270`

Only the 5,172 development pairs and their 88 development scenes may be used.
G2, navigation, held-out, sealed, production, or legacy runtime outputs must not
be listed, opened, indexed, or inferred.

The frozen action order is exactly:

`arc_left, arc_right, backward, forward_fast, forward_medium, forward_slow,
hold, yaw_left, yaw_right`.

Other lateral, event, or recovery entries present in the registry are excluded.

## Development-only counterfactual label adapter

Labels are derived and materialized once in a separate pre-GPU, model-independent
preflight without RGB decoding. The compact output is split into hash-bound train,
probability-calibration, and checkpoint-selection files. Geometry for all three
roles is necessarily opened during this one source scan, but no model exists and
no model-dependent score is computed. The training runner later consumes the
three frozen label files in the role order below. Labels never enter the model
call graph.

Before this preflight starts, its execution binding must enumerate every allowed
per-scene source record from the raw manifest as exact path, byte count, file
SHA-256, scene, family, role, and purpose. Only the named `frames.jsonl`, render
summary, and scene manifest records may be opened; directory walking or discovery
is forbidden. The three role files and their manifest are written exclusively as
canonical, content-hashed artifacts and become immutable training inputs.

For every unique current endpoint in train, probability-calibration, and
checkpoint-selection roles:

1. Read its role, scene, family, endpoint identity, image path, and image hash from
   the bound development endpoint index.
2. Parse `(frame_index, env_index)` from the canonical rendered filename.
3. Use only the bound per-scene render summary to recover the unique timestamp and
   image commitment.
4. Select exactly one row from the bound source `frames.jsonl` by
   `(frame_index, env_index, timestamp_ns)`.
5. Reconstruct the frozen endpoint identity from role, scene, episode ID, episode
   step, frame index, environment index, timestamp, and image hash; its SHA-256
   must equal the endpoint identity.
6. Verify the canonical pair and endpoint content hashes and this exact join
   matrix: pair to source frame for episode ID, environment index, and reset
   count; endpoint filename to source frame for frame/environment index; endpoint
   plus render summary to source frame for timestamp and image SHA; pair
   `frames_jsonl_sha256` to the inventory-bound source-frame file SHA; the
   inventory scene-manifest file SHA to the bytes actually opened; pair
   `scene_manifest_sha256` to `manifest_sha256(parsed_manifest)`; and
   role/scene/family across pair,
   endpoint, and shard, plus parsed-manifest scene/family where defined.
7. Require the render-summary schema/status and its committed source-frame and
   scene-manifest paths and hashes to equal the raw-manifest inventory.
8. Read only the base position and yaw needed by the offline label adapter.
9. Hash-check and parse the bound scene manifest, geometry contract, directional
   footprint policy, and primitive registry.

The frozen geometry sources are:

- geometry V2: `e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52`
- geometry V2 content: `e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca`
- directional policy: `750d8afe47ee3edd5988cdea443f19703efad7a3266218932671b9fdfbe43828`
- directional-policy content: `c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc`
- directional-policy ID/profile: `go2-directional-observed-max-margin-v1` /
  `observed_max_plus_margin`
- directional geometry implementation:
  `lewm/planning/oriented_footprint.py` at file SHA-256
  `5831379e52eb0eaa1c2cf8d195b6d46b29ad8b66dbadc98f51629f22bc656b37`
- primitive registry: `cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8`

For each of the nine actions, the adapter integrates the fixed five-command
velocity block from the source pose. It then evaluates a straight remote corridor
along the candidate's resulting heading at eleven stations from 1.45 m through
3.45 m in exact 0.20 m increments. Candidate yaw means the recorded starting yaw
plus the nominally integrated candidate yaw; it is not claimed as an executed
measurement.

Integration starts at `(0,0,0)` in body coordinates and applies every registry
command in order as
`x' = x + (vx*cos(yaw)-vy*sin(yaw))*0.10`,
`y' = y + (vx*sin(yaw)+vy*cos(yaw))*0.10`, and
`yaw' = wrap_angle_pi(yaw+w*0.10)`, using Python float64 values. Each immediate
command segment and the blind bridge call the clean
`ManifestDirectionalFootprintFeasibility.interpolated_sweep` with keyword values
`maximum_corner_step_m=0.025` and
`maximum_yaw_step_rad=math.radians(5.0)`; repeated segment-boundary poses are
tested once. The blind-bridge end is exactly
`post_xy + 1.45*(cos(post_yaw),sin(post_yaw))` at unchanged yaw.

The straight remote intervals use an enumerated specialization that cannot vary
with floating-point `ceil`: station zero is the single predicted-next-frame pose
`(290/200,0,0)`. For interval `s=1..10`, the nine poses `k=0..8` are exactly
`((290+40*(s-1)+5*k)/200,0,0)`, evaluated in that order using Python float64
division. These are eight exact-real 0.025 m subintervals. The C-order little-
endian float64 `[91,3]` pose rows preceded by little-endian int64 offsets
`[0,1,10,19,28,37,46,55,64,73,82,91]` have SHA-256
`df96a4d23e9f2a297467c7384e54e9d7f8eac64609e937392f0db51e3c87abc3`.
World label poses are those local poses transformed by the nominal post-action
SE(2) pose. Station zero checks its single footprint; each later station checks
all nine sampled polygons. The blind bridge is recorded separately and is
deliberately excluded from the remote-corridor label and model score. It is
neither trained nor claimed here.

Labels test the frozen sampled directional polygons against manifest collision
boxes. Each model mask rasterizes the union of closed-cell intersections with
those exact same sampled polygons. Thus label and score share poses and polygons;
pose-only masks are forbidden. The label records projective-support
station/interval safety, remote-corridor safe-prefix length, separately reported
immediate-primitive and blind-bridge feasibility, colliding object IDs where
applicable, and exact provenance. No continuous-clearance claim is made.

Immediate primitive and blind-bridge feasibility are reported and may be used by a
later learned-map safety veto, but neither is predicted or claimed by the remote
corridor score in this probe. Primary paired utility is evaluated only on states
where every non-HOLD candidate's immediate primitive and blind bridge are
feasible. This all-candidates condition is fixed before scores exist and supplies
no candidate-specific oracle filter. Outside that subset, any selected infeasible
primitive is assigned utility `-1` and reported, but it cannot create positive
planner evidence.

`raster_labels.u1` remains observable physical camera evidence only. It may ground
the shared semantic representation, but it is not configuration occupancy and is
never described as body-safe supervision.

The label preflight must prove:

- exactly 5,172 current states and 46,548 action rows;
- exact per-role counts of train `4,262 / 38,358 / 72`, calibration
  `415 / 3,735 / 8`, and selection `495 / 4,455 / 8` for
  states/action-rows/scenes;
- exactly 512,028 action-station labels;
- every endpoint identity reconstructs exactly once;
- all source and geometry hashes match;
- train, calibration, and selection roles remain scene- and endpoint-disjoint;
- no G2, held-out, sealed, navigation, checkpoint, or RGB payload is opened;
- at least 512 informative train states and at least one eligible ranking pair for
  every non-HOLD action;
- at least 512 of the frozen 16,000 scheduled presentations are informative and
  every non-HOLD action participates in an eligible ordered ranking pair on at
  least 32 scheduled presentations; exact scheduled counts are recorded before
  RGB access;
- at least 128 informative calibration states and 128 informative selection states;
- every registered selection family has at least eight informative states;
- every role-local scene has at least two distinct current endpoint identities for
  the wrong-RGB derangement;
- each non-HOLD action and station has both safe and unsafe support in train and in
  the combined calibration-plus-selection population; and
- each informative state has all non-HOLD immediate primitives and blind bridges
  feasible, a positive oracle remote-corridor prefix, and at least two non-HOLD
  corridors with different remote-corridor prefix lengths.

The preflight also runs the frozen calibration, selection, utility, and bootstrap
implementation with oracle station probabilities equal to the exact labels. It
must attain precision/unsafe-recall/safe-recall `1.0/1.0/1.0`, normalized utility
`1.0`, and nonempty admissions in every family. Failure is a metric-pipeline STOP
before GPU use.

Failure is a pre-GPU data-contract STOP, not a model result. It permits only a new
materially justified preregistration; it does not consume the training attempt.
No architecture, loss, threshold, mask, gate, schedule, or population change may
be made from label values after this preflight. The only permitted preflight
decision is exact PASS or STOP against the requirements already written above.

## Frozen model and score path

The neural model is the unchanged
`GeometryAnchoredDeformableBevLiftJointJepaV1` architecture initialized as above.
For batch size `B`:

1. `current RGB -> online BEV latent z_t`;
2. `z_t + each candidate action -> nine predicted latents z_hat_(t+1,a)`;
3. reshape and decode all nine latents through the one shared semantic head;
4. convert UNKNOWN/FREE/OCCUPIED logits to per-cell FREE versus UNSAFE evidence;
5. aggregate UNSAFE evidence over the fixed eleven directional-footprint masks;
6. obtain eleven candidate corridor-safety probabilities and a calibrated safe
   prefix; and
7. select the non-HOLD action with the longest predicted safe prefix, using frozen
   vocabulary order for ties.

There is no direct action-affordance head, global pooled shortcut, pose input,
geometry input, label input, or action-only learned bypass. Predictor output must
causally change the candidate score.

The predicted-next corridor-mask stack is defined in predicted-next body frame at
yaw zero over the model's exact 64 x 64 lattice. A cell is included when its closed
0.10 m square intersects any directional polygon sample for that interval. Samples
are ordered station-major, then row-major; the materialized mask is C-contiguous
`uint8`. Station zero contains 49 cells, each later swept interval contains 61,
the stack contains 659 set cells, and its byte SHA-256 is
`63648c9c157d032db943b4dea5168879c287c847101606c56c97688f06e69da4`.
All 659 cells are inside the frozen 1,964-cell learned-support mask.

### Exact differentiable score

For semantic logits `(u, f, o)` at one cell, define FREE log-odds
`ell = f - logaddexp(u, o)`. For one interval mask with `N` cells, define its
safe logit as the temperature-eight smooth minimum
`m = -logsumexp(-8*ell_mask)/8 + log(N)/8`, and probability `p = sigmoid(m)`.

For candidate action `a`, the differentiable prefix utility used by ranking is
`U_a = sum_s product_{j<=s}(p_(a,j)) / 11`. The binary target utility uses the
same formula with zero/one station labels. For every ordered pair of non-HOLD
actions whose target utility is strictly larger for the first action, the ranking
term is `softplus(-8*(U_better-U_worse))/log(2)`. `R` is the equal-row mean over
eligible pairs, then the mean over rows that have at least one eligible pair.
Rows without a pair are excluded from that mean. If an entire microbatch has no
eligible row, `R` is the exact differentiable zero `0*sum(U)`. `R` is explicitly a
sparse auxiliary; dense all-row/all-action supervision comes from `Q`. The number
and fraction of microbatches with active `R`, plus eligible rows and pairs, are
recorded as diagnostics and are not promoted into an extra post-hoc gate.

`Q` is `binary_cross_entropy_with_logits(m, station_safe)` averaged equally over
rows, all nine actions, and all eleven intervals, then divided by `log(2)`.

At evaluation threshold `t`, a candidate's discrete remote-corridor safe-prefix
is the number of consecutive probabilities from station zero satisfying `p>=t`,
stopping at the first rejection. An empty prefix scores zero. The non-HOLD action
with the largest prefix is selected; ties use the frozen action order. States with
zero oracle prefix are excluded only from normalized-utility and paired-control
statistics, are counted and reported, and remain included in station calibration.
The oracle action is the non-HOLD action with the longest binary target prefix,
using the same action-order tie break. On the primary subset utility is selected
target prefix divided by oracle target prefix; elsewhere a selected action whose
immediate primitive or blind bridge is infeasible receives utility `-1`.

## Joint objective

All trainable online components update together from update 1 through update 1000.
There is no separately trained or post-hoc predictor.

The fixed loss is:

- `S`: equal-row, equal-present-class current/next semantic NLL, divided by
  `log(3)`;
- `P`: executed-action per-cell channel-normalized Smooth-L1 JEPA energy to the
  stop-gradient EMA next latent, divided by the detached same-microbatch EMA
  persistence energy defined below;
- `Q`: equal-action, equal-station binary corridor loss through the decoded
  semantic masks, divided by `log(2)`; and
- `R`: equal-row pairwise remote-prefix ranking loss over eligible action pairs,
  divided by `log(2)`.

`L = S + P + Q + R` with no annealing, adaptive weighting, focal term, class
weight, auxiliary head, or post-start change.

Let `e(x,y)` be the unchanged model helper `latent_energy_per_row`: move the 64
channels last, independently LayerNorm each cell across channels, apply beta-one
Smooth-L1, and mean over channel and the 64 x 64 cells. For each size-four
microbatch `mu`, compute stop-gradient EMA latents `zbar_t` and `zbar_next`, then
`b_mu = clamp(mean_i(e(zbar_t_i,zbar_next_i)).detach(), min=1e-6)` and
`P_mu = mean_i(e(zhat_executed_i,zbar_next_i))/b_mu`. No quantity is pooled
across microbatches. The update loss is the arithmetic mean of the four
microbatch losses.

The online encoder, lift, semantic head, and predictor receive finite nonzero joint
gradients. The EMA target is detached, has no optimizer membership, and updates
once after every successful online optimizer step with momentum 0.996.

Optimizer groups and clipping remain those of the reviewed geometry model runner:

- encoder AdamW learning rate `1e-4`;
- lift plus semantic head AdamW learning rate `3e-4`;
- predictor AdamW learning rate `3e-4`;
- betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`;
- one L2 clip at `1.0` over encoder+lift+semantic parameters and one L2 clip at
  `1.0` over predictor parameters; and
- no combined global clip or optimizer rebuild.

## Fixed execution

- experiment/training RNG seed: `20260728`;
- unchanged fresh-component constructor seed: `20260712`;
- unchanged frozen schedule seed: `20260713`;
- paired-bootstrap seed: `20260728`;
- one attempt, no retry or resume;
- exactly 1,000 updates;
- effective batch size 16 from four microbatches of four;
- exactly 16,000 train-pair presentations from the frozen schedule prefix;
- exactly four separately built joint microbatch graphs and four backward calls,
  followed by one optimizer step, per update;
- fixed final checkpoint at update 1000; no checkpoint search;
- train RGB/raster/counterfactual labels only before model freeze;
- probability-calibration RGB/raster/counterfactual labels only after update 1000
  freeze;
- checkpoint-selection RGB/raster/counterfactual labels opened last for the final
  development decision; and
- G2, navigation, held-out, sealed, and production access counts remain zero.

Update 0 is structural/synthetic only. Before update 1000, only nonfinite state,
zero intended gradients, target leakage, or exact accounting failure may stop the
attempt. There is no discretionary early scientific or collapse gate. Scientific
PASS or FAIL is decided only after the fixed update-1000 model is calibrated and
scored.

## Frozen controls and calibration

All controls use the identical semantic decoder, corridor geometry, closed-cell
raster rule, station aggregation, calibration algorithm, candidate set, and
tie-breaking:

- full: action-conditioned predicted next latent;
- coordinate-matched persistence: for every sampled predicted-next polygon point
  `q_next`, compute `q_current = t_action + R(yaw_action)*q_next` using the exact
  nominal five-command integration above, transform the polygon yaw by the same
  `yaw_action`, and rerasterize the transformed polygons onto the current-frame
  lattice with the same closed-cell SAT rule; score the current semantic field
  with that mask. Discrete masks or logits are never warped or grid-sampled;
- shuffled action: fixed forward cyclic derangement
  `(predicted slot for a) = (a+1) mod 9` while retaining the original candidate
  label and mask;
- wrong RGB: within each role and scene, sort unique current endpoint identities
  lexicographically, map each identity to the next identity with last-to-first
  wrap and no skip, and give all nine rows for that state the mapped endpoint's
  RGB; then run the unchanged full predictor and original candidate actions. The
  canonical mapping rows `(role,scene,current_endpoint,wrong_endpoint)`, sorted in
  that tuple order, are content-hashed in the execution binding before RGB/model
  access; and
- action prior: train-role-only per-action/per-station safe frequency with no RGB.

The persistence masks are C-contiguous `uint8 [11,64,64]` in the frozen action
order. Their per-station counts and byte hashes are:

| action | station counts | total | SHA-256 |
|---|---|---:|---|
| `arc_left` | `49,64,63,64,64,64,63,62,64,60,64` | 681 | `ea6e49053b653dd84250647f6ca51d5aa929df7cf84a214203a6c5822f186740` |
| `arc_right` | `50,63,61,63,62,66,62,64,63,64,63` | 681 | `77bf4e01900e559387a11f36a2c66a9859caee93c139032bb7e74c2296f3a1c2` |
| `backward` | `49,61,61,61,61,61,61,61,61,61,61` | 659 | `dfc0aeac0f6f8b44a8e37c7eac16dcfbd06ee98a7e1e3bf308f78413a472b08f` |
| `forward_fast` | `51,63,63,63,63,63,63,63,63,63,63` | 681 | `17a8e0b66a03c8d0210a7b0bf1665daa71ba8d355df2d344d5bf06feb3f6f773` |
| `forward_medium` | `49,61,61,61,61,61,61,61,61,61,61` | 659 | `4b78889928776d40f0c344d37dd942f91356da333b5a98ebb843fc966bb617d9` |
| `forward_slow` | `49,61,61,61,61,61,61,61,61,61,61` | 659 | `f651df5fead03d200477f1bfc418f17ed3bd613918c77a7615d65fbfdc75853f` |
| `hold` | `49,61,61,61,61,61,61,61,61,61,61` | 659 | `63648c9c157d032db943b4dea5168879c287c847101606c56c97688f06e69da4` |
| `yaw_left` | `50,64,63,63,65,63,66,60,65,61,63` | 683 | `bcba50e628bd4557840db74e2e47b9a0513d5bd0b454cd3c863d4883e1d1e6f2` |
| `yaw_right` | `49,63,62,66,62,64,62,64,63,62,61` | 678 | `c91dc19501891039bee3d3b9a536de655a243f9b3e4e74a88e9d9da2888f180f` |

The C-order concatenated `[9,11,64,64]` stack has 6,040 set cells, zero outside
the 1,964-cell projective-support mask, and SHA-256
`983577015f2822bbf60d89cd633baa9958afd624410e1a3e4390422647e59e34`.

These are inference controls. They do not replace the later mandatory matched
no-JEPA training arm.

For each arm, enumerate finite unique probability-calibration scores plus exact
thresholds `0.0` and `nextafter(max_score,+infinity)`. Discard thresholds with zero
admitted stations. Among thresholds
with safe-station precision at least 0.99, choose maximum safe recall, then maximum
unsafe recall, then maximum precision, then the numerically highest (most
conservative) threshold. Absence of an eligible threshold is an arm FAIL, never a
vacuous precision PASS. If this occurs for the full arm or any control, the whole
probe is terminally non-comparable and cannot PASS; there is no fallback threshold
or retry. ECE uses bins `[i/15,(i+1)/15)` for `i=0..13` and `[14/15,1]` for the last
bin, weighted by exact row count; empty bins contribute zero. Thresholds,
admission counts, Brier score, ECE, precision, unsafe recall, safe recall, and
per-family counts are stored before the checkpoint-selection role is opened.

## Final conjunctive PASS gates

All of the following must pass on the last-opened checkpoint-selection role:

### Integrity and representation

- exact source, input, role, schedule, seed, update, presentation, optimizer, EMA,
  and access accounting;
- all outputs and gradients finite;
- target gradients and optimizer membership exactly zero;
- intended online-group gradient nonzero at every update;
- training predictor-forward, predictor-objective, and backward counts all equal
  4,000 microbatches, while predictor optimizer-update count equals 1,000;
- forbidden input/bypass/open counts zero;
- current and paired RGB latents nonconstant and nonidentical; and
- the one-step footprint zero-support regression is witnessed while every frozen
  corridor mask is nonempty and wholly inside learned semantic support.

### Observable semantic retention

- three-class balanced accuracy at least `0.80`;
- FREE recall at least `0.85`;
- OCCUPIED recall at least `0.70`;
- UNKNOWN recall at least `0.90`; and
- rough-family OCCUPIED recall at least `0.65`.

### Calibrated projective-support corridor utility

- admitted station safety precision at least `0.99`;
- unsafe-station recall at least `0.95`;
- useful safe-station recall at least `0.90`;
- every family has nonzero admitted and safe station denominators, at least 32
  admitted stations, safe recall at least `0.70`, and unsafe recall at least
  `0.70`;
- mean selected non-HOLD action actual remote-prefix divided by oracle
  remote-prefix at least `0.90` on informative states;
- at least `0.90` of informative states overall and `0.80` in every family receive
  a nonempty selected prefix;
- selected-action admitted-station precision at least `0.99` overall and at least
  `0.95` in every registered family;
- the selected-action distribution's largest share is no more than the oracle
  largest share plus `0.20`; and
- full-arm remote-prefix utility beats coordinate-matched persistence, shuffled
  action, wrong RGB, and action prior with a paired scene-cluster bootstrap 95%
  lower bound strictly above zero and a positive family margin in at least six of
  eight registered families.

The normalized-utility, nonempty-prefix, selected-action-precision, action-share,
and paired-control gates all use the same informative, all-candidates-immediate-
and-blind-feasible primary subset defined before scores exist. Model probabilities
choose each action, but every reported selected/oracle prefix and normalized
utility uses the corresponding binary target labels, never predicted prefix as
ground truth. The separately reported outside-subset utility rule does not enter
these PASS gates.

Statistics use equal-weight scene means. The paired bootstrap draws exactly eight
selection scenes with replacement, preserves full/control pairing, computes the
equal-scene mean delta, and repeats exactly 10,000 times with seed `20260728`. The
95% lower bound is sorted replicate index 249 (zero based). Deterministic
evaluations are not counted as independent evidence. All aggregate and per-family
results are recorded, including failures.

## Decision and next authority

The first failed applicable conjunct closes this exact mechanism. Numerical or
integrity failure after training begins is terminal for this attempt. No retry,
resume, second seed, threshold retune, loss retune, schedule extension, or same-root
reuse is allowed.

A PASS authorizes only:

1. a separately preregistered matched no-JEPA development training arm;
2. complete canonical physical perception and calibration requalification because
   the encoder changed; and
3. if both pass, integration with reversible persistent belief and learned
   navigation scoring before any G2 request.

A PASS does not authorize checkpoint reopening by default, G2, navigation,
held-out, sealed, production, promotion, deployment, or a final JEPA-navigation
claim.
