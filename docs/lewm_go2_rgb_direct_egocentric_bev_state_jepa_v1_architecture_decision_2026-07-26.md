# RGB direct egocentric BEV-state JEPA V1 architecture decision

Date: 2026-07-26

Status: **source-only PREREGISTERED; pending implementation, source closure,
independent review, and separate one-attempt execution authorization**.

This record authorizes only source implementation, focused tests, source
closure manifest construction, and independent review. It grants no
generated-input, RGB, label-array, checkpoint, runtime-artifact, GPU,
training, evaluation, G2, navigation, held-out, sealed, promotion, deployment,
or production authority.

## Decision

The next and only selected perception mechanism is a direct egocentric
BEV-state JEPA learned from RGB:

```text
current 112x112 RGB
    -> patch-7 RGB encoder
    -> learned global-cross-attention 64x64 BEV decoder
    -> 1x1 UNKNOWN/FREE/OCCUPIED state head
    -> current three-logit BEV state

current three-logit BEV state + nine-way one-hot executed action
    -> learned residual transition with no warp or geometry input
    -> predicted next three-logit BEV state
```

The three UNKNOWN/FREE/OCCUPIED logits, and their probabilities, are the sole
JEPA, predictor, and later navigation state. The decoder's hidden 64-channel
features may not bypass the three-logit bottleneck. During training only, a
detached EMA copy produces the target states used by the JEPA objective.
Raster labels ground online outputs but never enter an encoder or the
transition.

The complete RGB-to-state call graph is frozen as follows. `O` is the one
weight-shared online encoder+decoder+state-head stack and `T` is its detached
EMA copy.

- `O(current_rgb)` supplies `G_current` and is the sole learned-state input to
  the causal predictor.
- `O(next_rgb)` is a training-only grounding call. It supplies `G_next` only;
  gradients may reach `O` only through `G_next`.
- `T(next_rgb)` supplies the detached true-next target to `J` and `C`.
- `T(current_rgb)` supplies the detached current-target negative to `C` on
  non-hold rows only.
- `T(fixed_negative_rgb)` supplies the detached mapped-negative target to `C`.
- `O(fixed_negative_rgb)` is an observation-time, no-gradient diagnostic call
  used only by the wrong-RGB grounding control.

No output or hidden feature from the `O(next_rgb)` grounding call or
`O(fixed_negative_rgb)` diagnostic call may feed `J`, `C`, the transition,
prediction, action retrieval, optimizer loss other than `G_next`, later
navigation, or deployment. The current-RGB-only declaration describes the
causal prediction and inference path. Next RGB, fixed-negative RGB, and hard
labels are unavailable to that path and to later navigation.

This is a change to the learned state and transition contract, not another
action-residual timing, loss-weight, schedule, seed, or encoder-tokenization
successor.

## Governing predecessor result

The committed V13 terminal audit is bound as follows:

- path:
  `docs/lewm_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_learning_curve_continuation_terminal_audit_2026-07-26.json`;
- commit: `e03f6eb2dbfadad188e2cb07d5451096b4179969`;
- file SHA-256:
  `1486a102b010d06dc8b8a91130eb6c79a95d9a8ca426dea9a4833fc4aee488d8`;
- content SHA-256:
  `eea443eca48a6cc85bd054f93fb38c94f3d7e6fd4050cd15f442406ffa09b28e`;
- byte count: `30116`; and
- classification:
  `VALID_SCIENTIFIC_GATE_FAILURE_AT_UPDATE_400_CLOSES_EXACT_V13_LEARNING_CURVE_AND_ALL_MASKED_PAIR_TUBELET_TIMING_SUCCESSORS`.

V13 validly reached 400 updates and 6,400 presentations. Generic visual rank
and same-action target retrieval improved, but action retrieval stayed near
chance, masked-future loss worsened, and shuffle/mean controls failed. The
remaining falsification target is therefore the action-relevant state and
transition contract.

## Frozen architecture

### Online state

- RGB resolution: `112x112`; patch size: `7`.
- Encoder: dimension `192`, depth `6`, heads `6`, MLP ratio `4`.
- Decoder: learned global cross-attention, internal dimension `64`, producing
  a `64x64` egocentric BEV over forward cell centers `[-0.95, 5.35]` metres
  and left cell centers `[-3.15, 3.15]` metres, both at `0.10` metre spacing.
- State head: one `1x1` projection to exactly three logits per cell in
  UNKNOWN/FREE/OCCUPIED order.
- No hidden-state, token, skip, Camera, rasterizer, or auxiliary-feature
  bypass around the three-logit state is allowed.

### Learned transition

The transition reuses the `BevResidualPredictor` shape with `bev_dim=3`,
`action_dim=9`, and `hidden_dim=128`. Its only causal inputs are the current
three-logit BEV and the nine-way one-hot executed action. The predictor's
legacy three-vector slot is constructed internally as an exact-zero vector;
it is not a runtime input. The transition never calls a warp and accepts no
commanded or realized pose or delta, odometry, attitude, camera calibration,
map, goal, clearance, swept path, or other geometry.

The final residual layer is initialized to exact zero. Update 0 must therefore
be exact persistence and action-symmetric before any optimizer update.

### Target and initialization

The EMA target copies exactly the encoder, decoder, and state head. It is hard
synchronized once at initialization, then updated with decay `0.996` after
each optimizer update. The predictor has no target copy.

Only the encoder is migrated from the original bound N320 encoder state. All
decoder, state-head, and predictor parameters receive one deterministic fresh
initialization with seed `20260712`; no V1-V13 runtime checkpoint or trace is
reused.

## Frozen data and loader boundary

The narrow `_row_array` loader may read only current RGB, next RGB, fixed
negative RGB, and `raster_labels.u1`. It may not invoke a general frame loader
or read any other supervision array. Raster labels are `uint8`, shape
`64x64`, with UNKNOWN=`0`, FREE=`1`, and OCCUPIED=`2` under schema
`lewm_go2_observable_camera_ray_raster_v4`.

The existing `target_raster_labels` endpoint view grounds both bound current
and next endpoints. Raw-V13 remains unchanged: exactly 4,262 train pairs from
72 scenes and 495 checkpoint-selection pairs from eight scenes, with its
existing endpoint mappings, fixed negatives, row ordering, action
permutation, family binding, and schedule. The frozen mapped-negative binding
is same-action eligible for 4,237/4,262 train rows and 494/495 selection rows.
Non-singleton `(scene, primitive)` groups map cyclically to the next row and
therefore preserve action. For a singleton group, the already-frozen rule maps
to the next row in the complete scene sorted by `content_sha256`; the remaining
25 train rows and one selection row are consequently same-scene fallbacks not
guaranteed to preserve action. Those fallback rows are included in training
`C` and in the wrong-RGB diagnostic exactly like every other row, but are
excluded from metrics named same-action. No rendering, relabeling, filtering,
resampling, new rows, role changes, or dataset changes are allowed. The
accompanying JSON preregistration binds every source hash.

## Frozen objective

Let the state probabilities be the softmax of the direct three logits. There
is no Camera head, ray/depth/ground objective, rasterizer, equivariance loss,
warp loss, variance loss, or auxiliary loss.

`G` is the mean of `G_current` from `O(current_rgb)` and `G_next` from the
separate weight-shared `O(next_rgb)` call. Each hard-label grounding loss uses
exact `hierarchical_raster_cross_entropy_v4` reduction on the direct
probabilities: balanced OCCUPIED-vs-rest BCE and balanced FREE-vs-UNKNOWN BCE
restricted to non-occupied labels, combined `0.5/0.5`.

`J` applies the analogous soft-target hierarchy from the executed-action
predicted probabilities to detached EMA next-state probabilities: BCE against
detached target occupied probability plus target-nonoccupied-weighted BCE on
conditional free probability, combined `0.5/0.5`.

`C` is one joint conditional NCE. Its positive is the executed-action
prediction against `T(next_rgb)`. Its negatives are all eight wrong-action
predictions against that same detached target, the executed-action prediction
against `T(fixed_negative_rgb)`, and, for a non-hold row only, the
executed-action prediction against `T(current_rgb)`. The mapped-negative
candidate is same-action where the frozen mapping is eligible and is the
frozen deterministic fallback on the 26 singleton-group rows; all rows remain
in `C`. The candidate count is therefore exactly `10` for hold and `11` for
non-hold. Candidate energy is the same soft-target hierarchical energy used
by `J`. Each NCE logit is negative energy divided by the detached mean
candidate energy for that row, clamped below at `1e-6`.

At each registered observation, the wrong-RGB control evaluates
`O(next_rgb)` and `O(fixed_negative_rgb)` under no gradient against the same
true-next raster label. For each of the eight selection scenes, separately
average the exact hard-label hierarchical grounding loss over all its rows;
that scene is a strict correct-RGB win exactly when the correct mean is lower
than the mapped-negative mean. Same-action target NLL and strict-win metrics
use only the 494 eligible selection rows; aggregate `C` and the wrong-RGB
control retain all 495 rows.

Normalize `G` and `J` by `log(2)`. Normalize `C` per row by
`log(candidate_count)` before row averaging. The total is exactly:

```text
loss = 1 * G/log(2) + 1 * J/log(2) + 1 * normalized_C
```

## Optimizer and cap

- AdamW in float32; betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay
  `1e-4`.
- Encoder learning rate `1e-4`.
- New decoder, state-head, and predictor learning rate `3e-4`.
- Clip encoder+decoder+state-head jointly to norm `1.0`; clip predictor
  separately to norm `1.0`.
- Preserve the exact Raw-V13 batch-16 schedule, seeds, ordering, and bound
  schedule prefixes. One scheduled pair is one presentation.
- One attempt observed at updates `0`, `100`, `400`, and `1000`.
- Hard caps: `1000` updates, `16000` presentations, and `60` GPU-active
  minutes.
- No retry, resume, repair, recovery, replacement, second seed, or V2.

## Frozen staged gates

Every condition at a stage is conjunctive. Comparators are strict or
non-strict exactly as written. Stop permanently at the first failure; a later
stage may be reached only after every earlier stage passes.

### Observation populations and reductions

At every registered observation, `G` and `J` are the plain row means over all
495 checkpoint-selection pairs, using the same formulas and calls as the
training objective under no gradient. Action retrieval uses all 495 rows. Its
nine logits are the nine action-candidate negative energies divided by the
same detached row scale used by `C`; cross-entropy targets the executed action,
top-1 uses lowest-vocabulary-index tie breaking, and macro balanced accuracy is
the arithmetic mean of the nine per-executed-action recalls. For each row the
hardest wrong action is the minimum-energy one of the other eight. A selection
scene has a positive hardest-wrong margin exactly when the row mean of
`hardest_wrong_energy - executed_energy` is strictly positive.

Correct-vs-deranged target NLL uses only the 494 same-action-eligible rows and
plain two-logit cross-entropy over the correctly ordered pair
`[-correct_energy/row_scale, -deranged_energy/row_scale]`. Strict win means
`correct_energy < deranged_energy`. A selection scene has a positive target
margin exactly when the eligible-row mean of
`deranged_energy - correct_energy` is strictly positive. The one fallback row
is absent from these same-action metrics but remains in `C` and all-row
controls.

Raster quality follows the established checkpoint-selection endpoint
protocol. Within each family, deduplicate the current and next endpoint
identities, sort them, and evaluate no-gradient `O(endpoint_rgb)` exactly once
against that endpoint's own `target_raster_labels`. The aggregate scope pools
all 924 unique endpoints; their ordered identity SHA-256 is
`dd84fc73e14056c9d6c8f7c066c2dcafe9726827193c42982d51f412ea744fa4`.
The rough scope pools the 123 unique endpoints whose frozen family is
`rough_local_dynamics`. These endpoint metrics are separate from the 495-pair
wrong-RGB control.

For each raster scope, predicted class is argmax in UNKNOWN/FREE/OCCUPIED
order, retaining the lowest class index on a tie. Publish the integer `3x3`
confusion matrix with target rows and predicted columns. Each class recall is
the diagonal divided by its target-row count; balanced accuracy is the plain
mean of recalls for target classes present in that scope. Raster NLL is the
plain per-cell mean of negative log direct softmax probability assigned to the
hard label, clamped below by float32 machine epsilon. Free recall and occupied
recall are the corresponding class recalls. The endpoint count, family
membership, label identity, confusion counts, and NLL count must be recorded.

### Update 0: integrity only

- Prove the exact three-logit bottleneck and absence of every bypass.
- Prove exact persistence, action symmetry, and chance action metrics.
- Prove target isolation and nonzero gradients on the intended online path.

### Update 100: directional gate

- `G < G_at_update_0` and `J < J_at_update_0`.
- Action NLL `< log(9)` and macro balanced accuracy `> 1/9`.
- Correct-RGB grounding strictly beats fixed-wrong-RGB grounding in at least
  `6/8` checkpoint-selection scenes.
- Registered states, losses, controls, and gradients are finite and the state
  is nonconstant.

### Update 400: mature mechanism gate

- `G <= 0.90 * G_at_update_0` and `J <= 0.90 * J_at_update_0`.
- Action NLL `< 0.99 * log(9)` and macro balanced accuracy `> 0.15`.
- Executed action strictly beats the hardest wrong action in at least `4/8`
  checkpoint-selection scenes.
- Correct-vs-deranged target NLL `< 0.99 * log(2)` and strict-win rate
  `>= 0.60`.
- Correct-RGB grounding strictly beats fixed-wrong-RGB grounding in `8/8`
  checkpoint-selection scenes.

### Update 1000: capped perception gate

- Aggregate raster balanced accuracy `> 0.9009460724448773`, free recall
  `> 0.91637020862468`, occupied recall `> 0.8059679976935274`, and raster
  NLL `< 0.18704089070408247`.
- Rough-raster balanced accuracy `> 0.7719525130620232` and occupied recall
  `> 0.4319466882067851`.
- Correct-RGB grounding strictly beats fixed-wrong-RGB grounding in `8/8`
  checkpoint-selection scenes.
- Action NLL `< 0.95 * log(9)`, macro balanced accuracy `> 2/9`, and
  hardest-wrong positive result in at least `6/8` scenes.
- Target NLL `< 0.95 * log(2)`, strict-win rate `>= 0.65`, and positive result
  in at least `6/8` scenes.

Update 1000 remains a perception gate, not navigation, held-out, physical, or
production qualification.

## Exact distinction from predecessors

- **Phase 2Z:** privileged occupancy, clearance, goal, and swept-path inputs
  are absent. V1 infers its state from RGB and predicts it with a JEPA.
- **Camera V1-V6 and multiresolution/overlap:** ray hazard, depth offset,
  ground clear, fixed evidence rasterization, and Camera objectives are absent.
- **Shared-V5:** its Camera deployment path, zero-weight occupancy side head,
  commanded-pose warp, hidden BEV bypass, and Camera-dominated joint objective
  are absent. V1 transitions only the direct three-logit state with action.

## Terminal consequence and later authority

Any staged failure permanently closes Direct Egocentric BEV-State JEPA V1 and
produces a complete failure receipt. No V2 tweak, retry, or replacement is
implied or authorized.

A full pass makes the frozen checkpoint eligible only for separately
authorized physical requalification and a mandatory matched no-JEPA
development arm. Both require their own frozen source, review, and authority;
neither is authorized here. G2, navigation, held-out, sealed, promotion,
deployment, and production access remain prohibited.

Before this preregistered mechanism can run, its implementation, narrow
loader, metrics, failure receipts, focused tests, and source closure must be
committed, independently reviewed, hash-bound, and granted a separate
one-attempt authorization.
