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
detached EMA copy sees the bound next RGB and produces the target next
three-logit state. Raster labels ground outputs but never enter an encoder or
the transition.

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
permutation, family binding, and schedule. No rendering, relabeling,
filtering, resampling, new rows, role changes, or dataset changes are allowed.
The accompanying JSON preregistration binds every source hash.

## Frozen objective

Let the state probabilities be the softmax of the direct three logits. There
is no Camera head, ray/depth/ground objective, rasterizer, equivariance loss,
warp loss, variance loss, or auxiliary loss.

`G` is the mean of the current and next hard-label grounding losses. Each
grounding loss uses exact `hierarchical_raster_cross_entropy_v4` reduction on
the direct probabilities: balanced OCCUPIED-vs-rest BCE and balanced
FREE-vs-UNKNOWN BCE restricted to non-occupied labels, combined `0.5/0.5`.

`J` applies the analogous soft-target hierarchy from the executed-action
predicted probabilities to detached EMA next-state probabilities: BCE against
detached target occupied probability plus target-nonoccupied-weighted BCE on
conditional free probability, combined `0.5/0.5`.

`C` is one joint conditional NCE. Its positive is the executed-action
prediction against the true detached next target. Its negatives are all eight
wrong-action predictions against that same true target, the executed-action
prediction against the bound fixed same-action deranged target, and, for a
non-hold row only, the executed-action prediction against the current target.
The candidate count is therefore exactly `10` for hold and `11` for non-hold.
Candidate energy is the same soft-target hierarchical energy used by `J`.
Each NCE logit is negative energy divided by the detached mean candidate
energy for that row, clamped below at `1e-6`.

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
