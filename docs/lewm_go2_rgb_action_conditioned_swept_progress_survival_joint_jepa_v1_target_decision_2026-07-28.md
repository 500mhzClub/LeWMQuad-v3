# RGB action-conditioned swept-progress survival joint JEPA V1 target decision

Date: 2026-07-28

Status: target and model-free selection census preregistered; neural source work,
RGB access, GPU use, and training remain unauthorized until the census passes.

## Decision

Terminate the fixed 1.45--3.45 m projective-support corridor mechanism. Its
corrected target produced 165 informative checkpoint-selection states overall
but 0 in `small_enclosed_maze`, because every fixed 1.45 m blind bridge failed.
The terminal result is commit
`9ab13f5b86d19d620965b0ad74d9a5eaab471623`, file
`docs/lewm_go2_post_action_projective_support_selection_admissibility_census_v1_result_2026-07-28.md`,
3,743 bytes, SHA-256
`b5c1b444c5918cd499783f362eb2dc45ab40c19cae97983044235890963c96f0`.

Test one materially different target: action-conditioned collision-free swept
progress from the robot, with partial progress credited before first contact.
This preserves the RGB-only, jointly trained JEPA navigation goal without
lowering a family gate or moving the failed bridge.

## Frozen geometry target

Use the same exact eight non-HOLD actions, primitive commands, source poses,
directional footprint, scene manifests, and sweep interpolation as V4.

For each current state and candidate action:

1. Integrate and collision-check the exact five-command immediate primitive.
2. Express its nominal post-action pose in world coordinates.
3. From that pose and its resulting yaw, collision-check fifteen consecutive
   straight 0.10 m segments: `[0.0,0.1]`, ..., `[1.4,1.5]` m. Each segment uses
   `maximum_corner_step_m=0.025` and
   `maximum_yaw_step_rad=math.radians(5.0)` with the exact directional footprint.
4. The safe-progress prefix is zero when the immediate primitive is infeasible.
   Otherwise it is the number of consecutive safe continuation segments before
   first contact, in the closed integer range 0--15. No later segment can restore
   a broken prefix.

This target makes no remote 1.45 m admission requirement, no projective-mask
occupancy claim, and no oracle candidate filter. Geometry supplies development
labels only and is forbidden as a model input.

A state is informative iff the best non-HOLD prefix is positive and the eight
non-HOLD prefixes contain at least two distinct values.

## One model-free screen

Before RGB or model work, run one checkpoint-selection-only census over the exact
495 states, 4,455 action rows, eight scenes, and eight registered families bound
by the V4 development inputs. It may open only exact raw metadata, geometry, and
the eight selection source pose/scene records; it must not open the schedule,
other role sources, RGB/image bytes, label roots, models, checkpoints, GPU/runtime
outputs, navigation, G2, held-out, sealed, or production material.

PASS requires at least 128 informative states overall and at least eight in every
registered family. Failure closes this target before neural implementation. A
PASS only permits source implementation and a complete train/calibration/schedule
model-free preflight; it does not itself authorize training.

Before any GPU use, the complete preflight must retain the existing floors of
512 informative train states, 128 calibration states, 512 informative
presentations in the frozen 16,000-presentation schedule prefix, and at least 32
unequal-prefix ranking participations for every non-HOLD action.

## Neural mechanism if the target passes

The candidate starts fresh from the accepted N320 encoder initialization. The
existing geometry-anchored encoder/lift, local action predictor, semantic
retention path, and EMA next-observation JEPA target remain one joint graph.

A single shared survival head reads each action-predicted latent before the
forced-UNKNOWN semantic decoder. It uses fixed sweep-aligned per-progress spatial
pooling and emits one immediate-primitive logit plus fifteen conditional segment
logits. At-risk binary loss supervises a segment only while its preceding path is
safe. Predicted survival is monotone by cumulative products; the action score is
expected safe progress. All candidate actions receive labels on every row.

Encoder, lift, predictor, and survival head train together from update one in one
backward pass with semantic retention, executed-action EMA latent prediction,
survival, and progress-ranking losses. There is no head warm-up, frozen backbone,
detach, separate optimizer/training stage, privileged-state reconstruction, or
post-hoc selector fit.

This is distinct from the failed Phase 2V/2X shallow source-only RGB-to-teacher-
state distillation and from global mean-pooled scalar affordance heads. Global
mean pooling, the old factorized teacher state, hand-weighted safety selectors,
and a separately trained encoder or predictor are forbidden repeats.

Any later neural attempt remains capped at one fresh seed, 1,000 updates, and
16,000 presentations. It must retain action-only, wrong-RGB, shuffled-action,
persistence, and matched no-JEPA controls. A failed capped attempt closes direct
N320 swept-progress survival without bin, horizon, loss-weight, seed, or duration
variants.
