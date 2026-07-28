# Go2 main-pool action/frame alignment audit — 2026-07-28

## Superseding causal correction

- This section supersedes the original no-rebuild/factorized-next decision
  recorded with this audit in commit
  `dab0e31dd25194b07efb9a542f3d677bb62201c9`. The measurements, populations,
  and numerical results below are unchanged; their candidate-planning
  interpretation was incomplete.
- Write `F(i,j)` for the `j`th post-request frame of primitive block `i`.
  V1 labels the interval `F(i,1) -> F(i+1,1)` with requested primitive `p_i`.
  That interval begins after `p_i` has already acted for one tick and ends
  after the unseen destination primitive `p_(i+1)` has acted for one tick.
- High separability of `p_i` over that mixed interval establishes correlation,
  but not a deployable candidate transition. At planning time the model must
  receive a state from before candidate `p_i` begins and may not target a
  state containing any effect of an action it was not given.
- The candidate-valid edge is therefore the same-episode boundary transition
  `F(i-1,5) -> F(i,5)`, labeled by requested primitive `p_i`. Six consecutive
  primitives require seven such shared boundaries. Rows without a real
  same-episode predecessor boundary must be filtered and deterministically
  backfilled; they must not be synthesized.
- Consequently, the V1 schedule is not valid evidence for candidate-planning
  action conditioning. Its completed model run remains a terminal V1 STOP,
  but one mechanism-identical, schedule-corrected V2 integrity replacement
  must precede any factorized conditional-increment model. The factorized
  proposal is a contingent fallback only if that corrected replacement still
  fails its unchanged action/history gates.

## Scope and custody

- This was a read-only semantic audit of the exact active development
  schedules after the factual shared-transition probe stopped on action,
  history, and all-hold gates.
- Inputs were limited to the bound train index SHA-256
  `f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b`
  (16,000 rows / 1,000 scenes), validation index SHA-256
  `86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880`
  (2,048 rows / 150 disjoint scenes), and the exact allowlisted
  `frames.jsonl` sources under `.generated/datagen_full/rollout/train` and
  `.generated/datagen_full/rollout/val` for those 1,150 scenes.
- The audit evaluated 96,000 train and 12,288 validation action edges and
  parsed 251,734 selected endpoint/neighbor metadata rows. It decoded zero RGB
  pixels and opened no checkpoint, tensor, training trace, label shard,
  navigation input, test, held-out, sealed, or other-role material.
- The frozen V1 adapter source was
  `lewm/datasets/go2_recurrent_h4_rgb_sequences.py`, SHA-256
  `3f8c2a89af2934e8225dd98447b952d9e5ce8bedac99a7f834118263957652e6`,
  27,386 bytes.

## Source contract

- Every selected metadata row used one uniform schema. Relevant fields were:
  - `base_pose_world.position.{x,y,z}` and orientation `wxyz`;
  - `base_quat_world_xyzw` and `base_rpy_rad.{roll,pitch,yaw}`;
  - `twist_body.linear/angular.{x,y,z}`;
  - command primitive, sequence ID, request timestamp, block size, command
    timestep, requested `vx/vy/yaw` arrays, source, and route context.
- All 108,288 index action IDs exactly matched the source frame's primitive
  context. All requested arrays matched the nine-primitive registry. Timing,
  context, schema, and action-ID mismatch counts were zero.
- A command is requested at time `t`; its five post-command BaseState frames
  occur at `t+0.1` through `t+0.5` seconds. V1 constructs an edge from the
  first frame of one run to the first frame of the next run. Each nominal
  0.5-second edge therefore contains 0.4 seconds under the indexed/current
  request plus 0.1 seconds under the destination request.
- `frames.jsonl` attaches the latest requested `command_block`, not the later
  `executed_command_block`. It does not carry executed/clipped arrays or the
  clipping flag. Exact corpus-wide requested/executed joining would require a
  separate scan of the hundreds-of-GB raw message streams and was not
  performed here.

## Requested-action separability

Realized body-frame `(dx,dy,dyaw)` was computed from endpoint poses. A
standardized nearest-class-centroid diagnostic was fitted on train scenes and
evaluated without fitting on the disjoint validation scenes. Balanced chance
for nine primitives is `0.111111`.

| Motion window labeled by | Accuracy | Balanced accuracy | Balanced standardized eta-squared |
|---|---:|---:|---:|
| V1 mixed 0.5 s, indexed/current primitive | 0.532715 | 0.479329 | 0.359102 |
| V1 mixed 0.5 s, destination primitive | 0.3447 | 0.2899 | 0.1171 |
| V1 mixed 0.5 s, previous primitive | 0.3395 | 0.2806 | 0.1238 |
| Pure within-current-block 0.4 s | 0.519613 | 0.467984 | 0.349227 |
| Corrected boundary-to-boundary current 0.5 s | 0.505691 | 0.452270 | 0.340344 |

- The current primitive is unequivocally the strongest lag despite the
  request-time mixture. There is no catastrophic one-step action-label shift.
- For the corrected boundary-to-boundary calculation, 620 of 96,000 train
  edges and 75 of 12,288 validation edges were excluded because their episode
  began without a real same-episode pre-command boundary. The retained counts
  were 95,380 train and 12,213 validation edges.
- Requested forward/yaw command to realized-rate correlations were:

| Window | Train forward | Val forward | Train yaw | Val yaw |
|---|---:|---:|---:|---:|
| V1 mixed 0.5 s | 0.652912 | 0.627939 | 0.771798 | 0.760251 |
| Pure within-block 0.4 s | 0.647963 | 0.622216 | 0.764126 | 0.750474 |
| Corrected boundary 0.5 s | 0.631695 | 0.607116 | 0.763810 | 0.754515 |

- Validation realized body-motion variance explained by the current command was
  `R^2=0.5528`, versus `0.1777` for the destination command and `0.5532` for
  both together. Requested lateral velocity was zero for every primitive, so
  its correlation is undefined.

## Per-primitive findings

Validation recall under the strongest V1 current-lag diagnostic was:

| Primitive | Recall |
|---|---:|
| `arc_left` | 0.466 |
| `arc_right` | 0.545 |
| `backward` | 0.762 |
| `forward_fast` | 0.037 |
| `forward_medium` | 0.509 |
| `forward_slow` | 0.140 |
| `hold` | 0.622 |
| `yaw_left` | 0.625 |
| `yaw_right` | 0.609 |

- Directional/yaw primitives and hold are strongly above chance. The three
  forward-speed classes overlap heavily: realized 0.5-second train `dx` was
  `0.077 +/- 0.068 m` for fast, `0.092 +/- 0.058 m` for medium, and
  `0.049 +/- 0.046 m` for slow. Medium exceeded fast on average.
- Other train mean `dx +/- SD` and `dyaw +/- SD` values were:
  - arc left: `+0.068 +/- 0.055 m`, `+0.091 +/- 0.081 rad`;
  - arc right: `+0.067 +/- 0.051 m`, `-0.173 +/- 0.093 rad`;
  - backward: `-0.052 +/- 0.034 m`, `+0.031 +/- 0.075 rad`;
  - hold: `+0.013 +/- 0.025 m`, `+0.003 +/- 0.059 rad`;
  - yaw left: `+0.016 +/- 0.026 m`, `+0.160 +/- 0.109 rad`;
  - yaw right: `+0.013 +/- 0.025 m`, `-0.185 +/- 0.109 rad`.
- High within-class variance is consistent with contacts, obstacles,
  controller limits, body inertia, and differing incoming motion. It is a
  genuine partially observed dynamics problem, not evidence that requested
  actions carry no physical signal.

## Decision

- The H6 V1 adapter has a real request-time endpoint impurity: 0.4 seconds of
  current request plus 0.1 seconds of destination request. Its raw timing and
  action IDs are otherwise correct.
- The original inference that lower action separability made a boundary
  replacement scientifically inappropriate is withdrawn. Actuator/body lag
  explains why the mixed V1 interval is more class-separable, but it cannot
  authorize conditioning on one already-observed action tick or targeting one
  unseen destination-action tick.
- Use exactly one reset-safe schedule-integrity replacement whose edge for
  requested `p_i` is `F(i-1,5) -> F(i,5)`. Preserve the V1 model, accepted
  encoder, seed, objective, optimizer, thresholds, observations, and
  1,000-update/16,000-presentation cap. Only the endpoint/index schema,
  deterministic same-seed quota backfill, schedule hashes, output identity,
  and receipt identity may change.
- Do not expose executed/clipped future commands to the world model. A
  navigation policy knows requested actions, not future post-controller
  outcomes. The RGB/action state must learn incoming velocity, controller lag,
  contacts, and action response from ordered visual/action history.
- The data contains requested-action signal: validation balanced separability
  is `0.4793` on the V1 mixed interval and `0.4523` on the corrected boundary
  interval, both far above `1/9` chance. Those diagnostics support attempting
  the corrected factual model, but they do not rescue V1's causal alignment.
- If and only if the one mechanism-identical, schedule-corrected V2
  replacement stops on the unchanged action/history gates, a separately
  preregistered factorized conditional-increment H4 JEPA may be reconsidered.
  No data-scale run, transition-depth tweak, trained corruption margin, retry,
  or V1 checkpoint reuse is justified before that result.
