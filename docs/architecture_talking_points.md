# LeWMQuad-v2 — Comprehensive Architecture Talking Points

A PhD-level walkthrough of every major component in the repository: data
collection, rendering, world-model architecture and loss, planning heads,
inference, and the failure analysis that has shaped the present design.
Cross-references point at the live source so the audience can follow along
in the editor.

---

## 0. The setting in one paragraph

LeWMQuad-v2 is a paper-faithful re-implementation of Maes et al.'s **LeWorldModel**
(arXiv:2603.19312) instantiated on a quadruped (mini-pupper) maze-navigation task.
A learned, joint-embedding world model (JEPA, no pixel decoder, no proprio leakage)
is trained on egocentric images from a closed-loop physics simulator. At test time
a model-predictive CEM planner samples command-block primitives, rolls them
forward through the predictor, and scores rollouts with several **learned energy
heads** trained on cached latents — safety, goal, exploration, place,
displacement, coverage-gain, escape-frontier. A persistent perception-only
keyframe memory router escapes local plateaus. The repository emphasises
*audit-first* design: every planner decision can be logged, the report is
brutal about what does not yet work, and the runbook is sized for a single
RX 7900 XTX (≈3-day full pipeline).

The deliverable is a stack that is intentionally close to the LeWM paper:
ViT-Tiny encoder, transformer predictor with AdaLN-zero action conditioning,
SIGReg regulariser, dual BatchNorm projector, and a planner that consumes
*projected* latents. Most non-paper additions live downstream of the world
model — energy heads, primitive bank, keyframe router — and can be ablated.

---

## 1. The pipeline and run-book

The README ([README.md](README.md)) defines a six-step pipeline. The talking
points map one-to-one onto these steps.

1. [`scripts/1_physics_rollout.py`](scripts/1_physics_rollout.py) — Genesis
   physics rollouts produce `chunk_*.h5` files containing trajectories
   (proprio, commands, dones, scene metadata).
2. [`scripts/2_visual_renderer.py`](scripts/2_visual_renderer.py) — multi-
   process renderer replays the scenes deterministically and writes 224×224
   egocentric RGB into the same HDF5s under the `vision` dataset.
3. [`scripts/3_train_lewm.py`](scripts/3_train_lewm.py) — JEPA training of
   the LeWorldModel with SIGReg.
4. [`scripts/4_train_energy_head.py`](scripts/4_train_energy_head.py) —
   latent extraction and head training (10 phases including AR finetuning).
5. [`scripts/5_*`](scripts/) — auxiliary visualisation / debug.
6. [`scripts/6_infer_pure_wm.py`](scripts/6_infer_pure_wm.py) — closed-loop
   maze inference with the CEM planner.
7. [`scripts/7_aggregate_inference_runs.py`](scripts/7_aggregate_inference_runs.py)
   and [`scripts/aggregate_plan_audits.py`](scripts/aggregate_plan_audits.py)
   — aggregate metrics + per-replan audits into reports.

The repo bakes in the recommended **pure-perception regime**:
`seq_len=4`, `temporal_stride=5`, `command_representation=active_block`,
`action_block_size=5`, `macro_action_repeat=5`, `mpc_execute=1`,
`--planner_action_space primitives`, `--memory_router_mode keyframe`. See
the README for justification (closed-loop replan every macro step instead
of executing a five-step open-loop plan).

---

## 2. Data collection — `scripts/1_physics_rollout.py`

The rollout script wires three things together: a Genesis physics scene with
parallel envs, a closed-loop maze teacher policy that produces *privileged*
high-level commands, and a labeller that emits dense supervisory targets the
heads will eventually consume.

### 2.1 Scenes

Three scene families share a common `ObstacleLayout` representation
([`lewm/obstacle_utils.py`](lewm/obstacle_utils.py)):

- **Free obstacles**: a small number of axis-aligned boxes scattered in an
  open arena. Useful for camera robustness because it gives the dog clear
  sightlines but ensures it sometimes faces walls.
- **Composite scenes**: hand-crafted templates from
  [`lewm/maze_utils.py`](lewm/maze_utils.py) — T-junctions, crossroads,
  S-bends, zig-zags, doorway/corridor patterns. Each generator emits walls
  with a single shared concrete-grey colour so the encoder cannot use wall
  hue as a corridor-identity shortcut.
- **Enclosed grid mazes**: `generate_enclosed_maze(...)` carves an
  R×C grid using iterative recursive-backtracking
  ([`lewm/maze_utils.py:812-1113`](lewm/maze_utils.py#L812-L1113)). Wall
  thickness defaults to `0.20 m` after the anti-clipping fix (the original
  `0.06 m` walls were a key reason the rendered camera was clipping into
  geometry — see §10). After carving, BFS finds dead-ends sorted by distance
  from the spawn cell; beacons are mounted on the closed wall faces of the
  furthest dead-ends. **Returned metadata** (`return_metadata=True`)
  includes:
  - `cell_centers_xy`: world-frame XY of every cell.
  - `adjacency`: open-neighbour lists per cell (used by the maze teacher).
  - `dead_end_cells` / `dead_end_distances`: BFS-sorted dead-ends and
    distance-to-spawn per cell.
  - `beacon_cells`: which cell each beacon panel sits in.
  - `world_bounds_xy`: the four world bounds, used by camera retraction
    safety.

This metadata is later **stored as HDF5 attributes** (`scene_seed`,
`scene_type`, plus dumped JSON for the maze graph). The `streaming_dataset`
uses `scene_seed`+`scene_type` as a stable scene id for scene-level holdout —
critical because earlier `(file, env)` IDs leaked the same scene into eval
(see §10).

### 2.2 Beacons

[`lewm/beacon_utils.py`](lewm/beacon_utils.py) defines `BEACON_FAMILIES`
(named identities such as `red`, `green`, `blue`, etc.) and
`make_beacon_panel(wall_pos, wall_normal, identity, rng)`. The panel is a
small textured rectangle attached to a wall face, and its normal points
*into* the cell so the camera looking forward can see it. Distractor patches
share the colour space of beacons but at random non-wall locations to test
that the encoder doesn't conflate colour with goal identity.

### 2.3 The maze teacher

`build_maze_teacher` and `compute_maze_teacher_commands` form a closed-loop
*privileged* policy that drives the dog to give the world model rich
supervision. There are three families, controlled by an external command
schedule:

- **`maze_teacher_beacon`** — pick the unvisited beacon nearest to the dog
  in graph hops, BFS the cell graph to it, convert the next cell waypoint
  to a body-frame `[vx, vy, wz]` command. The output is *the privileged
  shortest-path command at every step*.
- **`maze_teacher_frontier`** — same machinery, but the target is the
  furthest unvisited graph cell (frontier expansion).
- **`maze_teacher_explore`** — randomised next-frontier with cooldowns to
  avoid pinball oscillation.

The teacher operates on the cell-centred adjacency graph but outputs
*continuous* body-frame velocity commands suitable for the low-level PPO
controller. This produces dataset trajectories with the geometric semantics
that the planning heads (especially the `ExplorationBonus` and
`EscapeFrontierHead`) will eventually need to learn.

### 2.4 Commands and command patterns

[`lewm/command_utils.py`](lewm/command_utils.py) provides a small DSL of
command patterns (`COMMAND_PATTERNS`) — straight, arc, brake, sinusoidal,
random-walk, etc. — and a `OUProcess` (Ornstein-Uhlenbeck noise) that smooths
random commands. `build_mixed_command_sequence` and `sample_command_pattern`
mix the privileged teacher commands with structured patterns and OU-noise
exploration so the dataset spans both task-relevant and dynamics-relevant
behaviours.

This DSL is *also* what the inference-time primitive bank is built from
(`build_action_primitive_bank`,
[`scripts/6_infer_pure_wm.py:1156-1211`](scripts/6_infer_pure_wm.py#L1156-L1211)),
so the planner only ever explores command sequences from the same family
the world model was trained on. This is the single most important alignment
to call out in the talk: **planner and dataset share a primitive vocabulary,
which is why we can search a discrete library instead of the unsupported
75-D continuous space.**

### 2.5 The PPO low-level controller

[`lewm/models/ppo.py`](lewm/models/ppo.py) defines a tiny `ActorCritic`
(`obs_dim=50`, `act_dim=12`). Inputs: 50-D proprio = base orientation /
gravity, body-frame linear velocity, base angular velocity, joint positions,
joint velocities, last action, and the current 3-D high-level command.
Outputs: 12 joint position deltas applied around `q0`. This network is the
*only* thing the world model is *not* trying to replace; it is frozen
throughout the rest of the pipeline. The PPO controller runs four physics
steps per command (`decimation=4`) at ~0.04 s per substep.

Inside [`scripts/1_physics_rollout.py`](scripts/1_physics_rollout.py) the
`SimConfig` defaults make this controller robust:
- `safe_clearance=0.18` (collision margin used by collision detection).
- `soft_collision_prob=0.3` (occasionally allow soft contact so the dataset
  contains clean recovery sequences instead of being collision-free).
- Anti-clipping wall thickness `0.20 m` (was `0.06 m`).

### 2.6 Labels

[`lewm/label_utils.py`](lewm/label_utils.py) emits per-step dense labels
that downstream phases use as energy-head targets: clearance to nearest
obstacle, traversability heuristics, beacon-visibility flags, contact /
collision counters, and beacon identity. `compute_episode_labels` aggregates
these into per-episode arrays. These labels are **not** input to the world
model — only to the heads.

### 2.7 Reset semantics, episode IDs, scene IDs

Two subtle things matter for evaluation integrity:

- **Reset-separated `episode_id`**: `_build_episode_metadata` in
  [`lewm/data/streaming_dataset.py`](lewm/data/streaming_dataset.py) walks
  the `dones` array and assigns a new episode whenever a done is observed.
  This breaks the previous "(file, env)" episode definition that was
  treating multiple resets within one env as a single episode (which silently
  trained across resets and contaminated supervision).
- **Stable `scene_id` from HDF5 attrs**: `_build_scene_ids` reads
  `scene_seed`+`scene_type` from each file and rounds-up to one global
  scene id per (file, env) tuple. Holdout is then *scene-level*: held-out
  scenes are not seen by the encoder under any (file, env) pairing.

---

## 3. Rendering — `scripts/2_visual_renderer.py`

A multi-process worker pool replays the scene-id-keyed metadata
deterministically and writes vision frames into the existing HDF5 files.

Highlights to discuss:

- **Backend resolution**: the renderer auto-detects `cuda`/`vulkan`/`amdgpu`/
  `cpu`. Dual-renderer setups can use Vulkan for the fast worker and CPU for
  a fallback worker, which matters on AMD systems where ROCm and Vulkan
  share device state.
- **Egocentric camera** (see [`lewm/camera_utils.py`](lewm/camera_utils.py)):
  the camera is rigidly mounted on the URDF (`mount_pos_body =
  (0.09055, 0.0, 0.07)`, pitch 15°, FOV 58°, near plane 0.01–0.08 m). It
  is intentionally *not* moved inside the body — clipping is handled by:
  - **Frustum-aware multi-ray detection** (`frustum_min_hit_distance`,
    9 rays through near-plane sample points) — used to detect impending
    occlusion.
  - **Camera retraction** (`retract_camera_to_safe`,
    [`camera_utils.py:259-298`](lewm/camera_utils.py#L259-L298)) — pulls
    the camera back along `-cam_forward` by up to 8 cm if the frustum's
    minimum hit is below `safe_clearance`.
  - **Depth-buffer validation** (`depth_buffer_has_clipping`) — rejects
    any frame where >0.5% of pixels lie at or below the near plane.
- **Frame substitution fallback**: when retraction + depth check fail (very
  rare after the wall-thickness fix), the renderer copies the last known
  good frame for that env instead of writing a clipped frame. This shows
  up in inference logs as a `frame_substitution_count`.
- **Texture domain randomisation**: [`lewm/texture_utils.py`](lewm/texture_utils.py)
  randomises wall and floor textures *within colour bounds that do not
  overlap the beacon families*, so the encoder can't shortcut a beacon
  via wall hue.
- **`--reuse_vision_from`**: re-renders only labels (used after a labelling
  fix) by reading vision from a sibling dataset and copying it across.

Two common student questions:

1. *Why egocentric instead of bird's-eye?* The world model's strong claim
   is that egocentric pixels alone contain enough information for
   navigation. Bird's-eye would invalidate that claim. The cost is camera
   safety — hence the elaborate retraction logic.
2. *Why 224×224 with 14×14 patches?* That gives a 16×16 token grid, which
   is the largest feasible token count under a ViT-Tiny budget on the
   target GPU (RX 7900 XTX, 24 GB).

---

## 4. The world model — `lewm/models/`

This is the single most paper-faithful part of the codebase. Walk students
through the four sub-components in order.

### 4.1 Encoder (`lewm/models/encoders.py`)

`VisionEncoder` is a vanilla ViT-Tiny:
- 12 transformer blocks, 192-dim hidden, 3 heads (64 dim/head).
- 14×14 patch convolution, 16×16 patch grid, learned class token, learned
  positional embeddings.
- LayerNorm before each MHA / MLP, GELU MLP, residual connections.
- Output: the class-token's 192-D vector — the per-frame latent `z_raw`.

`JointEncoder` wraps `VisionEncoder` and optionally fuses proprioception
(`use_proprio=True`). For the *pure-perception* track that the planner uses,
proprio is disabled — the inference script enforces this with
`scripts/6_infer_pure_wm.py:2696-2707`.

### 4.2 Projector — the dual-BN pattern

`Projector(in_dim=192, hidden_dim=2048, out_dim=192)` is `Linear → BN →
GELU → Linear`. Two instances exist on the LeWorldModel:

- `enc_projector`: projects encoder latents into the population the loss
  is computed on (the *target* side of the JEPA loss).
- `pred_projector`: projects predictor outputs into the same space (the
  *prediction* side).

Both are trained jointly. They are *different* batch-norms because they are
fit to different populations: `enc_projector` sees observation latents,
`pred_projector` sees rollout latents. A BatchNorm's running stats track its
input distribution, and at eval time they are frozen — but those frozen
stats are now systematically misaligned because they were estimated on
populations that drift apart over training (the "projector-space mismatch"
addendum in `report.md`, 2026-04-15). The fix is in
[`lewm/models/lewm.py`](lewm/models/lewm.py):

```python
def pred_proj_from_raw(self, z_raw):
    return self.pred_projector(z_raw)
```

This re-projects observation-side latents through `pred_projector` *at the
planner cost site* so the rollout-space goal latent really is in the same
BN-normalised space as the predictor outputs. Under-appreciated point: this
is also why the inference script encodes the goal beacon and then calls
`world_model.pred_proj_from_raw(z_breadcrumb_raw)` to make `z_goal_pred_proj`
([`scripts/6_infer_pure_wm.py:2925-2927`](scripts/6_infer_pure_wm.py#L2925-L2927)).
The planner uses `z_goal_pred_proj` as the cost-side goal under
`terminal_cosine` / `terminal_l2`, while the *learned* goal head still
consumes `z_goal_proj` because that is the population it was trained on.

### 4.3 Predictor (`lewm/models/predictor.py`)

The `TransformerPredictor` is wider than the residual stream — a deliberate
DiT-style design choice:

- 6 layers, 16 heads × 64 dim/head → **1024-dim attention** vs. the 192-dim
  residual stream.
- 2048-dim MLP per block.
- AdaLN-zero conditioning on the action embedding (Peebles & Xie's DiT trick:
  the conditioning vector predicts per-block scale and shift parameters,
  initialised so the conditioned block starts as identity).
- Causal attention via PyTorch 2's `F.scaled_dot_product_attention(..., is_causal=True)`.
- A `rollout(z, cmd_seq, history_z=None, history_actions=None)` method that
  prepends history latents/actions before unrolling, so inference can
  condition on a 4-step observation buffer.

The action embedder is its own small subnet:
- `Conv1d(input_dim=3, smoothed_dim=10, kernel_size=3, padding=1)` smooths
  consecutive commands so the predictor sees temporally coherent
  conditioning rather than per-step jitter.
- 2-layer MLP up to 192-D.

When `command_representation == "active_block"`, a single planner step's
action is a *block* of 5 commands stacked into a 15-D vector. The action
embedder still ingests them as a (B, 15) input and projects to 192-D.

### 4.4 SIGReg (`lewm/models/sigreg.py`)

The single most theoretically interesting piece. It enforces an isotropic
Gaussian latent so the projector population doesn't collapse. The mechanics
(read this slowly with students):

1. Project the (B, T, D) projected latent through `M=1024` random unit-norm
   directions to get scalar projections `s = z·θ ∈ R^{B*T*M}`.
2. Test marginal Gaussianity of the *empirical CDF* of `s` against the
   standard-normal CDF using an Epps-Pulley test. The test statistic is a
   weighted L² distance between characteristic functions, with weight
   `w(t)=exp(-t²/2)` (a Gaussian window over frequencies).
3. Discretise the integral as a Cramér-Wold sum over `K=17` quadrature knots
   over `[0, 3]`. Because every projection direction is unit-norm and the
   target is `N(0, 1)`, the per-direction expected statistic has a closed
   form against which the empirical statistic is differenced.
4. Multiply by `B` so the lambda for `total_loss = pred_loss + λ·sigreg_loss`
   is constant across batch sizes.
5. Run the SIGReg term in **float32 under bfloat16 autocast**, since the
   character-function differences are numerically delicate.

This is the Balestriero & LeCun "SIGReg" paper (arXiv:2511.08544). The
default `λ = 0.09`. Empirically, raising `λ` past ~0.15 over-smooths the
latent and hurts prediction MSE; lowering it below `0.05` lets the BN+
linear projector collapse onto narrow directions.

### 4.5 Putting it all together: `LeWorldModel.forward`

Walk students through the loss path
([`lewm/models/lewm.py`](lewm/models/lewm.py)):

1. `z_raw, z_proj = self.encode_seq(vis_seq, prop_seq)` — per-frame raw
   and projected latents.
2. `z_pred_raw = self.predictor(z_raw, cmd_seq)` — autoregressive (in
   training: teacher-forced) rollout in raw space.
3. `z_pred_proj = self.pred_projector.forward_seq(z_pred_raw)` — project
   into the loss space.
4. The teacher-forced prediction and target are aligned at offset 1:
   `pred = z_pred_proj[:, :-1]`, `target = z_proj[:, 1:]`.
5. Mean-squared error between `pred` and `target`, weighted by a per-sample
   validity mask (the streaming dataset can mark resets/invalid steps), then
   averaged over `n_valid`.
6. Plus `λ · sigreg_stepwise(z_proj, ...)`.

There is no decoder, no pixel reconstruction, and no proprio in the loss for
the pure-perception checkpoint. That is the JEPA promise.

---

## 5. Streaming dataset — `lewm/data/streaming_dataset.py`

This deserves its own section because its bugs caused weeks of confusion
(see report addendum 2026-04-15).

- `StreamingJEPADataset(IterableDataset)` shards work per-(file, env) so
  multiple workers do not stream from the same env's HDF5 cursor.
- **`_open_h5_file`** opens HDF5 in compression-aware mode: when `vision`
  has a non-trivial filter chain (gzip, blosc, etc.), it sets
  `rdcc_nbytes` to ~256 MB. When vision is uncompressed, the rdcc cache
  is *off*, because the regression of 2026-04-15 was the dataloader being
  ~10× slower under uncompressed vision because the cache was being kept
  hot for nothing. This fix matches the runbook recommendation
  (`scripts/repack_h5_vision.py` to convert gzip→none after collection).
- **`OPEN_FILE_CACHE_SIZE = 2`**: bounded LRU of open HDF5 handles so we
  don't run out of file descriptors when we shard across hundreds of
  chunks.
- **Command reconstruction** with `command_latency=2`: the model sees the
  command that *was active 2 raw steps ago*, mirroring the latency of the
  PPO controller's response. Without this, supervision is misaligned and
  the predictor learns a noise-floor.
- **Active-block construction**: `seq_len=4`, `temporal_stride=5`,
  `action_block_size=5`, `window_stride=5`. Each model step covers a 5-step
  command block; consecutive model steps are 5 raw steps apart;
  consecutive sequences (windows) are also 5 raw steps apart. The planner's
  `cmd_dim=15` (5 command steps × 3-D `[vx, vy, wz]`) is exactly this
  active-block.

Two common questions:

1. *Why iterate instead of randomly index?* HDF5 random access is slow on
   compressed vision; iterating per-(file, env) sequentially is ~30× faster
   and amortises decompression.
2. *How do you do scene-level holdout in a streaming setup?* The dataset
   is constructed with explicit `train_scenes` / `eval_scenes` lists; the
   `_build_scene_ids` step assigns scene ids before the dataloader draws,
   and the streaming workers are filtered to only visit train scenes.

---

## 6. World-model training — `scripts/3_train_lewm.py`

Things to highlight on the slide:

- **AdamW**, `lr=3e-4`, `weight_decay=1e-4`, gradient clip 1.0,
  `CosineAnnealingLR` over `total_epochs`.
- **bfloat16 autocast** for everything except SIGReg (float32 under amp).
- **Gradient sanitisation**: `nan_to_num_` on grads after backward — a
  band-aid for the rare bf16 NaN that otherwise contaminates AdamW's
  running stats.
- **Scene-level holdout**: held-out scenes are passed in as a list and
  the dataloader is filtered before iteration. *Important caveat from
  memory*: the eval split was contaminated for the first 12 epochs (the
  metric appeared to fall, but it was on training data); from epoch 13
  onward the held-out split is clean. The post-epoch-13 rise in held-out
  loss is therefore *memory decay* (the model is forgetting the data it
  started training on as the cosine schedule attenuates LR), not genuine
  overfitting.
- **`run_eval` vs. `run_multistep_eval`**: `run_eval` is a one-step
  teacher-forced eval; `run_multistep_eval` rolls out *autoregressively*
  in `pred_projector` space for K = `seq_len-1` steps so the metric
  reflects what the planner will actually see at test time. This was added
  in the 2026-04-07 addendum because the report shows a sharp gap between
  one-step (clean) and multi-step (poor) metrics.
- **Logging**: `LOG_FLUSH_EVERY=10` so we don't trigger CUDA syncs every
  step; CSV logs include both eval modes.
- **Checkpoint metadata**: `cmd_dim`, `command_representation`,
  `temporal_stride`, `action_block_size`, `command_latency`, `image_size`,
  `patch_size`, `use_proprio` — every downstream consumer
  (`scripts/4`, `scripts/6`) re-derives parameters from this metadata and
  fails loudly on mismatches via `resolve_encoder_config`.

---

## 7. Energy heads — `scripts/4_train_energy_head.py` + `lewm/models/energy_head.py`

This script is the most procedural part of the codebase. It runs **ten
phases** sequentially, each freezing all upstream components.

### 7.1 Phase 1 — Latent extraction

- Loads the LeWM checkpoint, freezes encoder/projectors/predictor.
- Streams the dataset, computes both `z_raw` (per-frame) and rollout
  latents (predictor output, projected).
- For each window writes a shard with **two views** of every step:
  - `enc_projector(encoder(z))` — the population the heads were
    historically trained on.
  - `pred_projector(predictor(encoder(z), cmd))` aligned to label[t+1] —
    the population the planner actually sees.
- `CACHE_VERSION=12` so any change to extraction invalidates the cache.

### 7.2 Phases 2–9 — Per-head training

All heads share a common shape: small MLPs over 192-D inputs, BN+GELU+Dropout,
trained with AdamW + cosine LR. The differences are in the loss target.

- **`LatentEnergyHead` (Phase 2: safety)**: 192-D → softplus scalar. Target
  is `composite_safety_target` (clearance+mobility) or
  `consequence_safety_target` (contact+mobility), built per-step from
  Phase-1 labels. The "consequence" mode — contact below `contact_clearance=0.08 m`
  + low mobility — gave the most useful planner cost in ablations.
- **`GoalEnergyHead` (Phase 3)**: takes `[z_pred, z_goal, z_pred-z_goal,
  z_pred*z_goal]` (concat of latents, difference, elementwise product) →
  scalar. Trained on identity-conditioned positive/negative pairs. Crucially,
  the head is trained on `z_goal` from `enc_projector`, but the planner can
  use it on rollout latents from `pred_projector` — this is the cross-space
  framing fix from the 2026-04-16 addendum.
- **`ProgressEnergyHead` (Phase 4)**: auxiliary, deprecated; still trained
  and bundled but not consumed by the planner.
- **`ExplorationBonus` (Phase 5)**: random network distillation. Frozen
  random target net, trained predictor net, MSE between them per latent.
  Has an `online_update(z)` method that lets the planner adapt the
  predictor at *inference* time too (`--rnd_online_lr 1e-3` by default).
- **`PlaceSnippetHead` (Phase 6)**: takes a (T, D) snippet (default T=3),
  returns an L2-normalised embedding. Trained with triplet loss on
  same-episode positives within a pose radius and same-episode negatives
  beyond a wider radius. Used at inference for snippet-level visited-NN
  novelty.
- **`DisplacementHead` (Phase 7)**: takes (z_start, z_end), regresses the
  XY displacement (m) between the two observation poses. Pose-supervised.
  Used as a planner bonus for "did this rollout actually move?".
- **`CoverageGainHead` (Phase 8)**: token-network per step + mean/max/
  terminal pooling, regresses the m² of *novel* path area covered over
  the lookahead window. Targets are built from `densify_xy_path` and a
  novel-path proxy that excludes already-visited area within a context
  window.
- **`EscapeFrontierHead` (Phase 9)**: 5-way concat including a
  `terminal_delta = z_end - z_start` term. Targets are a frontier-value
  proxy (BFS distance to unvisited cells, masked by visibility).

All heads bundle into `TrajectoryScorer` and serialise into a single
`scorer.pt` file (the `--scorer_ckpt` argument of inference).

### 7.3 Phase 10 — AR finetuning

Teacher-forced training and deployment differ substantially when the planner
unrolls more than one step. Phase 10 fine-tunes the heads with rollouts
generated *autoregressively* by the predictor, closing the train-deploy gap
identified in the 2026-04-07 addendum. This is the phase that empirically
moved end-to-end planner success the most.

---

## 8. Inference — `scripts/6_infer_pure_wm.py`

The longest single file in the repo (~3950 lines). The right slide
decomposition:

### 8.1 The planner: `PureCEMPlanner.plan(...)`

`PureCEMPlanner` ([`scripts/6_infer_pure_wm.py:205-855`](scripts/6_infer_pure_wm.py#L205-L855))
runs CEM (Cross-Entropy Method) over command sequences for `cem_iters=30`
iterations, with `n_candidates=300`, `elite_frac=0.10`. Two action spaces:

- **Continuous**: Gaussian samples around the warm-started mean, clamped to
  `[cmd_low, cmd_high]`. The 75-D continuous space (15-D × 5 horizon) is
  too large to search effectively — see report addendum.
- **Primitives** (default and recommended): a categorical distribution over
  a `primitive_library_size=128` library built from the same DSL the
  dataset uses (see §2.4). Each CEM iteration:
  1. Sample `n_candidates` index sequences from the per-step categorical
     distributions.
  2. Optionally jitter the resulting commands by Gaussian noise scaled by
     `primitive_jitter_scale * init_std`.
  3. Roll out, compute cost, take elite, refit categorical from elite
     counts (with a 1e-3 smoothing constant).
  4. Warm-start the next call by shifting the best index sequence by one
     and repeating the last index.

The planner forwards the entire batch through a single
`world_model.plan_rollout(...)` call (note: rollout history conditioning is
threaded as `z_history_raw` and `action_history`).

### 8.2 The cost terms

The cost is a weighted sum (read with students from
[`scripts/6_infer_pure_wm.py:520-777`](scripts/6_infer_pure_wm.py#L520-L777)):

```
cost = safety_w · safety
     + goal_w · goal_term
     + route_progress_w · route_term
     − exploration_w · exploration_bonus_or_visited_nn
     − terminal_displacement_w · terminal_displacement
     − coverage_gain_w · predicted_coverage_gain
     − escape_frontier_w · predicted_frontier_value
     − displacement_w · predicted_displacement
     + action_penalty_w · ||a||²
     + kinematic_safety_w · max_aabb_penetration
```

The `goal_term` selector lives at lines 543–571:
- `terminal_cosine`: `1 − cos(z_pred_proj[-1], z_goal_pred_proj)` clamped at zero.
- `terminal_l2`: `MSE(z_pred_proj[-1], z_goal_pred_proj)`.
- `head`: `goal_head.score_trajectory(z_pred_proj, z_goal_proj)` (uses the
  *trained* goal head, with `z_goal` in `enc_projector` space).
- `off`: ignore goal.

This is also the place to talk about the **kinematic safety veto**: a
geometry-aware backstop. From the current pose+yaw the planner integrates
each candidate's substep-velocity sequence forward in time, checks AABB
penetration against `obstacle_layout` with a margin, and adds `kinematic_safety_w * max_penetration`
to the cost. This makes the planner hybrid — learned safety + geometric
collision — and is required because the LatentEnergyHead alone is too
local-scalar to veto walls.

### 8.3 The keyframe memory router

Activated by `--memory_router_mode keyframe`, the most important
non-CEM behaviour. Walk students through the data structure:

- A list of `KeyframeNode(score_latent, proj_latent, step, last_seen_step,
  visit_count, odom_xy, yaw_rad)`.
- A list of neighbour-sets indexed by node id: an undirected graph.
- Adjacency is incremented whenever the dog transitions from one
  current-keyframe to another within a single planner step.

Every step the dog observes the world, encodes it, and asks
`match_keyframe_node(...)` whether the score-latent is similar enough
(`keyframe_sim_threshold=0.985`) to an existing node *that has not been
visited recently* (`keyframe_min_step_gap=8`). If yes, the matched node
becomes the current node and the visit count increments. If no, and we
have not added a node in `keyframe_add_interval=24` steps, we insert a
new one.

Plateau triggers a route. If `stall_plateau_steps=200` go by without a
new prototype or a route/goal-progress improvement, the dog enters
**route mode**: BFS the keyframe graph, pick a target that either improves
goal similarity by at least `goal_route_improve_margin=0.03` or, failing
that, lies in the recent frontier window `subgoal_frontier_window_steps=800`
and has not been visited for `subgoal_min_age_steps=120` steps. The route
emits a sequence of waypoint latents `z_route_proj` that the planner
optimises against using the same cosine/L2 mechanics as the goal term.

This is the authors' recipe for "perception-only memory" — it never uses
ground-truth pose to plan (odometry is dead-reckoned only for
match-radius gating, which can be turned off entirely with
`--keyframe_match_radius_m -1.0`). The contrast with classical SLAM is the
right discussion point.

### 8.4 Visited-NN exploration bonus

`exploration_bonus_mode=visited_nn` replaces the RND bonus with a kNN
density estimate against the *executed* rollout bank. With
`visited_rollout_tail_steps>1`, this becomes a *snippet* novelty bonus
that the `PlaceSnippetHead` is used as a feature transform for. This is
strictly more aware of recent geometry than RND, which has a tendency to
chase pixel-level novelty.

### 8.5 Audit trail

`--audit_plan` writes `plan_audit.jsonl`: one line per audited replan
containing the top-K candidates (by cost), their per-component metrics,
their command sequences, and the actual outcome after executing the
chosen first macro step (delta latent, executed pose change, reward
deltas). `scripts/aggregate_plan_audits.py` post-processes these into
a stratified report: which cost terms predicted real progress, which
ones were noise, which terms over-fired on safety. This is the team's
primary feedback loop — without it the planner is a black box.

### 8.6 Stuck and recover modes

`stuck_window_steps=12` window of (commanded magnitude > `stuck_cmd_threshold`,
mean dead-reckoned XY motion < `stuck_odom_threshold`, latent displacement
< `stuck_latent_threshold`) flips the dog into a `recover` mode for
`recover_budget_steps=28` steps with a permissive command space and
exploration uplift. After recovery there's a `recover_cooldown_steps=24`
gate so we can't oscillate.

### 8.7 Success detection

There is no privileged goal-detection. Success is *perceptual*: when
breadcrumb similarity exceeds both `success_goal_sim_threshold=0.90`
(projected) and `success_goal_raw_sim_threshold=0.95` (raw) for
`success_hold_steps=6` consecutive frames the dog declares it's reached
the beacon. The simulator-side oracle goal — distance from the beacon
claim point to the dog body XY — is logged for evaluation only and never
used by the planner.

---

## 9. Aggregation and reporting

- [`scripts/7_aggregate_inference_runs.py`](scripts/7_aggregate_inference_runs.py)
  consolidates the per-seed metrics (success, time-to-goal, coverage,
  collisions) into a CSV.
- [`scripts/aggregate_plan_audits.py`](scripts/aggregate_plan_audits.py)
  aggregates `plan_audit.jsonl` into a stratified report.
- [`scripts/summarize_dataset_coverage.py`](scripts/summarize_dataset_coverage.py)
  reports per-scene, per-episode coverage statistics and is what the
  pipeline reads to decide whether the dataset is large enough.

---

## 10. Failure analysis (read this with the audience)

The repository's `report.md` is its single best teaching artefact. The
audience should leave with the following hierarchical failure tree:

1. **Insufficient scene diversity**: under ~500 unique training scenes the
   encoder over-specialises and held-out reconstruction loss does not
   reflect held-out *navigation* loss. Mitigation: composite scenes,
   enclosed mazes, free obstacles, scene-level holdout.
2. **Scene-leaky `(file, env)` evaluation**: the same scene was being
   sampled into both train and eval splits because the original
   `episode_id = (file, env)` definition let multiple resets share an
   id. Mitigation: reset-separated `episode_id`, scene-level holdout via
   `scene_seed`+`scene_type` HDF5 attrs.
3. **Reset-contaminated supervision**: training across reset boundaries
   inside an episode confused the predictor about temporal continuity.
   Mitigation: same as (2).
4. **Teacher-forced vs. AR mismatch**: training metric was teacher-forced
   one-step, deployment metric is AR multi-step. Mitigation: the
   `run_multistep_eval` introduced in the training loop, plus the AR
   finetuning Phase 10 in head training.
5. **Single-frame state aliasing**: any single 224×224 frame in a
   corridor looks like many other corridor frames. Mitigation: 4-step
   history at the predictor + the keyframe memory router at inference.
6. **Unsupported 75-D continuous action search**: CEM cannot meaningfully
   explore that space in 30 iterations × 300 candidates. Mitigation:
   `--planner_action_space primitives` over a 128-entry library.
7. **`mpc_execute=5` open-loop drift**: executing a 5-step plan open-loop
   compounds prediction error. Mitigation: `mpc_execute=1`,
   `macro_action_repeat=5` (replan every macro step).
8. **Local scalar objectives**: a single per-step safety scalar can't
   veto upcoming walls reliably. Mitigation: kinematic safety veto over
   integrated future XY, plus the `EscapeFrontierHead` for sequence-level
   value.
9. **Camera clipping**: thin walls (0.06 m) + small near plane (0.01 m) +
   forward-mounted camera let the camera see *into* walls, producing
   degenerate frames. Mitigation: walls 0.06 → 0.20 m, near plane 0.01 →
   0.08 m, forward mount offset 0.10 → 0.06 m, plus frustum-aware
   retraction and depth-buffer validation.

The report's 2026-04-15 addendum is the single most important
reading: **projector-space mismatch at the planner cost site**. The fix
(`pred_proj_from_raw`) and the dataloader rdcc regression are both
discussed there.

The 2026-04-16 addendum lays out the WM plateau hypotheses
(s1≈0.20, s2≈0.29, s3≈0.41 MSE) and the goal-head cross-space framing fix.

---

## 11. Hardware and runtime

- Single RX 7900 XTX (24 GB), ROCm 6.x. Genesis runs under ROCm or Vulkan.
- Full pipeline ~3 days: rollout (12h, can parallelise across CPU
  workers) → render (4–6h, multi-process) → train LeWM (24–36h, depends
  on epoch budget) → extract latents (1h) → train heads (4h, includes AR
  finetuning) → inference sweep (varies).
- The dataloader fix (compression-aware rdcc) and the
  `repack_h5_vision.py` gzip→none repack tool both directly affect this
  budget.

---

## 12. Reading order for new students

1. `README.md` — pipeline shape and runbook.
2. `lewm/models/lewm.py` — the world model in 278 lines.
3. `lewm/models/sigreg.py` — SIGReg in 90 lines.
4. `lewm/models/predictor.py` — DiT-style predictor with action
   conditioning.
5. `report.md` — every failure mode named in §10, with prose.
6. `scripts/3_train_lewm.py` — the training loop.
7. `scripts/4_train_energy_head.py` — the head training pipeline.
8. `scripts/6_infer_pure_wm.py` — the planner.
9. `lewm/data/streaming_dataset.py` — the dataloader.
10. `scripts/1_physics_rollout.py` and `scripts/2_visual_renderer.py` —
    data collection, last because the simulator is incidental to the
    science.

---

## 13. Discussion prompts for the seminar

- *Why projector-space mismatch keeps surfacing in JEPA-style work.*
  Hint: BatchNorm running stats track training-time populations that may
  differ from inference-time populations, and the standard "use eval()"
  fix doesn't address the mismatch — it freezes it.
- *When does primitive-space search beat continuous CEM?* Hint: when the
  data manifold (set of plausible commands) is much smaller than the
  ambient command space, and especially when the dynamics are non-smooth
  in command space.
- *Why is keyframe routing not SLAM?* The graph is built in *latent*
  space, not metric space; edges are temporal transitions; the only
  metric input (`odom_xy`) is dead-reckoned and used as a soft gate, not
  as a planning state.
- *Why do we need both `enc_projector` and `pred_projector`?* Why not
  share one projector? Discussion: BatchNorm population, gradient
  conflict between encoding and rolling out, and the empirical fact that
  one-projector ablations underfit on rollout MSE.
- *What would it take to drop the kinematic safety veto?* What would the
  `LatentEnergyHead` need to learn that it currently does not?

---

## 14. What's not yet in the model (honest framing)

- No language conditioning, no instruction following.
- No multi-robot coordination.
- No real-robot deployment (sim-only).
- No counterfactual training (the predictor is teacher-forced; the
  Phase-10 AR finetuning closes part but not all of the gap).
- No principled uncertainty estimate over rollouts (the planner's CEM is
  point-estimate; ensembles or dropout-MC could be added).
- The keyframe graph does not yet support pruning of stale branches
  beyond "redundant prototype replacement"; long sessions can grow it
  without bound.

These are the natural next-research-question hooks for the seminar.
