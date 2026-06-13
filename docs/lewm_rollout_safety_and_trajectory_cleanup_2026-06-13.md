# LeWM rollout safety probe and trajectory-cleanup plan — 2026-06-13

## Decision

Keep local geometry as the hard collision-avoidance layer. Frozen raw LeWM
rollouts may be used as an auxiliary progress/ranking feature, but this bounded
experiment does not support using them as the safety gate.

Runtime local geometry must enter through
`lewm.planning.local_obstacles.LocalObstacleModel`. Privileged manifest geometry
is an explicit deployment-invalid benchmark adapter; a pure-perception seek
result must use a perception-backed provider and the benchmark's strict
deployment-valid guard. That guard covers the local-obstacle source chain only;
remaining simulator-pose dependencies in heading control, stuck detection, and
evaluation still require a separate odometry/source audit.

An experimental rolling ego-depth provider is now wired through that interface.
It removes manifest-grid queries from seek-time safety, but the benchmark source
is still deployment-invalid because the depth is simulator-rendered, the map is
registered with simulator ground-truth pose, and the current platform contract
is monocular RGB. Strict mode rejects it. This is a trajectory-control
experiment, not a pure-perception result.

The next trajectory-quality change should replace the current quantized
feasible-fraction steering score with continuous swept-body clearance,
execution-calibrated primitive envelopes, and steering hysteresis.

## Motivation

The v43 physical topological-navigation demo completed the route-valid task:

- final goal distance: `0.187 m`;
- physical path: `8.79 m`;
- seek blocks: `103`;
- veto escapes: `4`;
- edge realigns: `1`;
- falls: `1`;
- display-replay correction snaps: `6`.

The v28-v43 controller is much better than the earlier binary veto/escape
behavior, but it still scores trajectories using a quantized fraction of
collision-free probe samples. It cannot measure the difference between a
comfortably centered trajectory and a legal wall skim, and it models commanded
primitive motion rather than the physical gait's drift and overshoot.

The open representation question was whether LeWM's predicted latent rollout
contains enough additional safety information to solve that gap.

## Registered bounded experiment

Script:

`scripts/probe_task_aligned_rollout_safety.py`

Artifact:

`.generated/task_aligned_policy_v0/rollout_safety_v1_seed20260613.{pt,json}`

The experiment uses the existing v1 task-aligned counterfactual contract and
the frozen seq11 LeWM checkpoint. No new rollout data was collected.

Train and held-out evaluation each use a fixed random sample of `4,096`
decisions covering all `32` scenes in their respective existing split. Each
decision contains the same nine candidate primitives and existing
inflated-grid `collided`, endpoint-clearance, progress, heading, and cost
labels.

Three equal-capacity multi-task probes are compared:

1. `current_raw`: current raw latent, goal interactions, and action;
2. `rollout_raw_h1`: `current_raw` plus the one-step predicted raw-latent delta;
3. `rollout_raw_h2`: `rollout_raw_h1` plus the second predicted delta.

Safety decodability is evaluated at each variant's best collision-AP and
best-clearance checkpoints. Deployed selection is evaluated separately at each
variant's best selection-cost checkpoint. This separation is required: an
initial run compared the conservative current-latent epoch 0 against trained
rollout epochs and falsely reported a large rollout safety gain.

Reproduction:

```bash
.generated/venvs/genesis_render_vulkan/bin/python \
  scripts/probe_task_aligned_rollout_safety.py \
  --train-data .generated/task_aligned_policy_v0/train32_v1_spatial2.npz \
  --eval-data .generated/task_aligned_policy_v0/val32_v1_spatial2.npz \
  --checkpoint models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12/lewm_seq11_e3.pt \
  --output .generated/task_aligned_policy_v0/rollout_safety_v1_seed20260613.pt \
  --horizon 2 --epochs 20 --hidden 128 \
  --batch-size 256 --rollout-batch-size 128 \
  --max-train-rows 4096 --max-eval-rows 4096 \
  --seed 20260613 --device cpu
```

## Results

### Safety decodability

| input | best collision AUROC | best collision AP | best clearance MAE |
|---|---:|---:|---:|
| current raw | 0.8015 | 0.5202 | **0.1991 m** |
| + rollout h1 | **0.8091** | **0.5309** | 0.2057 m |
| + rollout h2 | 0.8001 | 0.5247 | 0.2042 m |

Relative to current raw:

- h1 collision AP improves only `+0.0107`; clearance MAE worsens `+0.0066 m`;
- h2 collision AP improves only `+0.0045`; clearance MAE worsens `+0.0050 m`.

The registered rollout-decoding gate required collision AP `+0.02` and
clearance MAE improvement of at least `0.005 m`. It **failed**.

### Deployed candidate selection

| input/control | regret / random | selected grid collision | target progress |
|---|---:|---:|---:|
| current raw | 0.633 | **3.03%** | +0.0002 m |
| + rollout h1 | 0.601 | 8.23% | **+0.0264 m** |
| + rollout h2 | **0.570** | 7.35% | +0.0254 m |
| action-only `yaw_left` | 0.599 | 2.66% | 0.0000 m |
| random | 1.000 | 20.37% | +0.0189 m |

Rollout features make the scorer less conservative and improve progress and
regret. They do not improve safety. The h2 deployed scorer exceeds random
progress but misses both the `<= 0.50` regret-ratio and `<= 5%` collision gates.
The promotion gate **failed**.

## Interpretation

The frozen predictor exposes useful action-conditioned navigation signal, but
not additional fine clearance information under this contract. The result is
consistent with the previous finding that the latent is useful for
place/action-level reasoning but lacks reliable per-step metric resolution.

This result does not prove that physical collision information is absent. The
labels are still an unvalidated inflated-grid proxy:

- collision checks the idealized candidate center path;
- clearance is measured only at the endpoint;
- labels omit swept-body side clearance, gait sway, overshoot, contacts, and
  falls;
- the h2 latent feature is compared against a one-block candidate label.

The correct conclusion is narrower: **current predicted raw rollout deltas do
not add enough decodable information to justify replacing the geometric safety
gate.**

## Trajectory-cleanup implementation plan

### Phase 0 — preserve the runtime claim boundary

1. Keep `--require-deployment-valid-local-obstacles` enabled for any result
   described as deployment-valid or pure perception.
2. Replace simulator-pose map registration with onboard odometry/state
   estimation.
3. Preserve the current monocular-RGB sensor contract by deriving local free
   space from RGB/LeWM features, or explicitly approve and document a platform
   depth-sensor contract change.
4. Keep the ego-depth provider as an experimental upper-bound and controller
   debugging source until those two source requirements are met.

### Phase 1 — continuous geometric local control

1. Add a continuous obstacle-clearance query to `InflatedOccupancyGrid`.
   Preserve the obstacle-distance raster already computed during grid
   construction instead of retaining only the boolean free mask.
2. Replace `_feasible_fraction` as the WALK ranking score with full-trajectory
   statistics:
   - strict collision feasibility;
   - minimum swept-body clearance;
   - low-percentile swept-body clearance;
   - endpoint and heading progress.
3. Score candidates using a smooth near-wall barrier plus progress, heading,
   curvature, and primitive-switch penalties. Keep collision feasibility as a
   hard gate.
4. Add steering hysteresis: retain the previous arc/straight choice unless the
   new candidate wins by a configured margin.
5. Expand the near-wall candidate bank with gentle/strong arcs and shorter,
   slower blocks. Use the existing coarse primitives in open space.

### Phase 2 — execution-calibrated envelopes

1. Mine physical replay transitions by primitive.
2. Measure longitudinal overshoot, lateral drift, yaw error, and swept
   nose/tail/side excursion.
3. Use a conservative percentile envelope per primitive when scoring
   clearance. This replaces fixed hand-tuned probe radii with measured gait
   behavior.
4. Add contact, minimum-clearance, and fall labels to future evaluation
   artifacts. Keep `grid_unsafe` explicitly named as a proxy until validated.

### Phase 3 — local path tracking

1. Generate a short free-space centerline/lookahead path for each topological
   edge.
2. Score progress toward the lookahead point, not only the direct next-node
   bearing. This prevents corner cutting when an edge's endpoint lies around a
   bend.
3. Allow LeWM rollout ranking only as a tie-breaker after candidates pass the
   geometric safety gate.

## Evaluation and promotion gates

Use v43 as the fixed-seed baseline, then run at least three physical seeds.
Report:

- route success and perceptual-stop correctness;
- falls and physical contacts;
- minimum and fifth-percentile swept clearance;
- veto escapes, backward/yaw/hold blocks, and edge realigns;
- primitive switches per meter;
- path length and seek blocks;
- replay correction snaps.

Promote the trajectory controller only if it preserves route success, has zero
falls/contacts across the evaluation runs, reduces veto escapes and correction
snaps, and does not materially increase path length or seek blocks.

## Stop conditions

- Do not tune more rollout-safety head capacity on the current proxy labels.
- Do not use predicted latent rollout collision probability as a hard safety
  gate.
- Revisit learned safety only after physical swept-clearance/contact labels
  exist, or after the representation is trained with explicit local
  geometry/depth/contact supervision.
