# Go2 Fully Learned Implementation Handoff

Date: 2026-06-28

## Goal

Turn the current clean Go2 all-beacon demo into a fully learned navigation implementation.

The target claim is not "learned locomotion." The low-level Go2 gait / primitive executor may remain fixed. The target claim is:

> A Go2 agent uses egocentric RGB through a frozen JEPA-style visual encoder, learned memory, and learned local/topological action selection to revisit and claim all remembered beacons in a maze, without privileged map geometry, beacon coordinates, hand-built standoff routes, or preplanned waypoint routes at runtime.

## Current Accepted Demo

Current best clean live closed-loop artifact:

- Video: `.generated/go2_memory_closed_loop/clean_go2_candidate_try010_strict_blockedarc_targetbearing018_arcmax035_fwdslow_y245_policy50.mp4`
- Result: `.generated/go2_memory_closed_loop/clean_go2_candidate_try010_strict_blockedarc_targetbearing018_arcmax035_fwdslow_y245_policy50_result.json`
- Quality: `.generated/go2_memory_closed_loop/clean_go2_candidate_try010_strict_blockedarc_targetbearing018_arcmax035_fwdslow_y245_policy50_quality.json`

Metrics:

- success: `true`
- beacons claimed: `red`, `yellow`, `blue`, `green`
- ticks used: `328`
- contact-like stalls: `0`
- hard stalls: `0`
- body-clearance violations: `0`
- fall/tip/unstable events: `0`
- translation share in EXPLORE: `0.5483`
- pure-yaw share in EXPLORE: `0.4517`

Compared with the previous accepted clean baseline:

- Old clean baseline: `340` ticks, zero stalls, near-target sweep `136.6 deg`.
- New clean demo: `328` ticks, zero stalls, near-target sweep `205.1 deg`.

Interpretation: the new demo is faster and cleaner on the primary live safety/success gate, but not smoother around targets. Keep both artifacts: use try010 as the current clean implementation baseline and the old 340-tick run as the smoother/orbit reference.

## Current Claim Boundary

The demo is real live closed-loop Go2 execution, not replay, and it uses learned perception/memory components.

Learned components currently in the loop:

- frozen Go2 JEPA RGB encoder:
  `.generated/go2_hidden_target_memory/go2_jepa_latent_encoder_medium_hidden_claim_seed20260628_img64_lat96_contrast02.pt`
- RGB/JEPA vector-memory controller:
  `.generated/go2_hidden_target_memory/go2_rgb_jepa_strict_exact_valuenorm_gate_neg6_pair8_nonforward_eval_seed20260825_h512.pt`
- primitive outcome / wall-action predictor:
  `.generated/go2_wallaware_learned/primitive_outcome_jepa_mixed_progress08.pt`
- narrow learned blocked-arc fallback in `scripts/benchmark_go2_memory_closed_loop.py`.

Still not fully learned:

- Runtime exploration still uses `--explore-standoff-route`.
- The standoff route uses manifest/grid geometry to create safe waypoint-like behavior.
- Claim thresholds, color switch thresholds, standoff lookahead, clearance weights, and route parameters are hand-tuned.
- The learned primitive outcome head acts as a guard/reranker, not as the full planner.
- The current demonstration is one selected maze, not yet a held-out maze suite.

## Fully Learned Runtime Contract

Allowed runtime inputs:

- egocentric RGB frames;
- frozen JEPA latent/spatial features from those frames;
- recurrent learned memory state;
- previous primitive/action and proprioceptive egomotion/odometry;
- learned primitive outcome/safety predictions from RGB/JEPA latent;
- fixed low-level Go2 primitive/gait executor.

Forbidden at runtime for the fully learned claim:

- manifest occupancy grid;
- true beacon coordinates or distances;
- hand-built standoff target solver;
- hand-authored route waypoints;
- route-table lookup built from privileged map geometry unless the table is itself produced by learned online inference from allowed observations;
- per-scene parameter tuning.

Allowed during training:

- privileged standoff route as a teacher;
- map geometry for labels;
- oracle collision/progress/safety labels;
- DAgger-style relabeling from failed learner rollouts.

## Definition Of Done

Minimum fully learned implementation gate:

1. Runtime runs without `--explore-standoff-route` and without any privileged map/standoff planner in EXPLORE.
2. Runtime policy chooses local primitives from learned RGB/JEPA/memory features.
3. All four beacons are claimed in the current reference maze.
4. Hard stalls: `0`.
5. Contact-like stalls: target `0`; acceptable first pass `<=1` only if no hard stall, no body-clearance violation, and no visible collision.
6. Body-clearance violations: `0`.
7. Fall/tip/unstable events: `0`.
8. Ticks: initially `<=400`; then beat clean scaffold baseline `<=328`.
9. Rendered policy-rate video exists for the exact live closed-loop run.

Generalization gate:

1. Train on multiple maze scenes.
2. Evaluate on held-out scenes not used for teacher data.
3. Report success rate, all-beacon claim rate, ticks, stalls, body/stability failures, orbit/sweep, and yaw/translation shares.
4. Require at least a small held-out suite before paper claims: e.g. `5/5` or `8/10` clean-ish passes depending on scene difficulty.

Paper-grade gate:

1. Compare against the standoff-route scaffold, geometry-only baseline, deterministic color-memory baseline, and no-memory ablation.
2. Show that the learned route/local policy is using memory: corruption/dropout of memory state must significantly hurt target ordering/claim success.
3. Show it does not rely on current visibility only: hidden target recall must remain above baseline after target leaves view.
4. Provide a clear ablation separating:
   - frozen JEPA encoder;
   - learned memory;
   - learned local planner/topology;
   - learned wall/outcome guard.

## Recommended Path

### Phase 1: Teacher-distilled learned local planner

This is the shortest path to a fully learned demo.

Use the current standoff-route controller as a teacher only for data generation. Train a learned local planner that maps allowed runtime features to the next primitive.

Inputs:

- frozen JEPA latent/spatial feature;
- vector-memory controller state / readout;
- active target color/query embedding;
- previous primitive;
- proprioceptive egomotion;
- primitive outcome predictions;
- optional compact recurrent state.

Labels:

- teacher requested primitive;
- teacher executed primitive after learned wall reranking;
- progress outcome;
- contact/stall label;
- claim/near-target state;
- route phase if useful as an auxiliary target, but not as a runtime input.

Initial implementation target:

- add/train a policy head or reuse `train_go2_closed_loop_learned_local_policy.py`;
- integrate through the existing learned-local hook in `scripts/benchmark_go2_memory_closed_loop.py`;
- run with the standoff route disabled and `--explore-goal-policy learned_policy` or equivalent;
- keep the learned primitive outcome guard active as a safety fallback at first.

Expected first milestone:

- all four beacons in the reference maze within `400` ticks;
- no hard stalls or body/stability failures;
- rendered video.

Then tighten to:

- zero contact-like stalls;
- `<=328` ticks;
- reduce near-target sweep below try010.

### Phase 2: DAgger on learner failures

Pure behavior cloning will likely drift because the teacher route visits a narrower state distribution than the learned policy.

Loop:

1. Run the learned policy live closed-loop without standoff route.
2. Collect failure windows: orbiting, wall-facing yaw loops, missed turn opportunities, slow near-target servo, stalls.
3. Relabel those windows with the scaffold teacher or a local oracle.
4. Retrain.
5. Repeat until reference-maze gate passes.

Keep the safety labels separate from action imitation. A policy that imitates route choices but ignores stall risk will look good offline and fail in Genesis.

### Phase 3: Replace scaffold with learned latent topology

Once local policy passes, step back and learn route structure more explicitly.

Options:

- recurrent latent graph: nodes are remembered observation/keyframe embeddings, edges predict reachability/progress;
- learned frontier/value head over memory slots;
- contrastive ordering objective where temporally/reachably close states are nearby and wall-separated states are not.

This is the more novel paper direction, but it should follow the local-policy demo. The local-policy route gives a concrete Go2 result while topology learning becomes the next research contribution rather than a blocker.

## Immediate Next Tasks

1. Add a clean `fully_learned` run preset/script.
   - It must fail fast if `--explore-standoff-route` or privileged grid routing is active.
   - It should write result/video/quality artifacts under `.generated/go2_memory_closed_loop/fully_learned_*`.

2. Build the teacher dataset.
   - Use try010 settings with standoff route enabled.
   - Generate multiple successful teacher traces across the reference maze and nearby maze variants.
   - Store per-tick RGB/JEPA latent features, memory state/readout, primitive outcome predictions, teacher requested/executed primitive, progress, stall, claim state.

3. Train the local planner.
   - Start with imitation of executed primitive.
   - Add class balancing so yaw/backward do not dominate.
   - Add auxiliary progress/stall prediction if needed.

4. Integrate and run the no-standoff benchmark.
   - Disable `--explore-standoff-route`.
   - Keep learned wall/outcome guard active.
   - Gate against the minimum fully learned implementation gate.

5. Iterate only against failures.
   - Do not chase every aesthetic metric initially.
   - Primary blockers are all-beacon success, hard contacts, body/stability failures, and no privileged route.
   - Orbit/time are second-stage once the fully learned runtime contract is met.

## Known Risks

- The current learned primitive outcome head is saturated near high blocked probabilities in some safe route states. It is useful as a guard but brittle as the sole planner.
- The clean try010 route still has high near-target angular sweep. A learned local planner may inherit this unless the teacher data is filtered or weighted.
- A single-scene policy may overfit. Generalization must be measured on held-out mazes before paper framing.
- Removing the standoff route will likely expose missing long-horizon structure. Do not mistake a first failure for evidence that learned memory is invalid; it means the local planner/topology layer is not yet carrying the route scaffold.

## Handoff Summary

Current state: clean live Go2 all-beacon demo is achieved, but it is scaffolded.

Next objective: remove or replace the privileged standoff-route scaffold with a learned local planner first, then a learned latent topology route.

Success criterion for the next milestone: a rendered live closed-loop Go2 video with standoff route disabled, all four beacons claimed, no hard stalls, no body/stability failures, and no privileged runtime map/route inputs.
