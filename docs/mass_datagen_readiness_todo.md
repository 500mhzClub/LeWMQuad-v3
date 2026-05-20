# Mass Data-Gen Readiness TODO

Purpose: track the work required before using Genesis-generated quadruped
rollouts as mass training data for a LeWM-style JEPA and downstream maze
solving.

Current status: **not ready for mass training**. The corpus blocker is
cleared (production_20260519T174524Z passes the audit and matches the
prior 247/312 sweep). Still open: multi-env RGB camera correspondence,
pilot end-to-end run with the spec mix, derived-label gating, P2 audits.

## P0 Blockers

- [x] Regenerate/reroll the production corpus until `audit_scene_corpus.py`
      reports `0 failure(s)`.
  - Prior failed corpus: `.generated/scene_corpus/production_20260518T231208Z`
    (7 `no_component_contains_all_beacons`, concentrated in
    `visual_sensor_stress`).
  - Diagnosis (2026-05-19): 6 of 7 failing scenes had `scene_seed_salt: 0`
    — `validate=True` did not run when that corpus was built. Replaying
    `_pick_valid_seed` for the val[0] slot under the current validator
    showed salt=0 fails and salt=2 succeeds, so a fresh regeneration on
    current `main` clears all 7 failures. No motif/validator fix required.
  - First rebuild: `.generated/scene_corpus/production_20260519T174524Z`
    — 350 scenes, passed the fragmentation-only audit, sweep 247/312.
  - **Current corpus**:
    `.generated/scene_corpus/production_20260519T220141Z` — rebuilt with
    the navigable-standoff validator (see Planner Yield Diagnosis below).
    350 scenes, `audit_scene_corpus.py` → `0 failure(s)`, sweep
    **267/312 = 85.6%**. This supersedes the 174524Z corpus; use it for
    pilot/production runs.
  - Acceptance:
    ```bash
    PYTHONPATH=lewm_genesis:lewm_worlds python3 scripts/audit_scene_corpus.py \
      .generated/scene_corpus/production_20260519T220141Z
    ```

- [x] Do not use inline multi-env RGB for training until per-env camera
      correspondence is fixed. (Decision: ship render-replay as the
      multi-env training path; defer inline-RGB fix until render-replay
      GPU cost becomes the bottleneck.)
  - Risk: `RolloutRunner._render_and_emit_rgb` sets a camera pose from env 0,
    so batched rollouts can produce missing or non-corresponding RGB for envs
    1..N.
  - Acceptance smoke (2026-05-19, `.generated/smoke_multienv_rgb`):
    2-env open_obstacle_field rollout → raw_pilot convert (audits pass)
    → render-replay env 0 and env 1 separately. Both renders return
    `invalid_frame_count=0`; env 0 camera pose (0.314, −2.483, 0.384)
    differs from env 1 (0.128, −2.698, 0.386); frame-0 RGB hashes
    differ (`f29198c4cab6` vs `23d5430d6822`).
  - Interim production path: run bulk rollout with `--no-rgb`, convert with
    `raw_training`, then render every env via render-replay.

- [x] Remove default privileged label leakage from render-replay RGB.
  - Fixed: target-route text is opt-in via `--overlay-target-label`.
  - Acceptance: training renders have no text overlay by default; any target
    overlay is opt-in for QA only, while `frames_rendered.jsonl` retains the
    label metadata.

- [x] Update the mass data-gen runbook to prefer MCAP-only rollout plus
      render-replay over inline multi-env RGB.

## P1 Training Corpus Requirements

- Mini-pilot (2026-05-19, `.generated/pilot_specmix`, 10 scenes ×
  4 envs × 500 blocks, default mix, `--no-rgb`) — all four pipeline
  gates green:
  - Convert with `raw_training`: 10/10 scenes pass (contract +
    data-quality audits).
  - Derive labels: 10/10 scenes (100,000 label rows total; missing
    command/episode_info counts both 0; local_graph_type histogram
    populated).
  - Spot render-replay (`replay_env_mode=batched`, 20 frames):
    `invalid_frame_count=0`, `camera_safety_unresolved_count=0`,
    `overlay_target_label=false`.
  - Realized mix on 20 000 blocks: route_teacher 27.1% /
    primitive_curriculum 22.5% / ou_noise 15.5% / loop_revisit 14.5% /
    frontier 10.4% / recovery 10.0%. Small-sample noise vs the
    spec — confirm at full-pilot scale.
  - Caveat: `--scene-limit 10` returned alphabetical scenes, all
    `large_enclosed_maze`. The full pilot must use a family-balanced
    selection (round-robin or per-family quotas).

- [ ] Scale beyond diagnostic corpus size.
  - Minimum serious split: `1000 train / 150 val / 150 test_id / 150 test_hard`.
  - Preferred split: `2400 train / 300 val / 300 test_id / 300 test_hard`.
  - Current production probe has `200` train scenes and is diagnostic only.
  - Scale-up bottleneck (2026-05-19 partial calibration —
    `.generated/calib_100_cpu`, killed at 6 of 100 scenes):
    `genesis_bulk_rollout.py` calls `build_scene_from_pack` per
    scene, which forces a Genesis kernel rebuild between scenes.
    Per-scene cost is dominated by that rebuild, not by simulation
    — sustained sim rate is ~13 k FPS on CPU (sweep), but the
    bulk-rollout per-scene wall is ~75–80 sec at `n_envs=4 /
    n_blocks=1000`. At that rate the 1450-scene minimum corpus
    takes ~30 hours serial. GPU at `n_envs=4` is *slower* than CPU
    (kernel launch dominates at small env count: 840 FPS vs CPU's
    13 k FPS), so GPU only pays off if `n_envs` is bumped enough
    to amortize launches.
  - Recommended scale-up tactic before launching the full run:
    1. Either raise `n_envs` to 16–32 and recheck wall on CPU
       (more sim per scene rebuild) or split the corpus into N
       shards and run N driver processes in parallel.
    2. Pick a `n_blocks` per scene that matches the average
       episode length you want (currently 1000 blocks = ~50 s of
       sim per env). Halving `n_blocks` halves wall cost.
    3. Add a family-balanced scene-selection mode to
       `run_mass_datagen.sh` (currently `--scene-limit` is
       alphabetical — first 100 of `production_20260519T174524Z`
       are all `large_enclosed_maze` / `local_composite_motifs`).
       The cleanest fix is to regenerate the corpus at the
       desired total via `standard_corpus_plan` and drop the
       limit.

- [x] Use the spec collection mix for the main JEPA corpus — default
      `DEFAULT_COLLECTION_MIX` in `lewm_genesis.collectors` already
      matches the §13 spec, so omitting `--collector-mix` from
      `genesis_bulk_rollout.sh` produces the spec mix.

- [ ] Produce optional balanced shards:
  - `success_route`: route-heavy expert success trajectories.
  - `explore_memory`: frontier and loop-revisit heavy.
  - `action_support`: primitive curriculum and OU heavy.
  - `contact_recovery`: recovery/contact heavy.

- [x] Gate every raw rollout with `raw_training` — driver
      `scripts/run_mass_datagen.sh` runs
      `convert_smoke_bag_to_raw_rollout.sh --quality-profile
      raw_training` per scene and aborts on any
      `contract_audit.pass != true` or
      `data_quality_audit.pass != true`.

- [x] Gate every rendered rollout with render-quality checks —
      new `scripts/audit_rendered_vision.py` checks
      `invalid_frame_count`, `camera_safety_unresolved_count`,
      `low_info_frame_count`, and `overlay_target_label`. Driver
      runs it after the render stage; both 2-scene and 10-scene
      smokes returned `0 failure(s)`.

- [x] Run derived-label generation for every raw rollout —
      `derive_raw_rollout_labels.py` produces `labels.jsonl` +
      `summary.json` with `cell_id`, `yaw_bin`, `local_graph_type`,
      `clearance_m`, `traversability_forward_m`,
      `landmarks[].{visible,bearing_body_rad,range_m,bfs_distance_cells}`,
      `integrated_body_motion_block/window`, `episode_id`,
      `episode_step`, `nearest_cell_distance_m`, `stuck_label`. New
      `scripts/audit_derived_labels.py` enforces schema +
      `missing_command_count == 0` per scene. Driver passes
      `--scene-id` from the rollout directory name to satisfy the
      resolver.
- [ ] Follow-up: per-row labels do **not** carry `command_context`
      (primitive_name / command_source). It is recorded on every
      render frame's `frames_rendered.jsonl` and on every
      `command_block` topic in raw_rollout, so downstream JEPA
      builders can join on `(timestamp_ns, env_idx)`. Decide
      whether to also denormalize it onto each label row before
      scale-up training, to avoid the join cost.

## P2 JEPA/H-JEPA Audits

- `scripts/audit_jepa_corpus.py` (added 2026-05-19) walks a
  `run_mass_datagen.sh` output root and emits a single audit JSON
  covering the first set of histograms. Validated on the 10-scene
  pilot (`.generated/audits/pilot_specmix_audit.json`):
  `scene_count=10`, `label_row_count=100000`,
  `effective_sequence_count=99370` (window=16), all primitives and
  command sources counted, local_graph_type histogram populated,
  `relaxed_claim_counts_by_family={large_enclosed_maze: 0}`.

- [x] Publish effective LeWM sequence counts after reset/window
      filtering — emitted as `effective_sequence_count`
      (per-row, episode_step ≥ window_ticks). Per-scene rollup
      can be added later if a scene-by-scene view is needed.
- [x] Publish per-family, per-primitive, and per-command histograms
      — `per_family`, `per_primitive`, `per_command_source`.
- [x] Publish local-graph-type coverage — `local_graph_type_counts`
      and `per_family_local_graph_counts`.
- [ ] Publish same-place positive and same-scene hard-negative pair
      counts — deferred; needs a richer landmark-visibility +
      pose-graph join than the per-row labels carry. Sketch a
      second pass that builds pair stats from
      `labels.jsonl` + `frames_rendered.jsonl`.
- [ ] Publish loop-closure and false-loop pair counts — same as
      above, deferred.
- [x] Publish goal-image observations per landmark and yaw bin —
      `goal_image_landmark_yaw_bins` keyed by
      `"<object_id>|yaw_<bin>"`.
- [x] Audit relaxed route-teacher claims separately from rendered
      landmark visibility labels — `relaxed_claim_counts_by_family`
      already separates the two; cross-checking against rendered
      visibility per frame is the deferred follow-up under
      Route-Teacher Patch Follow-Up.

## Planner Yield Diagnosis (2026-05-19)

Systematic per-family diagnosis of route-teacher beacon yield.
Sweep config: 3 scenes × 4 envs per family, route_teacher=1.0.

Root causes by family:
- **large_enclosed_maze / visual_sensor_stress**: block-budget limited.
  At 400 blocks 64% / 81%; at 1000 blocks **89% / 98%**. The long paths
  at 400 blocks were unfinished episodes, not planner bugs. Fix is a
  config change — probe + production at ≥1000 blocks (runbook §3 updated).
- **small_enclosed_maze**: narrow-corridor approach. Beacons reachable
  only via ≤0.35 m corridors that are grid-free but unnavigable for the
  trained PPO. Fixed with a family-scoped standoff corridor-width
  penalty in `route_teacher._standoff_corridor_penalty` (75% → 87.5%,
  no cross-family regression).
- **local_composite_motifs**: systemic motif/validator gap. Corpus scan
  found **50% of its beacons in single-LOS alcoves** (vs ≤7% elsewhere)
  — grid-reachable but the only sightline is through a sub-navigable
  corridor. Fixed by a new corpus-validator criterion
  (`min_navigable_standoffs_per_beacon`) that rerolls such seeds.
  Result on the rebuilt corpus `production_20260519T220141Z`:
  local_composite **75% → 100%**, medium **69% → 92%**, small
  **88% → 92%**, aggregate **80.1% → 85.6%**, no family regressed
  (loop_alias −2.1pp is single-beacon sampling noise).

Full benchmark with all changes (route_teacher=1.0, 3 scenes × 4 envs):

| family | baseline | nav@400 | nav@1000 | needs 1000? |
| --- | --- | --- | --- | --- |
| open_obstacle_field | 100% | 100% | 100% | no |
| rough_local_dynamics | 100% | 100% | 100% | no |
| loop_alias_stress | 93.8% | 91.7% | 95.8% | marginal |
| local_composite_motifs | 75.0% | 100% | 100% | no |
| small_enclosed_maze | 75.0% | 91.7% | 95.8% | marginal |
| medium_enclosed_maze | 68.8% | 91.7% | 91.7% | no |
| visual_sensor_stress | 81.2% | 81.2% | 97.9% | **YES** |
| large_enclosed_maze | 63.9% | 63.9% | 88.9% | **YES** |
| **aggregate** | **79.2%** | **85.6%** | **94.9%** |  |

- `baseline` = old corpus (`174524Z`, fragmentation-only) + planner
  without the corridor penalty @ 400 blocks.
- `nav@400` = navigable corpus (`220141Z`) + scoped corridor penalty
  @ 400 blocks.
- `nav@1000` = same as nav@400 but 1000 blocks/episode.
- **Scale-up only where necessary**: only `large_enclosed_maze` and
  `visual_sensor_stress` need 1000 blocks; everything else plateaus by
  400-500. Runbook §4 documents the tiered budget.

Validator change details:
- `audit_scene_reachability(..., min_navigable_standoffs_per_beacon=N)`
  requires each beacon to expose ≥N LOS-valid standoffs in a navigable
  (≥0.50 m corridor) cell of the canonical component. Default 0
  (backward-compatible); enabled =1 in `_default_scene_validator`,
  `_build_one` safety net, and `audit_scene_corpus.py` (`--min-navigable-standoffs`, default 1).
- Shared `corridor_width_m` helper in `planning_grid.py` is used by both
  the validator and the route teacher so they agree on "navigable".
- Reroll feasibility confirmed: 0/30 local_composite slots exhaust the
  salt budget (mean salt 3.2, max 17, budget 32).
- Remaining cross-family gap is large/visual budget — addressed by the
  ≥1000-block probe/production recommendation, not a code change.

## Route-Teacher Patch Follow-Up

- [ ] Keep the `small_enclosed_maze` arrival relaxation as a collection
      heuristic only.
- [x] Add summary telemetry for relaxed claims (`relaxed_claim_count` +
      per-event `relaxed_claim_events` carrying family/scene/env/beacon).
      Implemented in `RouteTeacher.relaxed_claim_count` /
      `relaxed_claim_events` and surfaced in `per_env_metrics`. Tests in
      `test_collectors.py` assert non-wedge families stay at zero.
      Verified by the post-edit sweep on
      `production_20260519T174524Z`: only `small_enclosed_maze` emitted
      relaxed claims (17 across 3 scenes × 4 envs); every other family
      was 0.
- [ ] Audit relaxed claims against derived/rendered landmark visibility
      per frame — needed in the pilot to confirm the relaxation isn't
      paying off only on frames where the beacon isn't actually rendered.
- [ ] Do not broaden relaxation to `medium_enclosed_maze` or
      `local_composite_motifs`; prior sweeps regressed those families.

## Suggested Execution Order

1. ~~Fix/reroll corpus validation failures.~~ (2026-05-19: corpus
   `production_20260519T174524Z` audits clean.)
2. ~~Run one full smoke suite over the new corpus.~~ (2026-05-19:
   sweep aggregate 247/312 = 79.2%, matches prior baseline.)
3. ~~Generate a small pilot shard with the spec mix.~~ (2026-05-19:
   `.generated/pilot_specmix`, 10 scenes, all four gates green.)
4. ~~Convert, render, derive labels, and run all quality audits.~~
   (2026-05-19: `scripts/run_mass_datagen.sh` chains all four with
   fail-fast audits via `audit_rendered_vision.py` and
   `audit_derived_labels.py`.)
5. **Next**: scale to the minimum training corpus
   (1000/150/150/150). Open question: CPU vs GPU backend for the
   full bulk rollout, and whether to run as one foreground job or
   in chunked overnight runs. Family-balanced scene selection
   needed — current driver uses alphabetical `--scene-limit`.
6. Run the P2 JEPA/H-JEPA audits after the minimum corpus exists.

## Design Decisions

### 2026-05-20 — Route teacher: reset-keep-collector (not revisit) for §13 mix

**Decision:** when the route teacher claims every beacon, the rollout
respawns the env from a fresh corridor cell **but keeps it assigned to
route_teacher** (`_check_and_reset_completed_envs` calls the current
policy's `on_episode_reset`, not `scheduler.on_episode_reset`).

**Why:** §13 specifies 30% route by *macro transition* (block). With the
original reset-on-completion *redraw*, completed route episodes were
handed to other collectors, so route accrued only ~18% of blocks (route
episodes finish fast). Two alternatives were tried and rejected:
- *deficit-balanced scheduler* — unstable feedback loop (route runaway to
  68%); also exposed that a blocks-on-reset metric mis-counts
  non-completing collectors. Reverted.
- *revisit-on-completion* (route re-targets claimed beacons) — overturned
  the deliberate, tested "claim-all-then-hold; do not re-target a visited
  beacon" contract (broke 7 collector tests) and only reached 22.9%.
  Reverted.
Reset-keep-collector preserves that contract (43 tests green), keeps
fresh-spawn diversity, and lifts route to ~26% of blocks (3-scene sample;
expected to average closer to 30% corpus-wide).

**How to apply:** treat ~26-30% route transition share as compliant;
§13 is advisory, and the hard gates (§10 situation coverage, §11 primitive
support) already pass with large margin. Do not switch route_teacher to
revisit mode — that is loop_revisit's role.

## Design Decisions / 2026-05-20 — Augmentation scope for minimum-tier generation

**Decision:** generate the minimum corpus with the **current** generation-time
domain randomization only — per-scene seeded color/HSV jitter, lighting
(direction/intensity/ambient/specular), camera-extrinsic mount jitter,
floor/obstacle friction & restitution, and visual-stress distractors
(`lewm_worlds/randomization.py`). No additional augmentation is added before
the run.

**Why:** the §14 items not yet implemented split by *where they belong*:
- *RGB sensor noise / blur / compression* — deliberately deferred to
  **train-time, per-window**. A JEPA predictor learns dynamics across a
  context+prediction window; photometric aug must be temporally consistent
  within the window or it injects spurious dynamics, and baking one fixed
  noise realization per frame wastes the augmentation. So clean RGB out of
  the data-gen pipeline is the intended contract, not a defect.
- *Robot mass/inertia/actuator strength + randomized command/sensor latency*
  — these shape the recorded proprio/base-state dynamics and **cannot** be
  added post-hoc, so they are the genuine gen-time gaps. Only a fixed
  one-step action latency exists today (`RolloutConfig.simulate_action_latency`).
  Deferred: out of scope for the minimum tier (no sim/robot-transfer claim yet).
- *test-transfer-visual / -physics holdouts* — §14/§17 make these conditional
  on a transfer claim; splits remain train/val/test_id/test_hard. Deferred.

**How to apply:** when a sim-to-real / robot-transfer claim becomes in-scope,
revisit body-dynamics DR (mass/actuator/latency) and carve held-out
visual+physics themes into test-transfer splits *before* regenerating. Do not
add RGB noise/blur to the renderer — that is the trainer's per-window job.

## Design Decisions / 2026-05-20 — Render-time CC0 surface textures

**Decision:** apply semi-realistic CC0 diffuse textures to floor / walls /
obstacles at render time. Assets (ambientCG, **CC0 1.0**, no attribution) live
in `assets/textures/{floor,wall,obstacle}/` (see that dir's README for
provenance). Scope is **floor + walls + obstacles, diffuse-only** (no
normal/roughness). Landmarks, distractors, and slick_patch stay solid color
(task identity / decoys / physics cue).

**Why / how it works:** validated against Genesis 0.4.6 empirically —
- the floor `Plane` textures and auto-tiles (~1 texel/m) directly;
- `gs.morphs.Box` **cannot** carry a texture (UV-less), so textured
  wall/obstacle boxes are rebuilt as UV-mapped cube **meshes**
  (`textures.cached_box_obj`, per-face UVs scaled to size for ~0.7 tiles/m,
  `collision=True`, `convexify=True`, `file_meshes_are_zup=True`);
- selection is deterministic per scene (`visual_seed`+`scene_id`), one map per
  category, so re-renders reproduce (§15).
Implemented in `lewm_genesis/textures.py` + `scene_builder.py`
(`build_scene_from_pack(apply_textures=...)`, default **False**). Render-only:
`render_replay_genesis.py` passes `apply_textures=True` (`--no-textures` to
disable); the rollout/physics path keeps fast box primitives unchanged.

**Validation:** all 8 families rendered egocentric → `invalid_frame_count=0`
(0.000%, gate <0.1%), `low_info_invalid=0`; audit gate green. Textured walls
also reduce wall-staring low-info frames. 166 base-python tests pass (+5 new
`test_textures.py`); rollout path byte-identical at `apply_textures=False`.

**How to apply:** keep textures render-side; never bake them into rollouts.
Adding/replacing maps = drop `*_Color.jpg` (CC0) into the category dir; tiling
density is `textures._DEFAULT_TILES_PER_M`.
