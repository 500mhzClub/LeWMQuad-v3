# Wide-maze physical demo: validated navigation demo candidate (2026-06-13)

**Current verdict:** v43 is the first successful physical seek with a
route-valid terminal beacon spur and a constrained goal target. It stops
perceptually at **0.187 m** after a 5.41 m initial separation, with 29 align
blocks and 4 veto escapes. Dense sampled-frame review shows forward-dominant
maze traversal, one brief alignment sequence, no sustained wall-walking or
spin loop, and a clear final stop at the blue beacon. It is a valid navigation
demo candidate, not yet a polished final demo: the physical run recorded one
fall, and the separately rendered replay applied 6 drift-correction snaps
across 103 seek blocks.

**Perception-nav claim boundary:** v43 used privileged manifest-grid queries for
runtime local collision avoidance. Its learned topological localization, route,
goal matching, and perceptual stop are perception-driven, but the result is not
yet a pure-perception seek. The enforced runtime obstacle-source contract is
documented in `docs/lewm_perception_nav_runtime_contract_2026-06-13.md`.

The original v9 user-review verdict remains valid historically: its endpoint
metrics passed while its behavior was unacceptable. Endpoint success alone is
still not sufficient.

**Current artifact:** `.generated/topo_nav/topo_nav_wide_maze_demo_v43.mp4`
(41.83 s, 502 encoded frames, 896x496) — full learned stack under the PPO gait
in the wide-corridor maze
(`.generated/scene_corpus/wide_maze_demo`, medium_enclosed_maze at ~1.21 m
cell pitch vs the standard ~0.94 m).
Results: `.generated/topo_nav/wide_maze_phys_v43.json` (one deterministic
physical validation trial, success). Harness: `scripts/benchmark_topo_nav_e2e.py`
(vulkan venv =
`.generated/venvs/genesis_render_vulkan/bin/python`).

The repro command is the v9 launch (seed 20260609):
`--goals-per-scene 6 --mode physical --apply-textures --policies topo
--goal-mode colourful --tour-max-cells 18 --tour-max-blocks 600
--tour-reprise-cells 3 --seek-max-blocks 700 --subgoal-budget 34
--edge-budget 26 --tour-capture-every 3 --demo-video <path>`.

Where the prior state was: 0/7 wide-maze trials (kinematic + physical), all
routes ~100% reversed edges, colourful goal contract never firing. Five
mechanisms were found and fixed, in order:

## 1. Tour-reprise loop closure (`--tour-reprise-cells 3`)

The memory is a time-DIRECTED chain; at the bare tour end every edge points
backward, so **no goal is forward-routable from the seek start** (measured:
every candidate route 100% reversed; per the bidirectional gap, reversed-edge
arrival never confirms). Re-walking the first ~3 outbound cells after the DFS
unwinds makes the §5.4 filter re-localize onto the chain's START;
`OnlineTopologicalMemory.update` mints the late→early MAP-transition edge,
putting the entire tour downstream of the seek start. Kinematic went 0/7 →
success (0.45 m) immediately.

## 2. Route-aware goal selection + per-candidate diagnostics

Goal selection now scores every candidate's planned route
(`navigator.plan_node_path`) for reversed edges and requires forward-mostly
routes; `[goal-cand]` lines print (cell, hops, sat, match, alias, rev/len).

## 3. Goal-image aliasing measured and filtered (`aliased_far_cosine`)

Bland tour-heading goal latents match far (≥2-hop) keyframes at 0.93-0.98 —
within ~0.01-0.02 of their own match — so a fixed tau_arrive cannot separate
true from false stops on bland corridors (false stops 2.4-4.1 m out, in both
modes). Selection rejects candidates with alias ≥ tau_arrive − 0.02; the
alias varies per tour draw and an unaliased candidate usually exists.
Landmark-STANDOFF goals (0.72 m, facing) fix saturation (0.48-0.55) but match
drops to ~0.7 < tau_goal (yaw-bound matching) → still unplannable; legible
colourful goals need landmark-facing views in memory first.

## 4. Physical-mode robustness

- **Fall recovery purges the belief window and re-aligns** (tumble frames are
  far outside the smooth-locomotion distribution — same logic as the Stage-4b
  freeze-during-ALIGN fix; replans clustered right after falls).
- **2-of-last-4 arrival streak** instead of 2 consecutive (gait bobbing
  flutters the cosine around threshold).

## 5. Two-tier goal-conditioned arrival (the decisive perception fix)

`tau_arrive_eff = min(tau_arrive, max(floor, goal_alias + 0.02))` per goal
(deployment-valid: alias comes from the robot's own memory + goal image).
Walk/servo blocks use tau_eff; ALIGN-scan blocks (unaliased goals only) use a
high bar `--tau-arrive-scan 0.95`, because the ADJACENT cell's align scan
matches a bland goal above tau_eff (measured stop at 1.86 m — alias only
bounds ≥2-hop views), while only the true goal-place align reaches ~0.99.
Measured profile on the winning trial: align 0.981@0.24m, 0.971@0.22m,
0.956@0.43m — nothing distant above 0.94. Under gait the WALK cosine never
reaches the kinematic 0.90 (walking camera pitch/height), which is why the
fixed bar missed physically-arrived robots twice (0.37 m, no stop).

## Honest caveats

- N=1 metric-passing physical trial in this scene; per-draw success ≈ 1/3
  kinematic, lower physical. The losing draws fail on long chain retraces
  (gait drift + falls) — routes here are full tour retraces, not shortcuts.
- The goal image is a bland corridor view (tour-heading contract); the
  colourful standoff contract fired once (1/6 draws) but failed on a 6-hop
  traversal. Legibility vs plannability is still the open tension.
- The cosine-resolution of perceptual arrival in this latent is ~1 cell; the
  scan tier (0.95) is calibrated on this scene's profile, not swept.

## Why it hits walls and spins (the actual blocker)

This is the already-registered Stage 4c gap, now visible on camera:

1. **WALK is open-loop under gait.** After ALIGN the controller just issues
   `forward_slow` blocks. The PPO gait's heading tracking drifts (the
   physical-mode groundwork already measured "headings/displacements diverge
   from every kinematically-calibrated constant"), so the robot veers into
   the corridor wall it was aligned parallel to. There is no closed-loop
   heading hold (IMU yaw vs the aligned bearing) during WALK.
2. **The kinematic veto reacts too late and with the wrong action.** When
   `_feasible_fraction` finally fails near a wall, the fallback is
   `yaw_right` — repeated, this is the on-camera "freak-out spin", often
   followed by edge-budget exhaust -> replan -> another full-revolution
   scan.
3. **Full-revolution ALIGN scans read as spinning** even when working as
   designed: every node arrival triggers up to a 360° scan + turn-to. Under
   gait the scans are slower and more frequent (more replans), so spinning
   dominates screen time.
4. Falls add stand-up resets mid-corridor that restart ALIGN (more
   spinning).

Concrete next steps (registered Stage 4c, in order): (a) closed-loop heading
hold during WALK — track the post-ALIGN IMU bearing with small yaw
corrections inside the walk blocks; (b) wall-aware veto response — back up /
steer away instead of yaw_right spam; (c) scan budget — early-out the
revolution at high keyframe cosine and cap scans-per-edge; (d) per-level
gait telemetry (alignment heading error, per-block displacement, filter
coherence under gait imagery) to verify each fix in isolation. Until (a)+(b)
land, physical wide-maze video stays non-demo-grade.

## 2026-06-12 update: collision-selection fixed (v12); blocker moves to perception

The wall-walking had a sharper root cause than "open-loop WALK": **the veto's
motion model could not represent a collision the gait was about to have.**
`_feasible_fraction` simulated ONE 0.5 s block (= 0.10 m of forward_slow) of
COMMANDED motion for the BASE CENTER against a 0.20 m-inflated grid; the Go2
nose extends ~0.25 m past base center and the gait overruns block boundaries,
so "feasible 1.0" and a wall strike were fully compatible — the planner then
kept selecting forward against the wall until the robot fell, and fall
recovery stood it back up at the contact point.

Fixes (all physical-mode only; verified kinematic semantics untouched):

1. **Capsule + horizon veto** — nose/tail probes (`--veto-body-radius-m
   0.25`) simulated over `--veto-horizon-blocks 2`.
2. **Proprioceptive stuck detector** — two consecutive forward blocks with
   <3 cm displacement = contact the grid missed -> `backward` off + re-align
   (never needed to fire in v12; the veto caught everything first).
3. **Escape set gains `backward`**; 3 consecutive vetoes on one edge ->
   re-scan instead of veto<->heading-hold oscillation.
4. **Fall recovery stands up at the nearest grid-free point**, not in the
   wall contact.
5. **Lost-replan fallback executes physically** (was a kinematic set_pos
   teleport mid-gait).
6. **Arc->straight fallback**: an arc's predicted sweep clips the inflated
   side band far more often than the straight line; try `forward_slow`
   before escaping (v11 interim run measured 56-71 % of budget burned in
   align/escape loops without this).

**v12 result (`wide_maze_phys_v12.{json,log}`, same seed/scene/config as
v9): 2 falls TOTAL across 6 trials (~2,500 gait blocks) vs v9's 6+1+3 across
3; zero stuck-pushes; heading RMS 0.12-0.14 rad; replay route-correction
snaps 0/700 (v9 needed the snap gate because the gait wandered).** The
collision-trajectory-selection defect is closed.

Task success was 0/6, and that is now a PERCEPTION/NAV story, all
pre-registered gaps: (i) bland tour-heading goals false-stop (3/6 draws,
worst alias 0.997 stopped 4.2 m out; live gait views cross alias bounds
measured on stored keyframes); (ii) align-scan dominance (up to 487/700
blocks — scans-per-replan cap still unimplemented, Stage 4c item c); (iii)
reversed-edge clusters near the goal (bidirectional gap). The colourful-goal
contract fired 0/6 draws in this maze — legibility vs plannability remains
the open tension. `topo_nav_wide_maze_demo_v12.mp4` shows the final
(failed-seek) attempt: clean corridor walking, no wall strikes, but no goal
arrival — demo-grade locomotion, not yet a demo-grade seek.

## v13-v27 (2026-06-12/13): beacon-goal campaign — first metric success

Goal contract upgraded per request: the goal must be one of the colourful
beacons hidden in the maze, photographed from an executed look-at standoff
(`--tour-landmark-lookat --goal-mode colourful --demo-require-colourful`).
Chain of defects found and fixed on the way (full diagnosis in memory
`project_topo_wide_maze_demo.md`):

- **Latent is colour-blind** — cross-beacon standoff close-ups at cosine
  1.000 (seq4_e9 encodes place/yaw, not hue). Fix: pixel-space colour gate
  (`_colour_signature` center-crop saturation+RGB; `_colour_match`) wired
  into the perceptual stop, alias scoring, and goal-node verification.
- **Look-at views must be `insert_spur` terminal nodes** (inline splicing
  detoured every route into the pillar: 195 escapes/48 realigns) **and
  `in_filter=False`** (in-filter spurs siphoned the posterior: subgoal
  arrivals 1/700 blocks).
- **Alias far-ness is metric** (>=1.5 cell pitches Euclidean), not graph
  hops (wall-adjacency artifact counted the goal's own node as a far alias).
- **Dead-edge `avoid_edges`** (+25 Dijkstra both ways) breaks the
  idempotent-replan loop; per-edge realign cap (2) -> replan.
- **DECISIVE: memory frozen during seek** (`memory.frozen = True`,
  localization-only). Mid-seek node commits were posterior black holes (MAP
  locked p*=1.00 on a fresh node); v25 align blocks fell 488 -> 71 and the
  colour-gated stop fired at the beacon for the first time (0.86 m).
- **Bounded final-approach release** (v27): after the arrival streak, servo
  at most 6 extra blocks and stop when the goal cosine falls 0.004 off its
  running peak (closest approach) or forward goes infeasible. v26's
  unconditional while-feasible suppression never released (drove through
  0.44 m, drifted out to 1.05 m).

**v27 result (`wide_maze_phys_v27.{json,log}`, seed 20260609, cell-22 blue
beacon, 5 hops, initial 4.99 m): final 0.46 m, `success_eval=True`,
`perceptual_stop_correct=True`, 0 falls, 0 fallback blocks, subgoal progress
6/6 (1.0), align 71 blocks, heading RMS 0.12 rad.** Cosine profile peaks
0.965@0.54m / 0.959@0.46m — the stop fired at closest approach. Goal-image
-> stop chain is fully deployment-valid (no privileged signals; colour gate
is raw pixels). The cell-6 second draw was skipped by
`--demo-require-colourful` (its beacon contract did not fire; that corner
remains the known hard-localization region, 4.67 m in v25/v26).
`topo_nav_wide_maze_demo_v27.mp4` (695 frames, ~58 s) is the demo-candidate
video — pending human review of behaviour on video before any success claim
is final.

This was later invalidated as a topological-navigation success: the synthetic
anchor-to-standoff spur crossed a wall. The learned final servo happened to
wander around the unrepresented route. v28 made that wandering longer and
exposed the defect with 34.41 m of path and 4 falls.

## v28-v43 (2026-06-13): route-valid terminal goal and bounded completion

The v28 clearance selector fixed the original binary veto behavior, but exposed
that the final goal spur was not a real free-space edge. The subsequent work
made the graph and terminal behavior explicit:

- Synthetic look-at spurs are accepted only when their centerline is free.
  When necessary, the mapping tour adds a short real graph excursion to a
  route-valid landmark anchor.
- Terminal spurs are goal-only nodes: excluded from localization and forbidden
  as transit shortcuts. Colourful stand goals constrain every plan/replan to a
  colour-matching terminal node.
- Frozen seek memory localizes without accumulating novelty state. Failed edges
  are penalized bidirectionally, alignment is bounded, and the initial seek
  preserves the validated tour-end MAP.
- Every directed mapping transition records its measured IMU bearing. Seek
  alignment uses that bearing instead of performing a visual revolution for
  every edge.
- Goal arrival no longer waits for lagging graph-index state. Two high-cosine,
  correct-hue observations arm a six-block, strict-feasibility-gated forward
  completion. v42 showed why: the robot saw the correct blue beacon at
  0.87/0.78 m while `at_final_node` remained false, then physically passed
  within 0.20 m and left.

**v43 result (`wide_maze_phys_v43.json`, seed 20260609, route-valid cell-32
blue-beacon stand, 7 graph hops): final 0.187 m, `success_eval=True`,
`perceptual_stop_correct=True`, 8.79 m physical path, 103 seek blocks, 0
fallback blocks, 1 fall, 29 align blocks, 4 veto escapes, 1 edge realign,
heading RMS 0.148 rad.** The constrained memory route is nodes 5→16 and the
terminal match is cell-correct. The final colour/cosine sequence rises from
0.842@0.866m to 0.989@0.187m and remains the correct blue hue.

**Remaining demo caveats:** N=1 for the corrected route-valid contract; one
physical fall remains; the display replay required 6 drift-correction snaps
over 103 seek blocks. The replay is useful for reviewing navigation decisions,
but those snaps must be removed before presenting it as an unqualified
locomotion-quality demo.
