# Wide-maze physical demo: NOT demo-grade (2026-06-11)

**Verdict (user review of the video): NOT a success.** The endpoint metrics
pass (final **0.22 m**, perceptual stop correct, 3 falls recovered,
all-forward route) but the BEHAVIOR is unacceptable: the robot walks into
walls essentially all the time, then "freaks out" and spins. Endpoint
metrics (final distance + stop flag) do not measure behavior quality —
grading on them alone was a mistake. The fixes below are real and keep the
*navigation* layer working; the *locomotion execution* layer under gait is
what fails (see "Why it hits walls and spins" at the end).

**Artifact:** `.generated/topo_nav/topo_nav_wide_maze_demo.mp4` (~61 s, 706
frames) — full learned stack under the PPO gait in the wide-corridor maze
(`.generated/scene_corpus/wide_maze_demo`, medium_enclosed_maze at ~1.21 m
cell pitch vs the standard ~0.94 m).
Results: `.generated/topo_nav/wide_maze_phys_v9.json` (3 trials recorded,
1/3 success — draws are reported honestly; the demo keeps the succeeding
trial). Harness: `scripts/benchmark_topo_nav_e2e.py` (vulkan venv =
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
