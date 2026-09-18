# Early heading-release ablation on the exposed turn-cycle layout

The preceding supervised trial reached the goal too late to return. Its arc
fallback was never used. During 300 seconds of frontier exploration, the
existing early heading-release rule reversed a latched recovery turn 88 times;
83 were followed by another latch within two seconds. This trial suppresses
only that early release, using the existing tested ablation mixin. Measured
target completion, translation-progress release, clearance-driven direction
switches and all outer selection and dispatch checks remain active.

The arc fallback remains enabled in both conditions, so the comparison changes
one rule. The checkpoint, six candidates, sensor parameters, 4800-tick budget,
layout/physics/appearance seeds, CPU group and timing remain unchanged. The
runtime's method order is checked to place the ablation at precisely the
existing release layer. The launcher reuses the preceding trial's collection
and evaluation functions and supplies explicit ablation metadata.

This is one exposed-layout execution, outside the completed fifteen-run fresh
cohort. Asynchronous trajectories need not match. Record every suppressed
release opportunity, actual commands, later heading completion, other recovery
activation, physical arrivals, contacts, tracking and timing. A success without
eligible release states cannot establish the rule's effect. No fresh-layout,
JEPA-superiority, real-sensor or hardware claim follows.

Plan: `docs/go2_view_arc_no_early_release_plan_2026-09-17.json`.
Launcher: `scripts/run_go2_view_arc_no_early_release_development.py`.
Root: `go2_view_arc_no_early_release_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001`.

Before collection, superseded depth from the completed declared-gap learned
success and clean-transfer learned layout-2 success was retired. All diagnoses,
non-depth evidence, active references and failures remain. Reclaimed 3,660,001,280
allocated bytes; recording volume then had 5,131,517,952 bytes free.

Launched in session 46683, owner PID 4085610, CPUs 8–15,24–31.
Native launch acknowledged. Owner exited 0 after full archival; the standard
full-mission evaluator completed successfully.

## Outcome and exercised treatment

**No goal, no return; budget failure at 480.84 simulated seconds.** Zero
disallowed contacts, no pipeline faults, 1190/1200 on-time plans. The complete
raw-depth failure recording remains retained. The ablation is not promoted as
a navigation improvement.

Seven early-release opportunities were suppressed, at frames 600,604,...,624.
Each retained a right turn and had matching commands physically applied. The
new measured-view arc fallback was never selected. The treatment therefore
was exercised, unlike the earlier unexercised heading-release comparisons,
but did not restore navigation. The reference reached the goal late and this
trial did not; different asynchronous trajectories preclude attributing that
whole outcome difference to the seven decisions.

There were 1045 hold selections. From frame 636 all six forecast paths became
blocked; subsequent measured-obstacle and latched-window vetoes prevented
movement. The first current obstacle veto was at 64.62 s using the 64.50-s
observation, with measured cell distance 0.44777 m. This is not a timing or
tracking failure: maximum registered-position error was 5.18 mm.

## Entry cause: incompletely observed wall clearance

The physical base centre first crossed the nominal 0.45-m wall margin at sampled
planning frame **576, before the first suppressed release at frame 600**.
At frame 576, physical nearest-wall distance was 0.44398 m while the stored
map's current clearance was 0.49419 m. Thus disabling early release cannot be
credited with causing the first margin crossing. The physically defined
circular margin is not the articulated collision geometry; zero contacts and
a nominal-margin violation can coexist.

Ten retained depth-pair samples were reconstructed with exact delivered-noise
digest checks. At sampled frames 560,568,576,584,600, neither camera's stride-4
point cloud contained above-floor points associated with the nearest wall
(world height 0.03–0.65 m, 15-mm wall-box association tolerance). The closest
wall point at base height projected behind both cameras. Many valid pixels
remained elsewhere: at frame 576, primary/auxiliary counts were 302016/307200.
This was incomplete directional coverage, not total depth loss.

As the robot turned, the wall entered the available view. At frame 620 the
nearest associated auxiliary wall point was still 0.63252 m away although
physical nearest-wall distance was 0.45184 m. By frame 630 the auxiliary sample
distance was 0.44935 m, close to physical 0.44883 m; at frame 632 it was
0.44660 m versus physical 0.44779 m. The new close observations account for the
late clearance rejection without requiring a large pose error. This is a
bounded sample diagnosis, not proof of complete surface visibility or an
exhaustive reconstruction of earlier map history.

Frames 700–4800 contain 1026 sampled physical clearances: minimum 0.44450 m,
median 0.44851 m, maximum 0.45678 m; 805 are at or below the nominal margin.
The late lock therefore cannot be dismissed as merely a conservative map.
No safe escape action or relaxed threshold is established.

Reader: `scripts/read_go2_view_arc_no_early_release_development.py`.
Evidence in the trial root: `clearance_entry_diagnosis_v1.json` and the paired
camera-coverage figure `clearance_entry_camera_coverage_v1.png` / `.svg`.
Native poses and wall geometry were used only after execution for evaluation.

The next scientific question is how the observed map and candidate-clearance
rules admitted motion near an incompletely observed wall. Reconstruct the
recorded map up to the first crossing to distinguish a missing wall span from
discarded or misregistered observations. Test a coverage or measured-surface
intervention only after that distinction; do not feed native wall geometry to
the controller or add another recovery override based solely on the late stall.
The original completed-cohort runtime remains the reference, and neither
exposed recovery trial establishes improved reliability or JEPA advantage.

That map reconstruction is now complete. It matched all 160 recorded map-count
and available current-clearance witnesses without removing any obstacle cell.
The missing geometry was the wall's nearer reverse face; older observations
represented its opposite face. See
`docs/go2_observed_wall_face_clearance_diagnosis_2026-09-17.md` for the wall-face
readout and the recorded footprint-coverage probe. No new navigation result is
claimed by that analysis.
