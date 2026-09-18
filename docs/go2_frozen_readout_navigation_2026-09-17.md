# Fixed four-arm native comparison of frozen motion readouts

The preceding prediction study found no JEPA advantage: on the same 3726
navigation windows, whole-window position RMSE was reference 7.14 mm, JEPA
readout 8.37 mm, supervised readout 7.62 mm and untrained-feature readout 6.90 mm.
The JEPA readout nevertheless improves on the original JEPA model used in the
current controller (9.76 mm). Better prediction alone does not establish mission
completion, and forecast errors also change recovery-action eligibility.

Run exactly four missions, in order: command history, JEPA readout, supervised
readout, untrained-feature readout. Use the existing `InterruptedViewRuntime`,
weak/strong visual thresholds 48/96, the same exposed layout 1, six candidates,
4800-tick budget, 2-mm depth noise, ideal gyro, CPU group and planning deadlines.
The command-history arm computes the JEPA readout but uses only the fitted
command-history prediction for planning. No model or controller changes are
allowed between assignments; no extra repetitions belong to this batch.

Primary outcome is independently verified physical goal-and-home arrival with
zero disallowed contact. Retain backtracking, tracking, timing, forecast errors
and every failure. Keep full batch depth while this comparison is active. Run
native jobs sequentially with no concurrent heavy analysis or training.

Plan: `docs/go2_frozen_readout_navigation_plan_2026-09-17.json`.
Launcher/evaluator: `scripts/run_go2_frozen_readout_navigation_development.py`.
Output roots: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_readout_navigation_{number:02d}_{arm}_noise_2mm_native_layout01_4800_v1_attempt_001/`.

This is a small development navigation diagnostic, with one execution per arm
on an already exposed maze. It cannot establish repeatability, a reliable
causal JEPA contribution, fresh-layout generalization or deployment readiness.
All four arms must be reported, including failures. The command-reference and
untrained-feature arms are controls; they do not replace the broader learned
world-model objective. No readout is promoted to the default by launching this
experiment.

## Complete outcome

All four assigned missions are independently evaluated: **0/4 goal arrivals,
0/4 round trips, zero disallowed contacts and no pipeline faults**. Every mission
exhausted its fixed 4800-tick budget. This exposed-maze diagnostic provides no
navigation or JEPA improvement claim. All four full failure recordings remain.

| Arm | Simulated seconds | Plans on time | Outcome |
| --- | ---: | ---: | --- |
| Command history | 480.90 | 1179/1200 | No arrival; sustained coverage hold |
| JEPA readout | 481.30 | 1163/1200 | No arrival; sustained coverage hold |
| Supervised readout | 480.84 | 1027/1198 | No arrival; active near goal at timeout |
| Untrained features + readout | 480.92 | 1117/1195 | No arrival; resumed movement late |

The post-batch depth probe identified an enclosing-rectangle floor-classification
problem shared by all four arms. The geometry correction and next prospective
mission are recorded in `docs/go2_polygon_floor_coverage_2026-09-17.md`.

## Assignment 1: command-history control

Launched in session 90211, owner PID 4127094. The launch receipt identifies
`InterruptedViewRuntime`, command-history planning, and the computed-but-unused
JEPA readout state `f372e75c1a5c4b3933beb9d59ee97158ce17a8a2b567a89c9be59b74cf8112a8`.
Console log is
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_readout_navigation_01_launch.log`.
Wait for owner exit and complete recording persistence before evaluation. A
mission-complete console message alone does not mean archival has finished.
Detailed `NAVIGATION_PROGRESS` lines are redirected to the mission's `worker.log`,
not the outer launch log. At the first detailed check it had reached frame 1700,
170.02 simulated seconds, with pose frame 1698 and phase OUTBOUND; no arrival
was reported. Earlier initialization-only status based on the outer log was
incomplete: the mission had already been advancing.

The live controller reached `MISSION_TICK_BUDGET_EXHAUSTED` at 480.90 simulated
seconds with no observed arrivals. It recorded 4805 camera frames, 1200 plans
(1179 on time, 21 late) and no reported disallowed contact. The live summary's
wall time was 621.29 s, before final archival. The owner remained live during
recording persistence; independent physical evaluation is still pending.

The owner subsequently exited zero; evaluation session 23676 completed and
exited zero after recording persistence. **No goal or home arrival, zero
disallowed contact, budget exhausted at 480.90 simulated seconds.** No pipeline
fault was recorded. All 251 plans from frame 3800 onward selected hold. Across
the whole mission there were 858 holds, 111 left turns, 134 right turns, 34 right
arcs, 25 left arcs and 38 forward plans.

The final recorded selection preferred forward. All six candidates passed the
recorded full-reserve obstacle predicate; all translating candidates also passed
the stopping projection. The translation-coverage stage rejected forward for
adding unknown cells [66,-13] and [66,-12] and selected hold. Current/hold baseline
had two unknown cells, which were not declared free. The unresolved coverage-view
target [66,-14] had been assigned a remote viewpoint [74,49] with a known-floor
route after a current-position view ended `FRESH_VIEW_PATCH_STILL_UNKNOWN` at
281.90 sensor seconds. The attempt to travel toward that viewpoint remained
coverage-blocked. This is a saved-state diagnosis of a coverage/view-route
conflict, not proof that executing an alternative would be safe or successful.
Preserve the full recording for post-batch diagnosis of the unobserved patch
and feasible viewing motion; do not change this batch's controller.

## Assignment 2: JEPA readout

Launched after assignment 1's owner and evaluator exited, in session 58896,
owner PID 4129049. Launch metadata confirms JEPA readout state
`f372e75c1a5c4b3933beb9d59ee97158ce17a8a2b567a89c9be59b74cf8112a8`
is used for control (`neural_reference_is_unused_for_control=false`). The
controller and all mission settings are unchanged. Detailed progress is in
this mission's `worker.log`; the outer log is
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_readout_navigation_02_launch.log`.

The mission reached its budget at 481.30 simulated seconds. The owner exited
zero after complete archival; evaluation session 2836 also exited zero.
**No arrivals, zero disallowed contacts, no pipeline faults.** Of 1200 plans,
1163 were on time and 37 late. The live summary recorded 4808 camera frames and
615.66 s of wall time before final archival. There were 998 holds, 63 left turns,
99 right turns, 15 right arcs, 11 left arcs and 14 forward plans. All 251 plans
from frame 3800 onward held.

The final plan preferred forward; all six candidates passed full-reserve
obstacle clearance and every stopping projection passed. The coverage filter
rejected all three translations for adding the single unknown cell [40,-14].
Current and hold footprints had no unknown cells. The active request to observe
that same cell had been routed toward viewpoint [1.875,-1.275] (33 route cells)
since 81.90 sensor seconds. A preceding current-position view had aligned at
81.10 s and ended `FRESH_VIEW_PATCH_STILL_UNKNOWN` at 81.50 s, using map frame
796. The preceding successful coverage request for [42,-16] completed at 75.90 s,
using map frame 740. This records the obstruction in the saved selection;
it does not establish that overriding the filter or a different action is safe.

## Assignment 3: supervised readout

Launched after assignment 2's owner and evaluator exited, in session 64727,
owner PID 4130825. The launch receipt identifies the supervised readout state
`5f2a862f1f7650c20655b4e99fe74ed5fcac1c6b76b2405ae260988d90e25c44`
and the unchanged `InterruptedViewRuntime`. The live process was rechecked
while its worker log advanced through camera frame 2500 (250.02 simulated
seconds), still outbound. No terminal outcome is inferred from that progress.
Outer log: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_readout_navigation_03_launch.log`.

The live simulation ended at its budget, 480.84 simulated seconds, with no
observed arrivals and no reported disallowed contact. Recording persistence
is still pending. There were 4805 camera frames, 1198 selections (1027 on time,
171 late), and no pipeline faults. The live wall time was 641.60 s before
archival. Actions were 245 holds, 328 left turns, 326 right turns, 76 right arcs,
44 left arcs and 179 forward plans. Unlike the first two arms, only two of the
249 selections at frame 3800 or later held. The frame-4800 progress record
reported 0.12327 m estimated goal distance; this is not a verified arrival.

Deduplicating saved coverage-view events gives eight requested patches observed,
two fresh-view failures and one weak-visual-support interruption. The final plan
selected right turn to view target [1,35], with route status
`FOOTPRINT_EXTENSION_REQUIRES_OBSERVED_VIEW`. The terminal-approach mode was not
active. This run remained active late in the mission rather than exhibiting
the first two arms' prolonged final holds. Physical evaluation remains pending.

The owner subsequently exited zero after persistence. Evaluation session 60904
also exited zero: **no goal/home arrivals, zero disallowed contacts, no pipeline
faults, budget exhausted at 480.84 simulated seconds**. All 1198 evaluated plans
are accounted for; 1027/1198 (85.73%) were on time. The full failure recording is
retained. The near-goal progress message does not change the failed outcome.

## Assignment 4: untrained-feature control

Launched after assignment 3's owner and evaluator exited, in session 38972,
owner PID 4132885, verified live after launch. The receipt identifies untrained
features with a fitted motion readout, state
`48ec5ad0c0120cb02e174e3eac699322227ae9344b35ca0d5dfb43e15ce87c2b`,
and the unchanged `InterruptedViewRuntime`. This is the final assigned mission;
no extra repetitions are planned. Outer log:
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_readout_navigation_04_launch.log`.
The output drive had 24 GiB free before launch. Wait for owner exit and evaluate
assignment 4 before executing the prepared saved-depth diagnostic.

The live simulation reached its budget at 480.92 simulated seconds with no
observed arrivals and no reported disallowed contact. The owner is persisting
the recording; physical evaluation remains pending. There were 4805 camera
frames and 1195 selections, 1117 on time and 78 late; no pipeline faults.
Live wall time before archival was 623.98 s. Actions were 726 holds, 111 left
turns, 160 right turns, 39 right arcs, 53 left arcs and 106 forward plans.
The last 100 simulated seconds included 56 holds and 93 forward plans; the
final action was forward on an observed-floor route to the goal cell.

The untrained-feature control also failed an in-place view of [40,-14], at
map frame 920 (93.90 sensor seconds). Its subsequent remote-view attempt
observed the patch at map frame 4064 (408.30 sensor seconds), after a long
stall. It observed five requested coverage patches in total and had one
recorded fresh-view failure. All four arms therefore initially failed a view
of [40,-14], but three eventually observed it; that distinction is included
in the pending depth probe. This does not establish that any model caused
the recovery or that the estimated map poses are identical across arms.

The owner then exited zero after complete persistence; evaluation session 35377
exited zero and confirmed no goal/home arrivals, zero disallowed contacts,
no pipeline faults and budget exhaustion at 480.92 simulated seconds. The
four-assignment comparison is complete; no extra repetitions were added.

## Post-batch diagnosis to retain

Source inspection confirms that camera-viewpoint routes use known centre cells
and obstacle clearance, while the final translation filter checks new unknown
cells across the entire predicted 0.48-m footprint. The route proposal does not
establish that later footprint-coverage checks will pass. Both completed arms
exhibit a coverage/view-route conflict, at different targets.

Also inspect why the requested in-place floor views stayed unknown. The current
floor classifier (`CurrentPlaneCoverageGeometry` / `BodyProjectedFloorGeometry`)
requires every valid pixel quad in the axis-aligned rectangle enclosing the
projected floor square to lie within a 10-mm floor-height band. The viewing
planner checks calibrated projection bounds and known map occlusion, not those
raw depth conditions. A saved-depth probe should separate invalid returns,
height-band failures inside the projected cell, and failures only in the extra
rectangle outside its projected quadrilateral. No cause has yet been established
from pixels; do not relax floor evidence based on projection alone. Probe JEPA
map frames 740 and 796 and the command-reference failed view around frame 2800
after completing the unchanged native batch. All needed recordings remain full.

Prepared `scripts/probe_go2_frozen_readout_floor_views_development.py` while
assignment 3 was running. It has not yet been executed. It reconstructs the
initial map basis/height and selected current-frame classifications, without
claiming to replay accumulated map history. It compares rejection counts for
image quads intersecting the projected floor square versus the extra enclosing
rectangle, checking decomposition against the existing classifier. The script
requires assignment 4's completed readout before running, to keep sensor replay
out of the timed native comparison. Only syntax was checked during the mission.

Cross-run coverage-event inspection adds a useful matched-location comparison:
**all three completed arms failed an in-place view of cell [40,-14]**. The
command-history arm failed at map frames 828, 2508 and 2516 before observing it
at frame 2528; the JEPA arm failed at frame 796 and stayed blocked; the supervised
arm failed at frame 1992, had a later weak-visual interruption, and observed the
cell at frame 2460. Supervised also failed [66,-15] at frame 2560 and observed it
at frame 4076. These are different trajectories, not matched camera poses or
proof of a model effect. The prepared probe now includes those failures and
later successful views to distinguish permanent unobservability from viewpoint,
classification and recorded-pose effects. It still has not been run.

After the last timed simulation ended, the probe's convex intersection function
passed a small isolated check covering separated, contained, edge-crossing and
corner-touching image quads, reversed polygon orientation, and translation
away from the polygon. No sensor replay was run during a timed mission.

The completed depth probe (session 22566, exit zero, 3.32 s) is saved at
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_frozen_readout_floor_view_probe_v1.json`.
Across 16 selected cell observations, ten failed the existing floor classifier;
nine failed only because of pixels outside the projected square. This includes
the first failed [40,-14] view in every arm and the command-history arm's final
blocking cells. The later command-history frame-2516 view also contained 172
bad pixel quads overlapping the projected square and remains rejected. Every
delivered depth digest and reconstructed old classification matched. These are
targeted diagnostics, not a population error-rate estimate or navigation proof.
