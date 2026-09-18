# Fixed fresh-maze comparison after stopping-projection diagnosis

## Completed twelve-assignment result

All twelve owners have exited. Independent physical evaluations, actual
controller-treatment checks, all applicable saved forecast evaluations, all
four per-layout comparisons and inspected PNG/SVG trajectories are complete.
The aggregate is
`go2_stopping_projection_transfer_four_layout_summary_v1_attempt_001/result.json`.

| Controller | Verified goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| Learned corrected XY and learned yaw | 4/4 | 4/4 | 0 |
| Fitted pose-command XY and command yaw | 4/4 | 3/4 | 0 |
| Instantaneous reactive | 3/4 | 3/4 | 0 |

These are four previously unused development mazes in the same family, one
frozen supervised training seed and one execution per assignment. The learned
controller completed end-to-end navigation on all four; statistical superiority,
JEPA benefit, isolated prediction-ranking benefit, calibrated real sensing,
wall-clock navigation and hardware readiness remain unestablished. The fitted
controller shares the predictive planner; the reactive controller differs in
predictive clearance and recovery as well as action ranking.

Completed simulated round-trip durations by layout 0/1/2/3 are learned
196.74/123.92/271.14/154.36 s; fitted motion 247.22/136.40/failed/149.62 s;
reactive failed/169.78/144.72/163.24 s. Learned's slow layout-2 run included
148.84 s of turn-only commands versus reactive's 43.14 s and longer travel
(21.240 versus 15.323 m). Thus the learned controller was not uniformly faster.
All four comparisons match shared settings and common source bindings, retaining
the documented first-pair annotation-only exception.

The two unsuccessful assignments remain fully retained: reactive layout 0's
mapped-clearance holding deadlock, and fitted layout 2's tracking-queue overflow
during return. Fitted layout 2 independently passed goal frame 1090 (maximum
dwell distance 13.085 mm), with 1,280 captured frames, 1,246 published poses,
maximum pose error 6.994 mm and zero contacts. Its owner exited 1 after 3:13.82,
without swap. Actual fitted treatment is confirmed on all 303 selected plans.
`terminal_tracking_queue_diagnostic_v1.json` records the original `Full()`
failure and the 32-entry tracking queue. Before frame 1000, median tracking
duration on the measured simulation clock was 64 ms. On frames 1091–1245 it
was 120 ms; 69.0% exceeded the 100-ms camera period, and completion age grew
to 3.324 s. Later clock-closed faults arose during teardown. These event times
are not a direct CPU profiler; exact underlying tracker cost remains to diagnose.

Fitted layout 3 passed both arrivals (frames 945/1493, physical dwell maxima
20.227/13.126 mm), with 1,495 poses, maximum error 6.363 mm, zero contacts,
342/366 on-time plans and owner exit 0 after 3:55.62, without swap.

The separate 60-second actual-host-deadline probe has now completed on exposed
layout 1. Attempt 001 exposed a command-history clock integration error and is
preserved. The corrected attempt 002 kept physical service history separate
from host deadlines: 148/150 plans finished on time, but all 3,005 requests were
zero. Camera acquisition blocked initial command service by about 43 ms against
the unchanged 1-ms tolerance. This establishes a simulator scheduling limitation,
not wall-clock navigation readiness. Full results:
`docs/go2_post_training_host_timing_2026-09-15.md`.

Fitted layout 2's exact replay and call profile are complete. Batching the same
gyro proposal fits preserved all 1,280 replay poses and floor constraints while
reducing return-section median tracking time from 106.924 to 84.265 ms. The
separate exposed-layout native follow-up independently passed a round trip
with 1,661 poses and zero contacts; it does not replace the fixed comparison
failure. It ran alone rather than with a second native owner. Details: `docs/go2_local_reference_tracking_cost_2026-09-15.md`.
Preserve the complete failure recording. Learned layout 2 remains full for its
long-turn diagnosis.

## Ten assignments evaluated; final fitted-motion pair running

Learned layout 2 passed both physical arrivals after its temporary progress
stall: 2,710 poses, goal/home frames 2195/2708, dwell maxima 16.944/13.110 mm,
maximum pose error 5.613 mm, zero contacts. Duration 271.14 simulated seconds,
654/670 plans on time; owner exited 0 after 6:27.63 without swap. All learned
assignments are now complete at 4/4 verified round trips on these four fresh
development mazes. This is one frozen training seed, not general reliability.

Reactive layout 3 also passed: 1,629 poses, frames 1125/1627, dwell maxima
3.611/17.642 mm, maximum pose error 5.261 mm, zero contacts, 163.24 simulated
seconds and 393/399 on-time plans. Owner exited 0 after 4:13.34 without swap.
The reactive roster is complete at 3/4 goals and round trips. Its original
layout-0 failure remains in the denominator and fully retained. All applicable
evaluations and actual treatment checks are saved for both fifth-pair outcomes.

Fitted motion on layouts 2 and 3 is now running in sessions 80538 and 56478.
These are the final two fixed assignments. Wait for both owner exits before
their complete per-layout comparisons and the four-layout aggregate. No model,
fit, controller or sensor change was made during the cohort. About 7.6 GiB was
available before this pair after retiring only two superseded successful probe
depth recordings; current first-success references and all failures remain full.

## Eight assignments evaluated; fifth pair launched

Reactive layout 2 and learned layout 3 both passed independent goal/return
checks with zero disallowed contacts. Reactive 2: 1,446 poses, goal/home frames
808/1444, physical dwell maxima 13.400/12.193 mm, maximum pose error 5.201 mm,
144.72 simulated seconds, 345/348 selected plans on time. Owner exited 0 after
3:52.85, no swap. Learned 3: 1,542 poses, frames 1021/1540, dwell maxima
7.548/15.986 mm, maximum pose error 5.225 mm, 154.36 simulated seconds,
356/378 plans on time. Owner exited 0 after 4:19.75, no swap. All applicable
evaluations and actual controller-treatment checks are saved.

Current evaluated totals are learned 3/3 round trips, fitted motion 2/2,
reactive 2/3, with all eight contact-free. They are incomplete condition
populations; the fixed twelve-run aggregate remains pending. The fifth pair
has launched unchanged: learned layout 2 and reactive layout 3. The final
pair is fitted motion on layouts 2 and 3 after both fifth-pair owners exit.

First-layout depth retention is complete: fitted motion 0 and learned 1 lost
only primary/auxiliary depth after their comparisons and diagnosis; all other
records remain. First full successes learned 0/fitted 1/reactive 1 and the
full reactive-0 failure remain. Inventory:
`.generated/depth_retirement_completed_transfer_first_layouts_2026-09-15/`.

## First six assignments complete; fourth pair running

Both first layouts now have complete three-controller comparisons and visually
checked PNG/SVG trajectory figures. Learned and fitted motion each passed 2/2
round trips; reactive passed 1/2. All six had zero disallowed contacts. Six
assignments remain, including the newly launched fourth pair: reactive layout 2
(session 74030) and learned layout 3 (session 23197). After both owners exit,
the fixed next pair is learned layout 2 and reactive layout 3, then fitted
motion on both layouts. No simulation/controller source was changed.

Reactive layout 0 exited 0 after 11:07.21 with a scientific budget failure,
4,805 published poses, no arrivals, maximum pose error 4.374 mm and no swap.
All 1,200 selections confirm the instantaneous controller with no predicted
outcomes. The saved `terminal_reactive_clearance_hold_diagnostic_v1.json`
identifies the holding mechanism: after right arcs at frames 564/568, current
stored clearance fell from 465.580 mm at frame 568 to 432.267 mm at 572,
below the selector's unchanged 450-mm threshold. Every remaining 1,058 plan
selected hold with all six candidates ineligible; 1,049 arrived on time.
Their stored clearances remained 388.691–432.267 mm, and route status remained
`ADDITIONAL_VIEW_REQUIRED`. This establishes the recorded selector deadlock;
it does not infer physical collision or a safe unexecuted escape. The full
failed recording is retained.

On layout 0, learned and fitted motion took 196.74 and 247.22 simulated seconds,
with paths 22.425/23.572 m. Translation command times were 122.00/120.22 s,
turn-only times 53.10/111.30 s, zero-command times 21.42/15.40 s. Thus this
duration difference reflects additional fitted-controller turning, unlike the
mostly zero-command difference on layout 1. Stopping projections changed 6/11
selected actions; neither predictive arm incurred an actual stopping veto.
All 12/13 completed camera views observed their requested patches; longest
events were 25.6/52.8 s. Both per-layout comparisons have 136 equal common
source bindings plus the documented annotation exception. These are single
executions per assignment; no statistical learned advantage is established.

## First complete three-controller maze: layout 1

Learned layout 1 exited 0 after 3:18.60, no swap, 1,238 accepted poses. Both
physical arrivals pass (goal/home frames 723/1236; quiet-dwell maxima
15.368/20.253 mm), maximum pose error 5.003 mm, zero contacts. All applicable
evaluations pass and the 302 selected plans match the learned XY/yaw assignment.

The layout-1 comparison is saved at
`go2_stopping_projection_transfer_comparison_layout01_v1_attempt_001/result.json`.
All three controllers passed round trips with no disallowed contacts. Learned,
fitted motion and reactive durations are 123.92, 136.40 and 169.78 simulated
seconds; path lengths are 14.661, 14.877 and 16.535 m. Shared settings match,
with 136 equal common-source bindings and the documented annotation exception.

The learned/fitted duration difference is almost entirely zero-command time:
11.84 versus 24.44 s, while combined translation/turn command times are 111.88
versus 111.80 s. Fitted motion had 38 late selected plans versus learned's 12.
This single timing-sensitive comparison does not establish a model-driven
efficiency improvement. Reactive used 93.44 s translation and 53.66 s turn-only
commands. All ten completed camera-view events per arm resolved by observing
the requested patch. Planned stopping changed zero learned and one fitted
selection; reactive had one actual stopping veto.

Reactive layout 0 remains live within its fixed budget. Five assignments are
physically verified so far; the fixed fourth pair awaits that owner's exit.

## Latest progress: second pair physically verified

Four of twelve assignments now have independently verified goal-and-return
arrivals, all with zero disallowed contacts. The second pair adds fitted-motion
layout 0 (2,470 poses, goal/home frames 1812/2468, physical quiet-dwell maxima
12.068/16.895 mm, maximum pose error 7.991 mm) and reactive layout 1 (1,696
poses, frames 1012/1694, dwell maxima 7.510/19.835 mm, maximum pose error
5.488 mm). Actual treatments are verified across 609 predictive and 410
reactive selected plans. All applicable evaluations are saved.

The fitted-motion owner exited 0 after 5:54.00, without swap. Both second-pair
owners are absent and their final recordings/results exist; the reactive
session handle is no longer available to retrieve its exit code. Its physical
arrival evaluation passed. The evaluator now accepts an absent motion-correction
field in reactive records; predictive records still require that field. This
changes only the result reader, not the frozen simulation or controller.

The next fixed pair is now launched: reactive layout 0 and learned layout 1.
Eight assignments remain to complete the cohort. These partial successes do
not yet establish learned prediction or JEPA superiority.

## First pair verified

Both owners exited 0 and both round trips passed independent physical arrival
checks, with zero disallowed contacts. Learned layout 0 accepted 1,966 poses,
goal/home frames 1298/1964, maximum quiet-dwell distances 19.444/12.473 mm,
maximum pose error 7.489 mm, 461/482 plans on time and 196.74 simulated seconds.
Owner time including recording was 5:02.80, maximum RSS 12,157,592 KiB, no swap.
Fitted-motion layout 1 accepted 1,363 poses, frames 732/1361, dwell maxima
19.228/9.473 mm, maximum pose error 7.982 mm, 289/327 plans on time and
136.40 simulated seconds. Owner time was 3:58.92, RSS 9,317,416 KiB, no swap.
Both had zero stopping-margin vetoes. These are different layouts; their times
are not a paired controller comparison.

Every saved selected plan confirms the assigned final XY/yaw source (482 learned,
327 fitted-motion plans). Original launch records remain intact; explicit
`launch_annotation_correction.json` records bind each original launch and the
verified treatment. The yaw evaluator uses this correction explicitly. All
four standard evaluations and `actual_controller_treatment_v1.json` are saved.
Use `scripts/evaluate_go2_stopping_projection_transfer_development.py` after
each owner exits; it checks actual treatments and evaluates all outcomes.

The original reactive terminal-feedback tests also passed on the newly composed
current-perception reactive runtime for forward pulse, turn and hold, with a
sentinel forbidding prediction-model access. No reactive native outcome is
available yet. Continue the fixed next pair: layout 0 pose-command, layout 1
reactive. Keep all sources/fits/settings unchanged after the documented
annotation-only correction.

Second pair launched: pose-command layout 0 in session 40341, PID 3650361;
reactive layout 1 in session 8293, PID 3650363. Actual launch records now have
the correct distinct treatment bindings: fitted XY/command yaw with stopping
projections versus no predicted XY/yaw and no stopping-projection planner in
the reactive arm. Both retain the local-reference tracker and committed views.
No final second-pair outcome is available yet. After both owners exit and their
outcomes are evaluated, the next fixed pair is reactive layout 0 and learned
layout 1.

Freeze the current implementation for twelve native assignments: learned
corrected XY/learned yaw, fitted pose-command XY/command yaw, and instantaneous
heading-first reactive control on each of four new development mazes. Do not
tune sources, model/fits, conditions or layout selection during the cohort.
Keep every failure in the results. Do not replace or repeat poor outcomes within
this fixed cohort.

The inventory is `docs/go2_stopping_projection_transfer_layout_inventory_2026-09-15.json`,
SHA-256 `ef34cccd5ce4a62bed742ee4a5eff02e54618c0ad1b81b6d0be787ebb089f06f`.
Construction seed 2026091553 selected four eligible layouts from five candidates,
with four distinct topologies and embeddings, disjoint from the explicit
72-layout development registry. Physics seeds are 2026098000–03; appearance
seeds 2026098100–03. These are within the same maze family, not sealed or
out-of-family evaluation. Geometry is available only to construction/evaluation;
the controller receives public mission coordinates and sensed observations.

All arms use `LocalViewRevisitMotion`, robust floor registration, the same
independent depth observer and current-plane mapper, persistent routing memory,
committed camera views, six actions, measured-simulation timing, synthetic
2-mm depth noise, ideal gyro, 480-second budget and physical arrival criteria.
No floor-reacquisition intervention is included. The two predictive arms share
the complete planner and stopping-projection checks. Both compute the same
frozen supervised model, but use their assigned final motion forecasts for
scoring and geometric checks. Contact scoring is disabled.

The reactive arm uses the existing instantaneous waypoint/heading controller
with matched current perception, routing and viewing modules. It loads no
neural world model and evaluates no candidate future outcomes. Its current
clearance and actual dispatch guards remain; prediction-based clearance,
stopping anticipation and predictive recovery are absent. Thus its comparison
is between complete predictive and instantaneous controllers, not an isolated
prediction-ranking ablation. It does not establish a JEPA training effect.
Timing differs naturally with the computation performed by each controller.

Use `scripts/run_go2_stopping_projection_transfer_development.py` with
`--layout-index 0|1|2|3 --condition learned|pose_command|reactive`.
Fix the following concurrent pairs in order, waiting for both owner exits and
recording completion before advancing to the next pair:

1. Layout 0 learned; layout 1 pose-command.
2. Layout 0 pose-command; layout 1 reactive.
3. Layout 0 reactive; layout 1 learned.
4. Layout 2 reactive; layout 3 learned.
5. Layout 2 learned; layout 3 reactive.
6. Layout 2 pose-command; layout 3 pose-command.

Even layouts use CPUs 0–7,16–23; odd layouts use 8–15,24–31. At most two
native owners including archiving coexist. Run independent physical-arrival
and navigation evaluations on all outcomes; saved XY/yaw evaluations apply to
predictive arms only. Compare actual assigned forecasts, current shared
settings, physical success, contacts, outages, stopping vetoes and durations.
Do not turn same-window alternative forecasts into hypothetical navigation
successes. Retention follows the explicit per-layout policy: preserve every
failure and all results; analyzed success depth may be retired after comparison,
retaining at least one full success per controller. Hardware, real sensor
calibration, wall-clock qualification and multiple training seeds remain open.

Later retention update: the old learned-0, fitted-1 and reactive-1 success-depth
pins have ended under `docs/go2_development_artifact_retention_2026-09-14.md`.
Their completed comparisons and all non-depth records remain, with current full
controller references in the multiseed and shared-recovery cohorts. Every
failure remains full. The old first-success depth retention above is superseded
for those three named successes only.

The first pair has launched: learned layout 0 in session 10018 (PID 3648791),
pose-command layout 1 in session 2072 (PID 3648790). Actual launch records match
the new inventory, twelve-assignment roster, local-reference tracker, committed
views and planned stopping projections. Controller-label correction is described
below; actual forecast receipts must confirm the assigned motion sources.
No floor reacquisition is enabled. Both owners must exit including archiving
before evaluations and the next fixed pair. No final outcomes are available yet.

### First-pair annotation correction

The shared cohort writer rebinds its annotation's `CONDITION` global to the
model-training label `supervised_rollout`, overwriting the new launcher's intended
`learned` or `pose_command` label. Thus the first two launch records incorrectly
label their motion source as supervised_rollout and yaw as command. Runtime
construction uses `args.condition` separately and is unchanged. Do not use those
first-pair launch fields as proof of the actual scored forecasts: verify the
saved per-plan receipts at completion before counting the outcomes.

The annotation now uses a distinct `TREATMENT` binding. The original launcher is
preserved under `.generated/stopping_projection_transfer_annotation_fix_2026-09-15/`.
This is a reporting-only correction; no runtime, model, fit, layout, sensing,
action-selection or dispatch function changed. Keep the two original launches
and add explicit corrections after their actual receipts are verified. Remaining
assignments use the fixed writer. Record this launcher-source exception when
comparing source bindings; do not substitute or rerun the first two outcomes.
