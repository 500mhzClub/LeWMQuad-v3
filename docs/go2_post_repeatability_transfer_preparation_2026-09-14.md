# Next independent development-layout roster

Prepared four new layouts while the fixed repeatability study runs. Construction
seed 2026091417 accepted the first four structurally eligible candidates from
five examined; one failed the existing route-structure requirement. No candidate
was selected or rejected using a native outcome. All four abstract topologies
and metric embeddings are distinct from each other and the explicit 60-layout
development registry. This is not a claim about every historical maze.

The source is `lewm/post_repeatability_transfer_layouts_development.py`, using
the existing sixteen-cell tree generator, geometry parameters and farthest-cell
goal rule. Physics seeds are 2026097400–2026097403; appearance seeds are
2026097500–2026097503. Only the goal/return coordinates and return requirement
leave the scene/evaluator boundary through `public_mission`; graph structure
and shortest routes are not planner inputs.

Frozen inventory:
`docs/go2_post_repeatability_transfer_layout_inventory_2026-09-14.json`,
SHA-256 `021ed31825a0b59ab2cfa7e5cdf8ad02f0f2ff5816ba0c4330301555373487d7`.
No native collection, checkpoint loading or navigation has occurred for this
roster. This is prospective development material, not sealed final evaluation.

Finish all four current fixed-controller repetitions first. Use this fixed
roster for the next independent-layout navigation comparison; do not replace
layouts based on failures. Controller/control-arm assignment and native launch
remain to be specified using the repeatability evidence. Preserve the need for
matched reactive/non-predictive and memory-causality comparisons, realistic
sensing/timing, and bounded hardware evidence. Layout preparation itself proves
neither navigation nor reliability.

Prepared launcher `scripts/run_go2_post_repeatability_transfer_development.py`
while the second repeat pair is still running. It supports the unchanged
`HeadingRecoveryRuntime` and a model-free `TransferReactiveRuntime` using the
same nearby-panorama frontier-view strategy. Both share fresh scene construction,
compact capture, physical CPU allocation by layout parity, stable visual/gyro
tracking, observed persistent maps, route preference, fine goal routing,
terminal translation pulses, action bank, mission and arrival thresholds.
The reactive arm has no model or residual and never evaluates future outcomes.
Prediction-dependent recovery and terminal overrides remain part of the learned
controller treatment; this comparison cannot isolate JEPA or memory causality.

All eight writer configurations were exercised without native collection or
model loading. Shared comparator fields match within each layout; generated
scene specifications equal the frozen inventory, and the acquisition class
resolves the new physical initializer with the new specification and pack.
No controller selection or native launch for these new layouts has occurred.
Finish and assess the four repetitions before fixing dispatch order.

Existing focused reactive-pulse and nearby-panorama tests passed: five tests
in 2.46 seconds. No shared runtime or active repeat source was modified.

## Fixed eight-run comparison

After all four repetitions: 3/4 verified round trips, with the failed run
retained and no reliability claim. Use the current unchanged learned controller
and the prepared reactive arm on all four frozen new layouts. Complete all
eight assignments without tuning, replacing layouts or selecting checkpoints
from interim outcomes. Dispatch two owners per batch: learned layouts 0/1,
reactive layouts 0/1, reactive layouts 2/3, then learned layouts 2/3. This reverses
arm order across the two layout pairs. Wait for both complete archives and
owner exits before each next batch.

Use 4,800 navigation ticks, 20-mm observed arrival and unchanged 40-mm physical
quiet-arrival evaluation. Record all failures, contacts, goal and round-trip
checks, travelled distance, timing and recovery activation. Evaluate native
physics independently after each completed run; compare trajectories per layout.
The comparison tests the complete predictive controller, including its
prediction-dependent recovery, against instantaneous waypoint feedback. It does
not isolate JEPA training or persistent-memory causality.

Before first dispatch, approximately 18 GiB of RecoveryStorage and 76 GiB RAM
were available. This is sufficient for the first pair, not all eight recordings;
perform eligible user-authorized depth retirement before later batches as needed,
preserving current comparisons and unresolved failure inputs.

The first learned pair was launched around 20:16 local log time: layout 0
session 98620 / PID 3448125; layout 1 session 34794 / PID 3448166. Both owners
were confirmed live, and actual launch records identify the new frozen inventory,
learned arm and assigned CPU group. Results remain pending. Next batch is the
reactive arm on these same layouts, after both archives and owner exits.

During the first learned pair, reviewed completed-diagnostic depth retirement
recovered 18.1193 GiB. RecoveryStorage then had about 36 GiB free, sufficient
for all eight planned recordings at the established size estimate. Current
comparisons and unresolved failure inputs were preserved. Details are in
`docs/go2_development_artifact_retention_2026-09-14.md`.

## First learned pair completed

Both owners exited 0 and all archived arrivals passed independent physics
checks, with zero disallowed contacts. Layout 0: 2,978 camera pairs, verified
goal frame 1,896 and home 2,976; 23.284 m travelled. Goal dwell 16.799–20.392 mm,
home 11.343–16.953 mm; both quiet with all requests zero. Layout 1: 3,029 pairs,
goal frame 2,091 and home 3,027; 25.208 m travelled. Goal dwell 18.983–21.871 mm,
home 13.225–15.650 mm; both quiet with all requests zero. New heading recovery
activated zero times in both runs. Maximum position errors were 9.866 and
10.064 mm. Evaluations, navigation summaries, recovery diagnostics and resource
records are saved in the two roots.

Owner elapsed times were 440.73 and 468.76 seconds, peak RSS 16,803,308 and
17,093,000 KiB, zero process swaps. Before dispatching the matched reactive pair,
both learned owners were absent, RAM available was about 76 GiB and
RecoveryStorage had about 32 GiB free. The model-free reactive pair on layouts
0/1 was then dispatched with the fixed launcher and same per-layout CPU groups.
Its results remain pending. Complete reactive 2/3 and learned 2/3 afterward,
without changing the controller or roster based on interim outcomes.

Reactive layout-0 session 13831 / PID 3451705 and layout-1 session 38888 /
PID 3451752 were confirmed live around 20:25 local log time. Actual launch
records identify the reactive assignment with both model/residual use and
future-outcome evaluation false.

## First two matched comparisons completed

Both reactive owners exited 0, retaining 4,805 pairs each. Independent evaluation
confirmed no reported arrivals and zero disallowed contact samples. Layout 0
physically approached within 13.588 mm of the goal but did not finish a settled
arrival; final distance was 77.400 mm, path 20.343 m, and maximum position error
30.581 mm. Layout 1 approached within 29.768 mm, ended 58.783 mm away, travelled
23.711 m, and had maximum position error 12.674 mm.

Late reactive decisions were predominantly arcs and turns at the exact goal,
with nominal current clearance available. Both reached the goal vicinity;
these are terminal-convergence failures, not evidence that reactive exploration
could not find the goal. This limits attribution of the current learned 2/2
versus reactive 0/2 round-trip difference. Preserve the fixed study; investigate
a stronger terminal reactive control separately afterward if needed.

Both paired comparisons passed shared settings and common source identity
checks. Their `result.json` and visually inspected PNG/SVG trajectory comparisons
are in `go2_post_repeatability_transfer_comparison_layoutXX_v1_attempt_001`
for layouts 00 and 01. Reactive terminal-selection diagnostics, physical
evaluations, navigation summaries and resource records are in each run root.
Owner elapsed times were 681.05 and 715.19 seconds, peak RSS 25,075,448 and
25,094,984 KiB, zero process swaps.

With those owners absent, about 25 GiB disk and 76 GiB RAM available, the fixed
reactive layout-2/3 pair was dispatched next (sessions 49979 and 16732). Their
results remain pending. The final batch is learned layouts 2/3, unchanged.

Reactive layout-2 owner PID 3456218 (session 49979) and layout-3 owner
PID 3456243 (session 16732) were confirmed live about one minute after
20:38 local log startup. Actual launch records show the correct reactive
assignment and CPU groups.

## Remaining reactive pair completed

Both owners exited 0 with 4,805 camera pairs, no disallowed contacts and no
round trip. Layout 2 had no arrival, minimum goal distance 1.112 m, path
13.735 m, median/maximum pose error 11.142/49.694 mm. Layout 3 achieved a
verified outbound arrival at frame 971 (9.978–12.469-mm dwell, maximum
100-ms speed 0.01310 m/s, all requests zero), but failed the home approach:
final physical home distance 58.964 mm, path 27.240 m, maximum pose error
12.872 mm. All four reactive trials therefore have one verified goal and
zero round trips. These outcomes remain part of the complete fixed study.

Layout 2 has a separate perception failure: 683 of 684 unavailable/stale
requests had no current obstacle observation, versus one aged observation.
There were 13,213 later latched-veto requests. At inspected frames 2070, 2080,
2100, 3000 and 4800, all 307,200 primary pixels were below the configured
0.20-m minimum optical depth, giving zero valid primary points. Auxiliary
depth still had thousands of points and a floor fit, but the paired observer
requires nonempty clouds from both cameras and returns None. Preserve the
recording for this unresolved blindness/dispatch diagnosis; invalid rays are
not free space and no depth-validity or clearance threshold was changed.
The two diagnostic JSON files are in the reactive layout-2 root.

Elapsed owner times were 699.98 and 728.64 seconds, peak RSS 25,289,952 and
25,369,696 KiB, zero process swaps. With both owners gone, approximately
18 GiB disk and 76 GiB RAM available, the final learned layouts 2/3 were
dispatched unchanged. Their results remain pending.

Final learned layout-2 owner PID 3461044 (session 9447) and layout-3 owner
PID 3461085 (session 98469) were confirmed live around 20:52 local log time.
Actual launch records identify the fixed JEPA assignment and correct CPU groups.
The reactive layout-2 primary RGB frame 2070 was visually inspected: a nearby
wall fills the view, consistent with the recorded below-minimum optical depths.

## Complete fixed eight-run result

All eight owners exited 0. The learned controller achieved four verified goals
and four verified round trips; the reactive controller achieved one verified
goal and zero round trips. All eight had zero disallowed contact samples.
The final learned layout 2 passed goal/home checks at frames 1971/3040, with
goal dwell 22.303–26.087 mm and home dwell 8.046–14.644 mm. Layout 3 passed at
1263/2576, with goal dwell 6.695–8.913 mm and home dwell 10.819–17.743 mm.
All arrivals satisfied quiet motion and zero requested commands. Owner elapsed
times for these final runs were 431.84 and 387.89 seconds, with zero swaps.

All four paired comparisons passed the shared settings/source checks, and all
four PNG/SVG trajectory comparisons were visually inspected. The combined
result is `go2_post_repeatability_transfer_summary_v1_attempt_001/result.json`
under the RecoveryStorage development artifact base. Each native root retains
its independent evaluation, navigation summary and process resource record.
New hold-relative heading recovery activated zero times in all four learned
runs, so these successes cannot establish that helper's benefit.

This is a four-layout development result, not established reliability. The
prior fixed repeatability study remains separately reported as 3/4 round trips.
Reactive layouts 0/1 failed terminal settling, layout 3 failed home settling,
and layout 2 lost the primary depth observation near a wall. Consequently the
result does not isolate predictive exploration, JEPA training or memory causality.
Proceed to the separately prepared stronger terminal-reactive control without
altering or discarding any original outcome. Realistic sensing/timing and
bounded hardware validation remain outstanding.

The original reactive layout-2 blindness diagnosis now includes
`pre_blindness_command_diagnostic_v1.json`: the final second before the first
missing obstacle observation requested 0.60 s of left turn and 0.40 s of hold,
with no translation. A late right-turn plan preceded the missing observation;
its unexecuted benefit is unknown. The problem is therefore not explained
solely by forward overshoot. The primary camera sits 0.326 m ahead of the body
origin and has a 0.20-m minimum optical depth, so an ideal front-facing planar
wall can enter its blind range at about 0.526 m from the body origin, outside
the 0.45-m nominal disk. This simplified geometric observation does not certify
articulated clearance, and invalid depth remains unknown. Any subsequent
recovery must address observation loss during rotation as well as translation.
