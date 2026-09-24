# Fresh-maze transfer with the fixed successful controller

Final fixed-controller cohort: all four matched pairs completed and independently
evaluated. Learned achieved two outbound arrivals and one round trip; reactive
achieved no arrivals or round trips. All eight runs had zero disallowed contacts.
The learned round-trip result is 1/4 on this fresh cohort; reliability is not
established. The separate recovery-release revision has since completed all four
layouts, with 4/4 verified goal arrivals and 3/4 round trips.
Those revised outcomes are recorded in
`docs/go2_full_reserve_heading_release_hypothesis_2026-09-14.md` and are not
pooled with this unchanged cohort.

| Fresh layout | Learned outbound | Learned round trip | Reactive arrival |
| --- | --- | --- | --- |
| 0 | Verified | No | No |
| 1 | No | No | No |
| 2 | Verified | Verified | No |
| 3 | No | No | No |

Combined result:
`go2_fresh_stable_reference_matched_cohort_layout00_03_v1_attempt_001/result.json`.

After physically verified round trips on development layouts 4 and 6, test
four newly constructed mazes with the unchanged stable-reference, compiled-floor,
cached-route, arrival-entry-priority learned controller. Use indices 0–3 in order,
each with a matched reactive run. Retain all outcomes, including failures; do not
select replacement layouts or tune the controller inside this cohort.

`lewm/fresh_stable_reference_layouts_development.py` fixes construction seed
2026091407 and excludes abstract topologies and grid embeddings from the explicit
56-layout source registry (the previous registry of 48 plus eight independent
development layouts). All first four structural candidates were accepted. This
is a source-registry claim, not a claim to have examined inaccessible benchmarks.
The scene dimensions, goal-definition rule and sensor configuration are unchanged.

`scripts/run_go2_fresh_stable_reference_native_development.py` runs the existing
controller and acquisition wrappers, changing the scene initializer and instructed
goal coordinates only. Native graph geometry remains scene/evaluator-only.
The inherited arrival tolerances, 4,800-tick budget and timing are unchanged.
Use `--layout-index N`, adding `--reactive` for the paired baseline. All outputs
go to RecoveryStorage with distinct `go2_fresh_stable_reference_...` names.

Before native execution, source integration confirmed all four scene packs,
the intended initializer order, 56 excluded prior layouts and four distinct new
topologies. Public missions contain only instructed goal/home coordinates and
the return requirement. Native results remain pending.

The first learned trial (index 0) was launched after the layout-6 reactive owner
exited 0 and its physical evaluation/comparison completed. Output:
`go2_fresh_stable_reference_learned_round_trip_native_layout00_4800_v1_attempt_001`.
At launch, RecoveryStorage had 126.64 GiB free, RAM had 77.04 GiB available,
CPU utilization was 0.2%, and both GPUs were idle. Keep one native scene owner
at a time to preserve the established timing comparison; the existing twelve
archive workers run only after timed execution.

These are prospective development trials, not sealed final evaluation. They
test transfer of the complete controller; they do not isolate JEPA training or
memory. Realistic sensing/timing and real-platform evidence remain outstanding.

## First learned result: verified outbound, no completed return

Layout 0's owner exited 0 after its fixed budget. Physical evaluation verified
the outbound arrival at frame 4,732: the full one-second dwell stayed 28.01–29.46
mm from the goal, all requested commands were zero, and maximum 100-ms speed
was 0.0215 m/s. There were no disallowed contact samples. Return began but did
not complete before the budget ended; count this as a failed round-trip trial.

All 4,805 camera pairs and registered poses are retained. Median/maximum
position error was 14.82/29.31 mm. Plans on time: 1,176/1,196. Physical path
length was 22.626 m; timed wall duration was 481.887 s. During frames 2,000–4,000,
393 of 501 selected actions were in-place turns, compared with 79 translations
and 29 holds. This identifies a prolonged exploration/turning interval for
later diagnosis; it does not establish its cause. Keep the controller and
budget unchanged for the remaining cohort.

The run root contains `continuous_native_arrival_evaluation.json` and
`native_navigation_summary.json`. Its matched reactive trial was launched after
the learned owner exited and the two separate training-control correction fits
completed. Output:
`go2_fresh_stable_reference_reactive_round_trip_native_layout00_4800_v1_attempt_001`.
The cohort's controller/model/correction sources are unchanged; the new fits are
reserved for the later training-method controls.

## Post-hoc turning diagnosis; no cohort change

The saved-command alignment evaluator matched 364 executed waypoint turns during
frames 2,000–4,000. It compared the saved heading-error forecasts at 300/700 ms
with native motion toward the same body-relative waypoint. Of 188 turns predicted
to improve heading, none actually worsened it. This does not measure all forecast
components or hypothetical unexecuted actions.

The selection history contains 181 substitutions of one pure turn for the other;
180 occurred while a recovery-turn latch was active. The preferred turn already
passed full reserve clearance in 146 of those cases, and all 181 passed the
nominal footprint test. The latched rule intentionally persists until its stored
heading is reached or clear translating progress becomes available. The interval
also contains 14 recovery-direction switches. At frame 2,784, it retained a
right turn toward its stored heading even though the preferred left turn had
full reserve clearance. This is a specific mechanism for prolonged recovery,
not proof that removing the latch would improve a live trial.

Route targets were identical in 356 of 473 consecutive recorded target pairs;
the largest change was 0.2 m. Saved evidence:
`saved_waypoint_alignment_2000_4000_v1.json` and
`prolonged_turning_diagnostic_v1.json`. Test any recovery-release revision only
prospectively after the unchanged four-layout cohort, preserving its failures.

## First matched pair completed; layout 1 launched

The layout-0 reactive owner exited 0 with no arrivals and no disallowed contact
samples. Median/maximum position error was 4.33/9.41 mm, with 1,186/1,200 plans
on time. It selected hold for the final 693 plans (from frame 2,032). Stored
clearance was 0.4016 m against the 0.45-m nominal footprint, and all movement
and further survey turns were rejected. Closest physical goal distance was
1.256 m; final goal distance was 3.554 m. Path length was 8.309 m.

The paired comparison confirmed equal common source hashes and matched mission,
sensing, routing and timing settings. It is saved under
`go2_fresh_stable_reference_matched_comparison_layout00_v1_attempt_001/result.json`.
Outcome: learned verified outbound only; reactive no arrival; neither completed
a round trip. Both full recordings remain retained.

The comparison root also contains `native_navigation_comparison.png` and `.svg`,
generated from the saved physical trajectories and visually inspected. Both
controllers entered the same dead end. The learned controller subsequently
retraced that corridor and reached the goal by the longer route; the reactive
controller stopped in the dead end. This is observed physical backtracking,
not an isolated test of memory's causal contribution. The figure explicitly
labels the incomplete return and uses native geometry only for evaluation.

The unchanged learned controller was then launched on fresh layout 1:
`go2_fresh_stable_reference_learned_round_trip_native_layout01_4800_v1_attempt_001`.
The previous owner was absent, CPU utilization was 0.3%, RAM had 77.09 GiB
available, and RecoveryStorage had 119.84 GiB free. Both GPUs were idle.

## Layout 1 learned result: no arrival

The owner exited 0 at the unchanged budget with no arrivals and no disallowed
contact samples. All 4,805 camera pairs/poses are retained. Median/maximum
position error was 14.13/25.39 mm; 1,176/1,200 plans were on time. The robot
travelled 15.902 m, ending 3.672 m from the goal; closest approach was 1.283 m.
Timed wall duration was 481.890 s. Physical evaluation and summary are saved.

Applying the same frames-2,000–4,000 diagnostic as layout 0 found 359 pure turns,
120 holds and 22 translations in 501 plans. Of 173 opposite-turn substitutions,
171 had an active recovery latch; the preferred turn passed full clearance in
143 cases and nominal clearance in all 173. There were eight recovery-direction
switch events in that interval. The waypoint-heading evaluator retained 51
matching executed turns (survey turns are excluded): all 31 predicted heading
improvements also improved actual heading. This supports the repeated-recovery
hypothesis without establishing the outcome of a revised controller.

The matched reactive trial was launched after owner exit and evaluation:
`go2_fresh_stable_reference_reactive_round_trip_native_layout01_4800_v1_attempt_001`.
At launch, CPU utilization was 0.3%, RAM had 76.89 GiB available and RecoveryStorage
had 116.40 GiB free; both GPUs were idle. No cohort controller changes were made.

## Second pair completed; layout 2 launched

Layout 1's reactive owner exited 0. Physical evaluation confirmed no arrivals
and no disallowed contact samples. Median/maximum position error was 9.31/9.50
mm, with 1,184/1,200 plans on time. It held for the final 878 plans, from frame
1,292, with stored clearance 0.3914 m below the 0.45-m footprint requirement.
Closest/final physical goal distances were 1.298/2.496 m; path length was 5.651 m.

The comparison confirmed common source hashes and matched settings, and is
saved under `go2_fresh_stable_reference_matched_comparison_layout01_v1_attempt_001`.
Both arms failed to arrive on this layout. Across the first two fresh pairs,
learned has one verified outbound arrival and zero round trips; reactive has
zero arrivals and zero round trips. All four runs had zero disallowed contacts.

Fresh learned layout 2 was launched next, with the same controller and budget:
`go2_fresh_stable_reference_learned_round_trip_native_layout02_4800_v1_attempt_001`.
The prior owner was absent; CPU utilization was 0.5%, RAM had 76.74 GiB available,
and RecoveryStorage had 113.09 GiB free. Both GPUs were idle.

## Layout 2 learned result: verified round trip

The owner exited 0 after 2,361 camera pairs. Physical evaluation verified the
outbound arrival at frame 1,570 and home arrival at frame 2,359. The one-second
dwells stayed 24.76–29.22 mm from the outbound goal and 16.76–17.94 mm from home,
with all requested commands zero and speeds below the unchanged limit. There
were no disallowed contact samples.

Median/maximum position error was 8.05/14.14 mm; 573/581 plans were on time.
Path length was 20.990 m. Timed wall duration was 236.843 s; return took 78.9
simulated seconds after 157.0 seconds outbound. The controller, model, budget
and arrival tolerances were unchanged. Physical evaluation and summary are
preserved in the learned layout-2 root.

The matched reactive trial was then launched:
`go2_fresh_stable_reference_reactive_round_trip_native_layout02_4800_v1_attempt_001`.
The prior owner was absent, CPU utilization was 0.3%, RAM had 76.72 GiB available,
and RecoveryStorage had 111.49 GiB free; both GPUs were idle.

## Third pair completed; layout 3 launched

The layout-2 reactive owner exited 0 after archiving all 4,805 camera pairs.
Independent physical evaluation found no arrivals and no disallowed contact
samples. Median/maximum position error was 12.46/21.95 mm; 1,187/1,200 plans
were on time. Closest physical goal distance was 23.31 mm, but no observed
arrival dwell was completed; final goal distance was 75.01 mm. Path length was
21.984 m and timed wall duration was 481.521 s.

Saved planning records from frames 2,000–4,800 contain 701 terminal-position
plans: 333 right turns and 368 right arcs, with 693 plans on time. All 701
passed the current nominal disk-clearance check, and body-relative waypoint
distance remained 41.71–80.75 mm. The arcs used the existing 100-ms translation
pulses. This identifies persistent terminal turning/arcing rather than the
clearance holds seen on layouts 0 and 1. The saved diagnostic is
`reactive_terminal_action_diagnostic_v1.json`; these are selected-action counts,
not a reconstruction of every executed physics interval or a counterfactual
test of a different reactive rule.

The paired comparison confirmed matched settings and common source hashes:
`go2_fresh_stable_reference_matched_comparison_layout02_v1_attempt_001/result.json`.
Learned completed a verified round trip; reactive reached neither endpoint.
Across the first three fresh pairs, learned has two verified outbound arrivals
and one round trip; reactive has zero arrivals and zero round trips. All six
recordings and failures remain retained.

Fresh learned layout 3 was launched with the unchanged controller and budget:
`go2_fresh_stable_reference_learned_round_trip_native_layout03_4800_v1_attempt_001`.
The prior owner was absent, CPU utilization was 0.3%, RAM had 77.06 GiB available,
and RecoveryStorage had 108.22 GiB free; both GPUs were idle. One native owner
runs at a time to preserve the established timing comparison.

The layout-2 comparison includes visually inspected
`native_navigation_comparison.png` and `.svg`. Both controllers explored the
northern branch and retraced it before reaching the eastern goal corridor.
Only learned completed the arrival and returned home; its return followed the
known southern route without repeating that northern detour. This is physical
trajectory evidence, not an isolation of memory's causal contribution.

## Layout 3 learned completed; final reactive trial launched

The learned owner exited 0 after saving all 4,805 camera pairs. Independent
evaluation confirmed no arrivals and zero disallowed contact samples.
Median/maximum position error was 21.77/30.52 mm; 1,153/1,200 plans were on time.
Closest/final physical goal distance was 1.282/2.626 m, and path length was
22.333 m. Timed wall duration was 481.899 s. All four learned trials now have
complete physical evaluations: two outbound arrivals and one round trip.

The same frames-2,000–4,000 diagnostic found 389 pure turns, 78 translating
actions and 34 holds in 501 plans. All 173 opposite-turn substitutions had an
active recovery latch; the preferred turn had full reserve clearance in 138
cases and nominal clearance in 144. Ten events selected an alternative
recovery direction. Of 23 command-matched waypoint turns, 20 predicted heading
improvement and none of those worsened actual heading. Survey turns are excluded
from this heading diagnostic. The repeated recovery mechanism therefore also
appears on the fourth layout; no alternative native outcome has been established.
Saved evidence: `recovery_turn_latch_diagnostic_v1.json` and
`saved_waypoint_alignment_2000_4000_v1.json`.

The final unchanged reactive trial was launched:
`go2_fresh_stable_reference_reactive_round_trip_native_layout03_4800_v1_attempt_001`.
The previous owner was absent; CPU utilization was 0.5%, available RAM was
76.42 GiB and RecoveryStorage had 104.80 GiB free. Both GPUs were idle. Complete
this run and its paired evaluation before the prepared recovery-release trial.

## Cohort complete; recovery-release experiment started

The layout-3 reactive owner exited 0 with all 4,805 camera pairs retained.
Independent evaluation confirmed no arrivals and zero disallowed contacts.
Median/maximum position error was 7.58/9.97 mm; 1,184/1,200 plans were on time.
Closest/final physical goal distance was 1.274/2.855 m; path length was 11.936 m.
Timed wall duration was 481.541 s. It held for the final 777 plans, starting at
frame 1,696, with stored clearance 0.43146 m below the 0.45-m nominal footprint.
The saved `reactive_stall_diagnostic_v1.json` records rejection of all moving
and additional survey-turn candidates.

The fourth paired comparison confirmed matched settings and common source
hashes. All four comparison results are combined in the cohort result above.
The result compares complete selectors; it does not establish JEPA-specific
benefit, memory's causal contribution, repeatability or deployment readiness.
The layout-1 paired trajectory figure was also generated and visually inspected.

After all original outcomes were saved, launched the prewritten one-change
recovery-release trial on the same layout 0:
`go2_heading_release_fresh_learned_round_trip_native_layout00_4800_v1_attempt_001`.
The prior owner was absent, CPU utilization was 0.2%, available RAM was 76.83 GiB,
and RecoveryStorage had 101.90 GiB free; both GPUs were idle. Run this revision
on indices 0–3 in order, preserving all outcomes and the original comparisons.
