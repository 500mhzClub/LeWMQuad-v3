# Prospective release of a recovered heading action

Final four-layout result: revised controller 4/4 verified goal arrivals and 3/4
round trips, versus original controller 2/4 goal arrivals and 1/4 round trips.
All eight runs had zero disallowed contacts. Layout 0 arrived 102 seconds earlier
but still missed the return deadline. These are single development runs per
controller/layout, not established repeatability or final evaluation.

Subsequent evidence: the first shared-launcher JEPA training-control run on
layout 0 failed to reach the goal despite using the same assigned model and
correction. It spent 88.92% of plans turning and had 48.03-mm maximum position
error. The earlier cohort result below remains valid for those four runs, but
does not establish reliable repeated performance. See
`docs/go2_matched_motion_residual_controls_2026-09-14.md`.

| Layout | Original goal / round trip | Revised goal / round trip | Releases |
| --- | --- | --- | --- |
| 0 | Yes / No | Yes / No | 17 |
| 1 | No / No | Yes / Yes | 7 |
| 2 | Yes / Yes | Yes / Yes | 0 |
| 3 | No / No | Yes / Yes | 58 |

Combined result:
`go2_heading_release_matched_cohort_layout00_03_v1_attempt_001/result.json`.

Fresh layouts 0 and 1 repeatedly retained a recovery turn after the originally
preferred turn had full reserve clearance. The hypothesis is that this persistent
recovery state wastes exploration time. The saved analyses identify the action
substitutions; they do not establish how an alternative live trajectory would end.

`lewm/full_reserve_heading_release_development.py` prepares one narrow variant.
Outside terminal approach, release an active pure-turn latch only when the
originally preferred opposite turn has full reserve clearance, better current
utility, and predicted heading improvement. For survey turns, its scan utility
must also beat hold. Preserve nominal/reserve thresholds, footprint, duration,
arrival conditions, subsequent holds and terminal behavior. Reset only the
released latch, recording the reason and before/after action/utility.

Three focused tests passed in 2.11 seconds. They cover release without mutating
the saved input, rejection when only recovery clearance is available or heading
does not improve, and survey/hold behavior. These are implementation checks,
not native navigation evidence.

This variant was not used in the original four-layout cohort. Finish all original
learned/reactive pairs first. Then test this one change prospectively with new
output identities and the same four layouts, model, sensing and budget. Preserve
all outcomes and assess repeatability before attributing an improvement to the
change. Do not widen arrival or clearance limits or choose replacement layouts.

The separate launcher is `scripts/run_go2_heading_release_native_development.py`,
using `--layout-index 0` through `3` in order. It retains the original fresh-maze
scene, acquisition and controller configuration, adds the release runtime, and
writes exclusive `go2_heading_release_fresh_learned_...` outputs with the source
identity and corresponding original baseline root. Its CLI/import check passed.

All original four learned/reactive pairs have now completed and been evaluated:
learned 2/4 outbound and 1/4 round trips; reactive 0/4 arrivals. All eight runs
had zero disallowed contact samples. The first recovery-release trial was then
launched on layout 0 under
`go2_heading_release_fresh_learned_round_trip_native_layout00_4800_v1_attempt_001`.
This uses the source prepared before the final original cohort outcomes, with
the same model, sensing, timing, footprint, budget and arrival requirements.
Proceed through indices 1–3 without outcome-based changes to the revision.
The first result and next live trial are recorded below.

## Layout 0 result: earlier verified goal, return still incomplete

The owner exited 0 after all 4,805 camera pairs were archived. Independent
physical evaluation verified the outbound arrival at frame 3,712, versus 4,732
for the original controller: 102 seconds earlier within the same budget. The
one-second dwell stayed 10.22–15.30 mm from the goal, with maximum 100-ms speed
0.01175 m/s and all requested commands zero. There were zero disallowed contacts.
Return did not complete: final physical home distance was 0.49985 m. Both
original and revised trials therefore remain failed round trips.

Median/maximum position error was 3.23/9.81 mm, and 1,172/1,196 plans were on
time. Path length was 30.988 m, including the substantially longer return phase;
timed wall duration was 481.897 s. The release rule activated 17 times, all on
time, at frames 1,672–2,016. Pure turns decreased from 815/1,196 selected plans
(68.14%) to 610/1,196 (51.00%). These whole-mission counts include different
amounts of return travel; they are descriptive, not an isolated causal estimate
of turn reduction or a reconstruction of every executed interval.

`scripts/compare_heading_release_navigation_development.py --layout-index 0`
confirmed equal non-treatment launch settings and unchanged original source
hashes. Physical outcomes and selected-action mechanism counts are saved in
`go2_heading_release_matched_comparison_layout00_v1_attempt_001/result.json`.
This first result is consistent with useful recovery release, but establishes
neither improved round-trip success nor repeatability.

The same revision was then launched on layout 1:
`go2_heading_release_fresh_learned_round_trip_native_layout01_4800_v1_attempt_001`.
The prior owner was absent; CPU utilization was 0.2%, available RAM was 76.77 GiB
and RecoveryStorage had 98.58 GiB free. Both GPUs were idle. Indices 2 and 3
follow without changing the controller, budget or arrival requirements.

## Layout 1 result: verified round trip where original had no arrival

The revised owner exited 0 after retaining all 4,801 camera pairs. Independent
physical evaluation verified outbound arrival at frame 3,400 and home arrival
at frame 4,799, within the unchanged 4,800-tick budget. The one-second dwells
stayed 3.44–6.61 mm from the goal and 2.87–14.28 mm from home. All requested
commands were zero during both dwells, maximum 100-ms speeds were 0.00967 and
0.02117 m/s, and the full run had zero disallowed contacts. The original layout-1
trial reached neither endpoint.

Median/maximum position error was 5.33/10.04 mm, with 1,168/1,192 plans on time.
Path length was 32.943 m; final home distance was 2.69 mm. Timed wall duration
was 481.465 s. Return took 139.9 simulated seconds after the 340.0-second
outbound arrival. This was a narrow budget margin; the result establishes this
trial's completed round trip, not robust timing margin or repeatability.

The recovery-release rule activated seven times, all on time. Pure turns were
547/1,192 selected plans (45.89%), versus 880/1,200 (73.33%) in the original
failed run. These whole-mission proportions include different trajectories and
return phases. The comparison confirmed unchanged original source hashes and
equal non-treatment settings, and is saved in
`go2_heading_release_matched_comparison_layout01_v1_attempt_001/result.json`.

The unchanged revision was launched next on layout 2:
`go2_heading_release_fresh_learned_round_trip_native_layout02_4800_v1_attempt_001`.
The prior owner was absent; CPU utilization was 0.3%, available RAM was 76.77 GiB,
and RecoveryStorage had 95.24 GiB free. Both GPUs were idle. Finish layout 2,
then layout 3, before interpreting the full revision cohort or starting the
prepared matched training-method navigation controls.

The layout-1 root also contains visually inspected `verified_round_trip.png`
and `.svg`, showing the full physical outbound and return paths. The return
retraced the long known route around the maze walls, without repeating the
initial dead-end excursion. These trajectories do not isolate memory's causal
contribution.

## Layout 2 result: successful round trip preserved; release inactive

The owner exited 0 after archiving 2,505 camera pairs. Independent physical
evaluation verified outbound arrival at frame 1,664 and home arrival at frame
2,503. The one-second dwells stayed 22.74–25.27 mm from the goal and
21.49–22.08 mm from home, with all requested commands zero and maximum 100-ms
speeds 0.00663 and 0.00568 m/s. There were zero disallowed contacts.

Median/maximum position error was 7.61/12.62 mm, and 606/619 plans were on time.
Path length was 21.529 m; final physical home distance was 21.55 mm. Timed wall
duration was 251.317 s. Outbound and return took 166.4 and 83.9 simulated seconds,
respectively. The original controller also completed this layout, at frames
1,570 and 2,359: this revised run's home arrival was 14.4 seconds later.

The release rule activated zero times. This run therefore preserves the observed
success on this layout but provides no evidence that release caused its timing
or trajectory differences. The matched comparison confirmed unchanged original
source hashes and equal non-treatment settings and is saved in
`go2_heading_release_matched_comparison_layout02_v1_attempt_001/result.json`.

The final unchanged revision trial was launched on layout 3:
`go2_heading_release_fresh_learned_round_trip_native_layout03_4800_v1_attempt_001`.
The prior owner was absent; CPU utilization was 0.5%, available RAM was 76.63 GiB,
and RecoveryStorage had 93.56 GiB free. Both GPUs were idle. After its evaluation,
complete the four-layout revision comparison and begin the prepared matched
training-method controls without selecting or replacing layouts by outcome.

## Layout 3 result: verified round trip; revision cohort complete

The owner exited 0 after retaining 4,730 camera pairs. Independent physical
evaluation verified outbound arrival at frame 3,540 and home arrival at frame
4,728. The one-second dwells stayed 18.58–22.81 mm from the goal and
21.15–25.18 mm from home. All requested commands were zero during both dwells,
maximum 100-ms speeds were 0.00842 and 0.00863 m/s, and the full run had zero
disallowed contacts. The original run reached neither endpoint.

Median/maximum position error was 8.92/12.15 mm; 1,158/1,175 plans were on time.
Path length was 33.345 m, final home distance was 25.34 mm, and timed wall
duration was 474.320 s. Return took 118.8 simulated seconds after the
354.0-second outbound arrival. The release rule activated 58 times, 57 on time.
Pure turns occupied 47.32% of selected plans, versus 70.50% in the original run;
as with prior layouts, these counts include different trajectories and phases.

The fourth matched comparison confirmed unchanged original source hashes and
equal non-treatment settings. All four results are combined in the result above.
This supports continuing with the recovery revision, while retaining layout 0
as an unresolved round-trip failure and recognizing narrow time margins on
layouts 1 and 3. JEPA-specific advantage and memory's causal contribution remain
untested by this controller comparison.

The first matched training-method trial was then launched on layout 0 with the
same recovery revision and the frozen JEPA model/correction:
`go2_matched_training_jepa_heading_release_native_layout00_4800_v1_attempt_001`.
The previous owner was absent; CPU utilization was 0.6%, available RAM was
76.87 GiB and RecoveryStorage had 90.24 GiB free. Both GPUs were idle. Continue
JEPA/direct/supervised-rollout per layout in index order using the shared
launcher documented in `docs/go2_matched_motion_residual_controls_2026-09-14.md`.
