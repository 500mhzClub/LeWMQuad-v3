# Directed revisits near completed panoramas

Status: all four native trials and both matched comparisons are complete. The
baseline achieved one goal and one round trip in two trials; the directed-view
variant achieved neither goal in two trials. All four had zero contacts. One
directed revisit executed and completed, but the experiment does not establish
improved navigation. Do not promote the variant on these results.

Prospective development hypothesis: repeated full panoramas from nearly the
same position consume navigation time without proportionate benefit. The
survey-reposition layout-0 trial spent 176.8 seconds on six frontier panoramas;
two repeated northern dead-end surveys alone consumed 62.0 seconds, with their
viewpoints moving only 7.8 and 11.9 cm. Goal arrival succeeded but return exceeded
the fixed budget. These receipts do not prove the additional views were useless.

The new variant keeps the first full nine-stage panorama. For a later frontier,
use one fresh directed view if its target and current viewpoint are each within
0.25 m of a completed panorama's target and start/end viewpoints respectively.
The radius is fixed prospectively at half the existing 0.5-m frontier standoff;
it is not a visibility certificate. Require all nine earlier stages to have
completed using measured headings and subsequent maps. Directed views cannot
themselves seed reuse. Retain the original remaining-route-length standoff,
requested heading and post-alignment map requirement. Completion excludes only
the original 0.10-m neighbourhood of observed floor from frontier selection;
floor remains traversable and no unknown space becomes free.

This isolates survey scheduling. Both arms use the survey-reposition JEPA
controller, frozen weights and correction, original clearance reserves, action
repertoire, timing and 4,800-tick budget, 20-mm observed arrival and 40-mm physical
one-second quiet dwell. Native geometry remains evaluator-only.

Run layouts 0 and 1 concurrently with the established compact recorder, using
CPU groups 0–7,16–23 and 8–15,24–31 respectively. First run the unchanged survey
controller on both layouts, then the directed-revisit variant on both after
the first pair's archives finish. Both arms must use the same compact execution
profile. Preserve every failure and compare full goal/return outcomes, contacts,
survey time and completions, rule activations, requested sequences, localization
and deadlines. Single episodes and asynchronous timing do not establish causal
improvement or repeatability. No parameter search is planned for this pair.

Implementation: `lewm/nearby_panorama_directed_view_development.py`.
Launcher: `scripts/run_go2_nearby_panorama_comparison_development.py`.
Seven focused tests passed in 1.76 seconds, including the existing frontier
tests: full completion before reuse, fresh heading/map for a directed revisit,
unchanged floor/exclusion behaviour, distant targets/viewpoints retaining full
panoramas, and no shortcut across a long route detour. All four launcher writer
configurations were exercised without native execution.
This experiment remains ideal-sensor simulation development; the wider goal
still requires reliable independent-layout completion, current matched controls,
memory causality, realistic sensors/timing and bounded real-platform evidence.

## Baseline pair launched

After the serial layout-1 owner exited and its independent evaluation finished,
launched both baseline trials on 2026-09-14 at approximately 19:00 local log time:

- Layout 0: `go2_nearby_panorama_baseline_native_layout00_4800_v1_attempt_001`,
  PID 3422321, tool session 2359, CPU group 0–7,16–23.
- Layout 1: `go2_nearby_panorama_baseline_native_layout01_4800_v1_attempt_001`,
  PID 3422340, tool session 27562, CPU group 8–15,24–31.

Both owners were confirmed live with the assigned disjoint CPU affinity. Before
launch, available RAM was 76.18 GiB and RecoveryStorage had 49.01 GiB free;
startup left 69.11 GiB available RAM. Outcomes are pending. Wait for both owner
exits and complete archives before launching the directed pair. Preserve these
roots; an observation timeout does not authorize restart.

## Baseline pair complete

Both owners exited 0 and saved complete archives. Layout 0 physically verified
goal frame 3,604 and home frame 4,756, with zero contacts. Goal dwell distance
was 17.82–25.25 mm and home dwell 10.10–17.03 mm; maximum 100-ms speeds were
0.01762/0.01491 m/s and all requested dwell intervals were zero. Median/maximum
position error was 7.82/14.01 mm. Path length was 31.494 m. Of 1,163 plans,
1,151 were on time. Five full frontier panoramas consumed 163.2 seconds.

Layout 1 exhausted the budget without goal arrival, with zero contacts and
median/maximum position error 7.73/8.57 mm. Path length was 5.304 m. Of 1,200
plans, 1,151 were on time. Its first panorama completed in 27.6 seconds; its
second remained pending for 400.4 seconds, at view stage 7. All 501 plans over
frames 2,000–4,000 held (485 on time). At frame 3,000, hold minimum clearance
was 0.46628 m and preferred right-turn clearance 0.46652 m; no moving forecast
passed full reserve. This stall occurs during an unfinished panorama, and is
distinct from repeating a completed nearby panorama. Saved evidence:
`middle_window_hold_diagnostic_v1.json` in the layout-1 baseline root.

These outcomes differ from the earlier serial survey-reposition trials. They
do not isolate a compaction effect or establish repeatability. Use the newly
matched compact baselines for the directed-view comparison.

The baseline processes took 669.39/712.81 seconds including archive writing;
maximum RSS was 25,029,096/25,320,596 KiB, with zero swaps. Physical evaluations,
navigation summaries, survey diagnostics and resource records are saved in
both roots. The comparison helper now includes pending survey time as well as
completed surveys, preventing a long unfinished survey from disappearing from
the time-use comparison.

After both baseline owners exited and their evaluations completed, launched the
prepared directed pair with the unchanged launcher and CPU groups. Tool sessions
are 39098 (layout 0) and 67172 (layout 1). Output roots:
`go2_nearby_panorama_directed_native_layout00_4800_v1_attempt_001` and
`go2_nearby_panorama_directed_native_layout01_4800_v1_attempt_001`.
Before launch, RAM available was 76.39 GiB and RecoveryStorage free was 42.13 GiB.
The directed outcomes remain pending.

After completion, evaluate each root and run
`scripts/compare_nearby_panorama_navigation_development.py --layout-index N`
for N=0 and 1, then create the existing comparison plots. The helper checks
equal non-treatment settings and source identities, measures completed/pending
survey time, checks requested sequences during directed views, and records
whether execution diverged before the first directed revisit. Do not infer
causal episode improvement from an unpaired trajectory difference.

## Baseline stall-entry diagnosis while directed trials run

The layout-1 baseline's longest hold streak spans frames 1,304–4,800 (875
plans). In the saved entry window, frames 1,180–1,340, 35 of 41 plans matched
the requested sequence through 700 ms. The 25 matched moving plans were turns;
their median/maximum corrected path XY error was 5.42/11.41 mm. Errors were not
scored for interrupted command sequences. Several interruptions around entry
were caused by stale observations, veto latches or missing on-time plans.
The physical base-centre-to-wall minimum in this window was 0.47678 m; this is
evaluator-only centre geometry, not articulated-body clearance. Evidence:
`clearance_stall_entry_diagnostic_v1.json` in the baseline layout-1 root.

A separate saved-state hypothesis examines preferred survey turns when every
moving candidate is currently ineligible. Requiring every individual forecast
segment to be no worse than hold admitted zero of 501 middle-window states.
Using the existing hold-relative minimum-path comparison instead, requiring
no endpoint degradation and improved survey utility, admitted 452/501 states;
median endpoint clearance gain was only 0.232 mm. At frame 3,000, one segment
was 0.0062 mm worse than its hold counterpart although the whole-path minimum
improved. These distinctions are far smaller than measured forecast errors;
they are selection hypotheses, not accuracy or safety guarantees. Both rules
admitted zero states in the distinct supervised-layout-3 hold failure.

The minimum-path hypothesis would replace the existing millimetre clearance-gain
requirement for pure turns with angular progress, while preserving nominal
footprint and the hold-relative minimum. It changes recovery eligibility and
does not restore full reserve. It has not been implemented or executed. These
repeated saved states are not independent trials. Preserve both eligibility
reports; neither is part of the running directed-view experiment.

## Directed pair complete: no arrivals

Both owners exited 0 with all 4,805 camera pairs and complete archives. Physical
evaluation confirmed no arrival and zero contacts in either layout. Layout-0
median/maximum position error was 4.97/6.64 mm; layout 1 was 3.09/5.12 mm.
Path lengths were 7.628/5.082 m. On-time plans were 1,179/1,200 and 1,153/1,200.
Process durations including archives were 681.01/711.86 seconds, maximum RSS
25,421,716/25,398,972 KiB, and swaps zero.

Layout 0 completed three full panoramas and one directed revisit, totalling
97.2 seconds. The directed revisit ran from simulator time 164.7 to 166.3 s;
all four associated plans had their full requested sequences through 700 ms
observed. The first command difference from the baseline was already at 15.42 s,
so neither the episode difference nor the total survey-time difference can be
attributed to that intervention. The reference completed five full panoramas
in 163.2 seconds and verified its round trip.

Layout 1 completed two full panoramas in 56.4 seconds, with no directed-revisit
activation. Its execution first differed from baseline at 14.92 s. The baseline
had spent 428.0 seconds in completed and pending surveys; the inactive treatment
cannot explain that difference. Both layout-1 episodes failed to reach the goal.

Both directed runs subsequently selected hold on all 501 middle-window plans
while requesting routes to frontiers, with no survey pending. At frame 3,000,
preferred right turns predicted improved waypoint alignment but lacked full
reserve. Saved minimum-path hold-relative heading eligibility was 472/501
states in layout 0 and 477/501 in layout 1. This extends the recovery hypothesis
to route turns; no such recovery was implemented or executed in these trials.

Comparisons preserve equal non-treatment settings and source identities:
`go2_nearby_panorama_comparison_layout00_v1_attempt_001/result.json` and
`go2_nearby_panorama_comparison_layout01_v1_attempt_001/result.json`.
Both roots contain the complete outcome, timing, survey mechanism and command
attribution results. All four per-trial roots retain physical evaluations,
navigation summaries, survey diagnostics, resource records and recordings.
Both comparison PNG/SVG pairs were generated and visually inspected. Layout 0
shows the directed variant stopping at the northern dead end while the reference
returns from the goal. Both layout-1 trajectories stop near the first lower
passage, consistent with the saved hold diagnostics.

Next scientific target: test a preferred pure-turn recovery that makes predicted
heading progress while preserving the existing hold-relative minimum clearance
and nominal footprint, with no endpoint clearance loss. This changes the
recovery gain criterion and does not claim restoration of full reserve. Keep
the experiment separate, retain the failed references, and assess actual
executed turns and complete navigation. More reliable repetitions and broader
independent-layout evidence remain necessary regardless of a local escape.
