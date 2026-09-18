# Survey clearance repositioning: prospective development hypothesis

Status: both trials finished: two verified goals and one verified round trip,
with zero contacts. All three repositioning command sequences executed. Both
trials diverged from their references before intervention, so the episode
improvements do not establish a causal treatment effect. Reliability remains
unproven.

The matched JEPA layout-0 failure spent frames 2,000–4,000 surveying without
advancing through the first passage. Saved executed turns generally had the
predicted alignment direction, but repeated clearance recovery selected turns
opposite the desired view. The direct layout-1 failure showed a similar pattern.
Many existing translating forecasts passed full reserve and improved clearance
over hold, yet received no survey score. Detailed evidence is in
`docs/go2_matched_motion_residual_controls_2026-09-14.md`.

`lewm/survey_clearance_reposition_development.py` adds one recovery choice after
the existing heading-release logic. It applies only during a survey with an
active recovery latch selecting the opposite pure turn from the preferred turn,
and never during terminal position approach. Among the existing forward/arc
actions, require finite eight-segment forecasts, the unchanged full reserve,
minimum clearance no worse than hold, and endpoint improvement over hold of at
least `max(1 mm, 0.1 * remaining hold reserve deficit)`. This gain rule follows
the existing hold-relative recovery rule. Rank eligible translations by minimum
then endpoint clearance, with existing action order resolving ties.

After selecting one translation, clear the turn latch and keep the unfinished
survey target. Re-evaluate from the next observation; do not declare the view
complete or substitute a new route. Current-observation command vetoes remain
active. Learned forecasts and observed obstacle memory supply eligibility;
native geometry/pose remains evaluator-only. Model weights, correction fits,
action repertoire, command durations, footprint, reserves and arrival checks
are unchanged.

Three focused tests passed in 2.20 seconds: preserve target/forecast evidence and
input records; reject missing, nonfinite, insufficient-reserve or non-improving
paths; leave route motion, unlatched turns and preferred turns unchanged. Saved
eligibility reports apply this rule independently to recorded states, without
propagating changed state or claiming counterfactual execution.

Over frames 2,000–4,000, the rule was locally eligible in 111 of 501 JEPA-layout-0
plans (107 forward, four left arc) and 35 of 501 direct-layout-1 plans (all
forward). Reports are retained as
`prospective_survey_reposition_saved_eligibility_v1.json` in each run root. These
counts describe the original recorded states only; a live reposition changes
future observations and subsequent eligibility.

After the current comparison, test this variant prospectively on the diagnosed
JEPA layout-0 failure, retaining the same mission budget and physical evaluation.
Assess full goal/return outcomes, contacts, localization error, survey progress,
recovery behavior and timing. A local escape is insufficient to establish the
navigation goal. Broader layout coverage, repeated execution, matched controls,
memory causality and realistic sensing/timing remain necessary.

The native launcher is prepared as
`scripts/run_go2_survey_reposition_native_development.py --layout-index 0`.
It uses the original serial `FreshCameraSession`, full default CPU affinity and
the matched frozen JEPA correction, matching the diagnosed layout-0 reference's
capture setup. The compact buffer is deliberately a separate change from this
first controller experiment. Output:
`go2_survey_reposition_matched_jepa_native_layout00_4800_v1_attempt_001`.
The launcher accepts only the two serial-reference layouts, 0 and 1; first use
is layout 0 after the complete training-method comparison. Writer metadata and
runtime method dispatch were checked without native launch. The matched
correction supplies construction/correction, and the new class supplies only the
survey recovery selection. A single successful new trial would still not
establish repeatability or isolate all effects of asynchronous timing.

## First native result: verified goal, incomplete return

Layout 0 exited 0 with all 4,805 camera pairs. Independent evaluation verified
goal arrival at frame 3,852, with a one-second dwell 13.21–18.95 mm from the goal,
maximum 100-ms speed 0.01782 m/s and all requests zero. Return did not complete:
final physical home distance was 1.509 m. Contacts were zero. Median/maximum
position error was 3.28/9.66 mm; 1,158/1,184 plans were on time. Path length was
29.738 m, timed execution 481.874 s and total process duration 646.26 s. Maximum
RSS was 39,903,116 KiB with zero swaps.

Launch comparison preserved all 85 baseline extra-source identities and every
non-treatment setting. The only additions were the prospective launcher and
repositioning source, plus treatment/identity metadata. All saved corrections
used the assigned JEPA model and fit. The new rule selected left arc at frame
1,844 and forward at 1,892; both plans were on time, and all actual requested
20-ms intervals matched the full known-prefix/candidate sequence through 700 ms.
Heading release did not activate. Both moves preserved the original full reserve.

However, the first requested-command difference from the recorded reference
occurred at simulator time 66.60 s, before the first reposition at 185.90 s.
Therefore the episode does not demonstrate that repositioning resolved the
reference's original stall. It demonstrates two executed eligible recovery moves
and a goal-reaching trajectory, with the required round trip still incomplete.
Keep timing variation and controller effects distinct in subsequent interpretation.

Comparison and diagnostics:
`go2_survey_reposition_comparison_layout00_v1_attempt_001/result.json`.
The unchanged rule will next run on serial-reference layout 1 under the same
model, correction, capture, mission budget and arrival checks.

The layout-0 comparison PNG/SVG was generated and visually inspected. It shows
the prospective run exploring the northern dead end, reaching the goal via the
outer route, and returning to the first passage before time expired. The
recorded reference remained at that first passage during its outbound mission.
This visual difference carries the same attribution limitation noted above.

After the layout-0 owner exited and its evaluation/comparison completed, launched
`go2_survey_reposition_matched_jepa_native_layout01_4800_v1_attempt_001` with the
unchanged serial launcher. PID 3416752 was confirmed live; available RAM was
76.59 GiB. Its outcome remains pending.

## Layout 1 complete: verified round trip

The owner exited 0 after 547.64 seconds including archive writing, retaining
4,061 camera pairs. Independent physical evaluation verified goal frame 2,595
and home frame 4,059. One-second dwell distances were 11.01–13.79 mm at the goal
and 5.73–14.03 mm at home, maximum 100-ms speeds 0.01794/0.03250 m/s, with every
requested interval zero. There were zero contacts. Median/maximum position
error was 4.37/7.92 mm. Of 1,008 action plans, 983 were on time. Path length was
30.875 m; timed execution took 407.278 seconds. Maximum RSS was 34,087,588 KiB
with zero swaps.

All 85 baseline source identities and non-treatment settings were preserved.
All corrections used the assigned JEPA model and frozen fit. Repositioning
activated once, selecting forward at frame 848; the on-time plan's full 700-ms
requested sequence was observed. Its predicted minimum clearance was 0.48343 m,
with the unchanged full reserve. Heading release did not activate.

The first requested-command difference from baseline was simulator time 14.92 s,
well before repositioning at 86.30 s. The prior reference reached the goal but
did not return; this new run verifies a round trip, without establishing that
the added rule caused that difference. Comparison and execution diagnostics:
`go2_survey_reposition_comparison_layout01_v1_attempt_001/result.json`.
The comparison PNG/SVG was generated and visually inspected: both runs traverse
the outer route to the goal, and the new trial completes the physical return
where the reference exhausts its budget along the lower passage.

Layout 0 completed six full frontier panoramas totalling 176.8 seconds; layout 1
completed three totalling 88.0 seconds. In layout 0, two successive dead-end
panoramas began only 7.8 and 11.9 cm from the preceding viewpoint, taking 29.6
and 32.4 seconds. This motivates a separate prospective test of fresh directed
revisits near completed panoramas; see
`docs/go2_nearby_panorama_directed_view_hypothesis_2026-09-14.md`.
