# Heading recovery within the predicted hold clearance

Status: both prospective native trials exited 0 and physically verified their
goal and return arrivals, with zero contacts. The added recovery rule activated
zero times in both trials. This pair therefore demonstrates navigation capability
under the fixed setup, but neither a recovery-rule benefit nor repeatability.

Prospective development hypothesis: a pure turn can make useful heading
progress without increasing clearance by the millimetre amount demanded by
the existing reserve-recovery rule. The two completed directed-view trials
failed with long route-turn hold streaks despite small localization errors.
Their preferred turns often predicted a minimum clearance no worse than hold,
but insufficient endpoint clearance gain. See
`docs/go2_nearby_panorama_directed_view_hypothesis_2026-09-14.md`.

The new helper applies only when hold is selected and every moving candidate
is ineligible. Consider only the existing preferred pure turn, never a new
translation or action. Require finite eight-segment paths, hold minimum above
the existing 0.45-m nominal footprint, turn minimum at least the hold minimum,
and turn endpoint at least the hold endpoint. Require improved selected
utility over hold; for route turns also require reduced predicted heading
error over the committed action interval. Survey turns use their existing
angular utility. Re-evaluate each plan and clear any prior recovery latch.
Exclude terminal position approach.

This changes recovery eligibility: angular progress replaces minimum endpoint
clearance gain for this narrow case. It does not restore or certify the full
0.48-m moving-action reserve. The 0.45-m footprint radius, stored forecast error
reserves, actual observation vetoes, action bank, pulse durations, model and
correction remain unchanged. Point forecasts are not safety certificates, and
submillimetre clearance differences are smaller than measured prediction errors.
Native geometry remains evaluator-only.

Run one prospective pair on layouts 0 and 1 with the same compact recorder and
CPU groups as their completed directed-view references: 0–7,16–23 and
8–15,24–31. Keep the existing directed-view rule as a fixed background condition
to isolate this added recovery choice; this does not promote that failed variant
as a reliability improvement. Retain the same 4,800-tick budget, 20-mm observed
arrival threshold and 40-mm physical one-second quiet dwell. No parameter search
or automatic retry is planned.

Evaluate full goal/return outcomes, contacts, localization, deadlines and actual
requested turn sequences. Count recovery activations, clearance loss or gain
and heading progress only on matching executed windows. Compare against the
recorded references, including first execution divergence before intervention;
single runs cannot establish a causal episode benefit or repeatability.
Preserve failures. Reliable independent-layout replication, matched control and
memory studies, realistic sensing/timing and hardware evidence remain open.

Implementation: `lewm/hold_relative_heading_recovery_development.py`.
Launcher: `scripts/run_go2_hold_relative_heading_recovery_development.py`.
Output roots: `go2_hold_relative_heading_recovery_native_layoutXX_4800_v1_attempt_001`.

Four focused tests passed in 1.79 seconds: route/survey selection, forecast and
radius preservation, rejection of clearance loss or invalid paths, ineligible
contexts, and terminal-approach exclusion/latch behaviour. Applying the actual
helper to saved middle-window selections matched the prospective counts exactly:
472/501 eligible states on layout 0 and 477/501 on layout 1. These are repeated
recorded states, not executed counterfactuals or independent trials.

Both launcher writer configurations preserved all 90 reference source identities
and non-treatment settings; additions are the new helper/launcher and treatment
metadata. No previous native owner remained live. Before dispatch, available RAM
was 76.69 GiB and RecoveryStorage free was 35.28 GiB. The planned pair requires
approximately 7 GiB of output, within that available space.

The native pair launched on 2026-09-14 around 19:31 in the process logs.
Layout 0 owner PID 3432987, tool session 94594; layout 1 PID 3433007,
tool session 52484. Both were confirmed live with their assigned disjoint CPU
groups. Outcomes are pending. Poll these exact handles; wait for owner exits
and complete archives before another native batch or edits to bound sources.

The prepared comparison helper is
`scripts/compare_hold_relative_heading_recovery_development.py --layout-index N`.
It scores a recovery only when the final selected action remains that recovery
and the saved requested sequence through 700 ms matches. Heading progress uses
the frozen observed target and physical poses at dispatch (300 ms) and commit
end (700 ms). It also measures physical centre-to-wall clearance over that
interval, without equating centre distance to articulated-body safety. A
read-only probe on 25 previously executed turns reproduced the existing endpoint
XY errors; no historical trial records were changed or relabelled as recovery
trials. Complete navigation, contacts and first command divergence remain part
of the comparison.

## Both trials complete: verified round trips, no recovery exposure

Layout 0 retained 3,951 camera pairs and verified goal frame 2,789 and home frame
3,949. Goal dwell distance was 7.17–9.03 mm and home dwell 21.69–24.83 mm; maximum
100-ms speeds were 0.02435/0.00924 m/s with all requests zero. Median/maximum
position error was 5.99/10.22 mm. Of 980 plans, 962 were on time. Timed execution
took 396.205 seconds; process duration including archives was 557.54 seconds,
maximum RSS 21,208,040 KiB, swaps zero.

Layout 1 retained 4,591 pairs and verified goal frame 3,016 and home frame 4,588.
Goal dwell distance was 3.82–6.15 mm and home dwell 13.89–17.32 mm; maximum 100-ms
speeds were 0.00594/0.01428 m/s with all requests zero. Median/maximum position
error was 3.55/8.45 mm. Of 1,140 plans, 1,109 were on time. Timed execution took
460.470 seconds; total process duration was 670.81 seconds, maximum RSS
24,378,508 KiB, swaps zero. Both episodes had zero disallowed contact samples.

Neither run selected the added recovery: zero activations and zero executed
recovery windows. The two recorded directed-view references failed to reach
their goals, but this difference cannot be attributed to a rule that never
activated. Keep every failed reference and distinguish capability from reliable
repeated performance. This is not evidence that the proposed recovery works.

Next: freeze this controller and run two further prospective repetitions on each
of the same two layouts, as two parallel pairs. Fix all four repetitions before
their outcomes and complete the roster without tuning between trials. Report
the four new repetitions separately from this already observed successful pair.
These are execution-repeatability trials on known development layouts, not new
independent-maze evidence. New layouts and current control/memory comparisons
remain necessary after the repeatability result.

Both matched comparison roots are complete:
`go2_hold_relative_heading_recovery_comparison_layout00_v1_attempt_001` and
`go2_hold_relative_heading_recovery_comparison_layout01_v1_attempt_001`.
They preserved all 90 predecessor source identities and non-treatment settings.
First requested-command differences were at simulator 15.42/15.02 seconds;
neither trial had any recovery intervention. Path lengths were 30.026/31.772 m.
Both comparison PNG/SVG pairs were generated and visually inspected, showing
the complete outward and physical return trajectories. All four prior/new
outcomes remain retained; no native owner is still running from this pair.
