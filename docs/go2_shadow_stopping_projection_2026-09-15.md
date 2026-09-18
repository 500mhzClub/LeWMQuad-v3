# Planned stopping enforcement within the same frozen controller

Run supervised prediction seed 2026091001 on exposed shared-recovery layout 0,
then layout 1, once each. Before any new run, the saved references showed five
stopping interventions among 584 selected plans on layout 0 and none among
405 on layout 1. This reverses the initially proposed execution order so the
first test uses the maze with observed intervention exposure; both remain in
the comparison. The completed supervised runs on these same layouts
are the enforcement-on references. Keep frozen model/correction, sensing,
noise, action scoring, predicted-path clearance, reserve/heading/arrival
selection, routing memory, perception/recovery, timing and actual dispatch
guards unchanged. Compute planned stopping checks and their proposed action
changes in shadow, but retain the preceding selection and recovery-turn state.

This isolates the extra planned stopping intervention, not all online rollout
or predictive ranking. These are two repeated development mazes, not new
independent layouts. Retain both outcomes irrespective of success. Native
runs remain sequential on the same CPU group as each reference. Asynchronous
execution means trajectories and timing need not match exactly.

Sources: `lewm/shadow_stopping_projection_development.py`,
`scripts/run_go2_shadow_stopping_projection_development.py`,
`scripts/evaluate_go2_shadow_stopping_projection_development.py`.
Root: `go2_shadow_stopping_projection_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001`.

Evaluate physical goal/home arrivals, contacts, tracking/timing failures and
recorded interventions that would have changed selected actions. The actual
sensor dispatch safeguards remain active. A lack of intervention exposure
would limit interpretation; shadow actions alone are not navigation outcomes.

Focused stopping and existing auxiliary dispatch checks: 14 tests passed.

Layout 0 completed: 2,870 poses, physical goal/home arrivals at frames 2,275
and 2,868, zero contacts, maximum pose error 11.450 mm. All 683 selected plans
used the frozen learned model/correction and left the planned stopping action
change unapplied. There was one proposed intervention at frame 504, right arc
to right turn. Duration was 287.12 simulated seconds versus 245.58 reference;
519/683 plans were on time versus 518/584. Maximum recorded simulator lag was
21,632.589 ms. Owner completed archival and exited 0 after 6:58.88, no swaps.
All 172 common runtime sources match the retained reference. The intervention
was not necessary for this successful mission; different trajectories and
timing limit the performance inference. At the proposed intervention, all 20
associated 20-ms intervals applied translation; minimum observed stopping
clearance was 0.484149 m. Its on-time, committed action was therefore an actual
execution exposure. The two-panel PNG/SVG comparison is complete and inspected.

Layout 1 also completed: 1,601 poses, physical goal/home arrivals at frames
1,020 and 1,599, zero contacts, maximum pose error 10.347 mm. All 392 selected
plans used the same frozen learned model/correction and disabled stopping
enforcement. Duration was 160.44 simulated seconds versus 167.28 reference;
383/392 plans were on time versus 398/405. Maximum recorded simulator lag was
482.380 ms. Owner completed archival and exited 0 after 4:06.10, no swaps.
All 172 common runtime sources match the reference. Comparison and inspected
PNG/SVG figures are complete for this layout too.

Its one proposed intervention, frame 1,072, would change a right arc to a right
turn. The plan was on time and committed, but the actual stopping guard vetoed
it at 109.00 s and latched the other 19 intervals; none applied translation.
Signed recovery requested a -45-degree view. Translation resumed at 113.00 s,
with 2,320 subsequent translation intervals before the verified return.
`shadow_stopping_dispatch_exposure_v1.json` and
`shadow_stopping_veto_recovery_diagnostic_v1.json` record this distinction from
layout 0. The forecast anticipated a real veto, but the downstream guard and
recovery handled it without mission failure. These observations do not prove
that every future obstacle or unsafe approach is recoverable.

## Completed result

| Layout | Enforcement on: round trip / simulated s | Enforcement off: round trip / simulated s | Proposed off interventions |
| --- | --- | --- | ---: |
| 0 | Yes / 245.58 | Yes / 287.12 | 1 (translation executed) |
| 1 | Yes / 167.28 | Yes / 160.44 | 1 (actual guard vetoed translation) |

All four outcomes have zero contacts. The two off runs are followups on exposed
mazes, with one model seed; they are not new independent layouts or replacements
for any failed mission. There is no demonstrated success advantage from this
extra planned stopping intervention, and inconsistent timing/trajectory changes
do not establish a performance advantage. Keep both current off recordings and
the on references full for now; older redundant successful depth retirement
reclaimed 2,772,365,312 allocated bytes during this comparison, with every
failure and all non-depth evidence preserved.

The aggregate is
`go2_shadow_stopping_projection_complete_comparison_v1_attempt_001/result.json`.
The next scientific comparison must address future-outcome action ranking,
not add another stopping/recovery variant. This narrow ablation leaves other
predicted-path and recovery selection active, so it does not complete the
same-controller full online-rollout requirement. RGB dependence, calibrated
sensing/timing and hardware validation remain unresolved. No simulation is
running after these two completed owners.
