# Training-method comparison with current noisy perception

The current-plane learned cohort completed three verified round trips on four
development layouts. Layout 3 stopped at the unchanged full-3D-speed limit.
The stronger reactive comparison is ongoing and takes scheduling priority.
These layouts are development-exposed; this study cannot establish unseen-maze
generalization or hardware readiness.

Fix twelve subsequent assignments: JEPA, direct and supervised rollout on each
of layouts 0–3, using seed 2026091001 and full inputs. Include new JEPA runs
through the shared launcher, preserving the earlier cohort separately as a
repeatability reference. Run conditions in that order within each layout;
the even CPU group runs layouts 0 then 2, and the odd group runs 1 then 3.
At most two native owners, including archive phases, may coexist. Finish the
reactive roster first. Preserve every outcome without tuning, retries or
replacement of unsuccessful assignments.

`scripts/run_go2_current_plane_matched_training_noise_development.py` takes
`--layout-index 0..3 --condition jepa|direct|supervised_rollout`. It reuses the
exact existing `RoutingMemoryRuntime`, which already includes
`MatchedMotionResidualRuntime`. No new action selector, recovery rule or
perception class is introduced. Each model uses its corresponding frozen
visual-motion correction in `FITS`; do not reuse the JEPA correction for the
other models. The original matched fits used the same training/validation
window populations and visual labels; no new fitting is needed here.

Keep cached visual tracking, paired raw-height registration, independent depth
observation, current-plane floor mapping, persistent memory, 2-mm depth noise,
camera/gyro streams, six actions, command durations, 4,800-tick budget,
300-ms delay, arrival dwells, physical speed/contact guards and all planner
recovery policies identical across methods. The model training objective and
its matched correction are the intended differences. This tests complete
frozen training pipelines with matched correction procedures, not JEPA alone
with every numerical coefficient held equal. One training seed and four
development layouts provide limited evidence; further seeds and independent
layouts remain necessary.

A focused three-condition integration check passed before any new simulation:
the full metadata writer preserved each model/condition/correction assignment,
shared perception and recovery fields matched the completed learned layout 0,
all three actual fit files passed their frozen identity checks, and the runtime
resolves correction through the existing matched implementation. This is wiring
evidence only. Check actual launch and prediction-correction receipts after
execution, then independently evaluate physical arrivals and retain contacts,
pose/floor availability, execution stops, path lengths and timing for all runs.

Roots:
`go2_current_plane_matched_training_<condition>_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
No assignments from this launcher have started at preparation time. Ensure
space for the next pair using the existing authorized depth-retention policy;
retain current comparisons, frozen models/corrections and training inputs.
Completed, superseded recordings can lose regenerable depth after retaining
their outcomes, recipes and diagnoses. Do not turn storage preparation into a
new infrastructure project.

Before launching: reactive layout 3 exposed a reproducible partial-floor
transport conflict at frame 405 (two residuals above 3 mm, maximum 3.018 mm).
Diagnose how temporary measurement disagreement should affect continuous
perception before committing to the larger twelve-run study. No training
comparison runs have started; preserve this prepared design, and explicitly
revise shared perception for every condition if the diagnosis warrants it.

## Decision to execute after diagnosis

The reactive roster and declared-gap probes are complete. Exact sensor replays
localized the temporary floor conflict and later visual-support rejection;
the latter has no evidence supporting relaxed admission. The learned declared
gap probe resumed and completed its round trip, while reactive recovery remains
an open system limitation. These results do not require changing this study's
frozen pipeline. Proceed with the twelve assignments as specified above,
including their existing failure behavior, without incorporating the newer
floor-reacquisition runtime or injected gaps. This comparison measures the
current complete training pipelines, not a claim of solved perception.

Start JEPA layouts 0 and 1 on the even and odd CPU groups respectively, then
follow the recorded within-layout condition order. Approximately 20 GiB free
is sufficient for this first pair; manage retained recordings before subsequent
pairs if necessary. The prepared integration checks already passed; no new
test suite or source export is required to execute the unchanged launcher.

Both first assignments launched successfully: JEPA layout 0 in session 36871,
layout 1 in session 46277. Live progress reached frames 400 and 300 respectively;
actual launch records bind the intended model, correction, cached tracker and
current-plane mapper. Outcomes remain pending. After each owner exits, run
`evaluate_continuous_native_arrivals_development.py` on its exact root and save
the normal live navigation summary. Direct on the same layout is next, followed
by supervised rollout; then advance that CPU group to its second layout.

The existing `compare_matched_training_navigation_development.py` now accepts
`--current-plane-noise` to summarize each completed three-condition layout.
It retains the existing all-common-launch-field/source equality and actual
motion-correction receipt checks, with an explicit condition check for the
new roster. Only result summarization changed; the running scientific sources
and launcher were not edited.

## First two completed assignments

JEPA layout 0 exited 1 after 399.80 s including archive (maximum RSS
15,816,424 KiB, zero swaps). Independent evaluation verifies outbound arrival
at frame 2140: maximum dwell distance 21.386 mm, maximum 100-ms speed
0.01800 m/s, all requests zero. Return failed at native time 280.258 s when
full 3D speed reached 0.301271 m/s, above the unchanged 0.3 m/s guard;
horizontal speed 0.289224 m/s, vertical velocity -0.084342 m/s. It remained
in domain and had zero disallowed contact samples. The terminal sample, not
the unguarded startup settling maximum, is used in the saved speed diagnostic.
There were 2,788 camera pairs, 2,787 registered poses, median/max position error
4.316/7.776 mm, path 17.408 m, 684 selected plans and 677 on time. All 684
actual correction receipts match the frozen JEPA assignment. Preserve this
failure; the earlier successful layout-0 cohort is not substituted for it.

JEPA layout 1 exited 0 after 530.72 s including archive (maximum RSS
19,576,600 KiB, zero swaps). Independent evaluation verifies goal frame 2110
and home frame 3572: maximum dwell distances 18.627/17.928 mm, maximum 100-ms
speeds 0.015979/0.022697 m/s, all dwell requests zero. All 3,574 poses published;
median/max position error 5.115/9.246 mm, zero disallowed contact samples.
Both roots contain independent arrival evaluations and live navigation
summaries. This first pair has two goals and one verified round trip; ten
assignments remain, so no training-method conclusion is available.

Direct layout 0 launched on the freed even group in session 54409; direct
layout 1 launched after JEPA layout 1 exited in session 85463. Their outcomes
remain pending. Next on each group is supervised rollout on that same layout,
then the already specified sequence on layouts 2 and 3 respectively.

## Direct layout 0 completed; layout 1 recording pending

Direct layout 0 exited 0 after 674.37 s including archive (maximum RSS
25,386,732 KiB, zero swaps). Independent evaluation verifies goal frame 3520,
maximum native dwell distance 17.331 mm and quiet speed 0.021204 m/s, with
all dwell requests zero. It exhausted the navigation budget during return;
all 4,805 poses published, median/max position error 2.148/5.999 mm and zero
disallowed contact samples. Its last selected action at frame 4800 is forward:
the unsuccessful return was still moving, rather than a terminal holding loop.
All 1,197 actual selected-plan correction receipts match the direct model.
Supervised rollout layout 0 started next on the freed even group, session 31483.

Direct layout 1 finished navigation with a budget stop and no reported arrival;
owner exit and independent evaluation remain pending at this entry. Its saved
planning records already explain its terminal hold: after the left arc at
frame 3480, all 330 remaining selections are hold, already selected before
the memory-clearance filter. All six candidates pass the full-reserve map
check on every one of those decisions. Forward forecasts predict 48.5–73.6 mm
progress but contact scores 0.0441–0.1625; their combined distance, heading and
contact utility remains below hold by 0.00059–0.10380 m. This identifies the
action-scoring mechanism, not calibrated collision probability or proof that
an unexecuted moving action would be safe. Evidence is the layout-1
`terminal_model_hold_diagnostic_v1.json`. No policy, penalty or model is changed.

Direct layout 1 subsequently exited 0 after 697.57 s including archive
(maximum RSS 25,444,712 KiB, zero swaps). Independent evaluation confirms
no arrivals, zero disallowed contact samples and all 4,805 poses published;
median/max position error 8.855/10.808 mm. All 1,200 selected-plan correction
receipts match the frozen direct assignment. The supervised-rollout layout-1
run started next in session 57711 on the freed odd group. Both supervised
assignments are now running (layout 0 session 31483). Four of twelve outcomes
are evaluated: JEPA has two goals/one round trip, direct one goal/no round trips
on layouts 0/1, all with zero disallowed contacts. Finish all assigned layouts
before drawing the training-method comparison.

## Executed forecast diagnostics for the first four completed runs

The existing `evaluate_saved_executed_motion_forecasts_development.py` evaluated
saved raw and corrected XY predictions only where the complete requested
700-ms sequence actually occurred. No predictions were recomputed, no
unexecuted action was scored against imagined motion, and no native state
entered control. Results are saved per root as
`saved_executed_motion_forecast_evaluation_v1.json`.

| Condition / layout | Executed windows | Corrected endpoint XY RMSE | Translation-only raw / corrected RMSE | Windows with any corrected XY error >30 mm |
| --- | ---: | ---: | ---: | ---: |
| JEPA 0 | 661 | 7.45 mm | 22.66 / 9.55 mm | 0 |
| JEPA 1 | 852 | 8.69 mm | 22.78 / 9.27 mm | 0 |
| Direct 0 | 1,178 | 10.18 mm | 43.04 / 11.16 mm | 18 |
| Direct 1 | 1,189 | 7.42 mm | 39.63 / 10.20 mm | 9 |

These are each controller's own executed trajectories and action mix, not
paired prediction errors on identical observations. In particular, direct
layout 1 has 777 hold windows out of 1,189; its low overall error does not
imply navigation success. Of the 27 direct windows above 30 mm, 24 are holds,
one is a turn and two are translations. Windows overlap and are not
independent trials; these results do not certify clearance-reserve coverage.
The corrected JEPA layout-0 XY forecasts remained accurate despite the
physical speed stop: XY displacement accuracy alone does not establish a
bound on full 3D gait speed. Complete the supervised and remaining-layout
results before any aggregate training-method interpretation.

## Supervised layout 0 and completed first-layout comparison

Supervised rollout layout 0 exited 0 after 512.64 s including archive
(maximum RSS 19,792,628 KiB, zero swaps). Independent evaluation verifies goal
frame 2259 and home frame 3633, maximum dwell distances 17.073/8.351 mm,
quiet speeds 0.015579/0.017665 m/s and all dwell requests zero. All 3,635
poses published, median/max error 2.703/6.037 mm, no disallowed contacts.
Path length 23.570 m, selected/on-time plans 899/894, final home distance
7.554 mm. Its executed forecast analysis covers 894 windows: corrected
endpoint XY RMSE 9.432 mm, four windows with any corrected error over 30 mm;
translation-only raw/corrected RMSE 27.150/9.383 mm with no such exceedance.

The first complete three-method comparison passed all non-treatment launch
field/source equality and actual correction-binding checks. Result:
`go2_current_plane_matched_training_comparison_layout00_v1_attempt_001/result.json`.
All methods reached the goal. Only supervised rollout completed the return;
JEPA hit the speed guard and direct exhausted the budget. This outcome does
not support a blanket JEPA superiority claim. JEPA layout 2 launched next on
the freed even CPU group in session 34658.

Supervised layout 1 has reported goal frame 3017 and home frame 4280, but its
owner is still saving the recording at this entry. Independent evaluation and
the complete layout-1 comparison remain pending; session 57711 remains the
same running owner. JEPA layout 3 is next after that owner exits.

Supervised layout 1 subsequently exited 0 after 626.24 s including archive
(maximum RSS 22,946,584 KiB, zero swaps). Independent evaluation verifies both
arrivals, maximum dwell distances 17.251/8.185 mm, quiet speeds
0.022139/0.010940 m/s and all dwell requests zero. All 4,282 poses published,
median/max position error 5.587/13.004 mm and zero disallowed contacts.
Path length 27.108 m, selected/on-time plans 1,057/1,038, final home distance
7.962 mm. Its 1,026 executed forecast windows have corrected XY RMSE
10.405 mm and nine windows above 30 mm; translation-only raw/corrected RMSE
29.641/10.626 mm, with four such translation windows. All evidence remains
trajectory-conditional, with overlapping windows.

The complete layout-1 comparison passed the same common-setting/source and
runtime-correction checks and is saved in
`go2_current_plane_matched_training_comparison_layout01_v1_attempt_001/result.json`.
JEPA and supervised both returned home; JEPA reached both destinations earlier
(goal/home frames 2110/3572 versus 3017/4280). Direct never reached the goal.

Six of twelve assignments are now independently evaluated:

| Training condition | Goals on layouts 0/1 | Round trips on layouts 0/1 | Disallowed contacts |
| --- | ---: | ---: | ---: |
| JEPA | 2/2 | 1/2 | 0 |
| Direct | 1/2 | 0/2 | 0 |
| Supervised rollout | 2/2 | 2/2 | 0 |

These interim results do not establish a general training-method ranking.
JEPA layout 3 launched unchanged in session 40637 on the freed odd group;
JEPA layout 2 continues in session 34658. Direct then supervised follow on
each group as specified. No failed assignment is retried or substituted.

## JEPA layout 3 failure and remaining runs

JEPA layout 3 exited 1 after 86.03 s including archive (maximum RSS
4,744,664 KiB, zero swaps). Visual tracking became unavailable; no physical
stop was reported. Independent evaluation confirms no arrivals and zero
disallowed contacts, 450 camera pairs and 441 published poses, median/max
position error 1.813/3.892 mm. Path length 2.364 m, closest/final goal distance
1.282/1.386 m, all 110 selected plans on time and bound to the assigned
JEPA correction. Its 110 executed forecast windows have corrected endpoint
XY RMSE 7.210 mm and maximum 19.429 mm. These preceding accurate forecasts
do not make the later perception failure a success. The detailed raw tracking
rejection remains to be diagnosed from the retained recording after this
fixed study; no replacement trial is made.

Direct layout 3 started next in session 47493. JEPA layout 2 has reported goal
frame 1860 and home frame 3224, but remains in its recording phase in session
34658; its independent arrival evaluation is pending at this entry.

The first two complete layout comparisons now include PNG/SVG native
trajectory figures. The plotting helper labels the actual 2-mm synthetic
depth noise and includes all failed outcomes. Initial figures were visually
checked; legends obscured the layout-1 goal, so they were moved below the
axes and the final layout-1 figure was rechecked. Original drafts remain
under `native_navigation_comparison_initial_legend_overlap.*`; final figures
are `native_navigation_comparison.png` and `.svg` in each comparison root.

JEPA layout 2 exited 0 after 458.01 s including archive (maximum RSS
17,803,296 KiB, zero swaps). Independent evaluation verifies both arrivals:
maximum dwell distances 12.685/17.647 mm, quiet speeds 0.014236/0.019488 m/s
and all dwell requests zero. All 3,226 poses published; median/max position
error 3.717/7.130 mm, zero disallowed contacts. Path 23.820 m, selected/on-time
plans 786/778, final home distance 17.793 mm. All correction bindings match.
Its 763 executed windows have corrected endpoint XY RMSE 7.664 mm and maximum
27.190 mm, with no corrected-path error over 30 mm.

The complete new JEPA roster therefore has three goals and two round trips
on four layouts, zero disallowed contacts, and two preserved failures (speed
stop on 0 and tracking loss on 3). This is separate from the earlier 3/4-round-
trip current-plane JEPA roster; neither cohort replaces the other. Direct
layout 2 started next in session 45883 on the freed even group, while direct
layout 3 continues in session 47493. Eight of twelve assignments are evaluated;
the remaining four are direct and supervised rollout on layouts 2 and 3.

A targeted packet read of the new JEPA layout-3 failure provides counterevidence
to reusing the older near-range-blind-spot explanation: sampled frames 439–441
have all 307,200 primary pixels valid, 294,242–295,762 valid auxiliary pixels
and accepted independent floor planes. Last published pose is frame 440;
the last three selected actions are left turns. Saved evidence:
`terminal_sensor_availability_sample_v1.json`. Both delivered noisy packet
digests were checked. The exact tracker rejection remains unresolved.

The existing failure replay launcher now accepts `--root-name` and starts its
optional outer-candidate trace at that recording's last published frame. After
the fixed native cohort, run it with `--trace-candidates` and root
`go2_current_plane_matched_training_jepa_noise_2mm_native_layout03_4800_v1_attempt_001`.
This reuses the unchanged tracker and compares every recorded raw-pose witness;
no additional native trial, modified threshold or new replay framework is
introduced. The diagnostic has been prepared but not executed at this entry.

## Direct layout 3 verified; final supervised assignment started there

Direct layout 3 exited 0 after 530.70 s including archive (maximum RSS
19,680,664 KiB, zero swaps). Independent evaluation verifies goal frame 2038
and home frame 3600, maximum dwell distances 18.155/21.889 mm, quiet speeds
0.008173/0.016307 m/s and all dwell requests zero. All 3,602 poses published,
median/max position error 2.243/5.723 mm, zero disallowed contacts. Path length
20.390 m, selected/on-time plans 879/867, final home distance 22.013 mm.
All actual correction bindings match the direct model. Its 864 executed
forecast windows have corrected endpoint XY RMSE 11.627 mm and maximum
37.696 mm; 18 windows contain a corrected-path error over 30 mm. This
successful closed-loop result remains separate from forecast-reserve claims.

Supervised rollout layout 3 launched next on the freed odd group in session
22641. Direct layout 2 continues in session 45883 and has reported its goal
arrival and return phase; its final outcome is pending. Nine of twelve
assignments are evaluated. Remaining work in this fixed cohort is direct 2,
supervised 2, and supervised 3, then the full four-layout comparison.

## Direct roster complete; both final supervised runs active

Direct layout 2 exited 0 after 634.85 s including archive (maximum RSS
23,971,884 KiB, zero swaps). Independent evaluation verifies goal frame 2812
and home frame 4519, maximum dwell distances 24.838/12.993 mm, quiet speeds
0.016907/0.005192 m/s and all dwell requests zero. All 4,521 poses published,
median/max error 5.408/8.783 mm, zero disallowed contacts. Path length 24.619 m,
selected/on-time plans 1,115/1,106, final home distance 12.757 mm. All actual
correction bindings match. Its 1,096 executed forecast windows have corrected
endpoint XY RMSE 10.442 mm and maximum 35.838 mm, with 18 windows containing
any corrected-path error above 30 mm.

Direct and JEPA each finish at three goals and two round trips across four
layouts, all with zero disallowed contacts. Their failures differ: JEPA's
speed/tracking stops versus direct's incomplete return and learned-utility
holding loop. Equal counts do not establish equivalence or a general ranking.
Supervised rollout layout 2 started next in session 45110 on the even group;
supervised layout 3 continues in session 22641. Ten outcomes are evaluated
and all twelve assignments have now launched. Approximately 7.1 GiB remains
before the two final recordings are saved, sufficient at the observed archive
sizes; reassess before any subsequent native study.

## Completed fixed twelve-assignment study

All twelve owners have exited and independent physical arrival evaluation is
complete. Supervised rollout layout 3 exited 0 after 452.02 s including archive:
goal frame 1616, home 3052, 3,054 published poses, zero disallowed contacts.
Maximum goal/home dwell distances were 16.427/25.666 mm, with quiet speeds
0.030733/0.012254 m/s and all dwell requests zero.

Supervised rollout layout 2 exited 0 after 537.06 s including archive, zero
swaps, maximum RSS 20,724,140 KiB. Goal frame 2301 and home frame 3845 both
pass physical evaluation. All 3,847 poses published; median/max position error
3.929/7.021 mm. Maximum dwell distances 16.236/2.941 mm, quiet speeds
0.027289/0.011750 m/s, all dwell requests zero, zero disallowed contacts.
Its 946 executed forecast windows have corrected endpoint XY RMSE 9.416 mm
and maximum 35.298 mm; six windows contain any corrected-path error above
30 mm. Forecast windows remain overlapping and trajectory conditional.

| Frozen training condition | Goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| JEPA | 3/4 | 2/4 | 0 |
| Direct | 3/4 | 2/4 | 0 |
| Supervised rollout | 4/4 | 4/4 | 0 |

All four per-layout comparisons checked common launch settings/source bindings
and every selected plan's condition-specific correction binding. The aggregate
is `go2_current_plane_matched_training_noise_four_layout_summary_v1_attempt_001/result.json`.
Supervised rollout wins round-trip outcome on two layouts against each other
method and ties on two. JEPA/direct each win one layout against the other and
tie on two. This is one training seed on four development-exposed layouts;
it establishes neither statistical superiority nor equivalence. It does not
isolate training from the numerical condition-specific motion corrections.
Every original failure remains in the denominator and retained recordings.

The JEPA layout-3 tracking replay ran on the freed odd CPU group while the
last supervised layout-2 owner finished on the even group, advancing the
earlier plan to wait until all native runs ended. It reproduced all 441/441
published raw poses exactly and failed at frame 441 after 42.01 s. Evidence:
`terminal_candidate_trace_v1/result.json` and `outer_candidate_trace.json`
inside that failed run. Full depth availability did not imply usable image
correspondences: every outer candidate was rejected. A targeted unchanged
consensus-refit trace now distinguishes loss of minimum/majority support from
the earlier raw-pixel blind-spot mechanism. No threshold changes or replacement
native trial have been made.

The artifact drive has approximately 3 GiB free after this cohort. Retire
eligible completed predecessor depth before collecting another native batch;
the current twelve records and active failure inputs remain retained.
