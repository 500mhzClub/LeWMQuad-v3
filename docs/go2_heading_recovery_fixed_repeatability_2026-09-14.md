# Fixed-controller execution repeatability

Run four prospective trials, fixed before any of their outcomes: repetitions 1
and 2 on each of development layouts 0 and 1. Each repetition is a parallel pair
using the established disjoint CPU groups. Finish both archives before the next
pair. Complete the roster without controller tuning or outcome-dependent
substitutions. Preserve any operational failure; do not silently replace it.

The already observed original heading-recovery pair achieved two verified round
trips without activating the new rule. Report those two separately from these
four prospective repetitions. The study measures repeatability under actual
execution timing on known layouts; it adds no independent maze and is too small
to establish broad reliability. A favourable result still requires new-layout,
matched reactive/non-predictive and memory-causality studies, realistic sensors
and timing, and bounded hardware evidence.

Use the original `HeadingRecoveryRuntime`, fixed JEPA model and correction,
same procedural/appearance seeds, observed mapping and depth, action bank,
footprint/reserves, 4,800-tick budget, 20-mm observed and 40-mm physical quiet
arrival checks. The launcher reuses the original runtime and writers and checks
the settings/source identities against the original per-layout launch record.
Only output ownership and repetition metadata are new. No model or controller
selection is permitted between these four trials.

Launcher: `scripts/run_go2_heading_recovery_repeatability_development.py`
with `--layout-index 0|1 --repetition-index 1|2`.
Root names: `go2_heading_recovery_repeatability_repR_layoutXX_4800_v1_attempt_001`.
CPU group for layout 0 is 0–7,16–23; layout 1 is 8–15,24–31.

After each complete archive and owner exit, independently evaluate goal/return
dwells and contacts, save the navigation summary and resource result, and count
recovery activations and actually executed sequences. Report all four outcomes
including failures and the two trials per layout. Compare timing, trajectories
and stalls when outcomes differ; do not credit an inactive rule or infer causal
benefit from an early-diverging trajectory.

Before preparation, no native owners remained, available RAM was 76.82 GiB and
RecoveryStorage free was 29.38 GiB. Four maximum-length outputs need approximately
14 GiB; two simultaneous compact owners have previously completed with roughly
24 GiB parent peak RSS each. Reassess resources at dispatch.

All four writer configurations were exercised without native execution and
preserved the 92 original source identities and every non-ownership setting.
Only the repetition launcher is added to source metadata. Before the first pair
was dispatched, no native owner remained, RAM available was 76.31 GiB and
RecoveryStorage free was 29.38 GiB. First-pair tool sessions are 11262 (layout 0)
and 12878 (layout 1); owner details follow once startup is confirmed.

Both first-pair owners were confirmed live at startup around 19:48 in the logs:
PID 3438439 for layout 0 and PID 3438474 for layout 1, each with its assigned
CPU group. Actual launch records identify repetition 1 and the original
reference roots, with the original 92 sources plus the repetition launcher.
Outcomes remain pending. After both exits and complete archives, evaluate this
pair and then launch the same commands with `--repetition-index 2`, without
changing the controller or skipping the second pair based on the first outcome.

Repetition 1 completed: both owners exited 0 and both archives are complete.
Independent physics evaluation verified goal and home dwells on both layouts,
with zero disallowed contacts. Layout 0: 3,890 camera pairs, goal frame 2,821,
home 3,888; layout 1: 4,209 pairs, goal 2,725, home 4,207. The new heading
recovery rule activated zero times in either run, so these successes do not
establish its causal benefit. Navigation summaries, physical evaluations,
recovery diagnostics and GNU-time resource records are saved in both roots.
Owner elapsed times were 550.16 and 619.30 seconds, maximum RSS 20,921,368 and
22,529,504 KiB, zero swaps for both. Before repetition 2 dispatch, approximately
24 GiB disk and 76 GiB RAM remained available; sufficient for the final pair.
Repetition 2 was dispatched unchanged on the same two CPU groups. Its outcomes
remain pending.

Repetition 2 owner sessions are 96989 (layout 0, PID 3442853) and 55100
(layout 1, PID 3442882). Both were confirmed live about 4.6 minutes after
dispatch, with actual repetition-2 launch metadata and 93 extra sources.
Neither had a result or failure record at that observation.

## Completed prospective result

All four repetitions completed with owner exit 0: three verified goals and
three verified round trips, zero disallowed contact samples. Layout 0 passed
repetition 1 and exhausted the budget without arrival in repetition 2; layout 1
passed both. The original two successes remain separate from these four trials.
The unchanged controller is not consistently successful even on these two known
mazes. Broad reliability and causal benefit of the new recovery are not proven.

The sole new-recovery activation occurred in the failed layout-0 repetition at
frame 1724. All 35 requested 20-ms intervals matched the evaluated sequence;
actual heading error improved by 0.183908 radians during the 300–700-ms commit
interval. Corrected endpoint error was 6.381 mm. Minimum native base-centre wall
distance was 0.493216 m; this is not articulated-body clearance or a safety
certificate and does not establish a counterfactual benefit over holding.

Between frames 2000 and 4000, the failed run selected 255 left turns, 245 right
turns and one arc. Of 104 full-reserve heading releases, 103 were followed by
an opposite-turn selection within five plans. No panorama remained unfinished.
This identifies repeated recovery/heading direction reversal as a budget sink;
no single-event causal explanation is established. Preserve this failure and
its depth for further diagnosis.

Combined evidence: `go2_heading_recovery_fixed_repeatability_summary_v1_attempt_001/result.json`.
Per-layout comparisons: `go2_heading_recovery_repeatability_comparison_layoutXX_v1_attempt_001`.
Per-run evaluations, summaries, resource records and recovery diagnostics are
saved in all four roots. Repetition-2 owners took 673.73 and 663.29 seconds,
maximum RSS 25,244,928 and 23,821,264 KiB, with zero process swaps.

Next: assess the frozen controller against the matched reactive arm on the four
prepared new development layouts without further tuning to these two mazes.

Both per-layout PNG/SVG comparison figures were generated and visually
inspected. The failed layout-0 repetition eventually left the repeated-turn
region and advanced into the outer passage, but had insufficient remaining
budget to reach the goal. It travelled about 18.0 m; successful repetitions
traversed the complete outbound and home paths.

The saved physics and actual request stream confirm the failed repeat
oscillated during frames 2000–4000: 193.94 of 200 seconds requested pure turns,
with 230 turn-direction reversals when intervening non-turn requests are ignored.
Net horizontal displacement was 23.71 mm, within a 101.75 by 147.48-mm physical
XY extent. Accumulated absolute 10-Hz yaw change was 72.83 rad, versus net
-0.4377 rad. The 4.229-m 10-Hz centre path includes gait sway. These measurements
confirm wasted physical motion without assigning single-event causality.
`repeatability_physical_turn_oscillation_v1.json` and visually inspected
`physical_turn_oscillation_200s.png` / `.svg` are saved in the failed root.
