# Stronger non-predictive reserve and terminal feedback

Run fixed supervised seed 2026091001 on exposed shared-recovery maze 0, then
maze 1. Retain every outcome, with no tuning between missions. The model and
correction still compute and undergo validity checks, but no learned rollout
value enters action selection. Sensors, routing memory, measured recovery,
command windows, actual dispatch guards and physical arrival checks remain.

The new current-state control retains the predictive planner's 0.03-m action
reserve: current observed clearance must exceed 0.48 m for translations and
turns, while hold retains the 0.45-m nominal disk rule. Unlike predictive
reserve recovery, this has no forecast-based escape from a reserve deficit.

Within the existing terminal approach (0.10 m, short translation pulses), prefer
eligible translation that improves instantaneous distance utility over the
heading-preferred action. If no positive improvement exists, retain heading
feedback. No future pose, integrated command endpoint or predicted arrival is
used. This is a reasonable non-predictive counterpart to positional terminal
priority; it is not mathematically identical to a forecast-entry gate.

Compare with full predictive selection, instantaneous scoring with predictive
checks, and the original nominal-current-clearance off control. Both changes
address identified baseline weaknesses; their separate causal effects are not
isolated by this combined test. Two exposed mazes and one seed remain limited
development evidence, not general or hardware reliability.

Roots: `go2_current_reserve_terminal_feedback_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001`.

Seven focused tests passed, covering the current reserve for all moving actions,
terminal progress versus heading fallback, turn-only views, no reserve bypass,
and the inherited lack of forecast/selection-chain access.

Layout 0 completed with a budget failure: no arrivals, 4,805 poses, zero
contacts, maximum pose error 3.297 mm, 1,182/1,200 plans on time and 480.86
simulated seconds. Actual selections matched the non-predictive controller.
There were 1,106 consecutive hold plans from frame 380 through frame 4,800
because observed clearance was below the 0.48-m action reserve. At the first
and last blocked plans, stored/native wall clearance was respectively
0.478977/0.489985 m and 0.470170/0.482106 m. The nominal 0.45-m disk remained
clear, explaining why the inherited nominal-clearance diagnostic reports zero
blocked plans. No actual translation-veto view recovery occurred.

This exposes the current-state rule's inability to move out of its reserve
deficit. It does not isolate learned forecasting from geometric lookahead or
establish a safe alternative escape. The entire failure recording remains.
The four-controller layout-0 comparison is complete with 172 identical common
runtime sources. Layout 1 launched next with unchanged controller parameters.

Layout 1 also exhausted its budget without arrival: 4,805 poses, zero contacts,
maximum pose error 4.215 mm, 1,183/1,200 plans on time and 480.86 simulated
seconds. All 1,200 selections matched the specified non-predictive controller.
There were 1,101 reserve-blocked hold plans, starting at frame 384. Unlike maze
0, the blockage was not uninterrupted: the final translation request occurred
at 423.38 s. First/last blocked stored/native clearances were
0.469225/0.482700 m and 0.467987/0.484124 m. Nominal clearance remained clear;
there were no actual translation-veto view recoveries. Both native owners
closed with exit 0 and no swaps, after 10:45.74 and 11:29.18 including saving.
Clean process exits do not turn either budget failure into navigation success.

| Controller | Verified round trips | Contacts |
| --- | --- | --- |
| Forecast ranking and predictive checks | 2 / 2 | 0 |
| Instantaneous ranking and predictive checks | 2 / 2 | 0 |
| Nominal current-clearance feedback | 1 / 2 | 0 |
| Current reserve and terminal position feedback | 0 / 2 | 0 |

These eight outcomes reuse two exposed mazes and one training seed. The
reserve-and-terminal variant changes two rules together; neither run reached
the goal, so it does not establish the effectiveness of its terminal rule.
The gap is consistent with useful predictive feasibility/recovery but does
not isolate future prediction from the particular current-state rules. The
separate fitted-motion control also completed both mazes; learned RGB or JEPA
benefit remains unproven. Stop adding exposed-maze controller variants here
and move to the matched neural RGB contribution study. Preserve both complete
failure recordings and all previous comparison outcomes.

Both four-controller comparisons and inspected trajectory PNG/SVG figures are
complete. Aggregate: `go2_current_reserve_terminal_feedback_complete_comparison_v1_attempt_001/result.json`.
