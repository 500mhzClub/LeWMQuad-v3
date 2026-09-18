# Forecast action utilities versus instantaneous waypoint feedback

Run supervised seed 2026091001 once on exposed shared-recovery maze 0, then
maze 1. Compare with the retained forecast-scoring references. Keep the frozen
model/correction, actual forecast arrays, all clearance/recovery/arrival/stopping
checks, routing memory, camera-view acquisition, sensors, command interface,
CPU groups and timing unchanged. Runs remain sequential; retain every outcome.

Replace the main distance-and-heading utilities (including scan utilities)
with the instantaneous directional derivative of the same waypoint potential,
using current body-frame waypoint and requested command direction only. The
potential is distance plus min(0.35 m, current waypoint distance) times absolute
heading error. View-only scoring uses the original 0.35 m angular scale. Multiply
the derivative by each action's existing command duration: 0.4 s normally,
0.1 s for near-arrival translations. There is no trajectory integration or
learned forecast input to these utilities. Contact cost remains disabled.

Retain original prediction-derived fields for the existing recovery/arrival
gates and store the original main scores as reference evidence. These fields
remain forecasts; only `utility_m` and `position_contact_utility_m` are replaced.
This tests main action ranking with shared predictive safeguards, not a fully
non-predictive controller or full online-rollout-off requirement. The earlier
idea of an action-permutation control was not implemented or run: scrambling
view-turn outcomes would confound this comparison with camera-view acquisition.

Evaluate actual utilities against saved current waypoint/scan error, physical
arrivals, contacts, timing and interventions by retained predictive gates.
Two exposed mazes and one model seed cannot establish broad superiority.
Roots: `go2_instantaneous_waypoint_score_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001`.

Three focused tests passed: numerical directional derivative of the original
potential, scan/arrival command-duration behavior, and utility independence
from altered predictions while preserving forecast-derived gate evidence.

Layout 0 independently verified both physical arrivals (frames 1,788 / 2,507),
with 2,509 poses, zero contacts and maximum pose error 7.938 mm. All 620 main
utilities matched the specified instantaneous objective. Forecast and current
waypoint preferences differed on 177 plans; the final selected action differed
from instantaneous preference on only three. Duration was 251.24 simulated
seconds versus 245.58 reference; 471/620 plans were on time versus 518/584.
Maximum recorded simulator lag was 5,334.989 ms. Owner exited 0 after 5:58.73,
without swapping. This is substantial ranking exposure with a successful
mission, not proof of equivalence or broad reliability.

Layout 1 also independently verified both arrivals (frames 1,184 / 1,875),
with 1,877 poses, zero contacts and maximum pose error 10.160 mm. All 456 main
utilities matched the instantaneous objective. Preferences differed on 114
plans; final selection differed from instantaneous preference on seven.
Duration was 188.04 simulated seconds versus 167.28 reference; 443/456 plans
were on time versus 398/405. Maximum recorded simulator lag was 585.639 ms.
Owner exited 0 after 4:46.25, without swapping. Both native owners are terminal.

## Completed paired results

| Layout | Forecast / instantaneous round trip | Forecast / instantaneous simulated s | Forecast / instantaneous yaw-command reversals |
| --- | --- | --- | --- |
| 0 | Yes / Yes | 245.58 / 251.24 | 69 / 102 |
| 1 | Yes / Yes | 167.28 / 188.04 | 44 / 86 |

All four missions have zero contacts. The comparison verifies 172 unchanged
common runtime sources and shared settings per pair. Both PNG/SVG trajectory
comparisons are complete and inspected. Instantaneous paths were slightly
shorter: 22.953 versus 26.524 m on layout 0, 20.369 versus 20.558 m on layout 1.
Turn-only command durations were longer: 49.08 versus 43.18 s and 50.62 versus
39.14 s. Zero-command durations were 63.32 versus 53.86 s and 16.18 versus
15.30 s; zero command does not establish physical stationarity.

Command-reversal diagnostics count sign changes between successive nonzero
applied yaw commands, omitting zero-yaw intervals. Restricting to translating
arc commands gives forecast/instantaneous reversals 60/83 and 30/71. These
measure command switching, not physical instability. The instantaneous
directional derivative lacks finite-horizon overshoot evaluation, consistent
with increased switching; different trajectories and asynchronous timing mean
this is not an isolated causal explanation. Repeated trials and additional
independent layouts would be needed for a general performance claim.

The instantaneous baseline succeeded despite substantial differences in main
ranking (291 of 1,076 plans across the two runs). Forecast main utilities were
therefore not necessary for these two round trips with predictive safeguards
retained. Forecast ranking had shorter mission times and fewer command reversals
in both pairs; this is limited evidence about ranking behavior, not learning,
JEPA superiority, or full-rollout necessity. The final action overrode the
instantaneous preference on 10 plans; retained gates may still matter even
when they rarely change the selected action.

Aggregate:
`go2_instantaneous_waypoint_score_complete_comparison_v1_attempt_001/result.json`.
Current instantaneous recordings remain full. Redundant successful depth
retirement reclaimed 3,438,759,936 allocated bytes during this experiment;
every failure and all non-depth evidence remain. Storage afterward is about
3.84 GB, below the existing four-GiB launch reserve.

Next: test the remaining prediction-dependent selection stack against a
controller using instantaneous utilities and current observed clearance,
while preserving actual sensor dispatch guards, measured recovery, routing
memory and arrival verification. Audit remaining forecast consumers before
calling this a full online-rollout-off comparison. Do not repeat the completed
ranking or stopping experiments as progress. RGB dependence, realistic
sensing/timing and hardware evidence remain separate unmet requirements.
