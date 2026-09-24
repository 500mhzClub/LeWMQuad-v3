# Prospective maze layouts for the dense world model

Four same-family maze layouts were selected before the new dense predictor's
fit completed and before any navigation outcomes on them. They have distinct
abstract topologies and grid embeddings, disjoint from the explicit 105-layout
development registry. This is not a claim about every possible historical
environment or a sealed final benchmark. These layouts have not been used for
training or controller tuning. Prospective native execution began on September
22 under the fixed protocol below.

Construction seed: 2026091803. The generator accepted candidates 0, 2, 3 and 4
using the existing structural criteria. The registry contains 104 distinct
abstract topology groups. All four new layouts are connected sixteen-cell
trees with fifteen passages; cell pitch is 1.3 m. Geometry and routes remain
simulator/evaluator inputs. The navigation controller receives the public goal
and its own observations, not the maze map or shortest route.

| Layout | Shortest outbound route | Native execution |
|---|---:|---|
| 00 | 9.1 m | All four complete: command-history round trip verified; other three exhausted budget |
| 01 | 14.3 m | All four complete: every arm exhausted the budget without an arrival |
| 02 | 10.4 m | All four complete: command-history and action-conditioned round trips verified; reactive and no-future-action failed |
| 03 | 11.7 m | All four complete: command-history round trip verified; other three exhausted budget |

![Prospective maze layouts](go2_dense_world_model_maze_layouts_2026-09-18.png)

## Completed sixteen-run comparison, September 23

| Layout | Arm | Goal / return arrivals | Disallowed contacts / pipeline faults | Simulated duration | Outcome |
|---|---|---:|---:|---:|---|
| 00 | Action-conditioned | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted |
| 00 | No future action | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; all commands zero |
| 00 | Command history | 1 / 1, physically verified | 0 / 0 | 175.22 s | Completed round trip |
| 00 | Reactive feedback | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; near-goal stall |
| 01 | No future action | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; all commands zero |
| 01 | Command history | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; terminal clearance-filter hold |
| 01 | Reactive feedback | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; terminal low-visual-support hold |
| 01 | Action-conditioned | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; 404.42-s terminal zero-command interval |
| 02 | Command history | 1 / 1, physically verified | 0 / 0 | 330.92 s | Completed round trip; physical backtracking verified |
| 02 | Reactive feedback | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; 448.02-s terminal zero request under current-map clearance rejection |
| 02 | Action-conditioned | 1 / 1, physically verified | 0 / 0 | 359.62 s | Completed round trip; physical backtracking verified |
| 02 | No future action | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; all 1198 plans selected hold |
| 03 | Reactive feedback | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; 1164 hold plans; 465.22-s longest zero-request interval |
| 03 | Action-conditioned | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; 971 hold plans; 379.62-s longest zero-request interval |
| 03 | No future action | 0 / 0 | 0 / 0 | 480.32 s | Budget exhausted; all 1198 plans selected hold and all requests zero |
| 03 | Command history | 1 / 1, physically verified | 0 / 0 | 234.02 s | Completed round trip; all nine return edges reverse outbound edges |

All sixteen assignments have terminal native results and completed physical readers.
Command history completed layouts 00, 02 and 03 (3/4); action-conditioned
navigation completed layout 02 (1/4). Reactive feedback and no future action
each completed 0/4 round trips. The other twelve assignments failed navigation.
All sixteen had zero disallowed contacts and pipeline faults. These four
same-family layouts, with one execution per arm and layout, do not establish
reliable navigation or a JEPA advantage. The blind arm remained stationary on
every layout and is not a competitive moving-policy comparator.

The complete per-assignment summary is
`go2_dense_world_model_maze_cohort_complete_result_2026-09-23.json`.
The original failures remain included. Historical progress notes below describe
earlier partial states; this completed table supersedes their running totals.

### Completed same-window motion comparison

Command-history translation RMSE is lower than recorded neural RMSE on all
twelve trajectories with translation windows, by ratios of 4.63--15.03 in
neural/command-history error. Each row compares forecasts against the same
executed 700-ms endpoints within that trajectory. Different controllers execute
different trajectories, and their windows overlap; these are not twelve
independent layouts or alternative-policy outcomes. Reactive control does not
use the recorded neural forecast to select actions. The four stationary
no-future-action runs have no translation windows and each selected 1198/1198
holds. All sixteen records, including their yaw errors, are retained in
`go2_dense_world_model_maze_motion_comparison_2026-09-23.json`.

| Layout | Executed controller | Translation windows | Neural XY RMSE, mm | Command-history XY RMSE, mm | Hold plans / all plans |
|---|---|---:|---:|---:|---:|
| 00 | Action-conditioned | 116 | 68.48 | 10.55 | 504/1198 |
| 00 | Command history | 231 | 97.78 | 8.46 | 30/417 |
| 00 | Reactive feedback | 129 | 96.17 | 9.83 | 675/1198 |
| 01 | Action-conditioned | 28 | 65.20 | 6.80 | 1033/1198 |
| 01 | Command history | 57 | 93.01 | 7.75 | 544/1198 |
| 01 | Reactive feedback | 221 | 96.72 | 9.11 | 717/1198 |
| 02 | Action-conditioned | 346 | 73.70 | 12.96 | 117/888 |
| 02 | Command history | 315 | 101.91 | 6.78 | 359/808 |
| 02 | Reactive feedback | 62 | 93.81 | 9.86 | 1122/1198 |
| 03 | Action-conditioned | 79 | 69.09 | 14.93 | 971/1198 |
| 03 | Command history | 303 | 102.23 | 6.88 | 79/575 |
| 03 | Reactive feedback | 5 | 61.85 | 4.27 | 1164/1198 |

This strengthens the descriptive evidence of a recurring motion-decoding
weakness, including on successful trajectories. It does not separate predictor
error from readout error, prove that the encoder lacks motion information, or
assign the clearance-filter failures solely to learned predictions. The fixed
maze-view readout experiment addresses one candidate explanation while keeping
encoder and predictor unchanged.

### Final assignment details

Assignment 16 completed in 234.02 simulated seconds and 2049.00 wall seconds
before persistence. The coordinator recorded successful native-owner and reader
completion and then exited. Both arrivals passed the 4-cm, one-second physical
quiet-arrival checks: maximum dwell distances were 14.82 mm at the goal and
7.46 mm at home. All 2341 camera poses registered; tracking median/max was
4.60/9.79 mm. There were 575 plans: 179 turns, 317 translations and 79 holds;
9540 of 11701 requests were nonzero. Longest zero-request interval was 25.2 s.

The saved backtracking reader confirms nine outbound and nine return directed
corridor edges, with all nine return edges reversing observed outbound edges
and zero invalid graph transitions. Local route-turn memory was active in
10/207 return plans and no outbound plans; this is not causal memory evidence.
On 561 overlapping 700-ms executed windows, neural XY/yaw error was
76.11 mm / 7.26 degrees versus command history 6.50 mm / 0.75 degrees. The
303 translation windows gave 102.23 versus 6.88 mm XY error. Those errors
describe this executed trajectory, not alternative-policy navigation outcomes.

Assignment 15 completed in 480.32 simulated seconds and 2606.14 wall seconds
before persistence. All 4804 camera poses registered. The owner and physical
reader completed before the coordinator launched assignment 16. Every one of
1198 plans selected hold, with all 24016 requests zero, no arrivals, zero
contacts and zero pipeline faults. Tracking median/max was 0.108/0.339 mm.
All 1197 matched 700-ms windows are holds: neural XY/yaw error is 8.38 mm /
3.94 degrees versus command history 3.54 mm / 0.31 degrees. The blind arm is
complete at 0/4, with stationary behaviour on all four layouts; its failure
does not provide a competitive moving-policy baseline. Action is complete at
1/4 and reactive at 0/4; command history remains 2/3 with its last trial live.

Assignment 14 persisted its complete 4804-frame native result after the original
coordinator disappeared. The original owner's exit code was not recovered;
its process was absent and the terminal result and complete recording were
present. The existing physical reader was subsequently run and exited zero.
It confirms no arrivals, zero contacts/faults, 148 turns, 79 translations and
971 holds. The longest zero-request interval is 379.62 s. Tracking median/max
is 4.70/6.36 mm. On 1192 overlapping 700-ms windows, learned XY/yaw error is
21.02 mm / 3.93 degrees versus command history 5.83 mm / 0.53 degrees; on the
79 translation windows, XY error is 69.09 versus 14.93 mm. The action arm is
now complete at 1/4 successes, reactive at 0/4. Neither pending baseline may be
assumed successful. The unchanged coordinator was relaunched, preserved all
fourteen completed assignments and started assignment 15; no simulation was
repeated. Its continuation log is `.generated/go2_dense_cohort_continuation_2026-09-23.log`.

The layout-03 action terminal-hold diagnosis covers 949 plans at frames
1008--4800, after the longest zero-request interval starts. All 949 admit every
non-hold candidate under the nominal 0.45-m footprint, but reject every one
under the full reserve/recovery filter. All candidates share a pre-dispatch
prefix below the moving-action 0.48-m requirement. Hold remains clear in all
949 plans, with minimum predicted path clearance 0.454702--0.467439 m and no
additional reserve. The stopping projection changes none of these selections.
Thus this terminal failure directly reflects a reserve-filter deadlock; it is
not evidence of all moving paths violating the nominal footprint, nor does it
establish that removing the reserve would yield safe or successful navigation.
Readout errors and the preceding trajectory remain possible contributors.
`terminal_reserve_hold_diagnostic_v1.json` preserves every examined plan's
flags, the final candidate clearances and the planning-file identity. The
active cohort remains unchanged.

Layout-03 reactive feedback completed 4804 registered camera frames and 1198
plans in 480.32 simulated seconds (4232.02 wall seconds before persistence).
There were no arrivals or disallowed contacts. Plans comprised 29 turns, five
right arcs and 1164 holds; the longest zero-request interval was 465.22 seconds.
Median/maximum tracking error was 1.05/1.66 mm. Reactive feedback is now 0/4
across the full cohort; the other arms still have only three completed layouts.
The coordinator recorded assignment 13 complete before launching assignment 14.

The completed layout-03 reactive diagnosis found that every one of its 1164
hold plans failed the current-map nominal-clearance predicate. Stored clearance
was 0.420632--0.443580 m, below the unchanged 0.45-m threshold. These plans
span frames 148--4800 and all request an additional view, yet the selector
zeros the desired command and marks every candidate ineligible. The last
requested viewing correction is 0.987708 radians. This repeats the controller/map
deadlock observed on layout 02; forecasts are not used by this reactive selector.
It limits interpretation of the reactive failure as evidence for predictive
planning. It does not establish a physical collision or prove that relaxing the
rule would succeed. Per-hold evidence and input/source identities are saved in
`terminal_hold_current_clearance_diagnostic_v1.json` under the layout-03 reactive
root. No active-cohort setting changed.

Layout-02 no-future-action navigation completed all 4804 frames, without an
arrival, disallowed contact or pipeline fault. Every one of its 1198 plans
selected hold, and all 24016 requests were zero. The physical reader found
median/maximum tracking error 0.104/0.352 mm. Median model inference was
1377.56 ms. All 1197 matched 700-ms windows are holds: neural XY/yaw RMSE
13.53 mm / 3.32 degrees versus command history 3.54 mm / 0.31 degrees.
This repeats the action-blind tie/hold behaviour on layouts 00 and 01; it is
not a meaningful moving-policy baseline by itself. The owner and reader
completed before the coordinator launched layout-03 reactive feedback.

Layout-02 action-conditioned navigation completed in 359.62 simulated seconds
and 3158.15 wall seconds before persistence, with 3597 camera frames, 888 model
calls and 17981 policy steps. Goal frame 2400 and home frame 3596 passed the
physical 4-cm radius and one-second quiet-dwell checks. Maximum native distances
during those dwells were 8.66/18.93 mm; final distances were 5.39/16.82 mm.
There were zero disallowed contacts and zero pipeline faults. Median/maximum
tracking position error was 8.05/10.62 mm. Plans comprised 415 turns, 356
translations and 117 holds. Median/p95 model inference was 2700.24/2758.00 ms.

The supplemental physical reader found thirteen unique outbound directed
corridor edges and eight return edges. All eight return edges reversed observed
outbound edges, with zero invalid graph transitions. Local route-turn memory
was active in 11/294 return plans and no outbound plans. Return routing used
218 frontier, 70 goal-cell and six visual-support plans. This demonstrates
physical backtracking with the shared map; it does not isolate a memory benefit
or demonstrate internal JEPA memory.

On 858 matched overlapping 700-ms windows, action-model XY/yaw RMSE was
52.72 mm / 7.58 degrees, versus command history 10.78 mm / 1.06 degrees.
On the 346 ordinary translation windows, XY RMSE was 73.70 versus 12.96 mm.
Thus the successful navigation does not erase the motion-readout limitation.
Command history also succeeded on this layout, 28.70 simulated seconds faster.
Reactive feedback and the blind control failed. This first
prospective dense-model round trip is positive execution evidence, without
reliable navigation or JEPA superiority established. The deployed mixed head
was unchanged; the newly evaluated horizon-trained heads were not used.

Full evidence is in
`go2_dense_world_model_maze_layout02_action_mixed_data_v1_attempt_001`, including
`dense_navigation_readout.json`, `dense_backtracking_readout.json` and
`physical_return_corridor_readout_v1.json`. The coordinator started assignment
12 only after the completed owner and physical reader.

Layout-02 reactive feedback completed in 480.32 simulated seconds and 4240.20
wall seconds before persistence. Its physical reader confirmed no arrivals,
zero disallowed contact samples and no pipeline faults. All 4804 camera poses
registered; median/maximum tracking error was 2.11/2.30 mm. It selected 14
turns, 62 translations and 1122 holds in 1198 plans. The terminal zero-request
interval lasted 448.02 simulated seconds, and 1122 plans recorded
`ADDITIONAL_VIEW_REQUIRED`.

All 1122 hold selections also recorded `current_nominal_disk_clear=false`.
Their current stored-map clearances ranged from 0.436131 to 0.449905 m, below
the nominal 0.45-m rule. At the final plan, the controller still requested a
viewing correction of 1.215 radians, but the clearance predicate produced a
zero desired command and no eligible primitive. This follows the existing
`select_reactive` predicate in `lewm/continuous_reactive_selection_development.py`.
The diagnosis is retained in `terminal_hold_current_clearance_diagnostic_v1.json`,
with the planning-file identity and every hold row. It concerns the observed
map and controller rule, not a true physical footprint violation; no contacts
were recorded. No gate, map, predictor or active-cohort setting was changed.

On 1195 overlapping matched 700-ms windows, unused learned forecasts had
XY/yaw RMSE 36.13 mm / 2.46 degrees, versus command history's 4.08 mm / 0.34
degrees. The population contains 1121 hold windows. On the 62 translation
windows, XY RMSE was 93.81 versus 9.86 mm. Forecasts were computed for workload
control but were not used for reactive command selection, so this navigation
failure cannot be attributed to those forecast errors. Together with the
preceding command-history success, it records a controller/map interaction
that affects this comparison alongside learned-readout limitations.

The completed failure and physical reader remain under
`go2_dense_world_model_maze_layout02_reactive_feedback_mixed_data_v1_attempt_001`.
The coordinator then launched layout-02 action-conditioned navigation (owner PID
143220 at launch), with the fixed assignment order and model identities. Its
subsequent successful outcome is reported above; the cohort is now on the blind
assignment. Do not pool unequal partial arms.

Layout-02 command history completed in 330.92 simulated seconds and 2892.60
wall seconds before persistence, with 3310 camera frames, 808 model calls and
16546 policy steps. Both one-second goal/home dwells passed the physical
4-cm radius and quiet-motion checks, with maximum native distances 16.02 mm
and 24.92 mm. There were zero disallowed contacts and no pipeline faults.
Median/maximum position tracking error was 2.64/9.00 mm. Plans comprised 121
turns, 328 translations and 359 holds; the longest zero-request interval was
137.60 seconds early in the outbound leg. Thus substantial stationary time
did not prevent completion on this layout.

The supplemental physical reader found ten unique outbound directed corridor
edges and eight return edges; all eight return edges reversed previously
traversed outbound edges, with zero invalid graph transitions. All 154 return
plans used observed-floor goal routing. Local route-turn memory activated on
zero plans in either leg. This is physical backtracking with the shared
explicit map, not an isolated memory benefit or evidence of internal JEPA
memory. Native state and the maze graph were evaluator-only inputs.

On 804 overlapping matched 700-ms windows, command-history XY/yaw RMSE was
5.67 mm / 0.59 degrees versus unused learned forecasts at 65.55 mm / 5.71
degrees. On 315 ordinary translation windows, XY RMSE was 6.78 versus
101.91 mm; thirteen translation-pulse windows are reported separately in the
saved reader. These forecasts were evaluated along the command-history
trajectory and do not establish how an alternative policy would navigate it.

Authoritative `result.json`, `dense_navigation_readout.json`,
`dense_backtracking_readout.json` and `physical_return_corridor_readout_v1.json`
are retained under
`go2_dense_world_model_maze_layout02_command_history_mixed_data_v1_attempt_001`.
The coordinator launched layout-02 reactive feedback (owner PID 129901 at
launch) with unchanged models and settings. At that point command history was
2/3, while each other arm had only two completed layouts. The later reactive
outcome is recorded above. No JEPA navigation advantage is demonstrated.

Layout-01 action-conditioned navigation completed in 4258.98 wall seconds before
persistence, with 4804 camera frames and 1198 model calls. The physical reader
confirmed no arrivals, zero disallowed contact samples and zero pipeline faults.
Median/maximum tracking position error was 2.13/2.85 mm. Plans comprised 137
turns, 28 translations and 1033 holds. The final 404.42 simulated seconds issued
zero requests. There were 748 `NO_CLEAR_CANDIDATE_ZERO_REQUESTED` plans; 986
hold selections had a higher recorded non-hold utility, so the failure is not
explained simply by a main-utility preference to remain stationary. These are
recorded gate decisions, not proof of a safe counterfactual action.

On 1195 overlapping matched 700-ms windows, learned XY/yaw RMSE was 13.22 mm /
3.26 degrees versus command history's 4.15 mm / 0.45 degrees. The population is
dominated by 1032 hold windows; on 28 translation windows XY RMSE was 65.20
versus 6.80 mm. The lower aggregate error than maze 00 is therefore not evidence
of improved navigation. The fixed learned planner has now failed both completed
prospective layouts; every comparator also failed layout 01. This remains a
partial four-layout comparison, with no demonstrated JEPA navigation advantage.

Results and failure recordings remain under
`go2_dense_world_model_maze_layout01_action_mixed_data_v1_attempt_001`.
The coordinator automatically launched layout-02 command history (native owner
PID 120773 at launch), with all settings and model identities unchanged.

The saved-plan common-prefix diagnosis further limits interpretation of this
failure. All six candidates shared identical first-three-segment clearances in
all 1198 plans. The terminal 404.42-s zero-request interval contains 1012 plans,
all selecting hold. In 748, every full path failed nominal-footprint clearance;
in 632, every candidate already failed within the shared 300-ms prefix. The
prefix minima in those 632 plans ranged from 0.43916 to 0.44999 m, below the
0.45-m nominal threshold. The final prefix minimum was 0.44168 m, with right
turn preferred before filtering. The first such prefix rejection was frame 868,
after the terminal hold interval had already begun; it does not explain the
initial onset by itself.

For these recorded prefixes, changing only candidate-dependent later segments
cannot satisfy either the full-path rule or the existing reserve-recovery
rules, which require the prefix to clear the nominal footprint. Improving the
learned prefix, avoiding the earlier state, or addressing the observed-map
interaction could still matter. This is not proof of the underlying map/pose/
forecast cause, not a safe gate-relaxation proposal, and not a counterfactual
navigation result. No sensor replay, fit or runtime change was performed.
`common_prefix_clearance_diagnostic_v1.json` retains the per-plan evidence in
the completed layout-01 action root. The relevant path/recovery predicates are
in `lewm/memory_forecast_clearance_development.py` and
`lewm/clearance_turn_recovery_development.py`.

A conservative distance bound further separates the current map state from
forecasted displacement. The first checked segment starts at the current
position. If its recorded distance to observed obstacle cells is d and its
predicted body-relative endpoint has norm L, current point clearance lies
between d and d+L (horizontal projection of a rigid rotation cannot increase
L). Applying this bound with 1e-9 m upper-bound padding proves current nominal
footprint deficit in 395 of the 1012 terminal hold plans. Their upper bounds
range from 0.44410 to 0.449999885 m; the first proved instance is frame 2060.
At the last plan, current clearance is bounded between 0.44168 and 0.44483 m.

Thus, for those 395 recorded position/map states, even changing the entire
forecast while retaining its current origin cannot make the nominal full-path
clearance rule pass. Earlier decisions, mapping or pose changes could prevent
reaching those states. The bound does not identify their cause or prove a true
physical footprint violation: native execution recorded zero disallowed
contacts. Rows not proved by this sufficient bound may still have deficits.
`current_clearance_bound_diagnostic_v1.json` retains the formula and every
per-plan bound. No source, threshold, active cohort or sensor stream changed.

Layout-01 reactive feedback completed in 4237.21 wall seconds before persistence,
with 4804 camera frames and 1198 model calls. Its physical reader recorded no
arrivals, zero disallowed contact samples and zero pipeline faults. Median/maximum
tracking position error was 5.10/7.49 mm. Plans comprised 259 turns, 222
translations and 717 holds. The terminal zero-request interval lasted 200.42
simulated seconds; 859 plans requested a better-supported visual view under
`LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW`. This is a completed navigation
failure, despite successful process execution.

On 1177 overlapping matched 700-ms executed windows, unused learned forecasts
had XY/yaw RMSE 43.81 mm / 6.61 degrees versus command history's 5.31 mm / 0.61
degrees. On 221 translation windows, XY RMSE was 96.72 versus 9.11 mm. Results
and the physical reader are retained under
`go2_dense_world_model_maze_layout01_reactive_feedback_mixed_data_v1_attempt_001`.
The coordinator automatically launched the layout-01 action arm (native owner
PID 106028 at launch). No experimental setting or model was changed.

Layout-01 command history completed in 4187.66 wall seconds before persistence,
with 4804 camera frames, 1198 model calls and 24016 policy steps. The physical
reader confirmed no arrivals, zero disallowed contact samples and zero pipeline
faults. Median/maximum tracking position error was 18.49/22.85 mm, and median
model inference was 2707.87 ms. Its plans comprised 597 turns, 57 translations
and 544 holds. The longest consecutive zero request lasted 212.82 simulated
seconds, from timestamp 269.0 to 481.82 s, through mission termination.

The saved-plan diagnosis localises this terminal stall to the clearance filter:
all 532 plans examined from frame 2676 onward preferred right turn before the
memory filter, but only hold passed the full-reserve clearance check. In 500 of
these plans all non-hold candidates passed the nominal-footprint check. At the
final plan, right-turn clearance was 0.47748 m against the 0.480 m requirement
(0.45 m nominal footprint plus 0.03 m reserve). Its higher utility therefore
did not produce a nonzero command. This records the filter behaviour; it does
not establish that lowering the reserve would be safe or complete navigation.
No gate or controller parameter was changed. Details are retained in
`terminal_hold_clearance_diagnostic_v1.json` under the completed root.

On 1197 overlapping matched 700-ms windows, unused learned forecasts had
XY/yaw RMSE 22.48 mm / 6.73 degrees versus command history's 4.33 mm / 0.61
degrees. On 57 translation windows, XY RMSE was 93.01 versus 7.75 mm.
The command-history baseline's maze-00 success therefore did not carry over to
this second layout, despite its more accurate motion forecasts on its executed
windows. These data distinguish a shared controller limitation from a general
claim that better motion prediction alone solves navigation.

Authoritative `result.json` and `dense_navigation_readout.json` are retained in
`go2_dense_world_model_maze_layout01_command_history_mixed_data_v1_attempt_001`.
The coordinator advanced to
`go2_dense_world_model_maze_layout01_reactive_feedback_mixed_data_v1_attempt_001`
(native owner PID 83308 at launch). The fixed experimental settings are unchanged.

Layout-01 no-future-action completed in 2552.77 wall seconds before persistence,
with 4804 camera frames, 1198 model calls and 24016 zero-command requests.
All 1198 plans selected hold under `INITIAL_PANORAMA_REQUIRES_VIEW`, with no
stopping-projection changes. Its physical reader confirmed budget exhaustion,
no arrivals, zero disallowed contacts and zero pipeline faults. Median/maximum
position tracking error was 0.103/0.402 mm; median model inference was 1378.49 ms.
On 1197 overlapping hold-only matched 700-ms windows, learned XY/yaw RMSE was
4.70 mm / 4.10 degrees versus command-history 3.54 mm / 0.31 degrees. These
stationary-window metrics do not establish performance along moving trajectories.
The completed failure is retained under
`go2_dense_world_model_maze_layout01_no_future_action_mixed_data_v1_attempt_001`
in the workspace development artifact root, including `result.json` and
`dense_navigation_readout.json`. The coordinator advanced automatically to
`go2_dense_world_model_maze_layout01_command_history_mixed_data_v1_attempt_001`
(native owner PID 70760 at launch).

The layout-00 no-future-action recording contains 4804 camera frames, 1198 model calls,
1198 hold plans, and 24016 zero-command requests. Every plan remained in
`INITIAL_PANORAMA_REQUIRES_VIEW`; no stopping-projection change occurred.
All six forecasts were exactly identical in each of the 1198 saved model calls,
and all three scan utilities tied on all 1198 plans. This directly confirms the
information-removal/tie-breaking mechanism described below. Zero collisions
in this stationary control are not evidence of successful navigation.

No-future-action execution took 2559.79 wall seconds before persistence;
median model inference was 1378.67 ms. Its identical candidate forecasts permit
one predictor evaluation per horizon, explaining a computational difference
from the action-conditioned arm; these remain untimed simulation runs, not a
matched wall-time or real-time claim. Median/maximum position tracking error
was 0.135/0.374 mm. On its 1197 matched hold-only windows, learned XY/yaw RMSE
was 7.38 mm / 3.86 degrees versus command-history 3.54 mm / 0.31 degrees.
Those stationary-window errors are not directly comparable to errors along
the moving action-conditioned trajectory.

The authoritative no-future-action records are `result.json`,
`dense_navigation_readout.json`, `planning.json` and `dense_model_calls.json`
under `.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_no_future_action_mixed_data_v1_attempt_001/`.
Its complete failure recording is retained. Assignment 3 subsequently completed
in `go2_dense_world_model_maze_layout00_command_history_mixed_data_v1_attempt_001`
(native owner PID 43136 at launch, now exited; coordinator session 64661).

Command history completed its round trip in 175.22 simulated seconds
(1486.89 wall seconds before persistence), with 1753 camera frames, 417 model
calls, zero disallowed contact samples and zero pipeline faults. The physical
reader verified outbound arrival at frame 1196 and return at frame 1752.
Native final distances were 21.31 mm and 15.97 mm respectively; both complete
one-second dwells passed the 40-mm physical radius, measured quiet-motion and
zero-request checks. Median/maximum position tracking error was 3.42/7.15 mm.
There were 30 hold plans, 124 turns and 263 translations, with a longest
consecutive zero-request interval of 1.90 s.

The supplemental physical backtracking reader found seven unique directed
corridor edges outbound and seven on return; all seven return edges reversed
observed outbound edges, with zero invalid graph transitions. Local route-turn
memory was active on 12 of 136 return plans (zero of 281 outbound plans), and
135 return plans used an observed-floor route to the goal cell. The saved
`dense_backtracking_readout.json` and `physical_return_corridor_readout_v1.json`
establish physical backtracking and recorded memory use, not a causal memory
advantage or a learned internal map.

On 405 overlapping matched 700-ms executed windows, learned XY/yaw RMSE was
75.15 mm / 7.29 degrees, versus 8.75 mm / 0.70 degrees for command history.
The learned forecasts were computed and retained, but did not supply this
arm's motion forecasts for selection. These within-trajectory comparisons
support the motion-forecast diagnosis; windows are not independent trials.

The selected-turn breakdown on this successful trajectory is less directionally
pathological than on the action-conditioned trajectory below. On 69 matched
left-turn windows, actual/learned mean yaw was +16.540/+9.513 degrees, with
learned/command-history RMSE 8.082/0.748 degrees and one learned opposite-sign
prediction. On 51 right-turn windows, actual/learned means were -16.918/-7.415
degrees, RMSE 10.317/0.594 degrees, and four learned opposite-sign predictions.
Every window exceeded one degree of actual rotation. These use the same saved
700-ms evaluator errors and selected neural forecasts; reconstructing actual
yaw reproduces saved command-history errors to 1e-10 radians. The much lower
right-turn sign-error fraction here (4/51 versus 180/300 on the action arm)
argues against a universal sign inversion. Dependence on observations and
motion history remains plausible, but the trajectories and committed prefixes
differ, so this comparison does not isolate a causal mechanism.

Grouping these same selected-turn windows by the last applied yaw command
before each forecast narrows the association further. A previous command with
absolute yaw below 0.01 rad/s is labelled near-zero; other commands are grouped
by whether their sign agrees with the selected turn. On the action-conditioned
trajectory, 171 of the 180 wrong-sign right-turn forecasts occurred after a
leftward last applied command:

| Trajectory / selected right turn | Last applied yaw | Windows | Mean actual / learned yaw | Learned / command-history RMSE | Learned wrong sign |
|---|---|---:|---:|---:|---:|
| Action-conditioned | Leftward | 197 | -8.303 / +7.071 deg | 16.097 / 0.708 deg | 171 |
| Action-conditioned | Rightward | 99 | -17.678 / -8.855 deg | 9.957 / 0.558 deg | 7 |
| Action-conditioned | Near-zero | 4 | -8.853 / +0.220 deg | 9.173 / 0.620 deg | 2 |
| Command history | Leftward | 5 | -6.855 / +0.554 deg | 10.711 / 1.146 deg | 1 |
| Command history | Rightward | 45 | -18.222 / -8.575 deg | 10.191 / 0.503 deg | 2 |
| Command history | Near-zero | 1 | -8.515 / +4.949 deg | 13.464 / 0.113 deg | 1 |

For selected left turns following a rightward command on the action trajectory,
there were 12 wrong signs in 188 windows, with actual/learned mean yaw
+8.054/+5.729 degrees and learned/command RMSE 3.963/0.994 degrees. Thus the
association is directionally asymmetric. The grouping uses saved
`dense_model_calls.json:last_applied_command` joined to planning records and
the existing evaluator's matched windows; it is not a reconstruction of the
whole preceding motion state. These overlapping, policy-selected observations
motivate a later controlled left-to-right transition diagnosis, but do not
separate command history, heading, scene, encoder, predictor or readout effects.
No new inference, training or runtime change was used for this grouping.

The saved final-stage horizon-predictor training inputs provide limited but
nonzero reversal coverage. Counting sign changes among non-negligible yaw
commands (absolute yaw above 0.01 rad/s), restricted to commands before each
sample's target horizon, finds 407 reversal-bearing samples among 29008 input
samples. They span 80 distinct source/trial/departure-frame contexts: 351
sample-horizon entries come from the switch source and 56 from short pulses.
At 700 ms, 40 of 3326 samples contain left-to-right reversal and 40 contain
right-to-left reversal. The saved 1760-by-16 training schedule actually presents
184 left-to-right and 181 right-to-left samples among 28160 presentations
(365 total, 1.30%); at 700 ms the counts are 39 and 37 out of 3520 presentations.
This is a count of the final horizon-extension stage, not all predecessor
training exposure or unique physical trials. Balanced but sparse directions
motivate controlled transition-coverage experiments; they do not explain the
directional asymmetry by themselves. Counts use the completed
`go2_horizon_dense_predictor_v1_attempt_001/samples.json` and `schedule.json`;
no training inputs or weights were changed.

A complementary count includes the command-history/future boundary, which
the within-future reversal count above excludes. The final past applied yaw
command is recovered from `control[-1][-1][1]` using the training normalization
mean 0.03156160613672535 and standard deviation 0.32782628369047895; its sign
is compared with the last applied future yaw command before the target. Both
use the same 0.01-rad/s near-zero threshold. Across all 29008 samples there
are 250 left-to-right and 247 right-to-left boundary-to-target pairs, each
direction spanning 50 distinct source/trial/departure-frame contexts. The
actual schedule presents 224 and 220 respectively out of 28160 presentations.
At 700 ms, the input pool contains 42 left-to-right and 41 right-to-left pairs;
the schedule presents 40 and 38 out of 3520 presentations (78 total, 2.22%).
Of those 78, 75 come from the switch population and three from short pulses.
Counting any sign reversal from the last past command through the future,
rather than only comparing its endpoints, gives 94 of 3520 presentations at
700 ms. These metadata counts confirm sparse, roughly direction-balanced
transition coverage in the final training stage. They do not establish matching
physical yaw, visual context, turn timing or predecessor-stage coverage, and
therefore do not establish that the navigation failures are out of distribution.
Only saved sample/schedule metadata and normalization statistics were read;
the ongoing comparison and all model weights remain unchanged.

This is a completed prospective same-family baseline success on a layout where
the action-conditioned planner failed under the shared mission budget. It
shows that the shared sensing, mapping and navigation system can complete
this particular maze using simpler motion forecasts. It does not establish a
JEPA benefit, isolate encoder versus predictor versus motion-readout causes,
or establish performance over the remaining layouts. The authoritative
`result.json` and `dense_navigation_readout.json` are retained in the
command-history root above. Assignment 4 subsequently completed in
`go2_dense_world_model_maze_layout00_reactive_feedback_mixed_data_v1_attempt_001`
(native owner PID 48201 at launch, now exited).

Reactive feedback exhausted the 480.32-s simulated mission budget with zero
arrivals, zero disallowed contact samples and zero pipeline faults. It acquired
4804 frames and made 1198 model calls in 4217.84 wall seconds before persistence.
The robot approached the goal but failed to finish the arrival manoeuvre, then
spent much of the remaining run near it. Its plans comprised 675 holds, 389
turns and 134 translations; the longest consecutive zero request was 35.40 s.
There were 641 `LOW_VISUAL_SUPPORT_REQUIRES_MEASURED_VIEW` plans and 138
`ADDITIONAL_VIEW_REQUIRED` plans. These localise the recorded failure behaviour
without establishing its cause. Median/maximum tracking error was 2.75/11.15 mm.

On the reactive trajectory's 967 overlapping matched 700-ms windows, unused
learned forecasts had XY/yaw RMSE 37.40 mm / 5.30 degrees, versus command-history
4.83 mm / 0.51 degrees. On 129 translation windows, XY RMSE was 96.17 versus
9.83 mm. Those are recorded forecast comparisons, not additional navigation
trials. The terminal `result.json`, `dense_navigation_readout.json` and complete
RGB-only failure recording remain in the reactive root above.

Thus the completed layout-00 comparison is one verified command-history round
trip and three budget failures. This supports the simpler predictive baseline
on this layout; it establishes neither reliable performance across layouts nor
a benefit from JEPA training, learned prediction or memory. The fixed cohort
continues without parameter changes, beginning layout 01 with
`go2_dense_world_model_maze_layout01_no_future_action_mixed_data_v1_attempt_001`.

## First completed prospective result, September 22

Assignment 1 (layout 00, action-conditioned predictor, fixed mixed-data readout)
completed normally but failed to reach the goal within the mission budget.
The native owner and physical reader completed successfully. Assignment 2,
the same-layout no-future-action control, subsequently started automatically.
No settings or checkpoints were changed between assignments.

| Measure | Layout 00, action-conditioned |
|---|---:|
| Goal / return arrivals | 0 / 0 |
| Terminal outcome | Mission tick budget exhausted |
| Simulated duration | 480.32 s |
| Execution wall time before persistence | 4235.20 s |
| Camera frames / model calls | 4804 / 1198 |
| Disallowed contact samples / pipeline faults | 0 / 0 |
| Hold / turn / translation plans | 504 / 578 / 116 |
| Longest consecutive zero command | 29.20 s |
| Median / maximum tracking position error | 4.26 / 11.53 mm |
| Median model inference | 2705.28 ms |

The robot spent much of the run turning or holding. It resumed translation
near the budget endpoint, but recorded no arrival. This is a completed
prospective navigation failure, not an execution fault or a successful
generalisation result. The three matched controls on this layout are still
needed to distinguish weaknesses in forecasting from shared navigation logic.

On 1197 overlapping, matched executed 700-ms windows, learned motion forecasts
had XY/yaw RMSE of 26.19 mm / 9.13 degrees, versus 6.06 mm / 0.82 degrees for
command history. On the 116 translation windows, XY RMSE was 68.48 mm versus
10.55 mm. These trajectory-conditioned diagnostics do not establish how the
command-history controller will navigate. Of 504 selected holds, 393 had a
higher recorded non-hold utility before later selection/projection logic;
599 plans recorded stopping-projection changes. Those counts locate useful
diagnostic evidence but do not by themselves identify the cause of failure.

Inspection of the saved planning records narrows the hold mechanism. The
stopping projection changed a translation to hold on 284 plans (frames
1492--4700); all three translation candidates failed that projection in every
one of these cases. On 275 of those plans, at least one turn remained eligible
under the memory-clearance filter, but hold had at least as much recorded
utility as the best eligible turn. The other nine had no eligible turn. Thus
these 275 holds arose from the interaction of translation rejection and learned
utility ranking of the remaining actions. This does not establish that the
rejected translations were physically safe or that removing the check would
improve navigation.

For example, at frame 1540 the learned forecast ranked forward first and
reported 0.491 m minimum path clearance, above the 0.480 m reserve requirement.
The stopping projection, which also includes nominal command travel and a
0.5-s stopping allowance, reported clearance as low as 0.295 m against the
0.450 m footprint requirement. Both turns remained eligible, but their
utilities were negative while hold's was +0.00205 m, so the fallback held.
The two checks cover different projected motion; their disagreement is not
alone a measured forecast error. Separately, 75 hold plans had no candidate
passing the memory-clearance selection. These are post hoc mechanism counts,
not additional trials or counterfactual navigation outcomes. Runtime settings
remain frozen for the ongoing comparison.

The executed-turn forecasts also show a directional asymmetry:

| Selected action | Matched windows | Mean actual yaw | Mean learned yaw | Mean command-history yaw | Learned yaw RMSE | Command-history yaw RMSE | Learned opposite-sign windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Left turn | 277 | +10.698 degrees | +7.526 degrees | +10.641 degrees | 4.886 degrees | 0.855 degrees | 12 / 277 |
| Right turn | 300 | -11.404 degrees | +1.724 degrees | -11.093 degrees | 14.283 degrees | 0.662 degrees | 180 / 300 |

These are the reader's matched 700-ms executed windows, including the committed
prefix, not just the subsequent 400-ms action interval. Selected forecast yaw
is `atan2(sin_yaw, cos_yaw)` from the saved neural forecast at index 6. Actual
yaw is reconstructed as the wrapped difference between that forecast and its
saved signed evaluator error; the command-history forecast follows from the
same actual yaw and its saved signed error. Action indices were checked against
the saved selections (left 4, right 5). Opposite-sign counts use actual absolute
rotation above one degree; all 577 turn windows met that descriptive threshold.
The corresponding command-history opposite-sign counts were 0 left and 2 right.
One of the 301 selected right turns lacks a complete matched window.

The asymmetry is evidence of a weakness in the predictor-plus-physical-readout
chain on the executed trajectory. It does not isolate which learned component
caused the errors or establish that a different controller would finish the
maze. The windows overlap and are selected by the policy; they are not 577
independent trials. No additional training or runtime modification followed
this diagnostic during the fixed cohort.

A follow-up using the same matched windows and native pose trace found that
the right-turn error is already present at the readout's trained 500-ms
horizon. Actual right-turn yaw averaged -6.169 degrees versus a learned
+1.364 degrees; RMSE was 8.589 degrees versus 0.524 for command history, with
181/300 learned predictions of the opposite sign. For the actual 300--700-ms
commit interval, actual right yaw averaged -9.264 degrees versus learned
+1.078 degrees; RMSE was 11.241 versus 0.622 degrees, with 170/300 opposite
signs. Left-turn learned RMSE was 3.295 degrees at 500 ms and 5.065 degrees
over the commit interval. Thus neither use beyond 500 ms nor inclusion of the
committed prefix alone explains the directional error. The diagnostic
`turn_horizon_diagnostic.json` in the completed root retains all six summaries;
its 700-ms errors reproduce the existing reader to 1e-10 radians. No new model
inference or training was performed, and the same overlapping-window and
component-attribution limitations apply.

A command-convention check found no left/right reversal in the inspected
training-to-navigation input path. Both select post-limiter forward/yaw
channels in that order. In all 1198 saved model calls, candidate 4 requested
+0.45 rad/s and candidate 5 requested -0.45 rad/s for all four commitment
ticks; their final committed applied yaw commands had the same respective
values. The training candidate generator uses those same signs. This checks
the command interface, not every possible learned-model or target error.

The added full-heading readout training population is directionally imbalanced:
of 2832 saved 500-ms motion targets, 1784 rotate left by more than one degree,
520 rotate right by more than one degree, and 528 have at most one degree of
absolute rotation. The collection schedule includes both turn directions at
each heading, but uses long left turns to transition between headings. These
counts concern the added population, not the full mixed training distribution,
and include overlapping windows and repeated physical trajectories across
appearances. They identify a plausible data-coverage contributor, not a causal
explanation of the right-turn error. Checkpoints and the ongoing cohort remain
unchanged. The following completed diagnosis compares true-future readout
errors with predicted-feature errors before choosing a training intervention.

The CPU-only `scripts/evaluate_go2_dense_maze_turn_readout_development.py`
evaluated eight equally spaced chronological windows from each of four groups:
selected left/right turns following a same/opposite-sign last applied yaw
command. Selection used the saved matched-execution population, not error size;
eligible group sizes were 82 left/same, 188 left/opposite, 99 right/same and
197 right/opposite. Both requested and applied command tapes matched at every
20-ms tick through 700 ms for all 32 selected windows. The existing reader's
700-ms yaw errors were reproduced using its stored sin/cos representation.
The unchanged mixed-data head then decoded the current and actual future RGB
features at 500 and 700 ms on CPU. The 96-image diagnostic completed in 137.15 s
while the separate maze-01 command-history simulation continued on GPU.

| Selected turn / previous command | Horizon | Saved prediction yaw RMSE | Actual-future readout yaw RMSE | Command-history yaw RMSE | Wrong-sign predictions: saved / actual-future |
|---|---:|---:|---:|---:|---:|
| Left / right | 500 ms | 4.14 deg | 4.98 deg | 0.95 deg | 1/8 / 1/8 |
| Left / left | 500 ms | 3.24 deg | 3.48 deg | 0.43 deg | 0/8 / 0/8 |
| Right / left | 500 ms | 8.86 deg | 4.63 deg | 0.44 deg | 6/8 / 5/8 |
| Right / right | 500 ms | 7.37 deg | 9.05 deg | 0.36 deg | 0/8 / 1/8 |
| Left / right | 700 ms | 4.19 deg | 5.66 deg | 1.27 deg | 1/8 / 1/8 |
| Left / left | 700 ms | 5.78 deg | 5.65 deg | 0.44 deg | 0/8 / 0/8 |
| Right / left | 700 ms | 15.04 deg | 13.66 deg | 0.60 deg | 6/8 / 7/8 |
| Right / right | 700 ms | 11.91 deg | 13.33 deg | 0.43 deg | 1/8 / 1/8 |

All selected actual rotations exceeded one degree. On right turns following
left commands at 700 ms, mean actual yaw was -8.19 degrees, saved forecast
+5.93 degrees, and actual-future readout +3.94 degrees. Thus a substantial
directional error remains even with real future features; it is not solely a
failure to predict those features. Actual-future errors combine representation
and readout limitations, and the difference between these RMSEs is not an
additive or causal error decomposition. The small, post hoc, policy-selected
sample does not establish independent navigation performance. Its primary
implication is to investigate the representation/readout interface and its
training coverage alongside predictor transition coverage, rather than assume
that predictor-only retraining will repair the planner.

`turn_true_future_readout_v1/plan.json` and `result.json` in the completed
layout-00 action root retain selected frames, head identity, all 64 horizon
rows, XY/yaw errors, and an identity-input control. No tensors or new depth
recordings were saved. A preparation-only comparison failure is retained in
`turn_true_future_readout_preparation_failure_v1.json`: raw yaw and the prior
reader's float32 sin/cos-decoded yaw differed by 1.42e-9 radians. The evaluator
was corrected to reproduce the prior reader's representation before inference;
no navigation or model settings changed.

The follow-up `turn_true_future_readout_v1/yaw_convention_comparison.json`
checks the body-relative yaw used by the added readout training targets against
the evaluator's wrapped world-heading difference on these same 64 horizon
rows. Their maximum absolute difference is 0.151 degrees overall and 0.034
degrees for the 700-ms right-after-left group. Under the body-relative target,
that group's actual-future readout RMSE remains 13.649 degrees (saved forecast
15.027 degrees; command history 0.591 degrees). Thus the target/evaluator yaw
convention difference does not explain the large error in this sampled group.

A saved-output-only identity-bias subtraction also fails to repair this sample.
Subtracting `head(current,current)` from each forecast reduces right-after-left
700-ms yaw RMSE only from 15.04 to 14.48 degrees for saved predictions and
13.66 to 13.09 degrees for actual-future readout. Wrong-sign counts remain 6/8
and 7/8 respectively. Continuing right turns worsen from 11.91 to 14.09 degrees
and 13.33 to 15.55 degrees respectively. Thus a simple scene-dependent
zero-motion offset does not explain the failure. This is an algebraic diagnostic
on the same saved 32-window population, not a controller change or training
intervention; full rows are retained in
`turn_true_future_readout_v1/identity_bias_subtraction_diagnostic.json`.

A further CPU-only temporal-swap diagnostic used the identical 32 selected
windows and actual future RGB at both horizons. The fixed head was evaluated
as `f(current,future)` and `f(future,current)`. World-heading yaw should reverse
sign when the pair is swapped. The diagnostic also evaluated half the wrapped
forward-minus-backward yaw, without fitting or changing the navigation model.
All repeated forward estimates reproduced the preceding diagnostic within
1e-6 radians. The 96-image encoding completed in 138.3 seconds on CPU while
the maze-01 command-history owner continued on its separate CPU/GPU allocation.

| Selected turn / previous command | Horizon | Forward yaw RMSE | Antisymmetric yaw RMSE | Forward-plus-backward yaw RMS | Wrong sign: forward / antisymmetric |
|---|---:|---:|---:|---:|---:|
| Left / right | 500 ms | 4.98 deg | 5.47 deg | 9.13 deg | 1/8 / 1/8 |
| Left / left | 500 ms | 3.48 deg | 9.09 deg | 15.81 deg | 0/8 / 1/8 |
| Right / left | 500 ms | 4.63 deg | 3.47 deg | 5.97 deg | 5/8 / 1/8 |
| Right / right | 500 ms | 9.05 deg | 12.10 deg | 9.14 deg | 1/8 / 2/8 |
| Left / right | 700 ms | 5.66 deg | 10.54 deg | 11.50 deg | 1/8 / 5/8 |
| Left / left | 700 ms | 5.65 deg | 12.92 deg | 16.45 deg | 0/8 / 1/8 |
| Right / left | 700 ms | 13.66 deg | 10.75 deg | 6.39 deg | 7/8 / 7/8 |
| Right / right | 700 ms | 13.33 deg | 17.27 deg | 10.05 deg | 1/8 / 3/8 |

Temporal reversal is inconsistent, and post-hoc antisymmetrization worsens six
of eight group/horizon errors. In particular, seven of eight 700-ms
right-after-left estimates retain the wrong sign. This rejects a simple
antisymmetric inference correction as a general repair on this sample. Swapped
pairs differ from forward-time training inputs, so this is not proof that the
encoder has lost geometry or that symmetry-aware training could not help.
Together with the identity-offset result, it motivates a controlled readout
training/coverage comparison rather than more algebraic output corrections.
No new navigation success or independent evaluation is claimed. Full rows and
identities are in `turn_pair_symmetry_v1/plan.json` and `result.json` under the
completed action root; the evaluator is
`scripts/evaluate_go2_dense_turn_pair_symmetry_development.py`.

A separate controlled readout continuation has now started on spare CPU cores:
`docs/go2_multihorizon_motion_readout_2026-09-22.md`. It compares fixed 500-ms
training with 100--800-ms training, using identical departure contexts and
optimization budgets from existing training-role recordings. No prospective
maze inputs enter fitting, and neither candidate replaces the running cohort's
head. Feature encoding is in progress; there is no result or promotion yet.

Authoritative records: `result.json` and `dense_navigation_readout.json` in
`.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_mixed_data_v1_attempt_001/`.
The complete failure recording is retained under the cohort's RGB-only archive
policy. The second assignment used
`go2_dense_world_model_maze_layout00_no_future_action_mixed_data_v1_attempt_001`
(native owner PID 34409 at launch, now completed; coordinator session 64661).

The next comparison should include the dense action-conditioned planner,
the matched action-blind treatment, the established command-history baseline
and reactive feedback, with shared perception, routing and physical arrival
criteria. This document fixes the layouts, not a completed execution protocol.
The model interface and measured inference costs must be established before
binding final runtime settings and the execution order. Do not imply the old
300-ms deadline is met by the larger model. Any untimed simulation treatment
must be explicit and shared across the comparison; it cannot support a
real-time or hardware claim. Sensor realism remains at the declared simulation
level while the immediate work targets the world-model scientific questions.

The full goal still requires independent closed-loop goal/home arrival,
exploration and backtracking evidence, matched planning and memory controls,
and a clear distinction between a frozen V-JEPA encoder's usefulness and a
causal benefit from JEPA representation learning. These new layouts alone
establish none of those outcomes.

Source: `lewm/dense_world_model_maze_layouts_development.py`.
Inventory: `go2_dense_world_model_maze_inventory_2026-09-18.json`.
Construction session 89276 exited 0. The figure was visually inspected.

## Fixed execution protocol, September 22

The mixed-data motion readout completed the exposed development maze with both
native-verified arrivals, zero contacts and no pipeline faults. Its round trip
took 409.62 simulated seconds; the reactive control took 249.22 seconds. This
establishes development execution, without a performance advantage.

`go2_dense_world_model_maze_execution_plan_2026-09-22.json` fixes sixteen
assignments, checkpoints, source identities and the existing layout inventory
before the first prospective native execution. All use the fixed mixed-data
readout, 4,800 navigation ticks, 2-mm synthetic depth noise, ideal body gyro,
the same observed map and arrival rules, and untimed synchronous simulation.

| Layout | Fixed arm order |
|---|---|
| 00 | action, no_future_action, command_history, reactive_feedback |
| 01 | no_future_action, command_history, reactive_feedback, action |
| 02 | command_history, reactive_feedback, action, no_future_action |
| 03 | reactive_feedback, action, no_future_action, command_history |

Each assignment runs once. Navigation failure does not remove the run or stop
the comparison; execution faults or insufficient resources stop the coordinator
for inspection. No model/controller tuning or success-seeking repetitions are
permitted within this cohort. All outcomes retain RGB, commands, physics,
perception records and depth hashes, without raw depth archives.

Interpret the no-future-action arm as an information-removal control, not as a
competitive non-predictive navigation baseline. Source inspection during
assignment 2 confirms that `HorizonConditionedDensePredictor.condition` zeros
future commands while retaining horizon and past-control context. The native
model evaluates one future per horizon and expands it identically across all
six candidates. Consequently, model-based candidate scores cannot distinguish
actions at a common horizon. In particular, the initial scan selector lists
hold before the two turns and uses a stable maximum, so equal scan scores
prefer hold. Shared later navigation/clearance logic remains present, and the
terminal execution record must still establish what actually happened.
A navigation difference against this arm alone would not demonstrate accurate
learned dynamics or a JEPA-specific benefit. The command-history and reactive
arms are the stronger practical comparisons. This interpretation changes no
runtime setting, assignment, or outcome criterion.

The existing runner now accepts `--prospective-layout 0` through `3`. Its scene
initializer uses the corresponding fixed layout; the controller receives only
the public mission and live observations. Inventory equality, initializer
bindings and CLI import checks passed before the first native launch. The
physical reader identifies prospective layouts separately from exposed runs.

The coordinator is
`scripts/run_go2_dense_world_model_maze_cohort_development.py --run-all`.
Session 64661 started assignment 1; its native owner is PID 20944, creation time
1790075444.44. First output root:
`.generated/navigation_development_artifacts_v1/go2_dense_world_model_maze_layout00_action_mixed_data_v1_attempt_001/`.
The same coordinator evaluates each completed recording before advancing.
Inspect live processes and terminal artifacts before restarting after a lost
tool handle. Existing roots are never overwritten.

Resource assessment found 16 physical CPU cores, 81.1 GiB available RAM and an
available AMD R9700 GPU with approximately 32 GiB VRAM; workspace free space
was 9.4 GiB. Native/GPU assignments run sequentially, using CPU cores 8–15 and
24–31. The existing 2-GiB prelaunch reserve remains in force. Historical failures
and reference recordings are retained.

This comparison tests generalisation within the maze family and the value of
future-action conditioning and forecast-driven control. It does not isolate
JEPA representation-training causality or persistent-memory contribution, and
does not establish real-time or hardware validity.
