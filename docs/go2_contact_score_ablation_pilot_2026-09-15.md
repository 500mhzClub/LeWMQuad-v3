# Contact-score contribution pilot

The completed eight-run XY-source intervention returned on four of four mazes
using pose/command XY, versus three of four with learned XY. Both arms retained
learned yaw/contact. The next question is whether learned contact scoring
improves actual navigation with the successful pose/command XY predictor.

A saved-plan readout on all four completed pose-command trajectories recomputed
only base candidate rankings for 2,976 non-scan, six-candidate plans. All six
candidates passed the recorded full-reserve clearance check in every included
plan. Removing contact changes 1,678 base rankings; commanded yaw instead of
learned yaw changes 595; both substitutions change 1,732. Original utilities
were reproduced from the recorded progress, alignment and contact components.
The report is `go2_remaining_neural_channels_saved_ranking_readout_v1_attempt_001/result.json`.
It excludes scans, does not replay downstream overrides, and establishes no
alternative execution or contact outcome. Contact's larger ranking effect
motivates testing that single component first.

Freeze four pilot assignments before the first dispatch: contact modes
`learned` and `disabled` on development layouts 0 and 1. These are the first
two indexed layouts, already exposed by both preceding comparisons. Run learned
then disabled within each layout, with layout 0 on CPU 0–7,16–23 and layout 1
on CPU 8–15,24–31. Keep at most two native owners, including recording writes.
Do not tune, replace, retry or silently expand the pilot after outcomes arrive.
Both current reference runs are included; preceding successes are context.

Both arms retain the frozen supervised model and correction, the fixed
pose/command XY fit, learned yaw, coherent perception, 2 mm depth-noise recipe,
mapping/routing memory, action bank and pulse logic, predictive path checks,
recovery, measured-simulation timing, and all arrival, footprint, clearance,
contact and speed guards. Both compute both contact alternatives. The disabled
arm supplies finite contact logits of -1000 to the existing scorer, making its
sigmoid contact term numerically zero. This disables a score; it does not claim
that an action is contact-free. The independent physical guards remain active.

`lewm/contact_score_ablation_development.py` changes only forecast channel 4.
It records complete upstream and applied forecasts so all four other channels
can be compared directly. The learned mode preserves the upstream prediction
exactly. The focused test checks invariance to changed neural contact logits
in the disabled arm, unchanged motion channels and input arrays, zero scored
contact probabilities, and a concrete hold-to-forward ranking change.

Launcher: `scripts/run_go2_contact_score_ablation_development.py`, arguments
`--layout-index 0|1 --contact-score learned|disabled`. Root template:
`go2_contact_score_ablation_pose_command_xy_<mode>_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
Keep all four outcomes and recordings. About 21 GiB storage and 76 GiB RAM are
available before launch; four runs at the existing maximum length fit the
observed storage envelope. Check actual remaining space between paired arms.
No model training or new fitting is involved.

After each owner exits and closes its recording, independently evaluate
physical arrivals and contacts, summarize navigation, and measure fully
executed forecasts. Compare actual forecast channels, shared settings and
source identities within each pair. Report the complete four outcomes, not
only successful runs. This pilot can expose a mechanism or failure; two maze
pairs cannot establish general reliability, statistical superiority, a fully
non-neural planner, real-time operation or hardware readiness. Learned yaw and
predictive planning remain in both arms. No hardware execution is authorized
or performed by this pilot.

The focused contact test and existing XY-channel property test both passed
(two tests, 1.76 s). All four launch-writer assignments were checked without
native execution or file creation, under their prescribed CPU affinities.
The actual first two launch records also match the frozen supervised model,
pose-command XY source, contact mode and four-assignment scope.

The learned-contact references launched on layouts 0 and 1 in sessions 41447
and 93525, respectively, with native owners 3604391 and 3604411. Both were
confirmed live. Disabled-contact counterparts follow after the corresponding
owner closes its recording and independent evaluation finishes.

`scripts/compare_go2_contact_score_ablation_development.py` is prepared for
the completed pairs. It checks all actual upstream/applied forecast channels,
the fixed pose-command and supervised fit bindings, probabilities consumed by
the scorer, shared settings and common runtime sources, and includes saved
physical outcomes and executed forecast errors. Its source syntax is checked;
no paired pilot report or outcome is asserted before both arms finish.


The saved ranking readout's `action_transitions.json` further localizes the
contact effect: 1,159 of its 1,678 changes replace a base-ranked hold with
movement (1,156 with forward); none replaces movement with hold. Another
301 changes replace a pure turn with its corresponding arc. The original
base winner matches the recorded final action in 1,664 of the 1,678 changed
plans. This suggests reduced waiting and more translating turns as concrete
behaviors to measure in the live disabled arm. It does not establish that the
alternative commands would be safe or that later overrides would retain them.
The live pilot and independent physical checks remain necessary.


## Reference outcomes verified; disabled arms launched

Both learned-contact owners exited zero with complete archives and zero swaps.
Layout 0: wall 541.50 s, peak RSS 20,782,156 KiB. Layout 1: wall 536.88 s,
peak RSS 19,823,724 KiB. Independent evaluation verifies both goals and returns,
with zero disallowed contacts in both runs.

Layout 0 has 3,817 poses, goal frame 2296 and home frame 3815, median/max
position error 5.153/8.805 mm and final home distance 14.952 mm. Its 927 fully
executed 700 ms windows give applied pose-command XY RMSE 9.960 mm, maximum
42.423 mm and fourteen path errors above 30 mm.

Layout 1 has 3,610 poses, goal frame 2456 and home frame 3608, median/max
position error 4.988/7.170 mm and final home distance 17.313 mm. Its 850 fully
executed windows give applied XY RMSE 9.936 mm, maximum 43.872 mm and eleven
path errors above 30 mm. These errors remain trajectory-conditional and do
not certify an action's clearance or instantaneous speed.

After these evaluations, disabled-contact layout 0 launched in session 68541
on the even CPU group and layout 1 in session 92370 on the odd group. About
16 GiB artifact storage and 76 GiB RAM were available; the two remaining
maximum-length recordings fit the observed storage envelope. No model, fit,
controller parameter, runtime source or physical guard changed between arms.
All four assigned outcomes remain fixed; the disabled-arm results are pending.

The paired analysis additionally reports selected-action counts and seconds
of applied translating, pure-turn and zero commands at the recorded 20 ms
cadence. These bins exhaust the command population; zero command is explicitly
not equated with physical stationarity. The binning was exercised on a
completed predecessor recording and its durations sum to the recorded command
intervals. The plotting helper now labels this experiment's two modes as
“Learned contact score” and “Contact score disabled”; it is not part of the
runtime source population.


## Recorded yaw accuracy, without changing this pilot

The new full-forecast receipts allow an executed-window yaw comparison using
`scripts/evaluate_saved_neural_command_yaw_development.py`. The evaluator uses
exactly the existing fully matched 700 ms command windows and the recorded
neural sine/cosine outputs. Native yaw targets are wrapped differences of
world heading, matching both causal-subtrajectory and moving-prefix training
target definitions. It also evaluates yaw change over 300–700 ms, the scored
candidate commitment interval. Future physics is evaluator-only; no inference,
fit or live yaw substitution occurs.

| Reference recording | Windows | Neural yaw RMSE at 700 ms | Command yaw RMSE | Neural commitment-yaw RMSE | Command commitment-yaw RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Layout 0 | 927 | 2.553 degrees | 1.504 degrees | 2.603 degrees | 1.479 degrees |
| Layout 1 | 850 | 2.292 degrees | 1.334 degrees | 2.228 degrees | 1.288 degrees |

The unfitted command integral has lower error in each of the six action groups
on both recordings. These are overlapping windows selected by the neural-yaw,
learned-contact controller; unexecuted actions and alternative navigation are
not evaluated. Saved per-window results are in each reference root as
`saved_neural_command_yaw_evaluation_v1.json`. The command predictor does not
model gait inertia or slip. This evidence motivates a future yaw intervention,
but the current four contact assignments and their learned yaw remain fixed.


## Completed pilot: mixed effects, retain the contact term pending diagnosis

All four native owners exited zero and all outcomes were independently
evaluated. The complete report is
`go2_contact_score_ablation_pose_command_xy_two_layout_summary_v1_attempt_001/result.json`.
Both paired reports check all actual contact/motion channels, scored
probabilities, frozen fit bindings, shared settings and 134 common runtime
source identities. Both PNG/SVG trajectory comparisons were visually inspected.
All four recordings, including the failure, remain retained.

| Contact scoring | Verified goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| Learned | 2/2 | 2/2 | 0 |
| Disabled | 1/2 | 1/2 | 0 |

Disabled layout 1 returned successfully: goal frame 1908, home 2773, 2,775
poses; median/max position error 6.879/11.436 mm, final home distance 16.736 mm.
Owner wall time 419.40 s, peak RSS 15,816,768 KiB, zero swaps. Recorded
trajectory time was 277.4 s versus reference 360.9 s. Applied zero-command
time fell from 150.02 s to 24.18 s, while pure-turn commands increased from
98.00 s to 147.52 s and translating commands changed from 112.92 s to
105.74 s. Thus reduced waiting outweighed additional turning on this maze.
Its 654 executed forecast windows have XY RMSE 6.063 mm, maximum 31.798 mm
and two path errors above 30 mm; these are a different trajectory population,
not an improvement to the unchanged predictor.

Disabled layout 0 exhausted the 4,800-tick budget without either arrival.
It recorded 4,805 accepted poses, median/max error 7.972/12.252 mm, final
physical goal distance 3.297 m and zero disallowed contacts. Owner wall time
669.24 s, peak RSS 25,429,740 KiB, zero swaps. Its 669 executed forecast
windows have XY RMSE 5.535 mm, maximum 17.442 mm and no path error above
30 mm, illustrating that small executed-motion errors do not imply navigation
success. Applied zero-command time was 236.20 s, versus reference 127.00 s;
this includes the long period without an available plan. It traveled 15.9 m
and explored a different branch before stalling; the reference traveled 23.3 m
and returned via the goal.

The failed root's `view_budget_stall_diagnostic_v1.json` records 685 selected
plans, 673 on time, and 515 `VIEW_BUDGET_EXHAUSTED` records. Exhaustion first
appears at frame 2540; the last selected plan is frame 3736, on time, with
`OBSERVED_COMPONENT_HAS_NO_FRONTIER`. The run completed nine frontier-view
events and excluded 58 frontier cells. Poses remained available, there are no
pipeline faults, and the mission ended outbound. The absence of a frontier,
and why prior views/exclusions did not produce a goal route, still require
map/frontier reconstruction; no root cause or remedy is claimed yet.

Interpretation: removing the learned contact score is not a general navigation
improvement in this pilot. It reduced waiting and completion time on layout 1,
but exposed an exploration failure on layout 0. Keep the contact-enabled
reference while diagnosing that failure. Two pairs do not establish general
superiority or calibrated contact probabilities, and all executed outcomes
were contact-free. The contact term could influence exploration through action
preferences rather than correctly predicting collisions; that distinction is
unresolved. No threshold, runtime or model changed during the pilot.

The same-window yaw analysis also completed on both disabled trajectories.
Layout 0 endpoint RMSE is 3.004 degrees neural versus 1.733 degrees command;
commitment-yaw RMSE is 3.028 versus 1.619 degrees. Layout 1 is 3.150 versus
1.575 degrees endpoint and 3.258 versus 1.491 degrees commitment. Command
integration remains more accurate in each action group on these recordings,
without establishing an alternative controller outcome. A yaw intervention
remains separate; first diagnose the observed exploration stall.

No simulation is running after this pilot. Artifact storage has about 11 GiB
free. The original goal remains active: broader reliability and generalization,
remaining prediction contributions, realistic sensing/timing and bounded
hardware evidence are not yet established.
