# Fresh-maze neural RGB contribution pilot

Completed September 16: all 36 fixed assignments terminated and were physically
evaluated: 35 verified round trips and one failure, all zero contacts. Full input
completed 18/18 and no-RGB 17/18; JEPA and direct each completed 12/12,
supervised rollout 11/12. The complete result is in
`docs/go2_neural_rgb_transfer_complete_result_2026-09-16.md`.

Assignment 22,
second-seed no-RGB supervised rollout on maze 0, stopped after 1,015 acquired
frames with a tracking fault (`measured visual pose unavailable`). Its physical
evaluation confirms no arrivals and zero contacts. Exact sensor replay identifies
an anchor/increment translation conflict; the diagnosis is recorded below.
Preserve this failure and its depth;
do not retry or reinterpret the completed frozen cohort. The prospective
coordinate-correction follow-up is separate and has not run yet:
`docs/go2_mission_coordinate_followup_2026-09-16.md`.

The first training seed's 12 assignments are complete and
physically evaluated, all successful round trips with zero contacts. Both RGB
treatments passed all six assignments. Full input was faster in five of six
pairs; effects vary markedly by method and maze, with terminal behavior a major
influence in several pairs. The second seed is also complete. This remains a
36-assignment pilot on only two independent maze layouts, with no established
general RGB or JEPA superiority. See the complete first-seed table below.

The full-input predictors have not outperformed the fitted-motion control in
recent navigation cohorts. Matching all nine no-RGB corrections gives similar
offline motion error. Test whether neural RGB changes actual goal-and-return
outcomes with the same predictive controller, rather than inferring navigation
benefit from those errors.

Fix 36 assignments: two fresh mazes, three training seeds (2026091001,
2026091401, 2026091402), three methods (JEPA, direct, supervised rollout), and
full/no-RGB treatments. For each seed and method in that order, run maze 0
full then no-RGB, followed by maze 1 no-RGB then full. All models and their
matching corrections are already frozen. No fitting, selection of a favorable
seed, retries or controller tuning during this cohort.

The maze inventory uses construction seed 2026091556 and excludes the explicit
80 previous development graphs. Physics seeds start at 2026098600 and appearance
seeds at 2026098700. Inventory:
`docs/go2_neural_rgb_transfer_layout_inventory_2026-09-15.json`, SHA-256
`6b572ea6cd183ec2fa04b4b345db7a1cfb6841a6d893091c388e4a46e13f105f`.
The two layouts are the environmental replication units. The 36 runs do not
create 36 independent mazes. This bounded pilot screens for a large input
effect before expanding the cohort; it has no claimed power for small effects
and cannot certify reliability or absence of an RGB benefit.

Keep the shared signed-view predictive runtime, combined tracker, transported
floor candidates, auxiliary depth, noise, ideal gyro, persistent map, action
bank, prediction horizons, disabled contact score, arrival dwells and physical
guards unchanged. The network input/training treatment, trained weights and
corresponding correction are the intended differences. No-RGB also removed
future RGB from the original JEPA training targets. Camera-based localization
and mapping remain in both treatments. Record the actual neural RGB nonzero
count on every forward call; require zero for the no-RGB treatment. Use the
same receipt overhead in both treatments.

Extend the existing correction assignment API to accept the separately frozen
no-RGB registry. Full-input assignments retain their original constructor path
and coefficients. No policy or perception algorithm changes. Both fresh arms
use these same sources; older native results remain contextual references.

Run native owners sequentially, retaining the existing layout-specific CPU
groups, because previous concurrent native runs incurred timing failures.
Evaluate each assignment before the next. Preserve every failure; retire
completed, diagnosed success depth under the existing retention policy after
paired comparisons. Keep results, RGB/body/gyro/poses, commands, physics,
models and correction fits. Four GiB free is required before each recording.

Launcher: `scripts/run_go2_neural_rgb_transfer_development.py`.
Evaluator: `scripts/evaluate_go2_neural_rgb_transfer_development.py`.
Report physical goals/returns, contacts, timing failures, path/time, actual
input and correction assignments, and every planned outcome. Simulation
uses measured simulation timing and synthetic 2-mm depth noise. This is not
calibrated sensor, hard-real-time or hardware validation. The broader goal
still requires independent generalization and deployment evidence.

Three focused tests passed for unchanged model outputs/input tensors, actual
forward receipts in both treatments, and rejection of nonzero RGB before a
no-RGB forward. An actual-artifact integration probe constructed all 18 runtime
assignments, checked every loaded coefficient against its assigned NPZ, executed
each neural model on the same retained causal policy input, rejected the other
input treatment's correction, verified launch bindings, and closed every runtime
thread. Full inputs were nonzero and no-RGB inputs zero in all tested calls.
The first fixed assignment, full-input JEPA seed 2026091001 on maze 0, launched
after those checks; its native outcome is pending.

The first full-input JEPA mission is complete: physical goal/home arrivals at
frames 1,192/1,688, both with the required quiet dwell; 1,690 poses, zero
contacts and maximum position error 9.210 mm. It took 169.16 simulated seconds;
408/416 plans were on time. Owner exit 0 after 4:24.47 including recording, with
no swaps. All actual prediction/correction and neural input receipts passed.
The matched no-RGB JEPA run on maze 0 launched next; no comparative conclusion
is available from this first success alone.

## First completed pair: JEPA seed 2026091001, fresh maze 0

The no-RGB counterpart also passed physical goal/home checks, at frames
1,756/2,232: 2,234 poses, zero contacts, maximum position error 7.677 mm,
545/551 plans on time, 223.52 simulated seconds. Owner exit 0 after 5:22.40,
with no swaps. All 551 actual neural calls received zero RGB; all 416 full-input
calls had nonzero RGB. Both model/correction assignments passed evaluation.

| Treatment | Round trip | Contacts | Simulated seconds | Path metres |
| --- | --- | --- | ---: | ---: |
| Full | yes | 0 | 169.16 | 16.440 |
| No-RGB | yes | 0 | 223.52 | 16.901 |

The pair has 175 identical common runtime sources and matching non-treatment
settings. The comparison and inspected PNG/SVG figures are under
`go2_neural_rgb_transfer_comparison_seed_2026091001_jepa_layout00_v1_attempt_001`.
This is one maze/seed pair, not evidence of a general RGB advantage.

No-RGB's extra 54.36 simulated seconds primarily reflect command allocation:
translation 87.40 versus 87.62 s, pure turns 86.92 versus 71.08 s, and zero
command 49.00 versus 10.22 s. Zero command is not a physical-stationarity
claim. All 101 no-RGB hold plans occurred during views; on 95 of them the
preferred pre-filter right turn was blocked by the predicted-clearance filter.
Full had seven hold plans, six during views. The saved
`hold_and_command_duration_diagnostic_v1.json` records the exact blocked frames.
The trajectories differ, so this is a recorded mechanism, not a same-state
counterfactual or proof that RGB improved future prediction accuracy.

No controller changes follow this observation. The next fixed assignment,
no-RGB JEPA on fresh maze 1, is running; its full-input counterpart follows.
Two of 36 missions are complete, both round trips, with no cohort-level
training-method or RGB conclusion yet.

The third assignment, no-RGB JEPA seed 2026091001 on maze 1, verified goal/home
at frames 1,064/2,013: 2,015 poses, zero contacts, maximum position error
5.060 mm, 477/487 plans on time, 201.62 simulated seconds. Owner exit 0 after
5:23.05 including recording, with no swaps. Every selected plan's actual RGB
and model/correction checks passed. Full-input JEPA on the same maze launched
next, as specified by the fixed reverse treatment order.

The first pair's terminal approach diagnostic separates its final 10 cm from
the rest of the mission. Full/no-RGB approach durations were 1.6/4.2 s outbound
and 2.3/2.6 s on return; neither arm left the 10-cm region after entering it.
Thus only 2.9 s of the first pair's duration difference lies in these terminal
windows. The main observed difference remains the extra no-RGB view holds and
turning. This diagnostic changes no controller rule or study assignment.

## Second JEPA pair and start of direct prediction

Full-input JEPA on maze 1 also verified the round trip: goal/home frames
996/1,588, 1,590 poses, zero contacts, maximum position error 5.013 mm,
378/390 plans on time and 159.14 simulated seconds. Owner exit 0 after
4:05.23 with no swaps. Actual input/model/correction checks passed. The pair
has 175 identical common sources; comparison PNG/SVG figures were inspected.
Full/no-RGB paths were 18.801/19.398 m and durations 159.14/201.62 s.

The maze-1 time difference has a different mechanism from maze 0. Full/no-RGB
terminal approaches within the final 10 cm took 1.7/8.2 s outbound and
2.3/43.6 s on return, without leaving that region. Their combined difference
of 47.8 s exceeds the whole-mission difference of 42.48 s; outside those
windows no-RGB was about 5.32 s faster. This second pair mainly reflects
terminal precision/settling behavior, not a broad route-planning advantage.
Both treatments passed both mazes; these four assignments do not establish
a general RGB or JEPA advantage. The controller remains frozen.

The fifth assignment, full-input direct prediction seed 2026091001 on maze 0,
also verified goal/home at frames 996/1,472: 1,474 poses, zero contacts,
maximum position error 7.683 mm, 357/361 plans on time and 147.54 simulated
seconds. Owner exit 0 after 3:39.12 with no swaps; actual treatment and
physical arrival checks passed. Its no-RGB counterpart launched next.
Five of 36 assignments are complete, all verified round trips; 31 remain.

The sixth assignment, no-RGB direct prediction on maze 0, verified goal/home
at frames 1,407/2,048: 2,050 poses, zero contacts, maximum position error
10.494 mm, 486/496 plans on time and 205.14 simulated seconds. Owner exit 0
after 4:57.79 with no swaps. Actual model/correction and zero-RGB receipts
passed. The direct pair has 175 identical common sources; its comparison
PNG/SVG figures were inspected. Full/no-RGB paths were 16.559/17.355 m.

Full/no-RGB translation command durations were 88.34/90.80 s, turn-only
49.40/89.14 s and zero 9.58/24.98 s. Terminal approaches took 2.3/3.5 s
outbound and 1.7/16.1 s on return, without leaving the 10-cm region. Those
terminal windows account for 15.6 s of the overall 57.6-s difference.
The greater difference was additional turning outside the terminal windows;
neither this trajectory comparison nor the earlier JEPA pairs isolate a
same-state forecast-accuracy effect. No controller tuning follows.

Six of 36 assignments are evaluated, all verified round trips with zero
contacts. The seventh fixed assignment, no-RGB direct on maze 1, is running;
full-input direct on maze 1 follows. Both treatments continue to use cameras
for localization and mapping. These are two environmental replication units,
not six independent mazes, and no general RGB superiority is established.

The seventh assignment, no-RGB direct on maze 1, verified goal/home at frames
996/1,568: 1,570 poses, zero contacts, maximum position error 5.294 mm,
375/384 plans on time and 157.20 simulated seconds. Owner exit 0 after
4:02.54 with no swaps; actual treatment and physical arrival checks passed.
Its full-input counterpart launched next. Seven of 36 assignments are
evaluated, all successful round trips. The completed JEPA maze-1 pair's depth
was retired after comparison and diagnosis under the declared policy; all
its non-depth scientific evidence remains, and the JEPA maze-0 pair remains
in full as the selected depth reference.

## First-seed JEPA and direct pairs complete: eight missions

Full-input direct on maze 1 verified goal/home at frames 1,448/2,007:
2,009 poses, zero contacts, maximum position error 7.025 mm, 470/479 plans
on time and 201.22 simulated seconds. Owner exit 0 after 5:06.00 with no
swaps; actual model/correction/input and physical arrival checks passed.
Its pair has 175 identical common sources and inspected PNG/SVG figures.
Full/no-RGB paths were 19.314/18.565 m.

| Method | Maze | Full seconds | No-RGB seconds | Round trips |
| --- | ---: | ---: | ---: | --- |
| JEPA | 0 | 169.16 | 223.52 | both |
| JEPA | 1 | 159.14 | 201.62 | both |
| Direct | 0 | 147.54 | 205.14 | both |
| Direct | 1 | 201.22 | 157.20 | both |

The direct maze-1 timing difference reverses maze 0: full is 44.02 s slower.
Full/no-RGB final-10cm approaches took 54.3/3.3 s outbound and 2.5/1.7 s
on return, without leaving the region. The terminal difference is 51.8 s;
outside those windows full was about 7.78 s faster. Full's long outbound
approach applied translation for 10.52 s and zero command for 43.78 s, with
no pure turns. It selected 114 forward and three hold plans; 114/117 selected
plans were on time, and 19 additional events were mission holds. The saved
`outbound_terminal_command_diagnostic_v1.json` retains the counts. A request
label of `NO_ON_TIME_PLAN` means no active plan and does not alone establish
a compute deadline miss. Arrival precision/settling is a major timing influence
in this pair; no claim of generally better route planning follows.

The descriptive executed-forecast readout is retained in
`docs/go2_neural_rgb_first_eight_executed_forecasts_2026-09-15.json`.
Across these eight trajectories, corrected endpoint XY RMSE spans
6.070–9.042 mm and is not consistently lower for RGB. Command-based heading
forecast endpoint RMSE is lower than learned heading in each recording
(command 1.333–1.646 degrees; learned 2.829–5.329 degrees). These are saved
forecasts for matched executed request prefixes through 700 ms; trajectories
and action populations differ between treatments, overlapping windows are not
independent, and unexecuted candidates are not evaluated. This does not prove
a causal RGB accuracy effect or a better alternative navigation outcome.

Eight of 36 assignments are evaluated, all round trips with zero contacts.
Full-input supervised rollout on maze 0 launched as assignment nine. All
models and controller settings remain fixed. The completed direct maze-1
pair's depth was retired after diagnosis, retaining all non-depth evidence
and the complete direct maze-0 pair as its selected depth reference. Neither
these four pairs nor repeated seeds constitute more than two independent
environmental units; reliability and general RGB superiority remain unproven.

Assignment nine, full-input supervised rollout on maze 0, verified goal/home
at frames 1,139/1,624: 1,626 poses, zero contacts, maximum position error
10.670 mm, 390/397 plans on time and 162.78 simulated seconds. Owner exit 0
after 4:00.01 with no swaps; actual model/correction/input and physical
arrival checks passed. Its no-RGB counterpart launched as assignment ten.
Nine of 36 assignments are evaluated, all verified round trips.

Assignment ten, no-RGB supervised rollout on maze 0, verified goal/home
at frames 1,300/1,852: 1,854 poses, zero contacts, maximum position error
7.808 mm, 439/443 plans on time and 185.52 simulated seconds. Owner exit 0
after 4:30.20 with no swaps; actual treatment and physical arrival checks
passed. The pair has 175 identical common sources; PNG/SVG figures were
inspected. Full/no-RGB paths were 17.275/17.871 m.

Full was 22.74 simulated seconds faster. Full/no-RGB command durations were
translation 92.36/94.08 s, turn-only 58.86/73.88 s and zero 11.34/17.36 s.
Final-10cm approaches were 4.3/3.8 s outbound and 1.8/10.3 s on return,
without leaving the region. The total terminal difference was 8.0 s, leaving
14.74 s of the mission difference outside those windows. This is a descriptive
trajectory comparison, not an isolated forecast-accuracy effect. All three
methods were faster with full input on maze 0 for this seed, but direct
prediction reversed direction on maze 1; no general RGB advantage is established.

Ten of 36 assignments are evaluated, all round trips with zero contacts.
No-RGB supervised rollout on maze 1 launched as assignment eleven, to be
followed by full-input supervised rollout on that maze. The older shared-recovery
supervised success's depth was retired under the policy after the fresh pair
completed; both fresh supervised recordings remain in full. Runtime settings
and fixed assignment order are unchanged.

Assignment eleven, no-RGB supervised rollout on maze 1, verified goal/home
at frames 953/1,525: 1,527 poses, zero contacts, maximum position error
4.372 mm, 364/373 plans on time and 152.80 simulated seconds. Owner exit 0
after 3:56.44 with no swaps; actual treatment and physical arrival checks
passed. Full-input supervised rollout on the same maze launched as assignment
twelve. Eleven of 36 assignments are evaluated, all verified round trips.

## First training seed complete

Assignment twelve, full-input supervised rollout on maze 1, verified goal/home
at frames 901/1,468: 1,474 poses, zero contacts, maximum position error
4.159 mm, 354/360 plans on time and 148.26 simulated seconds. Owner exit 0
after 3:49.63 with no swaps; actual treatment and physical arrival checks
passed. Its pair has 175 identical common sources and inspected PNG/SVG
figures. Full/no-RGB paths were 18.313/18.615 m. Final-10cm approaches were
1.7/1.8 s outbound and 1.8/1.8 s on return; neither left the region. Only
0.1 s of the 4.54-s difference lies in those terminal windows.

All first-seed outcomes and executed forecast summaries are saved in
`go2_neural_rgb_transfer_seed_2026091001_complete_comparison_v1_attempt_001/result.json`.
Every full/no-RGB assignment succeeded, with zero contacts; no outcome was
excluded. Positive differences below mean no-RGB took longer.

| Method | Maze | Full seconds | No-RGB seconds | Difference | Terminal component |
| --- | ---: | ---: | ---: | ---: | ---: |
| JEPA | 0 | 169.16 | 223.52 | +54.36 | +2.90 |
| JEPA | 1 | 159.14 | 201.62 | +42.48 | +47.80 |
| Direct | 0 | 147.54 | 205.14 | +57.60 | +15.60 |
| Direct | 1 | 201.22 | 157.20 | -44.02 | -51.80 |
| Supervised rollout | 0 | 162.78 | 185.52 | +22.74 | +8.00 |
| Supervised rollout | 1 | 148.26 | 152.80 | +4.54 | +0.10 |

The two maze layouts are the environmental units; six method/maze pairs are
not six independent environments. Neither treatment has a success advantage
in this seed. Full input's five faster pairs are preliminary efficiency
evidence, with different local behavior and terminal costs, not established
general RGB benefit. Model training/input and matching corrections differ;
both treatments retain camera-based localization/mapping. Command-based heading
forecasts have lower saved executed-endpoint error than learned heading in
all 12 recordings, but this establishes no alternative navigation outcome.

Twelve of 36 assignments are evaluated. Assignment thirteen, full-input JEPA
seed 2026091401 on maze 0, launched next. The same controller, models, fixed
assignment order and outcome criteria remain unchanged; no favorable seed or
method is selected and no policy tuning follows these interim results.

## Second training seed underway

Assignment thirteen, full-input JEPA seed 2026091401 on maze 0, verified
goal/home at frames 1,977/2,728: 2,730 poses, zero contacts, maximum position
error 8.479 mm, 628/662 plans on time and 273.12 simulated seconds. Owner
exit 0 after 6:29.37 with no swaps. Actual model/correction/input and physical
arrival checks passed. Preserve the timing limitation: 34 late plans and
maximum simulator lag 4,349.851 ms. This is measured-simulation navigation,
not hard-real-time qualification; the first seed's faster result is not used
to exclude this slower assignment.

Thirteen of 36 assignments are evaluated, all verified round trips with zero
contacts. Its matched no-RGB JEPA counterpart on maze 0 launched next. The
completed first-seed supervised maze-1 pair's depth has been retired after
comparison, diagnosis and aggregation; all its non-depth evidence and the
six selected first-seed maze-0 depth recordings remain.

Assignment fourteen, no-RGB JEPA seed 2026091401 on maze 0, verified
goal/home at frames 1,937/2,423: 2,425 poses, zero contacts, maximum position
error 7.887 mm, 586/592 plans on time and 242.84 simulated seconds. Owner
exit 0 after 5:48.18 with no swaps. Actual treatment and physical arrival
checks passed. Peak simulator lag was 548.674 ms. The pair has 175 identical
common sources and inspected PNG/SVG figures; full/no-RGB paths were
19.107/18.557 m.

No-RGB was 30.28 s faster, reversing the first seed's JEPA maze-0 direction.
Full/no-RGB terminal approaches took 1.7/28.6 s outbound and 25.8/4.1 s on
return; neither left the final-10cm region. No-RGB spent 5.2 s longer in these
windows in total, so its time outside the windows was 35.48 s shorter.
Full/no-RGB command durations were translation 96.32/93.42 s, turn-only
143.00/121.16 s and zero 33.60/27.84 s.

`paired_recorded_timing_diagnostic_v1.json` retains stage summaries and all
34/six late-plan frames. Full/no-RGB median planning service was 74/74 ms,
maximum service 616/464 ms and maximum completion age 1,158/654 ms.
Most full-input late plans clustered between frames 1,652 and 1,796, with
additional late events elsewhere. These are measured-simulation timestamps;
different trajectories and host timing do not isolate model-caused efficiency,
and the cause of the largest lag is not established. No controller change or
replacement run follows this result.

Fourteen of 36 assignments are evaluated, all round trips with zero contacts.
No-RGB JEPA seed 2026091401 on maze 1 launched as assignment fifteen. After
the completed pair's physical, terminal and timing analyses, its depth was
retired under policy; all scientific non-depth evidence remains and the first
seed's full JEPA maze-0 pair remains the selected depth reference.

Assignment fifteen, no-RGB JEPA seed 2026091401 on maze 1, verified goal/home
at frames 1,048/2,608: 2,610 poses, zero contacts, maximum position error
4.315 mm, 633/644 plans on time and 261.12 simulated seconds. Owner exit 0
after 6:31.72 with no swaps; actual treatment and physical arrival checks
passed. Peak simulator lag was 685.756 ms. The longer return remains part
of the outcome; full path and terminal comparison await its counterpart.
Fifteen of 36 assignments are evaluated, all verified round trips. Full-input
JEPA seed 2026091401 on maze 1 launched next, with unchanged settings.

Assignment sixteen, full-input JEPA seed 2026091401 on maze 1, verified
goal/home at frames 1,106/1,671: 1,673 poses, zero contacts, maximum position
error 7.133 mm, 396/408 plans on time and 167.76 simulated seconds. Owner
exit 0 after 4:16.60 with no swaps; actual treatment and physical arrival
checks passed. Peak simulator lag was 542.962 ms. The pair has 175 identical
common sources and inspected PNG/SVG figures; full/no-RGB total paths were
18.804/21.034 m.

Full was 93.36 s faster on maze 1, opposite the second seed's maze-0 result.
Full/no-RGB final-10cm approaches took 6.8/4.1 s outbound and 2.9/2.3 s on
return; neither left the region. No-RGB spent 3.3 s less in terminal windows,
so final settling does not explain its slower mission. The phase-path/command
diagnostic records a 56.5/156.0-s return after goal arrival, with pure-turn
commands for 4.7/101.8 s, translation for 47.60/49.18 s and zero for
4.20/5.02 s. Camera-sampled native return paths were 8.714/10.921 m;
these paths include body movement during turns and are not planned-route lengths.

The return selection diagnostic records 13 pure-turn plans with full input
versus 256 without RGB. In no-RGB, the selected action differed from the
pre-filter preference on 184 plans: 59 right-arc to right-turn, 100 right-arc
to left-turn and 25 right-turn to left-turn. The preferred candidate was
recorded as `BLOCKED` on 181 of those plans. Full had two right-arc to
right-turn changes, with neither preferred candidate recorded blocked.
At no-RGB frame 1,100, for example, preferred right-arc clearance was
0.468329 m against the 0.48-m reserve requirement; selected left-turn used
the hold-relative recovery rule. These records identify forecast-clearance
and recovery decisions accompanying the extra turning, not a same-state
counterfactual or proof of superior RGB forecast/clearance accuracy.

The comparison root retains `phase_path_command_diagnostic_v1.json`,
`return_selection_diagnostic_v1.json` and
`return_preferred_action_clearance_diagnostic_v1.json` alongside the terminal
diagnostic. Both completed successes' depth was retired afterward under policy,
with all non-depth evidence and selected first-seed depth references preserved.
Sixteen of 36 assignments are evaluated, all round trips with zero contacts.
Full-input direct prediction seed 2026091401 on maze 0 launched as assignment
seventeen. No model/controller tuning or outcome selection occurred.

Assignment seventeen, full-input direct prediction seed 2026091401 on maze 0,
verified goal/home at frames 1,888/2,415: 2,417 poses, zero contacts, maximum
position error 6.427 mm, 550/587 plans on time and 242.08 simulated seconds.
Owner exit 0 after 5:46.30 with no swaps; actual treatment and physical
arrival checks passed. Retain its 37 late plans and 3,761.408-ms peak
simulator lag; this is not real-time qualification. Its no-RGB counterpart
launched as assignment eighteen. Seventeen of 36 assignments are evaluated,
all round trips with zero contacts.

Assignment eighteen, no-RGB direct prediction seed 2026091401 on maze 0,
verified goal/home at frames 2,009/2,506: 2,508 poses, zero contacts, maximum
position error 8.274 mm, 602/619 plans on time and 250.98 simulated seconds.
Owner exit 0 after 5:58.91 with no swaps; actual treatment and physical
arrival checks passed. Peak simulator lag was 578.176 ms. The pair has 175
identical common sources and inspected PNG/SVG figures; full/no-RGB paths
were 18.403/19.065 m.

Full was 8.90 s faster. Full/no-RGB final-10cm approaches took 2.7/1.8 s
outbound and 3.8/1.7 s on return, without leaving the region. Thus no-RGB
settled 3.0 s faster in those windows, while its time outside the windows was
11.90 s longer. Full/no-RGB command durations were translation 92.66/95.60 s,
turn-only 121.90/141.62 s and zero 27.06/13.54 s. The saved paired timing
diagnostic retains all 37/17 late-plan frames and stage summaries: median
planning service 78/78 ms, maximum service 620/606 ms and maximum completion
age 1,162/748 ms. Different trajectories and host timing remain limitations
on interpreting this modest descriptive efficiency difference.

The pilot is halfway through: eighteen of 36 assignments are evaluated, all
verified round trips with zero contacts. The first seed contributes 12 runs
and the second six, on the same two independent development layouts. This
does not establish broad reliability or a general RGB/JEPA advantage. After
paired/terminal/timing analysis, the completed second-seed direct maze-0 pair's
depth was retired under policy, keeping all non-depth evidence and the
first-seed direct maze-0 pair in full. Assignment nineteen, no-RGB direct
seed 2026091401 on maze 1, launched next; its full-input counterpart follows.

Assignment nineteen, no-RGB direct prediction seed 2026091401 on maze 1,
verified goal/home at frames 1,040/1,570: 1,572 poses, zero contacts, maximum
position error 4.088 mm, 368/379 plans on time and 157.32 simulated seconds.
Owner exit 0 after 4:03.13 with no swaps; actual treatment and physical
arrival checks passed. Peak simulator lag was 509.331 ms. Nineteen of 36
assignments are evaluated, all round trips with zero contacts. Its full-input
counterpart launched as assignment twenty with unchanged settings.

Assignment twenty, full-input direct prediction seed 2026091401 on maze 1,
verified goal/home at frames 962/1,535: 1,537 poses, zero contacts, maximum
position error 4.362 mm, 366/377 plans on time and 154.02 simulated seconds.
Owner exit 0 after 3:58.30 with no swaps; actual treatment and physical
arrival checks passed. Peak simulator lag was 501.358 ms. The pair has 175
identical common sources and inspected PNG/SVG figures; full/no-RGB paths
were 18.383/18.531 m. Both runs had 11 late plans.

Full was 3.30 s faster, while full/no-RGB final-10cm approaches took 1.8/12.5 s
outbound and 1.6/1.9 s on return, without leaving the region. The terminal
difference was 11.0 s; outside those windows no-RGB was about 7.7 s faster.
Thus this small total-time difference primarily reflects settling and does
not demonstrate generally better route execution or RGB forecast accuracy.

Twenty of 36 assignments are evaluated, all verified round trips with zero
contacts. After physical/treatment, paired and terminal analysis, this completed
pair's depth was retired under policy, retaining all non-depth records and the
selected first-seed depth population. Assignment twenty-one, full-input
supervised rollout seed 2026091401 on maze 0, launched next. Models, controller
settings and fixed assignment order remain unchanged.

Assignment twenty-one, full-input supervised rollout seed 2026091401 on maze 0,
verified goal/home at frames 1,228/1,900: 1,902 poses, zero contacts, maximum
position error 6.607 mm, 452/459 plans on time and 190.40 simulated seconds.
Owner exit 0 after 4:37.17 with no swaps; physical arrival and actual neural
input evaluations are saved. Peak simulator lag was 447.317 ms.

Assignment twenty-two, its no-RGB counterpart, terminated with a tracking
fault after 1,015 acquired frames and 5,071 requested steps. The primary fault
is `measured visual pose unavailable`; mapping/planning subsequently report
closure of the simulation clock. The last progress record was outbound at
100.02 simulated seconds. This is a failed assignment, retained in the fixed
cohort. Completed physical evaluation confirms no arrivals, zero contacts,
1,013 registered poses and maximum position error 22.522 mm. Actual neural
input/model/correction treatment checks passed. Detailed failure diagnosis
remains pending; the fault alone does not establish why tracking failed.

The supervised seed-1401 maze-0 pair comparison retains 175 identical common
sources and inspected PNG/SVG trajectory figures. Full input travelled 18.059 m
and completed the round trip; no-RGB travelled 4.342 m with no arrival. Full
input's final-10cm approaches took 2.3 s outbound and 19.8 s on return, including
17.3 s of zero commands on return. The failed arm never entered either terminal
region, so its failure is not a final-arrival settling failure.

`tracking_failure_leadup_v1.json` records 252 no-RGB plans, 250 on time. Selected
actions were 202 pure turns, 43 translating actions and seven holds. There were
54 changes from the recorded preferred action; 51 preferred candidates were
recorded blocked. Frames 1,004/1,008 selected left turns while the preferred
right turn was blocked by predicted clearance. Every recorded registered pose
through frame 1,012 had an accepted floor; no floor-reacquisition hold was
recorded. This points to raw tracking rather than the downstream missing-floor
recovery path, but the precise rejection requires delivered-sensor replay.
Preserve this failure's full depth. Assignment twenty-three, the same no-RGB
model on maze 1, is running with the unchanged frozen controller.

Assignment twenty-three completed and physically verified goal/home at frames
945/1,543: 1,545 poses, zero contacts, maximum position error 4.353 mm,
366/379 plans on time and 154.88 simulated seconds. Owner exit 0 after 3:58.81
with no swaps. Peak simulator lag was 546.537 ms. Actual input and model/fit
checks passed. The raw-tracker replay of assignment twenty-two runs between
native owners; no controller or model changes are made to this cohort.

The exact delivered-sensor raw replay of assignment twenty-two completed in
104.84 s and matched all 1,013 recorded raw poses before reproducing failure at
frame 1,013. Both camera depth digests were checked for every replay packet.
`raw_tracking_failure_replay_v1/terminal_raw_snapshot.json` identifies
`ANCHOR_INCREMENT_CONFLICT`: current RGB-D fits against reference frame 484
(52.9 s old) and frame 1,012 (0.1 s old) disagree by 22.2497 mm against the
unchanged 20-mm limit. Rotation disagreement is negligible. Both fits passed
their individual gates: 17/57 inliers and 2.997/2.265-mm residual RMS for
anchor/increment. The primary conflict is terminal; auxiliary tracking was
not attempted. This is not absence of all current image correspondences or a
downstream floor-reacquisition failure.

Only after replay finished, evaluator-only physics measured the two rejected
hypotheses at that frame: the older-anchor estimate has 3.642-mm 3D error;
the recent incremental estimate has 21.290-mm error. The latter inherits the
previous accepted trajectory's drift. Thus more inliers, a lower pair residual
or simply preferring recent frames would not select the more accurate estimate
in this example. `conflict_native_accuracy_diagnostic_v1.json` preserves both
hypotheses and errors. No native truth enters tracking and no threshold is
relaxed. An observation-grounded conflict-resolution/relocalization successor
would require separate prospective validation; this failed assignment and its
depth remain retained. Assignment twenty-four, full-input supervised rollout
on maze 1, launched next with the frozen controller.

The recorded raw-pose accuracy trace confirms that this disagreement followed
accumulation of position error: 0.725 mm at frame 600, 4.969 mm at 700,
11.299 mm at 800, 15.088 mm at 900 and 20.763 mm at 1,000. The first recorded
raw error above 20 mm is frame 960, before the terminal conflict. Reference
frame 484 itself had 4.555-mm raw error. These evaluator-only values are saved
in `recorded_raw_drift_diagnostic_v1.json`; they do not grant an online accuracy
bound or justify accepting every revisited anchor.

Assignment twenty-four, full-input supervised rollout seed 2026091401 on maze 1,
verified goal/home at frames 904/1,505: 1,507 poses, zero contacts, maximum
position error 4.838 mm, 363/370 plans on time and 150.82 simulated seconds.
Owner exit 0 after 3:53.59 with no swaps; actual treatment checks passed.
Peak simulator lag was 601.098 ms. The pair has 175 identical common sources
and inspected PNG/SVG figures. Full/no-RGB paths were 18.548/18.528 m.
Full was 4.06 s faster. Final-10cm approaches were 1.6/2.4 s outbound and
2.0/2.5 s on return, without leaving the region. The terminal difference is
1.30 s, leaving 2.76 s outside those windows. This small descriptive gap is
not evidence of broadly superior RGB prediction.

## Complete second-seed comparison

All twelve fixed assignments for seed 2026091401 are evaluated. Full input
completed six of six round trips; no-RGB completed five of six, with the raw
tracking conflict above retained as the failure. All twelve had zero contacts.

| Method | Maze | Full outcome/time (s) | No-RGB outcome/time (s) |
| --- | --- | --- | --- |
| JEPA | 0 | Round trip, 273.12 | Round trip, 242.84 |
| JEPA | 1 | Round trip, 167.76 | Round trip, 261.12 |
| Direct | 0 | Round trip, 242.08 | Round trip, 250.98 |
| Direct | 1 | Round trip, 154.02 | Round trip, 157.32 |
| Supervised rollout | 0 | Round trip, 190.40 | Tracking conflict; no arrival |
| Supervised rollout | 1 | Round trip, 150.82 | Round trip, 154.88 |

Full was faster in four of the five pairs where both completed; no completion
time is assigned to the failed mission. Combined across the first two seeds,
full is 12/12 and no-RGB 11/12. These are repeated runs on only two independent
maze layouts, not 24 environmental replications. Models, corrections, controller
and assignment order remained frozen; no success-rate or JEPA-superiority
claim follows from this pilot.

`go2_neural_rgb_transfer_seed_2026091401_complete_comparison_v1_attempt_001/result.json`
retains all twelve rows, six pair comparisons, timing/terminal differences and
executed-forecast summaries, including the failed trajectory. Corrected XY
endpoint RMSE ranges from 5.275 to 8.741 mm across these different trajectories.
The command-yaw baseline again has lower endpoint error than learned yaw in
all twelve runs. These overlapping 700-ms executed-request windows do not
evaluate unexecuted candidates or establish an alternative navigation outcome.
The third seed started with assignment twenty-five, full-input JEPA on maze 0.

Assignment twenty-five verified goal/home at frames 1,164/1,663: 1,665 poses,
zero contacts, maximum position error 5.831 mm, 399/408 plans on time and
166.96 simulated seconds. Owner exit 0 after 4:05.43 with no swaps. Actual
neural input and model/correction checks passed; peak simulator lag was
457.825 ms. Its no-RGB counterpart launched as assignment twenty-six.
The completed second-seed supervised maze-1 pair's depth was retired after
comparison, keeping all non-depth records, the selected first-seed references
and the second-seed maze-0 failure/comparator depth.

Assignment twenty-six, no-RGB JEPA seed 2026091402 on maze 0, verified goal/home
at frames 1,714/2,323: 2,325 poses, zero contacts, maximum position error
6.923 mm, 558/564 plans on time and 232.88 simulated seconds. Owner exit 0
after 5:35.97 with no swaps; actual model/input/fit checks passed. Peak simulator
lag was 561.034 ms. Its full-input comparison has 175 identical common sources
and inspected PNG/SVG figures. Full/no-RGB paths were 16.542/18.010 m.

Full was 65.92 s faster, but 58.60 s of the gap occurred after first entering
the final 10 cm around goal/home. Full/no-RGB terminal windows were 1.6/46.2 s
outbound and 1.6/15.6 s on return. Neither left the region. Outside those
windows, no-RGB was 7.32 s slower. Its outbound terminal commands included
43.56 s of zero command; 95 of 107 selected plans were holds, all explicitly
selected by the predicted-quiet-arrival override. Of the 107 plans, 105 were
on time. These records are in `terminal_selection_diagnostic_v1.json`.

All 95 terminal-hold plans have an exactly matched requested prefix through
700 ms in the saved executed-forecast evaluation. They predict target distance
14.743–19.912 mm at that horizon. Actual body-frame displacement relative to
the original planned waypoint gives 17.329–22.551 mm: 17 of 95 lie outside
20 mm. However, 94 of 95 later mission observations remain outside 20 mm.
The 800-ms endpoint is not evaluated by this diagnostic. These overlapping
windows do not establish an alternative action's outcome.

The discrepancy exposes a coordinate-metric issue, not just forecast error.
At the same current registered pose, 85 of 95 plans have recorded planning
waypoint distance <=20 mm but mission-observed goal distance >20 mm. The
planning waypoint differs by 2.997–3.137 mm from the mission's initial-frame
horizontal goal projected into body XY. At frame 1,512 the three distances
are 18.311 mm (planned waypoint), 20.539 mm (initial-frame horizontal goal
projected into body XY) and 20.584 mm (mission observed distance). The runtime
plans in its floor-aligned map then projects to body XY, whereas mission
arrival uses initial-frame XY. A small projection difference can therefore
sustain holding at this strict arrival boundary. The comparison retains
`predicted_arrival_hold_diagnostic_v1.json` and
`terminal_coordinate_metric_diagnostic_v1.json`.

Do not repair this inside the frozen cohort or reinterpret all 65.92 s as an
RGB forecast benefit. After the fixed roster, a prospective correction should
make the planner's final-position objective and mission arrival use a consistent
coordinate metric, while preserving the physical arrival requirement and
reporting every previous result. Keep this pair's depth for the newly identified
issue. Assignment twenty-seven, no-RGB JEPA on maze 1, is running unchanged.

Assignment twenty-seven verified goal/home at frames 1,140/1,656: 1,658 poses,
zero contacts, maximum position error 6.397 mm, 398/407 plans on time and
165.96 simulated seconds. Owner exit 0 after 4:15.64 with no swaps; actual
input/model/correction checks passed. Peak simulator lag was 546.339 ms.
After retiring the reviewed old stationary no-RGB supervised adapter's depth
under policy, assignment twenty-eight, full-input JEPA on maze 1, launched.
Keep the current third-seed maze-0 pair in full for its terminal-coordinate
diagnosis; the second-seed tracking failure and its comparator also remain.

The later coordinate calculation reconstructs all 95 original terminal-hold
waypoints exactly from the public initial gravity vector, recorded registered
pose and mission target. Under the existing planar-motion approximation,
the target itself misses the mission's initial XY by 2.997–3.182 mm. Let
`B` map initial axes to the gravity-aligned map, `R` map current body to initial
axes, and `Q = B @ R`. A predicted body-XY displacement maps to mission XY as
`A = R[:2,:2] - outer(R[:2,2], Q[2,:2] / Q[2,2])`, assuming zero displacement
along the map's gravity axis. Using `current_initial_xy + A @ predicted_xy`
against the unchanged mission goal makes only 44 of the original 95 hold
forecasts pass both terminal-distance and speed tests. Thus 51 decisions would
no longer qualify as predicted arrivals under this consistent metric. No
alternate action or following observation is inferred. Vertical body motion
is not predicted by the existing XY model, so that approximation and the
remaining 44 holds still require examination. The saved
`terminal_coordinate_projection_analysis_v1.json` uses no native truth, changes
no forecast or controller, and verifies the corrected target identity to
5.3e-18 m. A future controller test must retain independent physical arrivals.

An evaluator-only check of the constant-gravity-height approximation on these
95 matched hold windows finds median/max XY projection error 0.00665/0.2042 mm
against recorded native displacement. This supports the approximation for
these held-action windows only, not for all translation/turning dynamics.
Of the 44 holds still eligible under the consistent projected metric, 43 have
the subsequent observed mission distance outside 20 mm. Coordinate consistency
is therefore a specific correction, not a demonstrated complete solution for
terminal prediction bias or closed-loop settling. The comparison retains
`terminal_planar_assumption_diagnostic_v1.json`.

Assignment twenty-eight, full-input JEPA seed 2026091402 on maze 1, verified
goal/home at frames 1,620/2,231: 2,233 poses, zero contacts, maximum position
error 6.737 mm, 530/540 plans on time and 223.78 simulated seconds. Owner exit
0 after 5:36.74 with no swaps; actual input/model/fit checks passed. Peak
simulator lag was 627.790 ms. The paired report has 175 identical common sources
and inspected PNG/SVG figures; full/no-RGB paths were 19.858/18.733 m.

No-RGB was 57.82 s faster on maze 1, opposite this seed's maze-0 result. Full/
no-RGB final-10cm windows were 70.9/14.2 s outbound and 9.0/1.7 s on return,
without leaving the region. Full spent 64.0 s more in terminal windows and
6.18 s less outside them. Full's outbound terminal commands included 63.60 s
of pure turning, versus 8.82 s for no-RGB. Of 173 full terminal plans, 170 were
on time and 163 selected right turn. The terminal-position override was
considered/changed on 172 plans; the later arrival-entry rule restored heading
guidance on 167. No-RGB had 33 outbound terminal plans, all on time, 22 pure
turn selections and 24 heading restorations. These records are saved in
`terminal_selection_diagnostic_v1.json`. This is evidence of terminal objective/
arrival-entry interactions, not a demonstration of generally worse RGB route
prediction. The selected original forecasts and outcomes remain unchanged.

All twelve JEPA assignments across three seeds and both mazes completed round
trips, with both neural input treatments 6/6. Full was faster in four of six
paired completions, but effects reverse by maze/seed and several are dominated
by terminal windows. There are still only two environmental units. Complete
the remaining eight direct/supervised assignments before concluding the pilot.
Assignment twenty-nine, full-input direct prediction on maze 0, launched next.

A post-hoc census includes all first 28 assignments, including the failure:
`go2_neural_rgb_terminal_metric_first28_diagnostic_v1_attempt_001/result.json`.
Twenty runs contain 162 predicted-quiet-arrival hold decisions in total. Three
runs have a planning waypoint already inside 20 mm while the mission's current
observed distance remains outside; five runs have a total of 69 hold decisions
whose existing forecasts fail the consistent mission-coordinate distance test.
All original planning targets reconstruct exactly from recorded public gravity,
pose and mission information. The changes comprise one full-direct seed-1001
maze-0 hold, 14 no-RGB JEPA seed-1401 maze-0 holds, two full-JEPA seed-1401 maze-1
holds, one full-supervised seed-1401 maze-0 hold, and 51 no-RGB JEPA seed-1402
maze-0 holds. This is a shared coordinate issue with uneven exposure across
trajectories, not proof of improved future navigation or a general RGB effect.
The census reads no native truth or raw depth and changes no forecast/runtime.

Assignment twenty-nine, full-input direct prediction seed 2026091402 on maze 0,
verified goal/home at frames 1,476/1,959: 1,961 poses, zero contacts, maximum
position error 5.491 mm, 475/482 plans on time and 196.52 simulated seconds.
Owner exit 0 after 4:45.26 with no swaps. Actual neural input and model/fit
checks passed; peak simulator lag was 433.389 ms. The completed third-seed
JEPA maze-1 pair's depth was retired after paired and terminal analysis, with
all non-depth records retained. Assignment thirty, no-RGB direct on maze 0,
launched next with unchanged settings.

Assignment thirty, no-RGB direct prediction seed 2026091402 on maze 0,
verified goal/home at frames 2,439/2,923: 2,925 poses, zero contacts, maximum
position error 11.251 mm, 713/724 plans on time and 292.98 simulated seconds.
The native owner is no longer running and its completed result and physical
evaluation are retained; its terminal process exit code was not recovered.
Actual neural input and model/fit checks passed. Peak simulator lag was
601.058 ms. Full RGB completed this pair 96.46 s sooner, with almost all of
the difference outbound; paired action/terminal attribution remains to do.
Thirty assignments are now evaluated: 29 round trips and the retained
second-seed no-RGB supervised tracking failure. Assignment thirty-one,
no-RGB direct on maze 1, launched with unchanged settings. Six assignments
remain including that active run; these still represent only two mazes.

The completed third-seed direct maze-0 comparison retains 175 identical
common sources and inspected trajectory figures. Full/no-RGB paths were
17.696/19.575 m. Terminal windows were 1.6/1.6 s outbound and 2.2/1.7 s
on return, so final stopping does not explain the 96.46 s full-input advantage.
Outbound command durations were 52.00/56.72 s translation, 88.02/140.90 s
pure turning, and 7.58/46.28 s zero command. Return durations were almost
identical, 48.2/48.3 s from the first return observation to arrival.

The stopping-projection selector changed five full-input plans (all on time)
and 113 no-RGB plans (111 on time). In no-RGB, 96 changes replaced translations
with hold: 71 right arcs, 12 forward commands and 13 left arcs; full had no
such hold replacements. At frame 1,436, for example, the preferred no-RGB
forward forecast had 0.49948 m minimum path clearance, above the 0.48 m
reserved requirement, but the added stopping projection reached only
0.39369 m, below its 0.45 m requirement, and selected hold. The guard uses
the forecast and recorded map; this does not establish that the rejected
action was physically safe, or what disabling that guard would do. The
paired report retains `phase_command_selection_diagnostic_v1.json` and
`stopping_projection_selection_diagnostic_v1.json`. This difference occurs
during exploration, unlike several earlier terminal-dominated pairs.

Assignment thirty-one, no-RGB direct seed 2026091402 on maze 1, verified
goal/home at frames 1,004/1,563: 1,565 poses, zero contacts, maximum position
error 6.801 mm, 374/383 plans on time and 156.86 simulated seconds. Owner
exit 0 after 4:01.91 with no swaps. Actual neural input/model/fit checks
passed; peak simulator lag was 535.369 ms. The completed third-seed direct
maze-0 pair's depth was retired under policy, preserving all non-depth
evidence. Assignment thirty-two, full-input direct on maze 1, launched next.

Assignment thirty-two verified goal/home at frames 1,084/1,596: 1,598 poses,
zero contacts, maximum position error 4.969 mm, 377/385 plans on time and
159.92 simulated seconds. Owner exit 0 after 4:06.32 with no swaps; actual
input/model/fit checks passed. Peak simulator lag was 543.531 ms. All twelve
direct-prediction assignments now completed round trips, as did all twelve
JEPA assignments; no training-method success advantage is established.
Assignment thirty-three, full-input supervised rollout on maze 0, launched
next. Four supervised-rollout assignments remain, including that active run.

The third-seed direct maze-1 pair has 175 identical common sources and
inspected figures. Full/no-RGB paths were 18.767/18.752 m, with full 3.06 s
slower overall. Its terminal windows were 18.2/1.6 s outbound and 2.6/1.5 s
on return, an excess of 17.7 s; outside those windows full was 14.64 s faster.
Full's outbound terminal window included 8.82 s pure turn and 8.08 s zero
commands. Of 36 selected plans, 34 were on time and 26 were pure turns;
18 terminal translation overrides were reversed by the arrival-entry rule.
No-RGB had one terminal selected hold plan per phase, both on time. Saved
terminal approach and selection diagnostics retain this distinction. All
twelve direct runs succeeded, full was faster in four of six input pairs,
and the effects again depend on maze/seed and terminal behavior; no independent
replication or training-method superiority follows from those counts.

Assignment thirty-three, full-input supervised rollout seed 2026091402 on
maze 0, verified goal/home at frames 1,516/1,997: 1,999 poses, zero contacts,
maximum position error 12.184 mm, 488/492 plans on time and 200.06 simulated
seconds. Owner exit 0 after 4:50.29 with no swaps. Actual input/model/fit
checks passed; peak simulator lag was 417.128 ms. Assignment thirty-four,
the matched no-RGB condition, launched next. The complete-pilot descriptive
summary script was checked against the first 24 evaluated assignments,
reproducing all 23 successes, the explicit failure, zero contacts and nine
full-faster comparisons among eleven pairs with two successful missions.

Prospective correction preparation, not active in this cohort:
`lewm/mission_coordinate_metric_development.py` implements the previously
derived body-XY to initial-XY planar displacement matrix from measured
orientations. Four focused tests pass: level-heading distance preservation,
agreement with an independent 3D plane-intersection solve under tilt,
reproduction/correction of the false-inside-arrival-radius geometry, and
rejection when body XY cannot parameterize a gravity-horizontal plane.
No current runtime imports this module. Integration must consistently use
the mission metric for terminal position scores, predicted hold distances
and the arrival-entry gate; merely changing the waypoint does not fix the
distance metric. Existing forecasts, clearance checks, measured settling and
physical arrival criteria must remain explicit. Navigation improvement and
the residual terminal forecast bias remain untested.

Assignment thirty-four, no-RGB supervised rollout seed 2026091402 on maze 0,
verified goal/home at frames 1,247/1,733: 1,735 poses, zero contacts, maximum
position error 6.178 mm, 420/428 plans on time and 173.62 simulated seconds.
Owner exit 0 after 4:13.91 with no swaps. Actual input/model/fit checks passed;
peak simulator lag was 417.022 ms. The completed third-seed direct maze-1
depth was retired, reclaiming 1.899 GB while preserving all non-depth records.
Assignment thirty-five, no-RGB supervised rollout on maze 1, launched next.

The third-seed supervised maze-0 pair retains 175 identical common sources
and inspected trajectory figures. Full/no-RGB paths were 17.468/17.490 m.
No-RGB finished 26.44 s sooner, with terminal windows almost equal:
1.8/1.9 s outbound and 1.7/1.7 s return. Outbound command durations were
49.20/52.24 s translation, 95.12/63.90 s pure turning and 7.28/8.56 s zero;
full's 31.22 s additional turning explains most of its 26.9 s outbound delay.
There were 241/162 outbound pure-turn selections and 371/375 versus 303/309
plans on time. Stopping-projection changes were seven/nine outbound and
two/three on return, unlike the preceding no-RGB direct maze-0 pair's many
holds. The saved `phase_command_selection_diagnostic_v1.json` keeps the
observed action differences without inferring unexecuted alternatives.

Assignment thirty-five, no-RGB supervised rollout seed 2026091402 on maze 1,
verified goal/home at frames 998/1,574: 1,576 poses, zero contacts, maximum
position error 5.805 mm, 375/382 plans on time and 157.76 simulated seconds.
Owner exit 0 after 4:03.83 with no swaps; actual input/model/fit checks passed.
Peak simulator lag was 498.529 ms. Assignment thirty-six, full-input
supervised rollout on maze 1, launched as the final fixed assignment.

Assignment thirty-six verified goal/home at frames 936/1,527: 1,530 poses,
zero contacts, maximum position error 4.389 mm, 364/374 plans on time and
153.36 simulated seconds. Owner exit 0 after 3:56.71 with no swaps; actual
input/model/fit checks passed. Peak simulator lag was 531.241 ms. The final
paired report has 175 identical common sources and inspected figures.
Full/no-RGB paths were 18.449/18.615 m; full finished 4.40 s sooner.
Terminal windows were 1.7/9.2 s outbound and 2.5/3.6 s return: no-RGB spent
8.6 s more inside those windows and 4.2 s less outside. The complete pilot
and third-seed aggregate are saved. No native owner remains active.

The completed third-seed supervised depth was retired under policy after
the pair and complete-pilot analyses, reclaiming 4,186,632,192 allocated
bytes while all 124 preserved JSON hashes remained unchanged. Selected
reference depth and the unresolved/current tracking and coordinate cases
remain retained. Free space is 8,393,068,544 bytes. Continue with the separate
mission-coordinate follow-up; the larger navigation goal remains active.
