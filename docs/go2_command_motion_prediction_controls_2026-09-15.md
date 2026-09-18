# Separating neural motion forecasts from pose and command prediction

The current sixteen native assignments remain unchanged. The analyses below
prepare a separate future control; none of these predictors is used by a
current native owner, and no current model or residual correction is refitted.

An unfitted body-twist integrator uses the exact recorded three-command prefix,
four candidate intervals (or the existing single translation pulse), and zero
tail. Exact constant-twist arc integration produces XY/yaw at 100-ms endpoints.
It estimates neither current pose nor contact probability. Two focused tests
verify pulse/tail timing and a rotating common prefix followed by an exact arc.
Source: `lewm/commanded_planar_motion_development.py`.

`scripts/compare_saved_command_motion_forecasts_development.py` compares only
the same saved windows whose complete requested 700-ms sequence occurred.
The original physics-evaluated endpoints are reused. Four-layout pooled XY
endpoint RMSE (mm) in the completed preceding training comparison:

| Trajectory condition | Windows | Command integration | Raw neural | Corrected neural |
| --- | ---: | ---: | ---: | ---: |
| JEPA | 2386 | 20.714 | 20.024 | 7.970 |
| Direct | 4327 | 18.225 | 25.054 | 9.898 |
| Supervised rollout | 3611 | 20.879 | 21.200 | 10.010 |

Each row compares identical windows within that trajectory population. Across
rows, actions, lengths and hold fractions differ; these pooled values cannot
rank methods on identical inputs. Windows overlap. The raw neural XY forecast
alone is close to, or worse than, simple command integration here, whereas the
corrected prediction improves substantially. That motivates a fitted control
which has the same causal pose/command information as the residual model.

## Fitted pose-and-command control

`scripts/fit_pose_command_motion_control_development.py` uses the original
four residual-training datasets, the unchanged validation dataset, masks,
stationary thinning and ridge penalty 1. It drops precisely feature columns
12–15 (neural XY and yaw), retains the 12 past-visual-pose and 26 known-command/
nominal-motion features, and directly predicts XY at each horizon. Targets are
the original subsequent registered visual pose deltas; no native truth is used
for fitting. The eight-horizon coefficients were saved before validation was
read. This is a separate fit for future use, not a modification of the current
sixteen-assignment cohort's frozen corrections.

Output: `go2_pose_command_motion_control_fit_v1_attempt_001/motion_fit.npz`.
SHA-256: `bc0c215b1a4ca171d63de7eb7be30f057dc4e6a97dc4ec0ea9f7ee4fb3bf00d6`.
The original development validation's 119 moving 700-ms windows give XY RMSE
8.225 mm for pose/command versus 8.033 mm for corrected JEPA. Across all 436
valid 700-ms windows the values are 5.619 versus 5.454 mm. These small differences
do not establish equivalence or statistical advantage.

`scripts/evaluate_pose_command_motion_control_development.py` then evaluates
the fixed control on the two completed new-layout JEPA recordings, using only
four available visual poses and the known action sequence as predictor inputs:

| New recorded layout | Identical windows | Pose/command XY RMSE | Corrected JEPA XY RMSE |
| --- | ---: | ---: | ---: |
| 0 | 635 | 8.886 mm | 8.611 mm |
| 1 | 647 | 7.981 mm | 7.654 mm |

These are retrospective development comparisons on JEPA-selected actions, not
live alternative-controller outcomes or untouched final evaluation. They do
not compare yaw/contact forecasts. Native endpoints are evaluator-only, and
future visual poses never enter the prediction features.

The next useful causal intervention, after the fixed native cohort, is a live
control preserving the same perception, map, candidate actions, commitment
ledger, routing, scoring and recovery implementation. Replacing only forecast
XY with this fitted control would isolate the incremental role of neural XY
prediction while retaining learned yaw/contact. A fully non-neural prediction
control additionally requires explicit yaw/contact treatment; the current
pose/command fit must not be mislabeled a complete model-free planner.

Keep this fit and its original small cached training/validation NPZ inputs.
All original native study results, including failures, remain unchanged.

## Runtime intervention prepared; no live use yet

`lewm/pose_command_xy_control_development.py` implements two fixed XY-source
choices, `learned` and `pose_command`, as a mixin for the existing predictive
runtime. Both choices compute the original neural forecasts/residual correction
and the pose/command alternative; the selected XY array alone changes what
downstream scoring, clearance, arrival-hold and recovery code consumes. Learned
yaw/contact channels remain exactly unchanged. The actual applied fit identity,
both alternative XY arrays and the original correction receipt are recorded.
It preserves the current commanded-prefix and terminal-pulse schedule.

A focused property test shows that changing neural XY cannot change the
pose-command arm's returned forecast, while yaw/contact and the input arrays
are preserved; the learned choice reproduces its input exactly. The loaded
runtime predictor also reproduced the previously saved numerical readouts
exactly for ordinary, translation-pulse and final sampled windows on both new
JEPA recordings. There is no new native launcher or live outcome for this
intervention yet, and no runtime in the fixed sixteen-assignment cohort imports
or selects it. The next experiment must label this as an XY-source ablation,
not a fully non-predictive or model-free controller.

## Paired live launcher prepared

`scripts/run_go2_pose_command_xy_ablation_development.py` now provides
`--layout-index 0..3 --xy-source learned|pose_command`. It uses the same frozen
JEPA model, original correction, coherent perception, map, memory, actions,
commitment timing, recovery and arrival logic as the current cohort. Both
alternatives are computed in both arms. Only the selected XY forecast changes;
the actual fit and both alternatives are recorded by the runtime mixin.

Plan eight assignments after all sixteen current assignments finish: both XY
sources on each of the same four now-exposed development layouts, keeping both
outcomes and every failure. The even and odd CPU groups retain their existing
layout split; run learned then pose_command within each layout. These are paired
development interventions, not another claim of unseen-layout evaluation.
No new fit or per-layout tuning is planned. Retain the current sixteen-trial
population and check available space before beginning this additional batch.

Both launch-writer assignments were checked without native execution or file
creation: fixed JEPA model/correction, coherent tracker, explicit XY source,
fixed control-fit hash and previously-exposed layout labeling all matched.
The first writer-only invocation lacked its required CPU affinity and stopped;
the same check with the prescribed affinity passed for both arms. The focused
XY-channel property test passed. No live ablation has started yet.

## Same-window evaluation now includes all four JEPA trajectories

The fixed pose/command control was evaluated on the remaining two completed
JEPA recordings, preserving both return-phase speed-stop failures. No fitting
or native execution occurred. Layout 2 has 611 executed windows: endpoint XY
RMSE 9.505 mm for pose/command and 9.101 mm for corrected JEPA. Layout 3 has
360 windows: 8.493 versus 8.190 mm. Unfitted command integration gives
22.018/22.152 mm on these two respective populations.

Pooling all four trajectories gives 2,253 overlapping executed windows and
endpoint XY RMSE 8.751 mm for pose/command versus 8.420 mm for corrected JEPA.
The control's maximum endpoint error is 27.764 mm. The small aggregate gap
continues to motivate the live intervention; it is not an equivalence result,
an independent-window significance test or a closed-loop outcome. These
recorded trajectories were generated by JEPA, and the control's yaw/contact
behavior remains untested. All per-window readouts are retained in the four
run roots, including the stopped trajectories.

## Existing spatial-memory evidence limits what needs repeating

The preceding completed memory experiment already supplies a matched routing
intervention: four persistent-memory round trips versus zero latest-mapped-pair
goals or round trips, with zero disallowed contacts in all eight assignments.
Its authoritative aggregate was inspected at
`go2_routing_memory_four_layout_summary_v1_attempt_001/result.json`; the study
and limitations are recorded in
`docs/go2_current_pair_routing_memory_preparation_2026-09-14.md`.
Both arms retained obstacle history for action clearance and other controller
state. This supports accumulated spatial routing evidence on those development
revisits, not a fully memoryless or model-internal-memory claim, and it does not
isolate memory specifically needed for return because the reduced arm reached
no goals. The next live prediction intervention should address the remaining
prediction question rather than repeat that completed routing-memory test.

The same fixed pose/command predictor was also evaluated on all four completed
direct trajectories and supervised layouts 0, 1 and 3. Per-run results remain
in their ordinary run roots; no fit or runtime changed. Direct's 3,778 executed
windows give pooled XY endpoint RMSE 9.769 mm for pose/command versus 10.343 mm
for corrected direct prediction; the simpler predictor has lower RMSE on each
of those four recorded trajectories. The three supervised trajectories give
2,416 windows and 10.781 versus 10.654 mm. Supervised layout 2 is still pending
at this entry and is not included in that partial aggregate. These remain
same-window, trajectory-conditional analyses, with untested alternative action
choices and no new yaw/contact evidence.

Supervised layout 2 subsequently completed and was included without changing
the fit. Its 985 executed windows give XY RMSE 11.076 mm for pose/command and
10.978 mm for corrected supervised prediction. Complete four-layout pooled
same-window results are now:

| Recorded controller | Windows | Pose/command XY RMSE | Corrected neural XY RMSE |
| --- | ---: | ---: | ---: |
| JEPA | 2253 | 8.751 mm | 8.420 mm |
| Direct | 3778 | 9.769 mm | 10.343 mm |
| Supervised rollout | 3401 | 10.867 mm | 10.749 mm |

Rows have different executed actions, durations and hold fractions; compare
predictors within a row, not the controller labels across rows. Failed and
budget-truncated trajectories remain included. The numerical gaps do not
establish navigation equivalence or a causal benefit of neural forecasting.

## Final live design: supervised reference selected before any ablation run

The fixed sixteen-assignment comparison is complete. Supervised rollout reached
and returned from all four mazes; direct returned on three, JEPA on two, and
reactive on two. Select the supervised model as the reference for the next
eight assignments to test the contribution of learned XY forecasting in the
strongest observed complete controller. This replaces the earlier unexecuted
JEPA-reference proposal. The choice uses development outcomes and is not a
blind model-selection claim; none of the original sixteen assignments changes.

`scripts/run_go2_pose_command_xy_ablation_development.py` now fixes
`BASE_CONDITION='supervised_rollout'`. Both arms retain frozen training seed
2026091001, supervised model and original supervised correction, learned
yaw/contact channels, coherent perception, map/memory, actions, timing, recovery
and all physical/arrival thresholds. Both compute the neural and pose/command
XY alternatives before selecting one. The pose/command fit remains exactly
`bc0c215b1a4ca171d63de7eb7be30f057dc4e6a97dc4ec0ea9f7ee4fb3bf00d6`;
the reference supervised residual remains
`413a3160c21dc43b4b8a1b87b4c092d59428158270d7b91179229d84b04fda1d`.
Both writer configurations were checked with the prescribed affinity and
correctly bind the supervised assignment, both fits and exposed-layout scope.
The existing runtime XY-channel property/numerical checks remain applicable;
the intervention implementation did not change with this reference selection.

Freeze all eight assignments now: learned and pose_command XY on each current
layout 0–3, with every success or failure retained and no within-study fitting,
tuning or replacement. Use the same even/odd CPU groups, layouts 0 then 2 on
even and 1 then 3 on odd; learned then pose_command within each layout. New
root template:
`go2_pose_command_xy_ablation_supervised_rollout_<source>_noise_2mm_native_layoutXX_4800_v1_attempt_001`.
Each pair requires new native runs because both alternatives are computed in
both arms. The original supervised successes are context, not replacements
for the new learned arm. These are previously exposed development layouts.

At preparation, the original owners and targeted replay have exited. About
75 GiB RAM and 26 GiB artifact storage are available. Retain two native owners
at most and monitor storage between pairs; all current sixteen recordings,
model/fit inputs and unresolved failure recordings remain pinned. No whole-tree
export or hardware execution is part of this study.

After independent evaluation of the sixteenth outcome and completion of the
four-layout aggregate, learned-XY reference layouts 0 and 1 launched in
sessions 70329 and 7058. Pose-command XY follows each corresponding completed
reference run, then the two sources on layouts 2/3. No new model or fit is
trained. All eight assignments remain fixed from this first dispatch.


## First two reference outcomes independently evaluated

Both learned-XY owners exited successfully (layout 0: 670.63 s, peak RSS
25,458,768 KiB; layout 1: 503.60 s, peak RSS 18,889,872 KiB; zero swaps each).
All native archive writes finished before evaluation and the paired launches.

Layout 0 reached the goal at frame 3163, but exhausted the mission budget
without a verified return. It recorded 4,805 poses, median/max position error
3.706/7.067 mm, zero disallowed contacts, and final physical home distance
20.416 mm. Being near home does not replace the required observed arrival and
one-second quiet dwell. The terminal return diagnostic records the actual
shortfall; retain this failure in the fixed population. Its 1,159 fully executed
700 ms windows give corrected XY endpoint RMSE 8.826 mm, maximum 37.571 mm,
and five windows with path error above 30 mm.

Layout 1 independently passed both arrivals (goal frame 2103, home 3412),
with 3,414 poses, median/max position error 2.474/6.506 mm, zero disallowed
contacts, and final home distance 18.099 mm. Its 813 fully executed windows
give corrected XY endpoint RMSE 9.783 mm, maximum 42.616 mm, and ten windows
with path error above 30 mm.

The learned reference therefore has one verified return from its first two
assignments. The earlier supervised 4/4 result does not establish repeatability:
these new reference outcomes differ despite the same model and controller,
with both forecast alternatives now computed and asynchronous host execution.
Do not attribute the difference to a particular timing mechanism without an
intervention. No settings or fitted parameters change during the eight runs.

Pose-command layouts 0 and 1 launched on the same respective CPU groups in
sessions 75854 and 47165 after both reference evaluations. Layouts 2 and 3
remain next, learned then pose-command on each. The approved retirement of
seven superseded successful predecessor depth recordings reclaimed 14.51 GiB;
all original sixteen and current eight recordings remain fully retained.


Layout 0's `terminal_return_dwell_diagnostic_v1.json` localizes the failure:
first observed return distance below 40 mm at frame 4709, minimum 18.120 mm,
only two consecutive observations within the 20 mm controller radius, and at
most one quiet interval versus ten required. Across the final 96 mission
observations, 23 selected plans comprise 16 forward, five right-arc and two
left-arc choices; all are on time and all six candidates pass full-reserve
clearance in each plan. This is a near-home settling shortfall, without a
verified causal remedy or any relaxation of the arrival criteria.


Both newly completed reference roots also contain
`saved_xy_alternatives_same_window_evaluation_v1.json`, comparing the two
alternatives actually recorded at planning time on identical fully executed
700 ms windows. Layout 0: learned 8.826 mm versus pose/command 8.873 mm
endpoint XY RMSE (1,159 windows); layout 1: 9.783 versus 9.832 mm (813).
In layout 0's 19 matched late-return windows from frame 4709 onward, the
values are 8.920 versus 9.282 mm. These overlapping, controller-selected
windows show no large aggregate accuracy separation; they do not establish
navigation equivalence or evaluate unexecuted actions. The live paired runs
remain necessary to measure consequences for action selection.


## First two paired interventions complete

Both pose-command owners exited with status zero and complete archives:
layout 0 wall 511.21 s, peak RSS 19,785,384 KiB; layout 1 wall 486.56 s,
peak RSS 18,078,768 KiB; zero swaps in both. Independent evaluations verify
both goal arrivals and returns, with zero disallowed contacts.

| Layout | Learned XY goal / return | Pose-command XY goal / return |
| --- | --- | --- |
| 0 | yes / no (return settling budget) | yes / yes |
| 1 | yes / yes | yes / yes |

Pose-command layout 0: goal frame 2024, home 3612; 3,614 poses;
median/max position error 1.355/3.615 mm; final home distance 16.437 mm.
Its 868 executed 700 ms forecast windows give applied XY endpoint RMSE
9.561 mm, maximum 41.399 mm, with nine path errors above 30 mm.
Pose-command layout 1: goal frame 2088, home 3251; 3,253 poses;
median/max position error 3.411/7.053 mm; final home distance 11.051 mm.
Its 777 executed windows give applied XY endpoint RMSE 9.602 mm,
maximum 41.329 mm, with five path errors above 30 mm.

The paired comparator completed for both layouts. It checked the actual
selected-plan XY against the assigned saved alternative, original supervised
model/correction bindings, fixed control fit, shared settings and common runtime
source hashes. Reports and visually inspected PNG/SVG trajectory comparisons
are in `go2_pose_command_xy_ablation_supervised_rollout_comparison_layout00_v1_attempt_001`
and the corresponding layout01 root. Layout 0's paths cover the same route,
with learned travel 25.7 m / 480.4 s versus pose-command 23.0 m / 361.3 s;
layout 1 shows similar exploration and return paths (20.1 m / 341.3 s versus
20.0 m / 325.2 s). These durations are recorded simulation time, not owner
wall time or real-time qualification.

On the two control-generated trajectories, recorded same-window alternatives
also remain close: layout 0 learned 9.566 versus pose-command 9.561 mm;
layout 1 learned 9.693 versus pose-command 9.602 mm. Saved readouts preserve
individual matched windows. Neither these errors nor two paired outcomes
establish statistical superiority, equivalence or a fully model-free result;
yaw/contact predictions remain learned in both arms.

The remaining learned-reference layouts 2 and 3 launched in sessions 5675
and 72704 on their prescribed even/odd CPU groups after the first paired
reports completed. Their pose-command counterparts follow, completing the
fixed eight with no tuning or retries. The final collector
`scripts/summarize_go2_pose_command_xy_ablation_development.py` is prepared;
it collects all four paired reports and both recorded forecast alternatives
on the same executed windows. Syntax was checked and its same-window metrics
reproduced the prior two independent reference readouts exactly. No incomplete
four-layout aggregate has been written.


## All four learned-XY references evaluated; final two controls running

Learned layout 3 completed first: owner exit zero, wall 434.36 s, peak RSS
16,458,268 KiB and zero swaps. Independent checks verify goal frame 1793 and
home 2908, with 2,910 poses and zero disallowed contacts. Median/max position
error is 9.126/13.718 mm, final home distance 6.638 mm. Both physical arrival
dwells pass despite the larger tracking error; goal dwell maximum physical
distance is 28.567 mm, below the unchanged 40 mm limit. Its 707 executed
700 ms windows give applied XY RMSE 10.745 mm, maximum 36.820 mm and ten
path errors above 30 mm. The layout-3 pose-command counterpart launched on
the odd CPU group in session 71128 after this evaluation.

Learned layout 2 subsequently exited zero: wall 543.34 s, peak RSS
21,054,568 KiB and zero swaps. Independent checks verify goal frame 2211
and home 3876, with 3,878 poses and zero disallowed contacts. Median/max
position error is 4.811/11.587 mm, final home distance 17.826 mm. Its 943
executed windows give applied XY RMSE 10.634 mm, maximum 42.115 mm and
twelve path errors above 30 mm. Its pose-command counterpart launched on
the even CPU group in session 78668 after this evaluation.

The complete learned reference now has four verified goals and three verified
returns from four assignments, preserving layout 0's return-settling timeout.
The two completed controls have two verified goals and returns; the final two
controls are running. Six of eight outcomes are independently evaluated, all
six with zero disallowed contacts. No full-cohort treatment conclusion is made
before the remaining outcomes. Runtime, fits, assignments and thresholds are
unchanged. Available artifact storage was approximately 27 GiB before the
last two control launches.


## Frozen-correction sensitivity while the final controls run

`docs/go2_frozen_motion_correction_xy_sensitivity_2026-09-15.json` records an
algebraic readout of the three already frozen correction fits, with their
existing file hashes. Source inspection confirms both ordinary and pulsed
correction add an affine residual to raw neural XY; features 12:14 are raw
XY and 14:16 are raw yaw sine/cosine. Holding causal poses, commands and raw
yaw fixed, the exact row-vector XY sensitivity is
`I + coefficient[h,12:14,:] / scale[h,12:14,None]`.

At 700 ms the supervised fit has singular values 0.485 and 0.063; JEPA
0.628 and 0.023; direct 0.535 and 0.135. Thus the correction attenuates raw
XY changes, strongly in one direction, but does not algebraically remove the
neural XY input. This is not a total model-influence attribution: the neural
yaw inputs also affect corrected XY, and learned yaw/contact remain in direct
action scoring. No observed forecast distribution, counterfactual navigation
outcome, new fitting or runtime change is inferred from this coefficient
readout. The live intervention remains the test of navigation contribution.


The coefficient sensitivity was additionally checked by a finite 1/2 mm raw
XY perturbation through the actual frozen ordinary and pulse correction
implementations for all three fits, holding other features fixed. The maximum
absolute discrepancy from the affine formula was 1.96e-18 m in float64.
This numerical check created no model or live-runtime change.

Pose-command layout 3 independently passed both arrivals (goal frame 1796,
home 3108) after owner exit zero, wall 462.56 s, peak RSS 17,435,328 KiB,
zero swaps. It recorded 3,110 poses, median/max position error 5.119/10.784 mm,
zero disallowed contacts and final home distance 26.320 mm. Its 741 executed
700 ms windows give applied XY RMSE 9.964 mm, maximum 33.604 mm and six
path errors above 30 mm. The layout-3 paired report passed shared-setting,
source and actual-treatment checks; its PNG/SVG trajectory comparison was
visually inspected. Both XY sources returned on this maze.

The last control, layout 2, subsequently reported a round trip and entered
archive writing. Its independent evaluation remains pending at this entry.
No complete eight-run aggregate is asserted before that check.


## Complete fixed eight-run result

Pose-command layout 2 exited zero, wall 511.31 s, peak RSS 20,013,828 KiB,
zero swaps. Independent checks verify goal frame 2076 and home 3659, with
3,661 poses, median/max position error 1.996/7.988 mm, zero disallowed contacts
and final home distance 19.286 mm. Its 890 executed 700 ms windows give
applied XY RMSE 9.865 mm, maximum 40.990 mm and eight path errors above
30 mm. Its paired report and visually inspected PNG/SVG comparison complete
all four layout comparisons.

The complete aggregate is
`go2_pose_command_xy_ablation_supervised_rollout_four_layout_summary_v1_attempt_001/result.json`.
All eight outcomes, including the learned layout-0 settling failure, are
included. All paired mission/settings and actual selected-plan treatment
bindings passed; 132 common runtime source hashes match across all layouts.

| XY source | Verified goals | Verified round trips | Disallowed contacts |
| --- | ---: | ---: | ---: |
| Learned supervised prediction plus frozen residual | 4/4 | 3/4 | 0 |
| Frozen pose/command predictor | 4/4 | 4/4 | 0 |

There are three paired return successes in common and one pose-command-only
return; no maze returned only with learned XY. Recorded trajectories cover
similar routes. The pose-command arm is faster in simulation time on layouts
0, 1 and 2, while learned is faster on layout 3; layout 0's learned time is
budget-truncated and must not be treated as a successful completion time.

Within the same fully executed 700 ms windows:

| Trajectories generated by | Windows | Learned XY RMSE | Pose-command XY RMSE |
| --- | ---: | ---: | ---: |
| Learned XY arm | 3622 | 9.919 mm | 9.977 mm |
| Pose-command XY arm | 3276 | 9.745 mm | 9.746 mm |

Compare predictors within each row; the two rows have different actions,
hold fractions and durations. Windows overlap. The aggregate uses both
alternatives recorded before action selection, not future information as an
input or recomputed alternative trajectories.

Conclusion: this fixed development intervention shows no navigation advantage
from retaining the learned XY channel in the selected supervised controller.
It demonstrates successful navigation on all four exposed mazes with that
channel replaced by the simpler predictor. Four pairs and one differing
outcome do not establish superiority, equivalence or general reliability.
The reference was selected using preceding development results and these
layouts were already exposed. Learned yaw/contact, shared perception, spatial
memory and predictive path checks remained active in both arms, so this is
not evidence that neural predictions or predictive planning as a whole are
unnecessary. It also supplies no JEPA-specific benefit or hardware claim.

All eight owners have exited, all recordings remain retained, and no native
simulation is running at completion of this study. The overall goal remains
open. The remaining neural yaw/contact contribution, broader maze/task
coverage and realistic sensing/timing are still separate scientific questions.
