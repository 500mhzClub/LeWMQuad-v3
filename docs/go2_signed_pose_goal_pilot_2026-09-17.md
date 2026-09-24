# Cost-only intervention using the existing signed goal readout

Status: **COMPLETE: 0/4 final arrivals, one transient visit, one contact**.
Native sessions 73606, 74627, 73826 and 54811 exited 0; reader 81929 exited 0.
No training occurred and no process remains live. This is not promoted as a
reliable controller.

| Exposed task | Original metric final cm / degrees | Signed pose cost final cm / degrees | Signed-cost contact | Signed-cost final arrival |
|---|---:|---:|---|---|
| fresh_00, left | 3.040 / 0.640 | 3.145 / 1.226 | No | No |
| fresh_01, right | 33.739 / 94.697 | 15.270 / 24.140 | Yes | No |
| fresh_02, left | 8.320 / 7.713 | 8.302 / 14.934 | No | No |
| fresh_03, right | 17.528 / 102.231 | 5.836 / 2.025 | No | No |

The original metric reference has no contacts; signed cost introduces one on
task 01. Signed cost's mean terminal error is 8.14 cm / 10.58 degrees versus
15.66 cm / 51.32 degrees for the reference. These means include a truncated
contact outcome and must not be presented as an overall navigation win.
Both methods remain 0/4 final arrivals. Direct visual feedback's retained
reference is also 0/4, with one contact; it was not rerun here.

Task 00 visits the goal then holds, but drifts outside tolerance. Task 01
initially selects forward rather than the original left-turn and eventually
makes contact during its fifth selected action. Task 02 still fails approach/
heading alignment. Task 03 substantially reduces the old wrong-direction error
but falsely latches arrival at tick 30, ending 5.84 cm from the target. Thus
correcting the scoring direction alone does not resolve forecast error,
collision avoidance or learned arrival recognition.

The controller keeps the encoder, adapted action-conditioned predictor, six
command primitives, applied-command reconstruction, observation schedule,
execution budget and arrival readout/rule unchanged. It replaces the learned
64-dimensional goal embedding distance with the sum of squared signed goal
readout estimates in the original 3-cm / 5-degree units. Both predicted and
observed feature inputs use the same already-fitted signed readout as direct
feedback. The inherited planner takes a mean over three components, which
has the same argmin; reported costs multiply by three to match the diagnostic.

Initial eleven-frame RGB prefixes exactly reproduce all four corresponding
reference runs. Task 01's first cost vector matches the preceding fixed
diagnostic within floating-point tolerance and selects the same forward action.
The original predictor and scalar-cost trials remain retained. These four
tasks informed the intervention; this is prospective execution on exposed
tasks, not independent confirmation or a JEPA-training advantage.

Runs used two CPU groups (4-7, 8-11), the shared R9700 and RGB-only recording.
Wall times were 29.81, 17.83, 36.31 and 34.42 seconds. The contact result is a
scientific failure with full prospectively specified recording, not an
infrastructure retry. Only unused depth from the completed, physically
duplicate old direct-feedback success was retired before launch; all failures
and non-depth evidence remain. The root output volume ended with about
683 MiB free, above the 512-MiB reserve.

Next address the measured goal-supervision gap, not another threshold change.
`go2_goal_metric_turn_separation_2026-09-17.md` shows that the original metric
fits within-trajectory turn distances while severely underestimating the
unsupervised distance between opposing trajectories, even on training images.
Test cross-trajectory training pairs using existing training-only RGB and
native pose labels, keeping encoder/predictor and fit budget fixed. Any such
fit still requires prospective navigation evidence; the present failure is
not erased by a component improvement.

Plan: `go2_signed_pose_goal_pilot_plan_2026-09-17.json`.
Results: `go2_signed_pose_goal_pilot_result_2026-09-17.json`.
Controller: `lewm/signed_pose_goal_control_development.py`.
Runner/reader: `scripts/run_go2_signed_pose_goal_pilot_development.py` and
`scripts/read_go2_signed_pose_goal_pilot_development.py`.
Output: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_signed_pose_goal_pilot_v1_attempt_001`.
