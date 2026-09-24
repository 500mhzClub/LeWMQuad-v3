# Why the dense visual-goal controller overshoots

Status: **COMPLETE**. All 12 matched native branches exited 0 (launcher session
95328); scoring exited 0 (session 72990). The four-worker collection took 50.4
seconds. No predictor training or controller change was made.

**At both tested states, perfect successor-image prediction would still choose
forward under the current dense-feature goal cost.** Forward is physically
worse than braking. The raw visual objective is therefore insufficient at these
near-goal states; predictor accuracy alone cannot correct these decisions.

The diagnostic replays the actual command prefix of action-model cases 0 and 3
to tick 35, then executes each of the six original candidates for 500 ms. These
states were selected post hoc to investigate the observed failure. Every branch
exactly reproduced the original native pose, joint and applied-command prefix
and all three predictor-context RGB images. The two forward alternatives also
reproduced the original successor poses and RGB exactly. Every branch completed
without contact. This is a matched counterfactual diagnostic, not a navigation
trial or independent benchmark.

## All outcomes

The prediction costs below were saved by the live controller before this
diagnostic. Actual visual costs use the same frozen encoder, normalization,
goal image and MSE. Native poses are used only for physical evaluation.

| Original case | Action | Predicted goal MSE | Actual goal MSE | XY error cm | Heading error degrees |
|---|---|---:|---:|---:|---:|
| 0 / cluster 02 | Hold | 0.43985 | 0.45387 | 4.828 | 2.006 |
| 0 / cluster 02 | Forward | **0.34297** | **0.45202** | 11.932 | 2.341 |
| 0 / cluster 02 | Left arc | 0.52460 | 0.67023 | 11.299 | 14.066 |
| 0 / cluster 02 | Right arc | 0.47213 | 0.60386 | 10.703 | 9.159 |
| 0 / cluster 02 | Left turn | 0.50789 | 0.65402 | 5.178 | 13.948 |
| 0 / cluster 02 | Right turn | 0.53877 | 0.63379 | 5.693 | 9.149 |
| 3 / cluster 03 | Hold | 0.37229 | 0.46379 | 2.938 | 2.374 |
| 3 / cluster 03 | Forward | **0.33440** | **0.38247** | 8.720 | 1.701 |
| 3 / cluster 03 | Left arc | 0.52364 | 0.64540 | 6.768 | 13.006 |
| 3 / cluster 03 | Right arc | 0.47378 | 0.60487 | 7.946 | 9.847 |
| 3 / cluster 03 | Left turn | 0.53492 | 0.63578 | 1.810 | 12.852 |
| 3 / cluster 03 | Right turn | 0.54261 | 0.65336 | 3.499 | 9.514 |

In case 0, the actual visual preference for forward over hold is small (0.00185),
while the predictor overstates it (0.09687). In case 3, the actual preference is
larger (0.08132), and hold alone satisfies the original joint 3-cm/5-degree
endpoint tolerance. Left turn has smaller XY error there but unacceptable
heading error. Thus physical position alone is not the complete goal criterion.
No case-0 candidate meets both tolerances at this departure; stopping earlier
or finer action timing may also matter after fixing the objective.

This does not prove that visual information lacks geometry or that JEPA is
unsuitable. It shows a mismatch between aligned-token feature MSE and the
desired physical goal relation. The goal image was acquired during gait. Body
pose, viewing geometry and image changes can matter to the visual distance
without corresponding to improvement in planar goal-reaching. Their individual
contributions have not been isolated here.

## Next scientific intervention

Keep the frozen encoder and action/no-action predictors unchanged. Fit a
goal-distance function using **training-layout image pairs only**, with physical
XY/heading relations as training targets. Include stopped and moving frames
and more than one temporal separation; do not fit a threshold or choose a
checkpoint on these exposed failures. Evaluate the same fitted cost with actual
future features first, then predicted features, preserving the current raw-MSE
cost and both live failures. This is a new goal-cost target justified by the
oracle failure, not another attempt to predict ego-motion from the same narrow
500-ms decoder fit. A calibrated cost is additional supervision; a resulting
navigation gain would not by itself isolate the JEPA training objective.

Once a useful cost exists, test it prospectively with the same controller
conditions and retain a strong non-predictive comparison. Full-maze exploration,
memory/backtracking, independent evaluation and deployment-valid timing/hardware
remain necessary for the thread goal. No goal-cost fitting was launched by this
diagnostic.

Resources: 72 GiB available RAM, 3.4 GiB output free space, no competing compute
jobs before collection. Four independent workers used cores 0-3, 4-7, 8-11 and
12-15; one GPU process then encoded the goal/successor images. All native traces,
RGB and failures are retained; no dense feature cache was written to disk.

Plan: `go2_dense_goal_overshoot_plan_2026-09-17.json`.
Result: `go2_dense_goal_overshoot_result_2026-09-17.json`.
Source: `scripts/diagnose_go2_dense_visual_goal_overshoot_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_dense_goal_overshoot_branches_v1_attempt_001/`.
