# Late-turn forecast diagnosis

Status: **COMPLETE**. Two hold counterfactuals reproduced the live causal
prefix exactly (native poses, joints, applied commands and three context RGB
images). Both completed the 500-ms suffix without contact. Worker sessions
1413 and 42059 exited 0 in 19.4 and 16.1 seconds; scoring session 35859 exited 0.
The previously executed right turns supply the matched alternative outcomes.
No training or navigation policy was changed.

At both states, the forecast reverses the hold-versus-turn ranking. Scoring
actual future images with the same learned cost correctly prefers hold.

| Case / departure tick | Action | Predicted learned cost | Actual-image learned cost | Final XY cm | Final heading degrees | Within original tolerance |
|---|---|---:|---:|---:|---:|---|
| 0 / 80 | Hold | 9.0354 | 2.9672 | 2.5506 | 3.6836 | Yes |
| 0 / 80 | Executed right turn | 8.7628 | 5.8977 | 2.5995 | 7.6035 | No |
| 3 / 55 | Hold | 8.4683 | 2.6463 | 2.5826 | 3.3721 | Yes |
| 3 / 55 | Executed right turn | 8.1449 | 5.8082 | 2.8744 | 8.0863 | No |

The physical cost (training metric's normalized XY and world-yaw target) also
prefers hold: 1.2632 versus 3.1025 in case 0; 1.1967 versus 3.5632 in case 3.
Thus the learned cost is sufficient to order these two actual outcomes; its
application to imperfect predicted features gives the wrong ordering. This is
a task-relevant forecast error / predicted-feature distribution gap, not proof
that any particular predictor architecture or training objective caused it.

These are two post-hoc states from exposed development layouts. Only hold and
the executed turn were compared, not all six actions. The hold counterfactuals
last 500 ms, so they do not establish that an indefinitely holding policy would
remain inside tolerance. The stronger result remains the prospective cost-only
comparison, including its final-arrival failures.

Next scientific step: a controlled training-only predictor intervention that
penalizes error in the frozen goal embedding alongside dense visual error.
Retain a continuation control with the same initialization, data and extra
update budget, so any improvement is not attributed to the auxiliary objective
when it could come from additional training. Keep the encoder and goal metric
fixed, retain action-blind and persistence comparisons, and evaluate prediction
and prospective decisions separately. These diagnostic frames must remain
outside fitting. Do not tune a stopping threshold on these two failures.
Before claiming navigation or JEPA superiority, independent full-maze tasks,
a strong reactive/non-predictive baseline, memory contribution and matched
representation-training comparisons remain necessary. Hardware and real-time
validation remain separate and incomplete.

Plan: `go2_dense_metric_late_turn_plan_2026-09-17.json`.
Result: `go2_dense_metric_late_turn_result_2026-09-17.json`.
Source: `scripts/diagnose_go2_dense_metric_late_turn_development.py`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_dense_metric_late_turn_v1_attempt_001`.
