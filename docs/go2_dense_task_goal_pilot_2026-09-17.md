# Matched predictor continuation: prospective live result

Status: **COMPLETE; neither continuation establishes an improvement over the parent**. All four intended task cells have complete terminal records. Both action models visit the goal region on both layouts but neither finishes within tolerance. No contacts occurred in the completed trials.

| Predictor, same learned goal cost | Goal visits | Final within 3 cm / 5 degrees | Contacts | Mean final XY cm |
|---|---:|---:|---:|---:|
| Previous frozen predictor | 2/2 | 0/2 | 0/2 | 3.160 |
| Eight extra epochs, dense L1 | 2/2 | 0/2 | 0/2 | 8.112 |
| Eight extra epochs, geometry-aware loss | 2/2 | 0/2 | 0/2 | 3.888 |

The auxiliary objective beats its equal-extra-training control on final position error, but both are worse than the previous model and neither solves final arrival. The auxiliary model also has worse final heading errors on these tasks. Thus it is not promoted; lower training task loss and slightly lower pooled transfer embedding error do not establish better navigation. Preserve all model families and outcomes.

| Case | Layout | Continued objective | Longest goal-frame streak | Final XY cm | Final heading degrees |
|---|---|---:|---:|---:|
| 0 | family_episode_026 | dense_action | 18 | 7.6495 | 6.7147 |
| 1 | family_episode_026 | metric_action | 8 | 3.3949 | 15.5410 |
| 2 | family_episode_003 | metric_action | 8 | 4.3818 | 9.0348 |
| 3 | family_episode_003 | dense_action | 4 | 8.5755 | 3.1631 |

All initial RGB frames reproduce the predecessor exactly. The encoder, goal metric, tasks, commands, timing, budget, seed and native stops remain fixed. No goal stopping rule was added. The unchanged action-blind uniform-tie policy uses the prior two contact-failure trajectories as explicitly reused reference, not extra independent repetitions. These are two exposed related local tasks; no full-maze, strong reactive-baseline, JEPA-training-superiority, real-time or hardware claim follows.

## Storage interruption retained

Six native executions were started for four intended task cells. Cases 0 and 1 completed normally (sessions 59180 and 35276, exit 0). Cases 2 and 3 stopped at the 512 MiB storage reserve (sessions 14771 and 76937, exit 1), after 12 and 11 decisions. Their full partial records and failure JSON remain in the original attempt. The initial output-space estimate was too low.

Only unused depth arrays from 14 completed, contact-free diagnostic replays were retired under the standing user cleanup instruction. This removed 1,278 arrays and freed 0.872 GiB. All diagnostic RGB, native poses, applied commands, metadata and results remain, as do every failed trial and all training data/checkpoints. Exact retired depth replay requires regeneration. Receipt: `.generated/depth_retirement_dense_goal_diagnostics_2026-09-17/`.

Only interrupted cases 2 and 3 were then executed afresh in a separate replacement root (sessions 79635 and 50634, exit 0). Output capacity was sized from the observed complete case sizes plus margin and reserve. The reader (session 67642, exit 0) confirmed identical actions, forecast costs and full native trajectory prefixes against both interrupted attempts. Cases 0 and 1 were retained without rerunning. The result JSON records both infrastructure failures and all four terminal task outcomes.

Next: fit/evaluate the prepared direct observed-image goal readout on long, bidirectional training pairs. This supports a stronger direct-control baseline and can assess goal-state estimation separately from world-model prediction. Do not continue tuning auxiliary-loss weights on these two exposed layouts. The original full-navigation objective remains incomplete.

Result: `go2_dense_task_goal_pilot_result_2026-09-17.json`.
Original plan: `go2_dense_task_goal_pilot_plan_2026-09-17.json`.
Replacement plan: `go2_dense_task_goal_pilot_storage_replacement_plan_2026-09-17.json`.
Component evaluation: `go2_dense_task_predictor_evaluation_2026-09-17.md`.

Figure: `go2_dense_task_goal_pilot_2026-09-17.png` (visually inspected).
