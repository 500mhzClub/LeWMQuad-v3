# Matched predictor continuation: fixed component evaluation

Status: **COMPLETE**, session 66935, exit 0; 42.96 seconds. No new navigation in this evaluation. The prospective action comparison is also complete; see `go2_dense_task_goal_pilot_2026-09-17.md`.

All models use the same frozen V-JEPA representation and learned goal metric. Each continued arm received exactly 1,760 additional updates. Training and exposed-transfer roles remain separate.

## Exposed transfer pulse panel

18 branches in six shared-history groups across two previously exposed geometries. Parent checkpoints are unchanged references. The branches are dependent within each history.

| Model | Dense MSE | Dense action retrieval /18 | Goal-embedding MSE | Goal-space action retrieval /18 | Goal-space centered effect ratio |
|---|---:|---:|---:|---:|---:|
| dense_action | 0.242430 | 18 | 0.551687 | 12 | 0.4540 |
| metric_action | 0.356119 | 18 | 0.540312 | 10 | 0.3974 |
| dense_no_future_action | 0.338514 | 0 | 1.409201 | 0 | 1.0000 |
| metric_no_future_action | 0.399047 | 0 | 0.783159 | 0 | 1.0000 |
| parent_action | 0.243141 | 18 | 0.756716 | 11 | 0.4617 |
| parent_no_future_action | 0.328250 | 0 | 1.187018 | 0 | 1.0000 |
| persistence | 0.387830 | 0 | 0.404631 | 0 | 1.0000 |

The auxiliary action arm reduces goal-embedding MSE by only about 2.1% versus matched dense continuation, while increasing dense MSE by about 46.9%. Goal-space retrieval falls from 12/18 to 10/18. Persistence still has lower pooled goal-embedding error than either action model. Centered action response and pooled factual error answer different questions. These results do not establish the new objective as better for planning.

## Previously diagnosed goal states

| State | Parent: all-six choice | Dense continuation: all-six choice | Geometry-aware: all-six choice | Actual-image oracle among measured outcomes |
|---|---|---|---|---|
| early case 0, tick 35 | right_arc | right_arc | right_turn | hold |
| early case 3, tick 35 | hold | hold | right_turn | hold |
| late case 0, tick 80 | right_turn | hold | right_turn | hold |
| late case 3, tick 55 | right_turn | right_arc | right_turn | hold |

Only hold and right turn were physically measured in the late-state diagnostic. Dense continuation improves the hold-versus-turn ranking at both states, but selects right arc when all six candidates are available in late case 3. That action has no measured counterfactual outcome at this state. Its zero pairwise regret is therefore not evidence of an optimal full action choice. The live trial must resolve the policy consequence. The auxiliary model chooses right turn at all four states; its early-case-0 physical regret is smaller than the parent but remains nonzero.

All causal forecasts were completed before loading future scoring images. Blind predictions were computed once per identical input group and broadcast. Recomputed parent dense scores match the previous evaluation within 1e-6. No model, coefficient or checkpoint was selected using these outcomes.

This is one seed, small exposed geometry families, and additional physical supervision through the frozen goal metric. It is not a JEPA representation-training comparison, independent-maze result, strong reactive-baseline comparison or deployment qualification.

Result: `go2_dense_task_predictor_evaluation_2026-09-17.json`. Training: `go2_dense_task_predictor_2026-09-17.md`.
