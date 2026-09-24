# Local repository mapping (read-only)

| Proposed component | Local evidence | Disposition |
|---|---|---|
| Frozen predictor and control history | `lewm/models/predictor.py`; predictor training/evaluation scripts; frozen counterfactual receipts | Reusable as-is for H1–H3 inputs; do not alter checkpoints. |
| Candidate bank and post-slew actions | `lewm/planning/primitive_bank.py`; `lewm/actions`; counterfactual builders | Reusable with a narrow waypoint-planner adapter. |
| Existing latent MPC seam | `lewm/planning/local_mpc.py`, `lewm/planning/costs.py` | Interface reusable; existing energy/goal cost is not the proposed factorised ranker. |
| Local obstacle contract | `lewm/planning/local_obstacles.py`, `depth_local_obstacles.py` | Reusable only when deployment-valid; privileged grids are evaluation-only. |
| Topological memory | `lewm/memory/online_topological_memory.py`, `topological_navigator.py` | Future subgoal producer; prior predictor-to-memory failures remain frozen. |
| Graph/BFS | `lewm_worlds/.../scene_graph.py` and navigator routing | Reusable for oracle waypoint construction and evaluation, not planner input. |
| Exact-reset branches | `lewm/benchmarks/counterfactual.py`, `go2_world_model_counterfactual_pilot_v1.py`, related scripts | Reusable where local-intent labels exist; old panel lacks complete topology for this task. |
| Safety/contact/clearance | local obstacle and benchmark/safety probe scripts | Partially reusable; must bind deployment-valid definitions and calibrate independently. |
| Existing CEM/MPC/planners | `lewm/planning/local_mpc.py` and named navigation integrations | Reference only; first experiment must exhaust the 12 candidates. |
| Waypoint producer/frontier | `frontier_viewpoint_information_gain.py`, `TopologicalNavigator` | Reusable later with a narrow local-intent contract; not first-pass planner input. |

## Smallest new modules

1. `local_waypoint_panel.py`: freeze oracle intents and collect exact-reset labels.
2. `factorised_trajectory_ranker.py`: model and independent heads.
3. `evaluate_safe_local_waypoint.py`: fixture, true-future, predicted, and closed-loop reducers.
4. `calibrate_planner_safety.py`: scene-disjoint residual and risk calibration.

No new generic benchmark framework, memory representation, utility scorer, or predictor is required.
