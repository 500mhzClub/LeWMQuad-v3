# Prospective learned goal-cost comparison

Status: **COMPLETE**. Four fresh native trials finished; action trials completed
the full 20-decision budget, while both action-blind trials stopped on contact.
All four process handles exited 0 (10567, 52848, 19801, 32987). The comparison
reader (90403) exited 0. Native contact failures remain in the results.

Replacing raw dense-feature MSE with a training-only learned physical distance
reduced mean final position error from **118.71 cm to 3.16 cm**. Both action
trials entered the original 3 cm / 5 degree goal region and stayed there for
30 or 55 consecutive camera frames, but **neither finished within tolerance**.
This is improved local reaching and retention, not completed maze navigation.

| Goal cost | Arm | Layout | Longest within-goal frame streak (10 Hz) | Final XY (cm) | Final heading (degrees) | Contact | Final within goal |
|---|---|---|---:|---:|---:|---|---|
| Raw MSE | Action | family_episode_026 | 2 | 129.279 | 19.255 | No | No |
| Learned | Action | family_episode_026 | 55 | 2.989 | 6.725 | No | No |
| Raw MSE | Action | family_episode_003 | 7 | 108.142 | 3.279 | No | No |
| Learned | Action | family_episode_003 | 30 | 3.331 | 8.055 | No | No |
| Raw MSE | No future action | family_episode_026 | 0 | 12.411 | 48.225 | Yes | No |
| Learned | No future action | family_episode_026 | 0 | 12.411 | 48.225 | Yes | No |
| Raw MSE | No future action | family_episode_003 | 0 | 14.783 | 47.406 | Yes | No |
| Learned | No future action | family_episode_003 | 0 | 14.783 | 47.406 | Yes | No |

The learned cost leaves encoder and predictor weights, scene, supplied goal
RGB, candidate actions, command limits, 500-ms horizon and execution blocks,
seeds, budget and native stops unchanged. No goal-based stopping rule was
added. Simulator poses are used only for evaluation; physical labels were
used in training the goal metric. Initial RGB matches exactly and first raw
forecast costs match within 1e-6 across the cost intervention. Both blind
trajectories exactly reproduce their predecessors, including contact stops.

Each action trial took about 40-41 seconds of wall time. Simulation pauses
during inference; this does not qualify real-time control. Two processes ran
concurrently on disjoint four-core CPU groups; sampled GPU usage was 63% with
about 6.2 GB total VRAM use. Output volume had 2.2 GiB free afterward.

The two tasks use related, previously exposed transfer geometries excluded
from native adaptation and goal-metric fitting. Neither is an untouched test.
Only local supplied-image reaching is tested, without exploration, memory or
backtracking. Action-blind uniform tie breaking is a weak control, not a strong
reactive baseline. Additional physical supervision in the cost means these
results do not isolate JEPA representation-training superiority.

Both action trials first approach with four left-arc blocks, then hold. Each
later chooses a right turn while already inside the goal region: case 0 at
tick 80, case 3 at tick 55. The completed two-branch diagnosis reproduced each exact departure and
executed hold. Both hold outcomes remain within tolerance; both executed turns
leave it. The learned metric prefers hold with actual future images but prefers
turning with predicted features. This isolates a task-relevant forecast ranking
error at these states. See `go2_dense_metric_late_turn_2026-09-17.md`.

Machine-readable result: `go2_dense_metric_goal_pilot_result_2026-09-17.json`.
Figure: `go2_dense_metric_goal_pilot_2026-09-17.png` (visually checked).
Plan: `go2_dense_metric_goal_pilot_plan_2026-09-17.json`.
Output: `/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_dense_metric_goal_pilot_v1_attempt_001`.
