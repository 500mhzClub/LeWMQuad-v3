# Why plan_cost Beats the GoalEnergyHead for Navigation

Date: 2026-06-09

The closed-loop visible-beacon benchmark consistently showed the model's built-in
`plan_cost` outperforming the learned `GoalEnergyHead` as the lewm planner's cost:

| cost | seq4 dead-ahead | seq4 jitter 0.7 |
|---|---:|---:|
| `plan_cost` (latent L2) | 0.92 | 0.73 |
| `GoalEnergyHead` (learned) | 0.67 | 0.33 |

This is counter-intuitive — the head was *added* to "replace the broken latent-L2
metric." The diagnosis is a use- and distribution-mismatch, grounded in the code.

## What each cost is

- **`plan_cost`** (`lewm/models/lewm.py:301`):
  `cost = ‖z_pred_proj − z_goal_proj‖²` — squared L2 in the projected latent
  space. It is **smooth and monotone in latent distance**, so over candidate
  action rollouts (each → a `z_pred`) it gives a clean, well-ordered servoing
  gradient: the action whose predicted latent is nearest the goal wins.

- **`GoalEnergyHead`** (`scripts/train_lewm_energy_head.py`): a learned
  **contrastive ranking** head. Training pairs are `(z_pred from recorded
  commands, z_goal = the SAME trajectory's future frame at horizon h)` as the
  positive; negatives are in-batch random goals **and** same-trajectory
  nearby-step frames (hard negatives), with distance-scaled margins. The
  objective (`energy_ranking_loss`) ranks the true goal below wrong goals. Its
  recorded `best_eval_ranking_acc = 0.984`.

## The mismatch

1. **Trained for goal discrimination, deployed for action servoing.** The head
   learns to rank *goals* for a fixed state ("which goal does this state lead
   to"). At deployment we do the opposite: fix the goal (the beacon) and rank
   candidate *action rollouts* (`z_pred`) against it. A head with 98.4% goal-
   ranking accuracy is *not* guaranteed to produce a smooth, monotone energy
   surface over `z_pred` for a fixed goal — and that surface is exactly what
   servoing needs. Plain L2 latent distance is monotone by construction; the
   contrastive energy is not.

2. **Goal-distribution mismatch.** The head's positives are *trajectory-future
   observations* (frames the robot actually reached). The deployment goal is a
   *constructed standoff render facing the beacon* — a different distribution.
   The head's learned energy is calibrated to the former, not the latter.

3. **Hard negatives sharpen discrimination, not the gradient.** Training the goal
   step to rank just below *adjacent* steps optimizes fine goal-vs-goal
   discrimination, which can make the energy surface jagged near the goal rather
   than a smooth basin — bad for the fine approach a servoing controller needs.

The decisive evidence is internal: the head is near-perfect at its own objective
(ranking acc 0.984) yet worse at navigation than bare L2. High retrieval
accuracy ≠ a good servoing cost.

## Implications

- **Use `plan_cost` for closed-loop navigation** with `seq4` (the benchmark's
  `lewm` policy without `--head-ckpt`). It is the better servoing metric.
- The `GoalEnergyHead` may still be useful for its *trained* job — goal
  reachability/retrieval (e.g. subgoal selection in the H-JEPA memory) — but it
  should not be the local servoing cost.
- If a learned servoing cost is wanted, it should be trained for the deployed
  use: rank candidate *action rollouts* toward a *fixed, deployment-distribution*
  goal, with a smoothness/monotonicity objective (e.g. regress true
  distance-to-goal), not goal-vs-goal contrastive ranking.

## Optional empirical confirmation (not yet run)

For a handful of states with a beacon goal, enumerate candidate primitives, roll
each out → `z_pred`, and compute the Spearman correlation of `plan_cost` vs
`GoalEnergyHead` energy with the *true* endpoint distance-to-goal. The prediction:
`plan_cost` correlates markedly higher (a better action ordering), directly
showing the head is the weaker servoing signal.
