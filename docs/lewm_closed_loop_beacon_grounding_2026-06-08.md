# Closed-Loop Visible-Beacon Grounding Test

Date: 2026-06-08

Grounds the offline task-aligned regret-ratio work
(`docs/lewm_task_aligned_v2_goal_controls_2026-06-08.md`) in *actual* closed-loop
navigation. The offline program concluded that visual goal-matching, not the base
representation, is the bottleneck. The simplest possible navigation check tests
that directly: in open space, can the model drive to a beacon that is fully
visible from the start?

## Setup

`scripts/benchmark_lewm_closed_loop_mpc.py --task visible-beacon` on held-out
`open_obstacle_field` scenes: it constructs a local open-space approach with
direct line-of-sight to a beacon, runs receding-horizon control (`--mode
kinematic`: score candidate primitive rollouts with the LeWM latent cost, execute
the first block, re-render, replan), and reports physical success/progress.

Policies: `lewm` (the model's image-goal cost) vs `bearing` (steer by true
bearing to the beacon — privileged geometry oracle) vs `hold`/`random` floors.
`--beacon-start-yaw-jitter-rad` sets whether the robot starts facing the beacon
(0.0, dead-ahead) or must turn to find it (0.7).

### Render-distribution fix

The benchmark built scenes with `apply_textures=False` (untextured box
geometry). `seq11_e3` is trained on `render_textured_v03`, so untextured frames
are out-of-distribution. Added `--apply-textures` (with `--backend vulkan`) so
the model is evaluated in its training distribution. All `seq11_e3` numbers below
use textured rendering.

## Results

| checkpoint / render | jitter | lewm success | lewm progress | bearing success |
|---|---:|---:|---:|---:|
| seq4_e9 (prior, untextured, N=10) | 0.0 | 0.78 | 0.781 | 1.00 |
| seq4_e9 (prior, untextured, N=10) | 0.7 | **0.00** | 0.347 | 1.00 |
| seq11_e3 (ours, textured, N=6) | 0.0 | **0.17** | 0.370 | 1.00 |
| seq11_e3 (ours, textured, N=6) | 0.7 | **0.00** | 0.264 | 1.00 |

`lewm`'s executed primitive mix is dominated by `arc_right / yaw_right / hold`
(circling and standing), with little forward motion: the planner spins instead of
approaching a beacon it can see.

## Conclusion

1. **Under any realistic heading offset the image-goal planner gets 0% — it
   cannot orient to a visible beacon.** This holds for *both* checkpoints
   (seq4_e9 N=10, seq11_e3 N=6) regardless of render/corpus. The "dog" can only
   walk straight at a goal that is already centered in front of it.
2. **Even dead-ahead our `seq11_e3` mostly fails (0.17)**, far below the prior
   `seq4_e9` (0.78). This comparison is confounded (different model *and*
   cost-head — `seq11_e3` used the learned `GoalEnergyHead`, the prior seq4 run
   used the latent-L2 `plan_cost` — *and* textured vs untextured render), so it
   is not a clean model verdict; but it flags that the checkpoint the entire
   offline program is built on is a weak image-goal planner.
3. **`bearing` solves every case (1.00).** Locomotion and the privileged geometry
   are fine. The broken link is converting a goal *image* into a direction to
   move — exactly the offline finding, now confirmed in closed loop.

## Implications

- The goal-localization translator (recover relative goal geometry from the goal
  image; `docs/lewm_task_aligned_v2_goal_controls_2026-06-08.md` Phase A/B) is the
  validated core fix. **Jittered visible-beacon success is now the real
  closed-loop metric** to optimize, more trustworthy than offline regret-ratio.
- **Re-examine checkpoint selection.** `seq11_e3` was chosen for rollout
  stability (the seq11 action-sweep), not goal-image planning. The clean control
  is to run the beacon test (and the offline goal controls) on `seq4_e9` under the
  identical textured pipeline and the same cost head, to decide whether `seq4_e9`
  is a materially better base for navigation before investing in the translator.
- Caveat for the offline program: it used `seq11_e3` frozen features throughout.
  The privileged-vector control showed those features suffice for action
  selection given goal *geometry*; the beacon test shows they fail given a goal
  *image*. Both point to the same translator, but a stronger base checkpoint may
  raise the ceiling.

## Artifacts

- benchmark: `scripts/benchmark_lewm_closed_loop_mpc.py` (`--apply-textures` added)
- results: `.generated/closed_loop_beacon/seq11_e3_tex_{j0,j07}.json`
- prior seq4_e9: `models/checkpoints_textured_v03_full_20260531/sweep_seq4/closed_loop_mpc_visible_beacon_e9_*_testid_open_obstacle_field.json`
