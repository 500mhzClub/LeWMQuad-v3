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

## CORRECTION (2026-06-09): the "both fail under jitter" claim was wrong

The table above mixes confounds: the `seq4_e9` rows used the **untextured** render
on a **different (non-textured) corpus** with `plan_cost`, while `seq11_e3` used
**textured** render with the learned `GoalEnergyHead`. A matched, controlled
follow-up — **same textured corpus, same 6 scenes, same `plan_cost` cost path,
same flags for both checkpoints** — overturns the headline:

| checkpoint (plan_cost, textured, matched) | jitter 0.0 | jitter 0.7 | lewm progress |
|---|---:|---:|---:|
| seq4_e9 | **0.83** | **0.83** | 0.83–0.88 m |
| seq11_e3 | 0.17 | 0.17 | −0.05 m |
| bearing oracle | 1.00 | 1.00 | — |

So **seq4_e9 navigates to off-axis (jitter 0.7) visible beacons ~83% of the
time**, near the bearing oracle — it does *not* fail under heading offset. The
earlier "0% under jitter" for seq4 was an artifact of the untextured/old-corpus
run; for seq11 the `GoalEnergyHead` further degraded the jitter case (0.17 with
plan_cost -> 0.00 with the head). The "image-goal planner can't orient to a
visible beacon" conclusion holds only for `seq11_e3`, **not** as a general claim.

## Higher-N confirmation + occluded probe (2026-06-09)

Re-ran on the textured pipeline with `plan_cost`, N=12 (visible-beacon,
`open_obstacle_field`):

| run | lewm | bearing | lewm progress |
|---|---:|---:|---:|
| seq4 visible, jitter 0.0 | 0.92 | 1.00 | +0.86 |
| seq4 visible, jitter 0.7 | 0.73 | 1.00 | +0.80 |
| seq4 visible, jitter 1.5 (~86°) | 0.58 | 0.75 | +0.76 |
| seq11 visible, jitter 0.7 | 0.25 | 1.00 | −0.07 |

seq4's visible-beacon capability is robust at N=12 and degrades gracefully with
heading offset (and at 86° even the oracle is only 0.75). seq11 stays weak
(0.25, negative progress). The visible-local case is settled: **seq4 + plan_cost
does the base's intended local goal-image servoing job.**

The `landmark` task (no line-of-sight guarantee) was intended as an occluded
probe but is **distance-confounded and uninformative**: it selected goals 3.6–8.9
m away (mean 7.6 m), while the 16-block budget allows only ~2.4 m of travel, so
even the bearing oracle reached 0/12 (it moved the right way, +1.19 m; seq4 went
the wrong way, −0.16 m). A proper local-occluded test needs *near* (≈1.5–2.5 m)
non-line-of-sight goals; that requires a max-distance cap the harness lacks. Note
the hint that on far goals seq4's image cost loses direction (negative progress)
— but it is confounded, not a clean result.

Per the topological-nav design (`docs/v3_topological_nav_plan.md`) the base is only responsible for
*local visible* subgoal servoing; topology/occlusion is the memory+subgoal
stack's job. seq4's visible-beacon result therefore validates the base's intended
role, and no encoder change or goal-localization translator is justified by this
evidence.

## Conclusion (revised 2026-06-09)

1. **`seq11_e3` cannot orient to a visible beacon** (0.17 dead-ahead, circles /
   negative progress). It is a weak image-goal nav base — consistent with its
   selection for long-horizon rollout stability, not local goal-image planning.
2. **`seq4_e9` *can*** navigate to visible beacons including off-axis ones
   (~0.83 at jitter 0.0 and 0.7, plan_cost, matched textured setup). Switch the
   navigation base to `seq4_e9`. Note `plan_cost` beat the learned
   `GoalEnergyHead` for both checkpoints — investigate why the head degrades nav.
3. **`bearing` solves every case (1.00).** Locomotion and privileged geometry are
   fine.
4. **Open question (untested here):** whether `seq4_e9` + `plan_cost` holds up on
   *harder, occluded / non-line-of-sight* goals (the `landmark` task and maze
   families) — that is where the offline maze/recovery failures live, and where a
   goal-localization aid would actually be justified. The visible-beacon test does
   not probe it.

## Implications (revised 2026-06-09)

- **Adopt `seq4_e9` as the navigation base.** It is dramatically better at local
  goal-image servoing than `seq11_e3` and already handles off-axis visible
  beacons at ~0.83 with `plan_cost`.
- **A goal-localization translator is NOT yet justified.** `seq4_e9` + `plan_cost`
  navigating off-axis visible beacons at 83% shows the frozen features already
  carry enough goal-direction signal for the visible case. The separate offline
  finding that a metric-vector *regression* from those features generalizes poorly
  (low held-out R^2) reflects an ill-posed regression target (goal image is the
  cell's arbitrary-yaw representative frame, not a goal-facing view) and the weak
  `seq11_e3` checkpoint — it is not evidence of a frozen-feature navigation
  ceiling.
- **The real open question is harder goals.** Re-run the beacon test on `seq4_e9`
  at higher N to confirm 0.83 is not a 6-scene fluke, and test the `landmark`
  task (no line-of-sight guarantee) and maze families to find where image-cost
  servoing actually breaks. Only a *well-posed* failure there (goal-facing image,
  occluded target) would justify a translator or any encoder change.
- Caveat for the offline program: it used `seq11_e3` frozen features throughout,
  now shown to be a weak image-goal nav base. Its conclusions should be re-checked
  on `seq4_e9` before being treated as properties of the LeWM substrate.

## Artifacts

- benchmark: `scripts/benchmark_lewm_closed_loop_mpc.py` (`--apply-textures` added)
- results: `.generated/closed_loop_beacon/seq11_e3_tex_{j0,j07}.json`
- prior seq4_e9: `models/checkpoints_textured_v03_full_20260531/sweep_seq4/closed_loop_mpc_visible_beacon_e9_*_testid_open_obstacle_field.json`
