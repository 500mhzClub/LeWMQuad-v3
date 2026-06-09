# Navigation-Base Synthesis: seq4 vs seq11, and the "frozen ceiling" question

Date: 2026-06-09

Closes the arc that began with a simple grounding question — *can the dog
navigate in open space to a visible beacon?* — and a chain of conclusions that
ended at "we've hit the ceiling of frozen LeWM features, unfreeze the encoder or
swap to DINOv2." This document records the corrected, evidence-based picture.

## What was tested

1. Closed-loop visible-beacon MPC (`benchmark_lewm_closed_loop_mpc.py`,
   `open_obstacle_field`, textured, `plan_cost`), seq4_e9 vs seq11_e3, with the
   `--apply-textures` fix.
2. Offline goal-source controls (none / route / local / privileged) rebuilt on
   **seq4_e9** features and compared to the existing seq11_e3 controls — to test
   whether the offline "image << geometry" finding was a seq11 artifact.
3. Line-of-sight check on the offline branch (goal-seeking) subgoals.

## Results

### Closed-loop visible beacon (success rate)

| checkpoint (plan_cost, textured) | jitter 0.0 | 0.7 | 1.5 (~86°) |
|---|---:|---:|---:|
| seq4_e9 | 0.92 | 0.73 | 0.58 |
| seq11_e3 | 0.17 | 0.25 | — |
| bearing oracle | 1.00 | 1.00 | 0.75 |

seq4 does the base's intended job — local **visible** goal-image servoing — well;
seq11 does not. `plan_cost` beat the learned `GoalEnergyHead` for both
checkpoints (the head degrades nav).

### Offline goal-source controls — branch (goal-seeking) regret-ratio

| goal source | seq11_e3 | seq4_e9 |
|---|---:|---:|
| none | 0.904 | 0.902 |
| route (final-route image) | 0.762 | 0.659 |
| local (matched-subgoal image) | 0.734 | 0.677 |
| privileged (geometry vector) | 0.517 | 0.450 |
| **image(local) → privileged gap** | **0.217** | **0.227** |

seq4 is uniformly better, **but the image-vs-geometry gap persists at ~0.22 — it
is not a seq11 artifact.**

### Line-of-sight

100% of sampled branch goal-present subgoals are in line-of-sight from the start
(straight-line path free in the un-inflated grid). **The gap is not occlusion.**

## Reconciliation and verdict

The two regimes disagree only in appearance:

- **Closed-loop beacon:** goal image *faces the beacon* (multi-view, goal-facing)
  → seq4 image servoing works (0.73–0.92).
- **Offline maze branch:** goal image is the subgoal cell's *arbitrary-yaw
  first-visit representative frame* → image goal underperforms geometry by 0.22.

The subgoals are visible (100% LOS) and the *same* seq4 frozen features drive
beacon navigation. So the persistent offline gap is most plausibly an artifact of
the **goal-image convention** (an arbitrary-yaw cell frame is a poor goal image;
the subgoal is also often outside the start camera FOV, so the robot must turn to
see it), **not a frozen-feature metric-geometry ceiling.**

Therefore:

1. **Adopt `seq4_e9` as the navigation base**, with `plan_cost` (not the
   `GoalEnergyHead`, which degrades nav — investigate separately).
2. **No encoder unfreeze, no DINOv2 swap, no goal-localization translator is
   justified by this evidence.** The "frozen ceiling" conclusion is refuted: the
   features already support image-goal navigation given a goal-facing image.
3. The whole offline task-aligned program was built on `seq11_e3` (a weak nav
   base) **and** an arbitrary-yaw goal-image convention; its "visual goal-matching
   is the bottleneck" headline does not survive grounding.
4. Architecture-aligned next step (`docs/v3_hjepa_plan.md`): the base owns local
   **visible, goal-facing** subgoal servoing — which works — and the H-JEPA
   memory/subgoal stack owns choosing visible subgoals and presenting goal-facing
   representative images. Effort belongs there, not in the base encoder.

## CONFIRMED: the goal-image convention is the lever (2026-06-09)

Tested directly in the closed-loop beacon framework (cheaper and cleaner than an
offline render pipeline): seq4, dead-ahead beacon (jitter 0, so the goal image is
the only variable), `plan_cost`, N=12, with `--goal-yaw-offset-rad` rotating the
goal *image* away from facing the beacon while holding the goal *position* (and
success criterion) fixed.

| goal-image yaw offset | lewm success | bearing |
|---|---:|---:|
| 0 (faces the beacon) | 0.92 | 1.00 |
| π/2 (90° off) | 0.00 | 1.00 |
| π (looks away) | 0.00 | 1.00 |

Changing only the goal image's orientation collapses image-goal navigation from
0.92 to 0.00 — and the planner is sensitive even at 90°. The frozen features are
unchanged; only the goal-image convention changed. **This directly confirms the
offline image-vs-geometry gap is a goal-image-convention artifact, not a
frozen-feature ceiling.** It also sets a concrete requirement for the H-JEPA
subgoal stack: a subgoal's representative observation must be roughly
approach/goal-facing, or local image-goal servoing fails.

## Artifacts

- closed-loop: `.generated/closed_loop_beacon/{cmp_*,hiN_*}.json`,
  `docs/lewm_closed_loop_beacon_grounding_2026-06-08.md`
- seq4 offline controls: `.generated/task_aligned_policy_v0/seq4_v2_goal_controls.json`,
  heads `head_seq4_v2_{none,route,local,privileged}_spatial2_seed20260608.*`
