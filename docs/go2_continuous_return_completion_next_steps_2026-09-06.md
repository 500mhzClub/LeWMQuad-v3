# Next: complete the actual return, then test JEPA and memory in mazes

The full scientific goal is unchanged: RGB plus deployment-valid sensors, JEPA
prediction and useful online memory for Go2 novel-maze navigation, independent
layout/seed comparisons, realistic sensing/timing and bounded hardware evidence.
The new coupled-feedback collection is terminal at0/3 returns. Both nominal
runs complete6/7 declared stages, including their signed half-turns and corner
return. The completed independent audit confirms 12/13 declared local holds
pass all native criteria; the right final corner hold fails at 0.0600518798 m
maximum position error. All signed-winding holds pass. The low-friction run
fails during its first leg. Preserve these outcomes.

## Immediate evidence-driven work

1. Preserve the completed and bound raw audit when scoring physical success.
   Retain every local native position/orientation/velocity AND signed-winding
   hold, all failed missions, command limits, durations and sensor failure
   records. The right mission has an extra clipped corner-return leg: stage
   count and local-hold count are not interchangeable. Do not describe the
   changed-start results as a causal improvement over the previous room batch.
2. Test bounded multiple-reference visual tracking under the existing gates.
   At the left failure, the active keyframe gives68/81 inliers but only5 current
   grid cells. Of the fixed eight most recent accepted keyframes, reference1654
   passes unchanged registration and translation-increment gates, with67/80
   inliers and6/6 cells. Its native position error is7.62 mm, scored only after
   reference/gate decisions. The low-friction failure similarly has60/75
   inliers and5 current cells; prior reference270 passes unchanged pair gates
   with2.57 mm evaluator-only error. Neither retrospective alternative is an
   executed recovery. Implement a distinct bounded reference buffer retaining
   accepted reference poses/features in the SAME uninterrupted frame. Select
   using sensor-only quality and consistency, never native error; reject
   conflicting/no qualified candidates, preserve gyro/increment checks, and
   do not reset pose or weaken the six-cell gate. Reconstruct entire recorded
   streams first, then test fresh closed-loop trajectories. A repaired estimate
   on recorded data is not the counterfactual trajectory of the new controller.
3. Correct return-goal semantics prospectively. The room scheduler sets final
   home heading0 even when the .4 m target is a CLIPPED intermediate subgoal.
   In the right run the home leg targets(.05550,.00891), not home(0,0), while
   already demanding heading0. This can require turning away from the direction
   of the remaining journey before reaching home. Intermediate return targets
   should retain a travel heading; only the actual final home target should
   demand the home heading. Likewise, preserve an approach-heading intent across
   clipped corner-return subgoals instead of deriving a new large turn from a
   centimetre-scale residual. Keep actual full goal/hold criteria and do not
   declare home from the clipped target or a route-stack pop. Add tests where
   home/corner distances are just above .4 m and where the intermediate arrival
   overshoots its target within the allowed local region.
4. Distinguish model error, search limits and action coverage. The right home
   leg stops after34 pulses with one pulse left, an empty best path and
   SEARCH_EXHAUSTED. This is not evidence that the robot cannot reach home.
   Finite beam search and an orientation-heavy cost can lose routes that must
   first turn away from the final heading. Test geometric reachability and
   lookahead on fixed synthetic cases with goals behind the current heading,
   including translation caused by turns, without using a synthetic pass as
   physical proof. Prefer fixing the intermediate-goal intent before increasing
   budgets or searching costs after failure. Backward/arc gait-bank commands
   remain an explicit missing response-coverage avenue: characterize them in a
   separately bounded protocol before adding them to a model/controller.

Freeze distinct source/protocol/output for any successor. Do not resume or
modify the completed assay. Keep final acceptance tolerances, failure records,
native supervision and realistic uncertainty limitations. An engineering
successor can combine justified fixes, but a scientific claim about either
mechanism needs matched ablations on prospectively paired scenes/starts/seeds.

## Scientific continuation and constraints

The [pulse-timed JEPA interface](go2_pulse_timed_jepa_interface_2026-09-06.md)
now represents2.2 s known pulse/brake endpoints without fabricated future
commands. It is untrained, not used in the controller and not yet paired with
a real-data training adapter. Reuse the existing RGB/body history and separate
target-side labels, adapt exact timestamps/censoring, and freeze layout splits
before learning. Compare direct, empirical/geometric, supervised rollout and
JEPA with matched sensors/data/actions/budgets; evaluate actual action choice
and completed missions, not latent losses alone. Keep online-rollout and memory
ablations separate from predictive-training claims.

After reliable continuous execution, connect actual branch and marker sensing
to episodic visits/attempts, physically backtrack and independently verify home
in a connected maze. The current room route is scripted; waypoint storage is
not a demonstrated memory advantage. Preserve UNKNOWN association in aliased
corridors and add independent maze layouts and training seeds.

Storage after this batch is about20 GiB free on the workspace filesystem; the
10 GiB reserve remains. The recovery/root filesystem has substantially more
space, but the current output/binding readers assume repository-relative,
nonsymlink paths. Do not silently relocate artifacts or delete old evidence.
Before another full collection, budget its size and, if needed, explicitly
support an audited external development-artifact root rather than bypassing
path checks. Narrow CPU/source work can continue without another large batch.

Hidden-robot ideal RGB-D/gyro, controlled floor, uncalibrated sensing/error
bounds and paused compute remain. Multiple references do not establish physical
uncertainty or body clearance. Reintegrate real optical/self-occlusion/near-field
and observed swept-body evidence, calibrated/noisy/dropped sensor tests, actual
wall-clock control and bounded hardware when accessible. Mechanical energy
remains unavailable without torque/power. No intermediate result completes the
original scientific goal; sealed roles remain inaccessible and V4 development-only.
