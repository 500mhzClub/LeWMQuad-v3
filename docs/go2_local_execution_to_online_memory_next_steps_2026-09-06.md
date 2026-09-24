# Next: carry local feedback into continuous navigation and memory

The full objective remains RGB plus deployment-valid sensors, useful JEPA
prediction and novel-maze navigation on Go2, with independently evaluated
memory, generalization, timing and bounded hardware evidence. A local servo is
an enabling baseline, not that objective. Ignore obsolete BEV directions.

## Current execution update

Latest: the [coupled feedback experiment](go2_coupled_room_return_result_2026-09-06.md)
is independently audited at 0/3 returns, with both nominal runs reaching the
final home stage and 12/13 native-passing local holds. Follow the
[continuous-return completion plan](go2_continuous_return_completion_next_steps_2026-09-06.md)
for bounded multi-reference tracking and correct clipped return-heading intent
before the next fresh closed-loop experiment. JEPA, useful physical memory,
maze generalization and deployment evidence remain unachieved.

The [room-return result](go2_room_return_pulse_result_2026-09-06.md) now completes
the raw-sensor/fresh-scene integration proposed below, with actual continuous
multi-leg simulation and exact independent replay. It yields **0/3 returns**,
not completion of this plan. Four of five declared local holds pass native
scoring. Turn-induced XY drift and lost signed-half-turn semantics move the
immediate priority to the [joint pose/action successor plan](go2_coupled_pose_action_planning_next_steps_2026-09-06.md).
The offline empirical planner is implemented but not physically deployed.
Live observed-branch/marker navigation and useful physical memory remain next
after reliable execution; all scientific comparisons below remain outstanding.

## Finish and preserve the present evidence

Pulse-feedback V1 is an [audited 0/4 result](go2_pulse_feedback_servo_result_2026-09-06.md).
The distinct goal-region successor removes its overly tight internal approach
threshold, not the scientific task criteria. Its [completed independent audit](go2_goal_region_pulse_servo_result_2026-09-06.md)
confirms3/3 nominal full-task successes and0/1 low-friction success. All864
decisions replay exactly;2,388tests pass. Preserve both studies without retries or
retrospective tolerance changes. Three nearby starts in one deterministic floor
environment cannot establish broad reliability or a causal effect size.

## Immediate implementation: a continuous, sensor-referenced mission executor

Implementation update: [continuous executor and memory bridge](go2_continuous_pulse_memory_integration_result_2026-09-06.md)
now exist and the parameterized local controller exactly replays864previous
decisions. This is source/interface progress, not a new physical mission.
Next wire a raw-sensor owner and fresh-scene collector with independent scoring;
the new session still needs a separately bound initializer. Retain the explicit
fixed-lookahead versus metric-return targeting gap rather than claiming home
from the provisional route stack. The requirements below remain the full scope.

1. Implement a distinct parameterized pulse waypoint/turn executor. The current
   controller hardcodes XY(.4,0), yaw+.3 and a one-leg excursion bound; it cannot
   simply be dropped into a maze loop. Accept goals explicitly anchored to an
   actual sensor observation, transform them once into the uninterrupted visual
   frame, and preserve that frame, failure latch and mission-wide budgets across
   legs. Do not reset pose at each arrival or infer progress from command-time.
   Keep leg completion separate from place identity and traversability.
2. Integrate the existing episodic visit/attempt memory with that executor and
   live RGB branch/marker observations. The source already contains
   `WholeTaskNavigation`, `EpisodicRouteHypotheses` and `ObservedExploration`;
   reuse their event semantics rather than building another disconnected graph.
   Store actual departure, scan, arrival and failure observations. An observed
   local goal completion records an attempted/achieved local motion, not an
   automatically recognized place, reverse edge or verified home.
3. Resolve the actual command-interface mismatch before integration. The old
   fresh-maze session rejects yaw magnitudes above .35rad/s, whereas the measured
   gait-bank pulse uses .45rad/s. Build a distinct explicitly bounded session and
   test requested versus slew-applied commands; preserve the platform limits and
   native stop supervision. Do not silently patch a frozen experiment or clip
   the command back into a poorly supported regime.
4. Freeze a prospective continuous multi-leg development test that includes
   both turn directions, maze-scale turns, stopping and actual return, followed
   by a small connected-maze exploration/hidden-marker/return-home test. Targets
   used online must come from current sensed branches/geometry or stored sensed
   observations, never evaluator world coordinates or maze topology. A scripted
   motion unit test may use declared targets but cannot stand in for navigation.
   Evaluate every arrival, executed edge, collision, marker claim and home claim
   independently from the full physical trace; retain incomplete missions.

## Address sensing limits without hiding them

The independently confirmed small positive turn does not prove that
RGB-D tracking survives90/180degree maze turns. The previous controller failed
three times at a combined consensus/grid/displacement gate; isolate those
causes in read-only diagnostics if they recur. Do not assume the gate alone
identifies the cause, relax it after failures, reset pose, fabricate missing
views or replace a physical return with a graph traversal. Keep UNKNOWN place
association available, especially for visually identical corridors.

Current controlled continuous floor and hidden-robot ideal camera are explicit
engineering assumptions. Reintegrate observed obstacle/body-sweep evidence;
native collision stopping is only external supervision, not online clearance.
Unobserved volume is not free space. Preserve missing physical uncertainty and
the known self-visible camera/aperture/raster and near-field limits until tested.
The low-friction failure remains a robustness gap, not a discarded outlier.
Nominal integration can proceed as a declared development condition while that
gap is retained; it must not be described as terrain-independent navigation.

## Establish the scientific contribution, not just a hybrid demo

Once a continuous mission is executable, freeze matched memory-on/local-only
arms using identical sensors, gait, local executor, observation opportunities,
action vocabulary and budgets. Test whether actual revisits/backtracking improve,
including repeated-looking branches, wrong-home claims and failed returns.
Appearance ranking or stack completion is not a verified place/home association.

Collect diverse task-driven sensor/action sequences, including onset, turns,
slip and stopping. Partition by independent layout before training/selection;
retain separate development and custodian-isolated final evaluation. Train and
compare geometric/persistence, supervised action prediction and JEPA using
matched data, sensory channels, action choices and resource budgets. Separately
ablate predictive training, online multistep rollout and persistent memory;
an encoder plus hand-coded servo does not establish any of those effects.
Use multiple training seeds and independent maze layouts, report uncertainty
at the layout/seed level, all mission failures and wall-clock/path costs.

Finally test optical realism, noise/dropout, timing with physics not paused,
sensor calibration/availability and bounded real-platform execution when actual
hardware access permits. Torque/power absence means energy is unavailable,
not joint travel relabeled as joules. No intermediate software or simulation
result completes the full scientific goal.
