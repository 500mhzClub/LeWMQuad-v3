# Next: joint pose/action prediction, then continuous maze navigation

The objective remains RGB plus deployment-valid sensors, JEPA-based predictive
navigation and useful online memory on Go2 in novel mazes, with independent
layout/seed comparisons and bounded hardware evidence. A model-generated room
trajectory is not that objective. Ignore obsolete BEV directions.

Implementation update: the joint feedback controller and three fresh physical
trials are now [complete and independently audited](go2_coupled_room_return_result_2026-09-06.md).
Both nominal missions reach the final home stage, but full returns remain 0/3;
12/13 declared local holds pass native pose/stopping/signed-yaw criteria.
The next actions are the [continuous-return completion plan](go2_continuous_return_completion_next_steps_2026-09-06.md),
not repeating the now-completed offline-only step. The scientific scope below
remains outstanding beyond this engineering progress.

## Evidence that changes the immediate action

The new continuous room-return collection is terminal, with zero complete
returns in three fixed trials. The nominal left and low-friction runs completed
one of seven sensor-defined stages; the nominal right run completed three.
All exhausted a local pulse budget. Preserve the independent raw audit and
per-leg full-physics scores in the companion room-return result document.

The continuous sensor/execution handoffs now operate in actual simulation, but
the controller treats heading correction and position correction separately.
Turning moves the body in XY, sometimes outside the 6 cm turn-goal region.
Further steering toward the old position also changes position, and can produce
long turning sequences. This is a coupled action-response problem; it is not
evidence for lowering visual acceptance gates or resetting localization.

An independent native calculation also places the low-friction forward hold at
0.0600306616 m maximum error, just outside the fixed 0.06 m limit despite visual
acceptance. Keep that failure. A near-boundary model/visual candidate is not a
robust physical arrival: the current estimator has no calibrated error bound.
Aim at the actual goal, evaluate sensor/response uncertainty on separate data,
and report native failures without adding a retrospective scoring tolerance or
feeding evaluator pose into the controller.

The old goal representation also wraps the final orientation. A request for
-pi does not preserve a clockwise winding requirement. The right-run half-turn
therefore cannot establish the signed maneuver stated in the protocol. Preserve
this semantic shortcoming rather than changing the frozen assay after seeing it.

## Immediate successor work

1. Finish and bind the current independent raw audit. Preserve every failed
   mission, every local native hold (including failures), the continuous pose
   errors, stop records and the winding limitation. Do not retry the old output.
2. Establish a transparent action-conditioned planning baseline. The new
   `coupled_pulse_rollout_development.py` composes measured start-body XY and
   yaw effects together, searches only the six measured forward/left/right
   2/5-tick pulse cells, and distinguishes orientation from net signed winding.
   Its bounded beam search and endpoint excursion bound are not completeness,
   collision checking or motion permission. Fit the initial table only from the
   older nominal pulse episodes; room responses are descriptive transfer data,
   not fitting inputs. Two samples per action cell are insufficient for a
   calibrated response or uncertainty model. The offline diagnostic is a
   software/planning check, not a physical counterfactual or learned-JEPA result.
3. Implement a distinct continuous feedback controller using this model as a
   baseline: observe current RGB-D/gyro pose, jointly plan XY and yaw, execute
   only the first supported pulse, brake until actually quiet, observe again,
   then replan. Preserve sensor failure latches, original final pose/hold
   criteria, mission budgets, native supervision and the same uninterrupted
   observation frame. Do not execute an open-loop plan or count predicted
   endpoints as arrivals. Explicitly accumulate gyro-supported signed yaw for
   winding tasks; orientation-only tasks must not imply a winding guarantee.
   Planning diagnostics must log model identity, selected action and predicted
   versus observed complete pulse/brake response. SEARCH_EXHAUSTED is not proof
   of physical infeasibility; handle it explicitly without fabricating progress.
4. Freeze a new bounded physical development protocol before launch. Include
   both quarter/half-turn directions and the full observed-waypoint return,
   actual fresh starts, separate nominal and low-friction outcomes, full native
   hold/signed-yaw/swept-body scoring and exact raw controller replay. Do not
   retrospectively compare changed scenes or starts as a causal success-rate
   improvement. For a controller comparison, preregister genuinely paired
   scenes, seeds, initial states, sensor opportunities, gait and budgets.
   Plan the total storage cost before new collection; retain the 10 GiB reserve.

The six-cell mean model omits state/history, action-onset/braking dynamics and
friction dependence. Nominal turn-translation prediction error is already about
6 mm per pulse, and the old forward-yaw means transfer worse than zero yaw in
both nominal room runs. Low-friction displacement errors are larger. Old
fitting endpoints use a fixed 20-tick brake; feedback responses may wait longer
for quiet. Do not interpret these mismatched horizons as a stationary physical
transition model or use their descriptive residuals as calibrated safety bounds.
If nominal feedback still fails, diagnose the logged action-specific residuals
and collect prospective state/history-conditioned transitions, not another
sequence of pose-filter or tolerance tweaks.

Backward and arc commands appear in the gait training bank but have not been
physically characterized by the current pulse studies. They are an explicit
coverage avenue, not permission to add unmeasured effects to the current table.
Use a new bounded characterization if the measured vocabulary proves inadequate.
Pure-turn position constraints are an assay choice, not a necessary feature of
all maze maneuvers; future path-based maneuvers may accommodate translation, but
must check body sweep and still verify the requested final/home pose. Do not
retroactively loosen the present turn criteria.

## Connect execution to the scientific experiment

Once continuous execution is demonstrated, wire actual sensed branches and
marker observations through the existing episodic visit/attempt machinery.
The room program stores measured metric waypoints, but its route order is
scripted and its separate raw route-bridge owner has not demonstrated a physical
memory benefit. Retain UNKNOWN place identity for aliased views. Require actual
backtracking and independently verified marker/home outcomes in a connected
maze; neither a stack pop nor a local goal is a verified edge or home arrival.

Build task-driven sequences with action, onset, settling, body state and RGB
history. Split by layout before training and selection. Compare persistence/
geometric and empirical dynamics, a supervised action-conditioned predictor,
and JEPA on matched sensory data, action vocabulary and resource budgets.
Evaluate both prediction quality and completed navigation; a predictor that
improves latent loss need not improve action selection. Separate predictive
training, online multi-step rollout, and persistent memory contributions with
matched ablations. Use independent layouts and multiple training seeds; report
uncertainty at those levels rather than treating correlated pulses as trials.
Keep legacy sealed roles inaccessible and V4 permanently development-only.

Restore actual optical/self-occlusion/near-field constraints and observed body
sweep; native collision stopping is external supervision, not online clearance.
Address sensor calibration, noise/dropout and timing with physics not paused.
Current ideal hidden-robot RGB-D/gyro and continuous floor cannot establish
deployment validity. Mechanical energy remains unavailable without torque/power.
Complete bounded real-platform tests when hardware access permits. None of the
intermediate software, offline planning or room results completes the goal.
