# From the verified startup turn to complete novel-maze missions

Latest update: the [causal RGB-D correspondence diagnostic](go2_rgbd_correspondence_motion_diagnostic_v1_result_2026-09-06.md)
passes synthetic tests but rejects all80A/Bimagepairs; the13Bweakpairs have no
keypoints. Source disables textures, and its alternate texture path changes
collision geometry type. Next is the [controlled visual-information/full-mission plan](go2_rgb_information_and_whole_mission_next_steps_2026-09-06.md):
new appearance-only rendering with matched collision geometry, textureless and
repeated-texture controls, then fresh motion and continuous complete missions.
No old source/result, threshold or uncertainty budget was changed. Tests1833
pass; fullmission0/2 and noJEPA/hardware claim remain unchanged.

Current evidence: [validation B](go2_action_motion_validation_development_v1_result_2026-09-06.md)
was executed once against the frozen A-only response. It failed the observer's
unchanged80mm uncertainty budget at5.3s after24/28 commands; depth was rank2 from
4.1s. Raw audit reconstructed the stop and actual zero tail without physical
violations. Learned body translation improved, but not overall articulated error
versus fixed posture. All37 moving loops exceeded100ms. Follow the new
[post-B plan](go2_post_B_observability_and_whole_mission_plan_2026-09-06.md): test
causal RGB-D motion constraints, preserve genuine missing information, add
uncertainty-reserve observation actions and integrate complete missions. No B
retry or budget increase. Whole-task0/2 and noJEPA/hardware claim remain. The
following updates are retained historical context, superseded by this status.

Latest physical development result: [motion identification A](go2_action_motion_identification_development_v1_result_2026-09-06.md)
completed all28 forward/turn/braking targets and the real zero tail. Independent
replay passed3000 physics samples,46 RGB-D frames and43 decisions without a
physical/sensor stop or padded-region violation. The same owner continued to6s.
However, the non-learned qdot predictor missed a foot centre by316mm at0.4s;
posthoc fixed posture still missed by71.7mm. No50mm motion error bound follows.
Next fit/freeze an action/state response model from A only, then run the fixed B
validation schedule in new source with complete outer-loop timing. Current
component sums180–265ms already exceed100ms and omit bookkeeping. No maze,
independent-layout, hardware or JEPA success is claimed. Tests1796/145files pass.

Latest implementation: [factored configuration and trajectory evidence](go2_factored_trajectory_progress_2026-09-06.md).
Obstacle conflicts, covered ground relations and positive residual visibility
are now separate, with contradictory ground witnesses retained. Time-indexed
body/joint predictions are checked at every node with explicit nearest-endpoint
intersample allowances. The included command/joint-velocity baseline is neither
learned nor execution-validated. Recorded reference comparisons pass, but no
future motion or support permission follows. Implement and execute the
[two-run motion identification/validation specification](go2_action_conditioned_motion_validation_next_execution_2026-09-06.md)
next, then integrate the complete mission. Do not substitute more static probes
or hypothetical error radii for physical prediction/braking evidence.

Latest diagnosis: [per-view ground-veto attribution](go2_configuration_ground_veto_progress_2026-09-06.md)
reproduces all original negatives and attributes every lower-leg veto at both
farther tangent queries to historical uncertainty plus plane-family returns.
The current 1-m view supports positive lower-leg separation under the supplied
model; the 0.75-m query still has incomplete visibility. Implement distinct
non-floor conflict, physical ground-relation and positive-coverage channels;
do not treat uncertain intersection as a measured wall or discard actual
contradictions. Pair that interface with action-conditioned body/foot/braking
predictions and fresh bounded execution validation, then a complete mission.
No additional static-offset sweep is needed to establish this failure cause.
The original consumer/outcomes stay unchanged. Tests: 1,749/142 files passed.

Latest implementation: [observation-bound residual/configuration queries](go2_observed_setup_configuration_progress_2026-09-06.md)
now combine the finite initial condition with retained depth, keeping whole-query
vetoes and current-pose cancellation. Recorded configurations beyond the starting
region obtain some positive sensor clearance but retain lower-leg/ground
ambiguities and unknown residuals. A separate gravity-tangent diagnostic removes
the pitched-translation foot-penetration flags, not all negatives. Next resolve
the explicit ground/contact relation and validate action-conditioned trajectories
and braking before integrating a new complete physical mission. Pose probes are
not executed trajectories or a replacement for the full objective.

Implementation update: [continuous owner and initial-clearance partition](go2_continuous_startup_handoff_progress_2026-09-06.md)
are now tested and verified on all 15 saved startup/tail observations. The same
state is retained through 2.9 s; all 27 current primitives are conditionally
covered by the supplied initial region, not by observed floor clearance.
Next connect residual-volume sensor queries and explicit support/motion models
to the complete mission controller. No fresh mission or learned-policy result
has been produced; the requirements below remain in force.

The objective remains RGB plus deployment-valid sensor navigation with tested
JEPA predictive-training and online-rollout contributions, online discovery/return
memory, independent-layout evidence and bounded hardware evidence when available.
The startup result changes one prerequisite, not that objective or the 0/2
full-mission record. This plan follows the
[actual turn and terminal-memory evidence](go2_startup_observation_turn_result_2026-09-06.md).

## Immediate implementation: continuous state and explicit evidence boundaries

Implement a new startup-to-navigation handoff with a single persistent sensor
observer and memory owner. Transfer state only after a valid completed turn,
consume every intervening stopping-tail frame exactly once, preserve the initial
velocity-prior contribution and remaining position uncertainty, and reject stale,
missing, wrong-episode or fault-latched state. Do not restart the frozen terminal
controller or reinitialize pose to remove uncertainty. A fresh experiment must
declare its own source, setup conditions and complete mission endpoint.

Separate three things in the navigation interface: supplied initial non-floor
clearance, sensor-observed new non-floor clearance, and explicitly modelled ground
support/contact. Five terminal views still supply no own-body floor coverage.
Therefore a blanket requirement to see the entire currently occupied body volume
cannot simply be carried into navigation and expected to pass. Conversely, a
plane hypothesis or absence of an obstacle return cannot fill unseen space.

The handoff must expose unknown space and validity/expiry explicitly. Evaluate
the initial known-clear region and newly observed swept volume as a union of
evidence with separate provenance, not their filled bounding box. In a new
static-maze protocol, any longer-lived initial-region assumption must be declared
up front, independently checked and identical across comparison arms; never
extend this trial's two-second condition after the fact. Dynamic-obstacle tests
must challenge that assumption. If deployment cannot supply the assumed start
and support conditions, test additional deployment-valid sensing or changed
camera coverage rather than treating evaluator truth as runtime evidence.

## Motion model and the next complete physical mission

The 0.7444-m all-joint/all-orientation radius is defensible for the startup
envelope but is too conservative to adopt blindly as a maze robot footprint.
For translation and narrow passages, implement action-conditioned short-horizon
body/foot motion and braking predictions. Keep measured current posture distinct
from predicted future posture. Validate prediction errors and contact outcomes
on fresh bounded development motions; report empirical coverage as empirical,
not a universal safety certificate. Do not spend further iterations tightening
static floor masks while leaving future motion or sensor blind space unresolved.

Connect these interfaces to the existing complete discovery/marker/return task,
including junction decisions, visited-place memory, backtracking and verified
home arrival. The next mission must run continuously from checked startup to a
task terminal, with native evaluation-only contacts and an actual stopping tail.
Do not count a turn, a few traversals or a checkpoint as mission completion. Use
new source/output and preserve all old failures. First test a complete development
mission; expand to independently generated layouts and starts after the failure
analysis, not merely more versions of the startup arena.

The measured loop is already over 100 ms before all execution overhead. Profile
and remove redundant depth/preparation/query work while preserving exact evidence
and decisions, then time the full sequential loop. Alternatively, preregister a
slower decision rate and rederive history, command, stopping and uncertainty
contracts together. Simply running at 10 Hz in simulation time does not solve
wall-clock latency or deployment staleness.

## Required scientific comparisons, not optional polish

Once continuous execution works, compare geometry-only, matched supervised
prediction and JEPA prediction using the same data, architecture capacity,
training budget, sensors, controller, setup assumptions and seeds. Separately
compare no learned lookahead, one-step prediction and genuine multistep online
rollout under matched compute. Rollout must propagate predicted state through
successive candidate actions and influence selection; scoring a single first
transition is not multistep planning. Include memory-disabled and memory-enabled
conditions to isolate return-navigation benefit.

Measure complete discovery-and-return success, collisions/falls, false home
declarations, path/time/energy or available proxies, interventions, uncertainty
stops and full-loop latency. Use independent layouts and multiple seeds with
layout-level uncertainty intervals. Include sensor noise/bias/dropout, texture
changes, altered starts and support/model mismatch; reserve final evaluation
outside the model-facing checkout. Keep the existing negative JEPA comparisons
visible until new matched evidence supersedes the particular claim tested.

Transfer the successful interface to real deployment sensors and a conservatively
bounded Go2 experiment when hardware access and operator safety arrangements
exist. No simulator contact check, ideal IMU, synthetic depth tolerance or
hand-designed planner is evidence of learned real-platform navigation by itself.
