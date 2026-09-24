# Next physical evidence: fixed contact-motion challenge

## Execution update

The challenge is now complete and audited; see the
[results and independence correction](go2_support_friction_challenge_result_2026-09-06.md).
Low friction produces major contact-estimator unavailability; the nominal early
physics trace repeats prior fitting data. Follow the
[contact-dropout multimodal plan](go2_contact_dropout_fusion_next_steps_2026-09-06.md)
next. The preparation instructions below are historical, not retry authority.

The [causal support diagnostic](go2_causal_support_kinematics_result_2026-09-06.md)
has useful fitting accuracy, but load persistence does not guarantee valid foot
kinematics. Do not spend another stage optimizing only that exposed recording.

## Implement and freeze before collection

1. Create a NEW bounded supervised collection and raw acquisition/audit path,
   recording the hypothetical ideal foot-vector sensor alongside RGB-D, joints
   and IMU at acquisition time. Keep force conversion on the simulator side;
   the consumer must receive no root pose, contact positions or terrain labels.
   Carry both unchanged V1 support hypotheses in shadow; do not let their
   nominal outputs authorize motion yet. Reuse the unchanged learned gait.
2. Predeclare a nominal-friction control and a lower-friction challenge with
   otherwise matched geometry, appearance, initial state and command schedule.
   Verify the actual native material parameters—not merely a configuration flag.
   Fix identities/seeds, friction values, output roots, resource bounds and
   independent stop-only supervision in the execution protocol before launching.
   This document is preparation, not a frozen executable protocol or permission
   to change a predecessor attempt. No hardware actuation is involved.
3. Retain the 1.3–1.5s initialization contract and record sustained forward,
   left/right turn and zero-command stopping intervals. Use a shorter new
   schedule sufficient to expose contacts and stopping, rather than copying
   the 26-second supervised prelude used to obtain historical floor coverage.
   Maintain the existing bounded command envelope and native speed/body/contact
   limits. Preserve partial trials and terminal stops; never restart because a
   challenge makes the model or physical guard fail.
4. Freeze scoring for all available frames and unavailable windows, by segment:
   velocity errors, cross-foot and RGB-D disagreement, force selection, contact
   point versus centre motion, stopping tails and sensor timing. Native contact
   velocity/location/type are evaluation only, loaded after predictions. Low
   cross-foot residual must not automatically count as correct motion: report
   common-mode error against native truth explicitly. Do not assume that a lower
   friction coefficient actually produces slip—measure the contact motion.

## Decision after the challenge

If contact hypotheses fail, determine whether the failure is insufficient
selected feet, changing contact normal, rolling/slip, initialization or sensor
error. Any revised weighting, robust consensus, contact-point model or fusion
rule must be a separately named fitting method with fresh reserved validation;
do not silently tune the frozen V1 gates. A larger error allowance fitted to
the largest observed residual is not a defensible clearance bound.

Use the evidence to implement the three separate local execution obligations:
current support consistency, observed prospective footholds, and body/leg plus
stopping sweep. An online observer may use load/kinematic observations as one
source and reject contradictions; none may fill unseen terrain or override
uncertainty with a simulator contact label. Resolve optical/aperture errors
before adding the proposed camera views to any policy. Keep hardware-feasible
sensor calibration separate from the ideal-transducer simulation arm.

The next navigation milestone remains a short sensor-only start/forward/turn/
brake run, without a privileged motion prelude, followed by persistent memory
and complete exploration, wrong-branch recovery, goal discovery and home return.
Then test JEPA predictive training and genuine online multistep rollout against
matched geometric/supervised baselines with identical sensors, gait, memory and
budgets on independent layouts and training seeds. Retain timing, robustness
and bounded real-platform evidence. This contact challenge is a prerequisite,
not a substitute for the scientific end state.
