# Next: retain motion observability when foot-contact hypotheses fail

## Execution update: frozen visual replay complete

Both unchanged visual estimators retained all 226 observations in each recording
and all 184 low-friction camera intervals containing contact dropout. The
independent accepted-pair/native/dropout audit passed. See the
[result](go2_friction_frozen_rgbd_dropout_result_2026-09-06.md) and the
[current integration plan](go2_visual_led_motion_integration_next_steps_2026-09-06.md).
Step 1 below is complete, not a request to rerun it. Implement visual-led motion
evidence with optional contact diagnostics; do not invent fitted contact weights
or make contact availability a new locomotion gate.

## Original staged plan

The [friction challenge](go2_support_friction_challenge_result_2026-09-06.md)
invalidates a contact-only continuity claim:815/1126lower-friction observations
lack consensus. The robot nevertheless completes the supervised tape. Keep
locomotion stability, contact-based odometry and environmental clearance distinct.

1. Replay the already frozen nominal joint-RGBD and gyro-RGBD pose models on both
   newly acquired recordings. No fitting, thresholds or keyframe changes. Save
   predictions before native scoring, retain terminal failures and score every
   missing-contact interval. Report image/depth observability, contact availability,
   drift and100ms displacement errors together. Neither camera estimates nor
   foot estimates are independent ground truth for one another. The nominal
  14.5sphysics prefix repeats fitting data and is not fresh dynamics validation.
2. Use those results to implement a NEW dropout-aware fusion procedure, keeping
   the old estimators immutable. Load/kinematic observations may be unavailable
   or contradictory; fuse them only with explicit uncertainty/robustness rules.
   Do not call missing force zero, demand a stationary-foot consensus at every
   walking phase, or label estimator availability as terrain/locomotion safety.
   Foot agreement cannot exclude common slip. A later changed fitting rule needs
   a fresh reserved validation with actual randomized physical initial states or
   conditions, verified from traces rather than seed numbers alone.
3. In the same staged execution work, fit a prospective action/brake response
   procedure that accounts for observed friction-dependent yaw and stopping.
   Freeze its fitting/validation split before new exposure. A scalar maximum
   error from two tapes is not a universal clearance radius. Current contact
   hypotheses do not establish future landing terrain or body/leg sweep.
4. Resolve optical aperture/clipping and remaining raster discrepancies before
   qualifying proposed camera mounts. Retain the current forward hidden-robot
   renderer only as a labeled compatibility diagnostic. Keep ideal three-axis
   load instrumentation separate from calibrated, hardware-available channels.

Then integrate a short sensor-only start/forward/turn/brake run with stop-only
native supervision and no privileged motion prelude. Complete persistent
place/branch memory, exploration/wrong-branch recovery/hidden-goal/home return.
Test JEPA predictive training, genuine online multistep rollout and memory
contributions against matched geometric/supervised baselines using identical
sensors, gait and budgets, across independent layouts and training seeds.
Full loop timing, robustness and bounded real-platform evidence remain required.
Do not replace those scientific aims with more isolated sensor assays.
