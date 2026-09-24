# Next: integrate visual-led motion without making contact dropout a stop rule

## Execution update

The interface is implemented. Actual stream replay preserved all 904 frozen
visual outputs and marked all 3,600 intermediate queries unobserved; no contact
weight or motion permission was introduced. See the
[integration result](go2_visual_led_motion_integration_result_2026-09-06.md).
The next implementation is the
[bounded closed-loop visual-servo stage](go2_bounded_visual_servo_next_steps_2026-09-06.md),
with controlled-floor and ideal-camera assumptions explicit. It is not a
sensor-qualified maze/hardware claim. The original implementation list below is
historical; do not repeat it or the completed estimator replay.

The [frozen friction replay](go2_friction_frozen_rgbd_dropout_result_2026-09-06.md)
shows that both visual estimators remain available throughout the observed
contact dropouts. The preceding contact-only diagnostic must not become an
unnecessary prerequisite for all navigation. RGB-D plus IMU is already a
multimodal baseline. The scientific destination remains learned predictive
navigation and useful memory, not an ever-expanding collection of sensor assays.

## Immediate implementation

Build a separately named causal motion-evidence interface around the unchanged
visual estimators. Keep pose, contact evidence and motion permission separate.

1. Each update exposes observation identity, calibration, measured/available
   times, visual pose/status, keyframe provenance and explicit unknown physical
   error bounds. Consume only current/past RGB-D and IMU. Terminal visual failure
   remains terminal; do not reset a failed chain or invent translation from
   integrated commands.
2. Contact observations are optional, co-timed, separately typed diagnostics.
   Missing or contradictory contact does not erase an available visual update.
   Do not substitute zero, require a static-foot consensus at every gait phase,
   or manufacture Gaussian uncertainty from inlier scatter. Preserve ideal-vector
   and feasible hardware-channel capability distinctions.
3. Implement temporal alignment and contact/visual disagreement reporting first.
   Retain the visual-only baseline. No weighted translation update is justified
   until a separately fitted uncertainty/robustness model improves an explicit
   outcome on genuinely fresh validation. Cross-foot agreement cannot rule out
   common-mode slip. At intermediate control times, expose stale/unobserved
   translation as such; gyro updates orientation, not otherwise unknown position.
4. Test sensor-only stream replay, dropped/stale/misaligned channels and terminal
   failure propagation. Freeze the new interface before any physical validation.
   The already exposed friction recordings are integration/fitting fixtures, not
   independent validation. Do not repeat the completed estimator comparison.

## Next physical milestone

Combine this interface with explicit current-support, future-footfall and
body/leg/braking evidence for a short sensor-only start/forward/turn/brake task,
with native state used only for stop supervision and subsequent evaluation.
Remove the need for a privileged motion prelude by supplying justified observed
near-field/support evidence—not by declaring unseen floor free. Resolve camera
aperture/self-visibility and the remaining depth/raster discrepancies before
using a proposed mount as a qualified sensing source.

Prospective action/brake response must account for friction-dependent yaw and
stopping. Define actual randomized initial states/conditions and distinct
fitting/validation roles before new exposure; verify the differences in the
recorded physical traces. Merely changing seeds did not create new dynamics in
the nominal challenge. Use bounded speed, distance, duration, collision stops,
resource budget and exclusive partial-result retention. Finite observed maxima
are not universal safety margins. Any honest task simplification must remain a
labeled stage, not a replacement for the full scientific objective.

## Return promptly to the scientific comparisons

After reliable local execution, integrate persistent place/branch memory and
complete exploration, wrong-branch recovery, hidden-goal discovery and return
home. Compare identical sensors, gait and budgets across geometric/supervised
and JEPA representations; independently ablate predictive training, genuine
online multistep rollout and memory. Use independent maze layouts and training
seeds, not correlated frames as experimental replicates. Measure complete-task
success, collisions/stops, path/energy/time cost, uncertainty and full-loop timing.
Test sensor degradation and obtain bounded hardware evidence when available.
None of those end-state requirements has been completed by the friction replay.
