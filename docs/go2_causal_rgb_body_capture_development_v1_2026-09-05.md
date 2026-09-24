# Causal RGB/body capture and command characterization V1

Specified before collection. Nine fresh development trials use the corrected
checkpoint gains 20/0.5, the pinned gait and existing physical/contact timing.
Commands are zero, +/-0.2 m/s forward, +/-0.3 rad/s yaw, and all four combinations
of those forward/yaw signs. Lateral command is zero. Each trial records 1.5 s
zero-command settling, 3 s fixed excitation, then 1.5 s zero-command release.
Fresh seed 2026090900+i and explicit trial identity; no snapshots or cached data.
An open 5-by-5-m box arena gives clearance for command characterization; this is
not a new maze or navigation evaluation. No adaptive command changes or retries.

Record native force/contact packets and all physical samples. Stop immediately
on the existing contact/body-stability criteria; retain partial data. Integrity
or infrastructure failure ends the study. These command responses may be
negative; no response criterion controls inclusion. Report mean body-frame
velocity and yaw rate over the last excitation second, tracking error, lateral
drift, and release endpoint/trailing-window motion. Do not confuse instantaneous
gait oscillation with mean command response or call this hardware calibration.

At 50 Hz, convert simulated body-origin angular velocity to body coordinates,
record named joint positions/velocities and construct ideal specific force as
R(t)^T * ((v(t)-v(t-20ms))/20ms - gravity). The first acceleration sample is invalid.
This is a causal backward-difference sensor model, not instantaneous point-IMU
ground truth. There is no lever arm, bias, noise, drift or transport delay.
Both ideal-sensor limitations and simulator reference values remain explicit.

Capture native fixed-mount RGB before each of the 45 post-settling command ticks
and once at the terminal boundary. Simulation pauses while rendering; measure
wall-clock capture time but do not claim real-time operation or hardware timing.
Image availability equals measurement time only in the simulated clock.
Store the 20-sample ordered gyro/specific-force/joint histories and 15-sample
past applied-command history, with time, validity and calibration identities.

Policy-facing artifacts contain only these causal sensor histories and RGB.
Keep world pose, velocity references, camera world transforms, geometry, contacts,
future commands and outcomes in separate audit/label artifacts. The observation
schema rejects extra fields and channels. No oracle local goal or scene ID is an
observation feature. No model is trained and these fixed open-loop probes do not
demonstrate a sensor-driven policy. The next dataset must additionally cover
scene diversity and alternative actions for meaningful JEPA/direct comparisons.

Audit source/gait/gain identity, raw contacts, command timing and tape, initial
state, every sensor value from the causal raw reference, every history prefix
and image binding. Include synthetic stationary/free-fall/rotated-body checks,
future-sample and privileged-field corruption controls. Completion means a
verified acquisition/interface package with measured command responses, not
the final navigation goal or deployment-valid physical sensors.
