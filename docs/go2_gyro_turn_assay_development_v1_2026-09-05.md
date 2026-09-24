# Paired ideal-gyro versus timed Go2 reorientation assay, V1

Specified before execution. Eighteen trials: +90°, −90° and +180° relative turns,
three initial world headings (−0.15,0,+0.15 rad), and two controllers. Each pair
shares seed2026091600+case_index (nine fixed cases), spawn(0,0,heading), the large
5-m enclosed arena from the earlier sensor probe, corrected gains20/0.5 and
the fixed gait. No prefix route, model fitting, seed replacement or coefficient
search. These are correlated actuator/sensor fixtures, not independent mazes.

After1.5s settling, the gyro controller uses only actual causal packet body rates
and the supplied relative target. It has the already tested yaw cap±0.35rad/s,
gain1.5, heading tolerance0.08rad, projected heading-rate tolerance0.1rad/s,
0.3s dwell and12s limit. The comparison requests signed0.35rad/s for
ceil(abs(target)/(0.35*0.1)) command ticks, then three zero ticks. Its scheduled
completion is not physical success. Both receive five further zero release ticks
and the same12s maximum controller budget,0.1s commands and0.002s physics.

Record every actual RGB/body/control packet, gyro decision/relative rotation,
requested/post-slew command and raw physical trajectory/contact packet. Simulator
contact/stability monitors remain outside the controller. Stop at first physical
violation. Sensor/timeout failures stop rotation and still issue the fixed safe
release if physics remains available. All failures count; no retries or longer
attempts. Integrity failures preserve partial artifacts and terminate the study.

Shared physical endpoint: controller finished (gyro COMPLETE or timed schedule
COMPLETE_TIMED), no physical stop/contact, final relative heading error≤0.12rad,
maximum base xy drift from turn start≤0.15m, full0.5s release whose last0.2s has
world xy speed≤0.1m/s and abs(world yaw rate)≤0.25rad/s throughout, and final
height≥0.2m with absolute roll/pitch≤0.5rad. Relative heading is the forward axis
of R_start^T R_current projected in the initial body xy plane—not world yaw.

Additionally measure the maximum gyro-estimated heading error against raw physical
relative rotation at every recorded controller boundary. The gyro estimate must
agree within0.04rad to support using the controller's heading tolerance; report
this separately from the shared physical endpoint, not as a favorable baseline
penalty. Report each paired case's heading error, drift, stopping, duration,
contact and reference agreement. No maze-level confidence intervals on this arena
fixture. No runtime tolerance/parameter changes based on results.

Freeze source/gait/spec hashes at launch. Audit all actual packet/controller
replays, raw time/command/contact/sensor/camera relationships, physical reductions
and exact paired settling physics/histories. RGB is recorded for prospective
branch inspection but is not steering the gyro controller, so rendering variation
must not be mislabeled a failure of gyro feedback; preserve original images.

This qualifies, at most, an ideal-sensor in-place-turn primitive under the stated
fixtures. It does not establish calibration/noise/bias robustness, translational
odometry, narrow-maze clearance, future feedback commands known to JEPA,
exploration/memory/return, or hardware transfer. Output exactly
`.generated/go2_gyro_turn_assay_development_v1_attempt_001`.
