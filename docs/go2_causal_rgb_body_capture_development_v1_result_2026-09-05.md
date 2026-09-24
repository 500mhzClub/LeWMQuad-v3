# Causal RGB/body capture V1: result

Nine fixed command probes completed without physical contact or early stop.
All passed the declared release-motion window. Collection produced **414 native
RGB/causal-history packets and 2,700 ideal simulated body-sensor samples** over
27,000 physics samples. Full raw-artifact audit PASS. See the
[protocol](go2_causal_rgb_body_capture_development_v1_2026-09-05.md).

## Measured command response

Means below use the last excitation second, in body coordinates. These are
descriptive development measurements, not independent-maze statistics.

| Probe | Requested vx / yaw | Mean measured vx / yaw |
|---|---:|---:|
| Zero | 0 / 0 | 0.0002 / 0.0007 |
| Forward | 0.20 / 0 | 0.1887 / -0.0078 |
| Reverse | -0.20 / 0 | -0.1874 / 0.0100 |
| Left | 0 / 0.30 | -0.0074 / 0.2436 |
| Right | 0 / -0.30 | 0.0131 / -0.2878 |
| Forward-left | 0.20 / 0.30 | 0.1814 / 0.2774 |
| Forward-right | 0.20 / -0.30 | 0.1987 / -0.2996 |
| Reverse-left | -0.20 / 0.30 | -0.2091 / 0.3033 |
| Reverse-right | -0.20 / -0.30 | -0.1955 / -0.2962 |

Units are m/s and rad/s. Every intentionally excited forward/yaw channel has
the expected mean sign. Tracking is not exact or symmetric: pure left yaw is
about 19% below its request, and reverse-right has mean lateral velocity
0.0340 m/s despite a zero lateral request. These residuals motivate measured
state feedback and action-conditioned physical prediction, not a claim that
body sensing or JEPA has already improved decisions.

Channels are nonconstant under excitation. Forward-left joint-position standard
deviations span approximately 0.016–0.076 rad across twelve joints. Ideal specific
force includes transients (body-z absolute maximum about 33.5 m/s² in that probe);
it is not simply a stationary gravity vector. This establishes signal acquisition,
not usefulness or fidelity to a physical IMU.

## Acquisition and policy boundary

The [live recorder](../scripts/run_go2_causal_rgb_body_capture_development_v1.py)
uses native read-back-verified checkpoint gains and captures RGB before each
command. The [sensor adapter](../lewm/simulated_body_observation_development.py)
feeds the causal buffer with ordered gyro, specific force, named joint
position/velocity, and separately identified past applied commands. Specific
force uses a causal 20-ms backward velocity difference; its first sample is
invalid. Simulation clocks imply zero transport latency and pause during
rendering. Physical IMU bias/noise, lever arm, calibration and synchronization
are not established.

The [policy-only loader](../lewm/causal_rgb_dataset_development.py) reads only
the explicit observation manifest, history arrays and RGB files. Extra tensor
fields, unknown channels, oracle metadata, future/late samples and path escapes
are rejected. World pose, velocity references, camera-world transforms, geometry,
contacts and outcomes remain separate audit artifacts. Reset identity is
bookkeeping, not a scene-ID feature. The loader does not discover datasets,
train a model, or supply hidden goals.

The [auditor](../scripts/audit_go2_causal_rgb_body_capture_development_v1.py)
reconstructed every 50-Hz sensor value from raw references with a separate
vectorized calculation and every history window with explicit causal indexing.
It verified source/gait/gain identities, native force attribution, command tape,
global clock, RGB pixels, physical capture boundaries and proper optical frames.
The policy reader was used for all 414 packets: the actual stored interface,
not only an in-memory fixture, was exercised.

Per-probe median capture wall times ranged from roughly 8.5 to 11.7 ms. This
includes local image capture/serialization, not complete policy inference or
real-time sensor-to-actuator latency. A turning image was visually inspected:
walls, floor and a corner are present. The arena is intentionally low-texture
and lacks maze diversity.

## Evidence identity and next action

Output: `.generated/go2_causal_rgb_body_capture_development_v1_attempt_001/`.
Result SHA-256:
`2225f0334dc1320cc06e47df9db8f4ce27dd8aebedb624c23a15ac337279c001`.
Audit SHA-256:
`f4df4f17ed08c9e4edf2d9daa8a1392b4af45e685e48e92e498281e123e99a10`.
The explicit combined suite passes **309 tests**. Previously bound executed
source and tracked frozen files remain unchanged.

This completes live ideal-sensor acquisition/interface work, not a sensor-driven
policy, JEPA benefit, hardware-valid sensing or maze navigation. Do not train a
generalization claim on these nine correlated open-loop probes. Next: attach
this recorder to fresh multi-junction development routes, count route failures,
and collect matched alternative actions with scene-level data-role separation.
That supplies observations and executed outcomes for direct-policy versus JEPA
comparisons.
