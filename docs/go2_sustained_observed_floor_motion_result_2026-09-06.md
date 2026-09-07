# Sustained motion V1: physical collection audited; full-body coverage not achieved

Two new supervised simulation trials completed and passed raw acquisition audit.
They establish useful sustained gait/turn/brake data and initially weak depth,
but **fail the intended full-body observed-floor objective**. Neither is a maze
mission, learned high-level policy, JEPA result or prospective safety validation.

## Evidence

| Measured quantity | Fitting trial | Development-validation trial |
| --- | ---: | ---: |
| Physics samples / RGB-D frames | 17,500 / 336 | 17,500 / 336 |
| Completed stimulus / extra zero ticks | 330 / 5 | 330 / 5 |
| Forward displacement over 15 s | 0.7161 m | 0.7257 m |
| Whole-tape path / net displacement | 1.2163 / 0.7900 m | 1.2219 / 0.8210 m |
| Left / right turn yaw change | +0.7838 / −1.0931 rad | +0.8462 / −1.0956 rad |
| Maximum active body speed | 0.1300 m/s | 0.1570 m/s |
| Native guard violations | 0 | 0 |
| Depth rank in every one of 335 motion pairs | 2 | 2 |
| Shadow accepted frames before budget failure | 36 | 36 |
| Maximum admitted position error | 1.803 mm | 2.081 mm |
| Maximum initial-view coverage, causal pose | 0/27 shapes | 0/27 shapes |
| Maximum initial-view coverage, evaluator pose only | 8/27 shapes | 11/27 shapes |
| Full-body initial-view coverage frames | 0 | 0 |

Both original shadow observers failed at absolute simulation time 5.1 s, 3.6 s
after the observation anchor. Failed owners were never invoked again. Remaining
motion was the fixed externally supervised collection tape, not navigation
continuing after a failed safety gate. No reset, extension, recovery or rerun occurred.

The extra 0.5-s zero tails displaced the body 1.021 and 1.154 mm. Maximum linear
speed over their last 200 ms was 1.810 and 2.041 mm/s; maximum angular speed was
0.01087 and 0.01491 rad/s. Braking response depends on preceding motion: the
fitting trial's first forward brake moved 25.919 mm, versus 4.955 mm in validation.
These measured outcomes are not worst-case stopping bounds.

## What the audit establishes—and does not

The new audit independently reconstructs native contact attribution, non-foot
ground guards, 50-Hz body and 500-Hz gyro samples, causal sensor histories, every
command and phase, 2-ms physics and 100-ms capture clocks, exact paired image/depth
conversion and hashes, body-camera transforms, setup admission, native actuator
and geometry identities, and procedural visual meshes. Analytic interior-depth
comparisons against the physical scene have maximum error 0.102 mm. All 672 saved
shadow outputs—including terminal and non-reinvoked states—match replay exactly.
The audit shares reviewed sensor/geometry functions with predecessor audits; it
is not an independently implemented simulator or sensor model.

Floor diagnostics retain the unchanged 0.1-mm range hypothesis, 1-mm surface tube
and 0.001 up-vector hypothesis. They use one initially measured surface, actual
measured joints, and **zero additional body-point error only for diagnosis**.
Causal queries use accepted sensor-estimated poses only. Evaluator queries use
simulator poses explicitly and cannot repair the failed estimator. The audit does
not establish arbitrary whole-history coverage, calibrated uncertainty, common
floor identity, non-floor clearance, or future gait/contact permission.

There are two physical instantiations with distinct declared seeds/poses, but
only **one geometry**. This is not novel-maze generalization. Validation was used
for acquisition audit and scoring the already-frozen original estimator, not
coefficient fitting. Its outcomes are now known; it must not be described as an
untouched test for later adaptive architecture/threshold selection.

## Why this did not meet the coverage objective

Requested speed is not achieved speed. The fitting trial travelled 0.7161 m under
a 0.12-m/s command held for 15 s: average net progress about 0.0477 m/s, not the
1.8-m displacement that command integration would predict. Raw command accounting
confirms the command was applied throughout; this is measured gait response.

A post-acquisition **fitting-only virtual fixed-posture design assay** queried the
same initial surface and initial joint posture at forward offsets 0–2.5 m in
0.05-m increments, with identity relative rotation and zero additional point
error. Coverage was 0/27 at 0.5 m, 7/27 at 0.75 m, 14/27 at 1.0 m, 25/27 at
1.25 m and 27/27 at 1.5 m. The first sampled full-coverage offset was 1.30 m.
This is not an actual executed configuration, a minimum over continuous distance,
a gait-envelope bound or a license to move. It explains why acquisition duration
must be designed from measured response and optical blind area, then verified.

The estimator has a separate limitation. At the fitting trial's last accepted
frame (5.0 s), accumulated point scale was 70 mm and inherited depth scale
8.874 mm, with zero surviving initial-velocity-prior radius. The unchanged
formula adds 2 mm for each point-complemented interval; at the next interval
the total crosses 80 mm. Depth was rank two from the first motion pair, and RGB
provided complementary constraints. This failure is not evidence of an observed
80-mm position error. Conversely, the small admitted native error cannot calibrate
or justify deleting the allowance, especially after censoring the long trajectory.

## Next execution

1. Use the **fitting trial only** to design a new explicitly named acquisition,
   not extend or relabel V1. A fixed 40-s forward segment at the same command is a
   candidate based on measured 0.0477-m/s response (~1.91 m extrapolated progress),
   with separately declared poses/seeds, turns, brakes and native supervision.
   Before launch, check the full declared travel envelope against walls, floor
   domain, camera visibility and disk capacity. Actual full-body coverage remains
   an acceptance measurement, never an assumed outcome. Do not increase speed.
2. Resolve the long-history estimator limitation in a **new model**, retaining
   V1's failure: evaluate RGB-D landmark/keyframe constraints for relative
   body-to-observed-surface pose while retaining separate global/return history.
   Compare against the unchanged incremental estimator. Do not reset global
   uncertainty, subtract arbitrary scalar radii, assume independent errors or
   loosen a threshold to make a run pass. Include correspondence failures and
   shared camera/gyro/depth/kinematic errors; freeze selection before new validation.
3. Fit prospective gait/turn/brake response using deployment-valid history and
   commanded actions, with native labels only for training/evaluation. Freeze
   features, fitting rule, horizon and error criteria before validation. Retain
   the earlier failed action-response validation; these two tapes do not erase it.
4. Integrate a common floor/non-floor and prospective-body interface, meet the
   timed loop, then complete real exploration, backtracking, hidden-marker
   discovery and home return. Establish JEPA versus matched supervised/geometric
   effects and genuine multistep/memory contributions on independent layouts and
   training seeds, robustness shifts, then bounded hardware when available.

## Reproducibility

Protocol: [sustained motion V1](go2_sustained_observed_floor_motion_development_v1_2026-09-06.md).
Collector: `scripts/run_go2_sustained_observed_floor_motion_development_v1.py`.
Auditor: `scripts/audit_go2_sustained_observed_floor_motion_development_v1.py`.
Output: `.generated/go2_sustained_observed_floor_motion_development_v1_attempt_001`.

Acquisition binds 501 sources and 6,659 inputs plus native/OpenCV dependencies;
2,072 expected output artifacts are present and bound. Audit binds 503 sources
and 2,074 acquisition inputs, plus its two detailed output reports. Source/input
bindings and detailed-report hashes were verified unchanged after audit.
Launched sources, protocol, acquisition and audit artifacts are immutable.

Fifteen collection tests and eight audit tests passed (including seven corruption
cases). The final combined 168-file regression passed **2,094 tests in 179.90 s**.
The preceding 167-file regression passed 2,086 tests in 186.04 s.
Physical acquisition overlapped regression; no deployment-latency claim follows.

- Launch SHA-256: `731656438060789d3d36b03cbd20f146d8b46d451d825e911254d59c70f3b575`.
- Acquisition result: `aa554d6398fadd5b21f0a298f2f463bd926aecc31d626609f062d7fa341aa1c9`.
- Audit launch: `ca75f35acebe8d4c72995155f1ca4591a6f20cf4fa60af348517362c5ebe913d`.
- Audit result: `ca84564e2139f50b12609b4ee53f5bbfb73a25f3387e500cae33cd532c7102b8`.

The full scientific goal remains active and unachieved. No JEPA training, learned
high-level policy improvement, successful maze mission or hardware actuation was
performed in this stage.
