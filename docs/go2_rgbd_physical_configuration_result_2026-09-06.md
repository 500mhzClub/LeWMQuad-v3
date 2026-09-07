# Physical configuration evidence: floor/obstacle separation implemented

The new RGB-D-owned adapter resolves the old blanket floor-return veto without
deleting pose uncertainty or authorizing motion. On the saved failed terminal
observation, seven supplied yaw configurations each have conditional non-floor
clearance for all 27 collision primitives. Twenty-one primitives have observed
physical floor separation, four are foot-contact candidates only, and two front
lower-calf primitives remain ambiguous. No turn or new mission was executed.

This advances the first step of the
[floor-factored navigation plan](go2_floor_factored_navigation_next_steps_2026-09-06.md).
It does not repair the recorded mission retrospectively or establish a future
gait, navigation policy, JEPA advantage, real-time operation or hardware readiness.

## Implemented interface and checks

`lewm/rgbd_physical_configuration_evidence_development.py` prepares the existing
`RGBDInertialRayMemory` owner's retained immutable depth/up observations. It
binds each observation's depth, clock, pose and error scales, selects one fixed
query-independent measured plane hypothesis per frame, and reuses exact URDF
support functions and observed floor/beam queries. It does not reintegrate gyro
or translation, reset history, add a setup-region exemption, or accept world
geometry as sensor input.

Queries preserve individual physical primitive identity and separately report:

- complete observed non-floor clearance and retained non-floor conflicts;
- unpadded physical floor gaps, observed floor-footprint coverage and
  incompatible/penetrating ground witnesses;
- contact candidates restricted to the four exact foot spheres, never generic
  calf links or height bands.

Missing views remain unknown. A wide historical gap interval does not itself
contradict another complete observation's positive separation. Covered definite
penetration and incompatible covered plane witnesses remain vetoes. An upward
plane is only a measured hypothesis; it supplies no standalone traversability
or contact permission. Every configuration result retains
`navigation_action_permitted=False` and `future_gait_qualified=False`.

The 25 new synthetic tests cover reference parity, immutable observation/pose
bindings, no reintegration or plane reselection, lifecycle refresh, missing views,
wall and low horizontal obstacles, elevated planes, penetration, exact foot roles,
proper transforms, error accounting and fault latching. Together with predecessor
geometry/aggregation tests, 87 focused tests passed. The full explicit 160-file
regression passed 1,995 tests in 173.01 s.

## Recorded diagnostic and independent reference replay

Protocol: [configuration evidence V1](go2_rgbd_physical_configuration_evidence_development_v1_2026-09-06.md).
Output: `.generated/go2_rgbd_physical_configuration_evidence_development_v1_attempt_001`.
Runner: `scripts/probe_go2_rgbd_physical_configuration_evidence_development_v1.py`.
Independent auditor: `scripts/audit_go2_rgbd_physical_configuration_evidence_development_v1.py`.

Both runs exactly reconstructed all 219 original controller decisions from saved
sensor packets, stopping at 23.3 s before the original zero tail. Forty retained
views were prepared. The terminal measured joint posture was queried at zero
translation and gravity-axis yaw 0, +30, -30, +60, -60, +90, -90 degrees. These
are seven queries on one previously seen development configuration, not seven
independent trials or a swept-motion experiment.

All original global-endpoint-sum pose envelopes remained unchanged. Additional
declared hypotheses were 5 mm endpoint error, .002 normal error, .001 up error,
1 mm plane-offset and range errors; non-floor padding remained 4 cm. These are
uncalibrated development assumptions, not estimates fitted to this run.

Every orientation produced the following counts:

| Evidence category | Primitives |
| --- | ---: |
| Conditional non-floor clearance | 27 |
| Non-floor conflicts | 0 |
| Observed physical floor separation | 21 |
| Exact foot-contact candidates with non-floor clearance | 4 |
| Non-foot floor intersection still possible | 2 |
| Covered definite penetration / incompatible ground / unknown floor coverage | 0 / 0 / 0 |

The unresolved shapes are `FL_calflower1:0` and `FR_calflower1:0`. At zero yaw,
their best covered witnesses are the initial 1.5 s observation. Their nominal
minimum floor gaps are 27.960 and 29.325 mm, but the full supplied-error intervals
are respectively [-3.713, 59.634] mm and [-1.898, 60.549] mm. Their status is
`GROUND_INTERSECTION_POSSIBLE`, not observed penetration, missing floor coverage,
or foot-contact permission. This distinction is precisely what the old padded
height-disk query could not preserve.

All seven complete outputs matched the independent reference beam and floor
implementations. The original failure, fusion state, gyro count and global error
scales were unchanged. The 478-source/6,127-input/native/OpenCV closure and output
bindings were checked before and after.

Recorded preparation took 1.279 s, first compiled configuration query 1.302 s,
and subsequent queries 260–275 ms. A CPU regression process was concurrently
active: these are diagnostic workload timings, not isolated deployment benchmarks.
They nevertheless do not demonstrate the required 100 ms end-to-end deadline.
Any optimization must retain reference agreement and undergo dedicated timing.

SHA-256 identities:

- Launch: `91df22609d12ed954ca66efa122bb92a48586c2c5d634fbee478d12c752c7c58`.
- Result: `d75e8a9f044fa22b5017bb17a5a796143cab30ace052230193cf9735718a85c3`.
- All-seven reference audit: `f915202645ffc6817d6222b259a22e3c77ae06655cd516fe8c0d606dd83d1059`.

## Next scientific action

The adapter exists and the failure is now factored, but physical action permission
is still unavailable. Next derive joint pose/plane sensitivity for the **actual
RGB-D estimator**, preserving shared image/depth/calibration/gyro errors and
velocity-prior history. Existing `relative_pose_uncertainty_development` and
`correlated_moment_sensitivity_development` provide tested mathematics, but the
latter's depth-only nominal integrator is not a drop-in RGB-D error model.

Use explicit shared sources and propagate them through the actual depth/point
registration and fusion, preserving correspondence/rank changes or rejection as
limitations rather than hiding them. A plane fitted from the same depth as motion
is correlated with that motion. Do not treat its error and pose error as independent
or claim first-order moments are calibrated safety bounds. Test shared-error
cancellation, rotating frames, bias accumulation, rejected matches and dropout;
validate coverage on independent development motion with sensor/calibration faults.

In parallel scientific scope, future body/foot response and braking must still
be measured and validated before connecting any configuration evidence to a
controller command. Then complete genuine discovery/backtracking/return missions,
and only then isolate predictive-training, genuine multistep-rollout and memory
contributions on matched independently generated layouts/seeds. The full goal and
all previous negative results remain intact.
