# Bounded rotation geometry adapter: numerical failure resolved in saved queries

The new adapter and recorded-output diagnostic are complete and audited. All
2,344 accepted pose states now support the unchanged strict geometry predicate.
All 708 formerly rejected gyro coverage queries have conditional results; all
previously available coverage decisions remain unchanged. Estimator matrices,
translations, keyframes, histories and physical error hypotheses were not changed.
No model or physical trial was rerun.

The adapter constructs a proper matrix Q and explicitly adds a per-shape bound
on the finite-map difference (R-Q)x. Its maximum added displacement allowance is
1.173e-12 m, about 1.2 picometres. This is numerical bookkeeping, not a calibrated
bound on the true robot. Unknown physical uncertainty remains unknown; the saved
query experiment retains its predecessor's explicitly unvalidated zero-additional-
physical-error hypothesis. It cannot authorize a command.

| Result | Fit joint | Fit gyro | Validation joint | Validation gyro |
| --- | ---: | ---: | ---: | ---: |
| Available conditional coverage queries | 586 | 586 | 586 | 586 |
| First full 27-shape coverage frame | 266 | 265 | 264 | 263 |
| Full-coverage frames | 320 | 321 | 322 | 323 |
| Estimated-covered / native-uncovered shape queries | 0 | 0 | 0 | 0 |
| Estimated-uncovered / native-covered shape queries | 41 | 38 | 20 | 17 |

The original gyro/no-coverage result was a numerical-interface failure, not a
scientific benefit of joint rotation estimation. The original failed coordinator
and its later explicit-status result remain immutable. This new diagnostic is
not an additional independent trial or fresh validation.

## Independent checks

Extended-precision calculations checked all 506,304 corners of the 27 body-shape
boxes across the four histories. Every corrected finite point map remained inside
its supplied numerical allowance. Actual maximum corner movement was 9.316e-13 m;
minimum containment margin was 2.563e-14 m. All 2,344 adapted coverage queries
matched direct image-cell enumeration instead of the production prefix counter.
Kinematic support boxes and measured-surface construction remain shared, so the
audit is not an independent physical sensor/robot calibration proof.

The 15 new adapter/startup tests passed; the combined focused run with existing
coverage/surface tests passed 42 tests. Full regression passed all 2,192 tests
across 176 explicit files in 179.69 s.

## Startup finding and next action

Both initial postures have zero camera-frustum-complete floor footprints. All 27
enclosures are entirely before the .2-m optical range boundary; their total
optical-depth span is approximately [-.713683,.019897] m. This is an actual
camera/geometry observability deficit, not estimator drift. The sensor packet
has no support/contact modality, and the renderer hides the robot.

Proceed with the [startup-support sensing implementation plan](go2_startup_support_sensing_next_steps_2026-09-06.md):
an explicitly new self-occlusion-aware sensor study and carefully separated
support-sensor evidence, followed by validated local action response. Do not
introduce an undocumented flat-floor prior or an open-loop motion prelude.
The JEPA/memory/rollout/novel-maze/hardware scientific goals remain incomplete.

## Identities

Root: `.generated/go2_bounded_rotation_geometry_adapter_v1_attempt_001`.
Launch binds 537 sources and 11,301 inputs plus native/OpenCV.

| Artifact | SHA-256 |
| --- | --- |
| launch.json | b907e7075397a1ca1671ff3866028f265a633f4d775f51592c53452f80673978 |
| result.json | 1984b2dcd11965e12012418a85cd5b9a1e41ac0acdba9d982e476fb236eae202 |
| predictions.json | 9d459ba1328e9153ada43ab0bfccfaa1ac2ca587cd9f60b7e515cff4f9f5b444 |
| corner_and_cell_audit_launch.json | e2f69d0231c83cfeae544059d5e053b792a1bbe5c1fe09486f94032920082f79 |
| corner_and_cell_audit.json | 3bf1c86951c340bf7c2df1625507903a45fa8b5a557a543d5e5c2d74591c845f |
