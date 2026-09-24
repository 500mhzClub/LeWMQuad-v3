# Multi-pixel plane: numerical improvement, noise-sensitive surface gate remains

Implemented a fixed 18x18-pixel measured-plane estimator and a physical-footprint
query that actually consumes its fitted plane. A recorded, audited comparison on
two saved development observations reduced the initial view's maximum depth-gain
factor step discrepancy from 1.320 mm to 0.0229 mm (about 58-fold). However, all
four signed 0.1-mm independent-noise patterns were rejected by the retained local
triangle-normal gate. This is not a validated uncertainty bound or clearance
repair. No new physics, controller action, JEPA training or mission was executed.

## Implementation and tests

Source: `lewm/multipixel_floor_plane_development.py`.
Protocol: [fixed-patch V1](go2_multipixel_floor_plane_development_v1_2026-09-06.md).
Runner: `scripts/probe_go2_multipixel_floor_plane_development_v1.py`.
Output: `.generated/go2_multipixel_floor_plane_development_v1_attempt_001`.

The estimator selects the median eligible cell and its fixed radius-eight patch
without trying alternatives. Every pixel contributes once; no outlier deletion,
hole filling or hidden/world-floor labels are used. Missing/ineligible cells,
poor tangent conditioning, excessive individual residuals and disagreeing local
triangle normals reject the entire patch. The explicit new query independently
checks every physical footprint cell and preserves penetration, missing coverage,
non-floor uncertainty and lack of contact/future-action permission.

Twenty-five new tests passed. The first expanded focused run passed 51 tests
before two additional runner tests were added. The full regression then passed
**2,042 tests in 164 explicit files**, 174.52 s. An initial border test fixture
contained wall data rather than eligible floor; it was corrected before launch.
No launched source or original experiment was changed.

## Recorded finite-perturbation findings

Seventeen fixed members per frame: nominal, signed gain/offset at three source
amplitudes, and signed independent patterns at one float32 ULP and 0.1 mm.
Every perturbation passed through the actual float32 representation. Relative
pose and measured terminal joints were held fixed; this isolates the plane
component and is NOT joint RGB-D uncertainty propagation.

| Initial-view metric | Three-point predecessor | Multi-pixel fit |
| --- | ---: | ---: |
| Maximum gain-factor discrepancy between .01/.005 steps, all 27 primitives | 1.319862 mm | 0.022865 mm |
| Front-left lower-calf discrepancy | 1.162631 mm | 0.022634 mm |
| Front-right lower-calf discrepancy | 1.078708 mm | 0.015476 mm |
| Maximum signed finite gap change at 0.1% gain | 0.360626 mm | 0.356358 mm |

Finite gain changes are similar even though small-step derivatives differ greatly.
This supports retaining finite represented-error analysis instead of interpreting
an arbitrarily small derivative as a calibrated confidence bound. The remaining
22.9-micrometre source-factor discrepancy is not zero or a general error guarantee.

Both views accepted nominal, all twelve shared gain/offset members and both ULP
patterns: 30 accepted fits total. All 15 accepted initial-view queries had full
physical floor-footprint coverage for all 27 shapes under the supplied diagnostic
hypotheses. Every accepted terminal-view query covered zero shapes: fitting an
infinite plane still does not make underbody floor visible. No rejected fit was
assigned a gap or a positive physical query.

All four 0.1-mm independent-noise patterns failed the .002 triangle-normal
difference gate. Independent arithmetic found maximum triangle differences of
.02757/.02873 in the initial view and .04128/.04093 in the terminal view, despite
maximum fitted-plane pixel residuals of only 38.6–45.5 micrometres. Three of these
four rejected members also changed the median eligible seed; all accepted members
kept their nominal seed. Thus the failure is categorical as well as numerical.
Two reused frames and four fixed patterns are not a measured rejection probability.

The diagnosis is narrower than “depth noise solved”: fitting over a longer spatial
baseline reduces quantization sensitivity, while demanding nearly identical normals
from adjacent tiny triangles remains sensitive to independent range noise. Large
triangle tolerances or small aggregate residuals alone would not prove absence of
small obstacles. No gate was loosened to make these cases pass.

## Audit, identities and limits

`scripts/audit_go2_multipixel_floor_plane_development_v1.py` checked all 34 saved
members against their represented depth and selected patch. Separate covariance
eigenvector arithmetic reproduced the 30 accepted SVD fits and physical gaps
within 1e-12, and independently reconstructed the local triangle discrepancies.
All 30 full reference queries exactly matched recorded cached queries. Perturbation
generation, physical support and reference-query utilities are shared with the
tested implementation; this is not an independent end-to-end experimental replication.

The launch binds 487 sources and 6,144 inputs plus native/OpenCV identities. The
audit binds its additional source and exact recorded artifacts. All were verified
unchanged after execution. The frame diagnostics took approximately 1.214/0.606 s
for their complete offline member populations; these are not controller-cycle or
deployment benchmarks.

- Launch SHA-256: `c2099caa658c4843e8d5d8f63100dc62eccc5668bfe869eeac646a6f079bcfa6`.
- Result SHA-256: `87ba058d9e989ee22d23f98e9c910b84dc4b54a8f6e3ffc6515d46c640a3e236`.
- Audit SHA-256: `7754f017d47785ebfac1214cde0510e0fa48c426acd5654c8b18770fc2c78012`.

## Next work and unchanged scientific requirements

Develop an explicit depth-error-aware measured-surface model, not just a larger
triangle-normal threshold. A candidate plane must be consistent with every
observed pixel's declared range interval; conditioning must bound possible normal
and offset variation. Keep holes, mixed/elevated surfaces, obstacles and uncertain
surface identity explicit. Quantization intervals are only one error component:
camera calibration, independent depth noise, shared bias, pose and kinematic error
cannot be inferred from the tiny residual of an ideal simulated patch.

Test surface/footprint bounds against analytical synthetic geometry and adversarial
discontinuities, then propagate finite joint errors through the actual RGB-D owner
on independent development motion, including weak-depth and rejected-point cases.
Do not use these two frames to select a noise multiplier that clears the calves.
Only after validation should a new execution consumer use the resulting bounds.

The two unresolved front lower-leg shapes, unvalidated future gait/braking,
unfinished exploration/return, latency, matched JEPA/multistep/memory comparisons,
independent layouts/seeds/robustness and hardware evidence remain outstanding.
The [execution plan](go2_floor_factored_navigation_next_steps_2026-09-06.md) and
ultimate scientific goal remain active.
