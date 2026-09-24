# Fixed multi-pixel plane V1: development diagnostic, no motion

Purpose: assess the diagnosed three-point/float32 plane failure without tuning
the clearance gate to the failed mission. This is an offline plane-only
diagnostic, not a rerun of the joint RGB-D estimator, calibrated uncertainty,
independent motion validation, or a navigation experiment.

Use only the initial and terminal observations (indices 0 and 218) of the
already audited fresh development maze. Query the terminal measured posture
using the saved nominal relative sensor poses. Initial up comes from the initial
specific-force average; terminal up is transported by the saved gyro orientation.
Do not read native world-floor/pose labels for plane fitting. Do not modify or
resume any predecessor source or run. Preserve their negative results.

## Fixed estimator and comparisons

Select the row-major median eligible measured cell once per input, then its
radius-eight-cell patch: 17x17 cells, 18x18 distinct pixels. No resizing, trimming,
outlier deletion, missing-pixel interpolation, RANSAC or alternate seed search.
All cells must pass the existing floor eligibility predicate. Fit all unique
points by centred SVD in float64 arithmetic AFTER their actual float32 range
representation. Orient the fitted normal toward observed up. Require smaller
tangent RMS >=5 mm and tangent singular-value ratio >=.02; normal dot up >=.97;
maximum individual pixel residual <=1 mm; every measured triangle normal within
.002 and anchored triangle offset within 1 mm of the fitted plane. Reject the
whole patch on failure. These are declared consistency hypotheses, not calibrated
noise bounds, and may reject noisy valid floor. A uniformly elevated plane is
still just a measured plane, never semantic ground.

Each frame has seventeen fixed members:

- Nominal.
- Signed persistent gain loadings .001*original float32 depth and offset loadings
  1 mm per valid sample, each at source-unit amplitudes .005, .01 and 1.
- Signed independent per-pixel uniform patterns, at one float32 ULP and at
  0.1 mm. Generator PCG64 via NumPy default_rng, seed 2026090620+frame index;
  generate one fixed 480x640 uniform[-1,1] array and reuse it across both sizes
  and signs. These finite patterns do not sample or bound a population.

All perturbations are cast back to float32. Preserve the original validity mask;
a valid range crossing [.2,5] is recorded as a categorical rejection, not clipped
or hidden. Record seed, fit status, tangent spread, residual, normal/anchor,
physical primitive minimum gaps and whole-footprint coverage. Compare the old
three-point seed plane under each SAME represented input. Report central factors
at .01/.005 and finite-amplitude changes at one source unit; do not select the
best step, calculate confidence intervals, or replace controller error scales.
For any rejected pair, factors remain unavailable. Record any seed change.

The new query uses the explicit fitted plane, not the predecessor's internally
reconstructed three-point plane. Every physical footprint cell must independently
belong to that plane family; a fitted plane cannot bridge an obstacle or unseen
region. Diagnostic query hypotheses remain normal=.002, up=.001, offset=1 mm,
point errors=0 (fixed pose/posture plane-only isolation). Those zeros are not
claims about physical sensor or action accuracy. Keep non-floor clearance,
contact permission, future-gait qualification and navigation qualification false.

## Recording and interpretation

Before launch, bind the new protocol, runner, implementation and focused tests
plus their narrow import closure to the predecessor's verified 483-source closure.
Bind predecessor launch/result and diagnostic artifacts; verify all inherited
inputs/native dependencies. Use a fresh exclusive output directory:
`.generated/go2_multipixel_floor_plane_development_v1_attempt_001`.
Freeze the launched sources. Write every categorical failure as evidence; an
infrastructure exception terminates with a failure record, never an implicit retry.
Verify bindings again after execution.

Synthetic tests must cover missing/step/spike patch rejection, no alternate-patch
search, conditioning and identity, rotated/elevated plane fitting, finite gain
through actual float32 against analytic camera-centred scaling, explicit-plane
query, full-footprint obstacle/missing data, behind-camera unknowns, preserved
penetration and cached/reference parity. Software tests and two reused frames do
not establish a sensor-error distribution, hardware timing or generalization.

If numerical behaviour improves, next validate finite shared AND independent
errors through actual RGB-D estimation on independent development motion,
including weak-depth/rejected-point intervals and noisy/discontinuous surfaces.
Do not relax triangle gates merely to accept the finite-noise cases. If the
strict local mesh is noise-fragile, develop and test an explicit bounded surface
model that retains obstacles and unknowns. Future action/braking validation,
successful maze discovery/return, matched JEPA/multistep/memory comparisons and
hardware evidence remain required.
