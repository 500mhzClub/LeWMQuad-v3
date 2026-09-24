# Bounded measured surface V1: fixed-reference tube, no motion

## First-principles model

Do not infer that the physical surface is one exact plane because noisy samples
have a good fit. Instead fit a deterministic reference (a,n), then bound actual
vertices relative to it. For optical range z_i, known body ray r_i and camera C,
p_i=C+z_i*r_i. With explicitly supplied |dz_i|<=e_i, every possible vertex obeys

    |n dot (p'_i-a)| <= |n dot (p_i-a)| + |n dot r_i|*e_i.

Here e_i is the declared non-quantization range bound plus the larger float32
half-bin width and an outward arithmetic allowance. Errors may be dependent.
No Gaussian assumption, fit-residual noise estimate, or finite-difference
covariance is used. Camera geometry is assumed exact for this conditional model.

Admit only vertices whose entire interval lies inside a fixed 1-mm residual tube.
Every convex interpolation within their triangles then remains in that tube.
A submillimetre variation is accounted for in clearance, not silently flattened;
a larger inconsistent step or missing pixel cannot be admitted. This still
assumes actual surface interpolation between adjacent samples: unobserved subpixel
holes/obstacles are not certified absent. A fitted plane is not semantic ground.

For each triangle, its unnormalized cross product is multiaffine in its three
positive ranges. Its projection along nominal up has extrema at the eight range
corners. The maximum cross-product norm also occurs at a corner (coordinatewise
convexity). Subtract up_error*maximum_norm plus an arithmetic allowance to check
that the entire family retains the same image winding. Use one common winding
from the seed, not a separately flipped sign per cell. This prevents a locally
folded or orientation-ambiguous mesh from being labelled observed support. It
does not require equal adjacent triangle normals.

The reference normal is fixed, so its own normal_error is zero BY DEFINITION,
not a claim of perfect physical floor orientation. Tube thickness enters the
physical minimum-gap and footprint bounds as plane_offset_error=1 mm everywhere
queried. The body-point errors must separately cover relative sensor pose,
kinematics, etc. Coverage uses every image cell in the conservative projection
rectangle of the uncertain full physical footprint. All its cells must have
bounded vertices and robust orientation. This is conditional mesh evidence, not
continuous scene/hardware certification, common surface identity, or permission.

## Frozen estimator and synthetic tests

Use the median eligible seed from the predecessor selection only as a fixed,
query-independent seed. Its radius-eight patch contains 18x18 pixels. Require
all patch pixels usable, a smaller tangent RMS >=5 mm and tangent singular-value
ratio >=.02. Fit all distinct measured pixels by centred SVD. Require normal dot
up minus up_error >=.97 and the camera centre strictly above the reference tube.
All patch cells must pass the NEW interval/tube/orientation checks. There is no
fallback patch search, trimming, inpainting or widening of the old .002 gate.
The old seed selection can still be noise-sensitive; report seed changes.

Declared V1 hypotheses: non-quantization range error 0.1 mm, surface tube 1 mm,
up error .001. Reject admissible range intervals crossing [.2,5] for positive
coverage; do not clip them. Require all possible admitted vertices below the
existing -0.15-m observed-up height boundary. None of these physical hypotheses
has been calibrated on real sensing.

Tests cover analytic floor, independent finite range perturbations, float32
rounding, missing/stepped fit and footprint cells, no unseen floor, penetration,
invalid contracts, degenerate triangles, sampled dependent errors/up variation
inside the derived eight-corner orientation bounds, and vertex/interpolated-point
tube enclosure. Reference rectangle enumeration must match the prefix index.

## One recorded development comparison

Use the same saved initial/terminal fresh-maze observations (0,218), each with
nominal and the two signed independent 0.1-mm patterns from the completed
multi-pixel V1 diagnostic. Six members total, not six independent trials. Use
the original generator and seeds; keep the actual float32 representation and
validity mask. Do not tune the tube or noise bound to the resulting clearance.
Record original V1 fit status, new status, seed and any seed change, candidate
cell counts and tube residual diagnostics.

At the original measured terminal joints and saved nominal sensor transforms,
perform two separate physical queries for each available surface: zero body-point
error for plane/surface isolation, and the EXACT per-shape/per-view point-error
values recorded in `configuration_00.json` of physical-configuration V1. Do not
replace the latter with zeros, fit residuals, or observed simulator errors. Verify
the matching raw-depth witness before reusing its allowance. Every query must
match the reference rectangle enumeration. No global estimator reintegration,
action, new physical mission or JEPA experiment occurs.

Freeze source/protocol/test narrow import closure before launch; inherit and
verify the previous 487-source/6144-input closure, bind its launch/result/audit
and completed frame artifacts and audit source. New exclusive output:
`.generated/go2_bounded_depth_surface_development_v1_attempt_001`.
Write results or an explicit terminal failure without resuming/replacing old runs.
Verify all source/input/native bindings again after execution.

Success here means only that the declared conditional mesh model is implemented
and its limited evidence is recorded, including failures. It does not validate
the assumed physical range/up/pose errors, action response, common floor identity,
navigation, generalization or a JEPA contribution. Next must include independent
development motion with joint sensor errors, weak-depth/point rejection, camera
and pose/kinematic errors; then prospective gait/braking, complete discovery and
return, matched predictive-training/multistep/memory comparisons, independent
layouts/seeds/robustness and hardware evidence.
