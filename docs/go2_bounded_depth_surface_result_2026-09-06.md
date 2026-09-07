# Bounded measured surface works conditionally; pose clearance still unresolved

Implemented a range-error-aware surface model that accepts the four fixed noisy
observations rejected by the preceding triangle-normal model. It preserves holes,
large steps, uncertain orientation and unseen-floor rejection without requiring
equal adjacent normals. With the original body-point error allowances retained,
front-left lower-leg clearance remains unresolved; front-right clearance changes
sign under one noise pattern. No controller, motion permission or failed mission
was changed. This is progress in conditional geometry, not navigation success.

## What changed scientifically

The fitted plane is now a deterministic reference, not an assertion that the
physical surface is one exact plane. Every possible measured vertex under an
explicit depth-error interval must lie within a fixed 1-mm residual tube around
that reference. Interpolated triangle points remain within the same tube. The
tube thickness is carried into physical-gap and full-footprint calculations.
Reference normal_error=0 is therefore definitional, not removal of an unknown
physical-plane error: the actual nonplanar surface is bounded by the tube.

Every admitted triangle must retain its common image winding for all eight range
endpoint combinations and the supplied up-error allowance. This follows from
multiaffinity of its projected cross-product and coordinatewise convexity of its
norm. Missing or inconsistent vertices remain excluded. Small surface variation
inside the tube is accounted for in clearance rather than silently flattened.

This formulation remains conditional on correct range/up/body-point bounds,
exact camera geometry and actual surface interpolation between adjacent samples.
It does not establish semantic ground, common physical surface identity, absence
of subpixel holes/obstacles, non-floor clearance, contact or future action safety.
Its numerical checks are not a real-camera calibration or formal interval-arithmetic
proof of every possible floating-point execution.

Implementation: `lewm/bounded_depth_surface_development.py`.
Protocol and derivation: [bounded surface V1](go2_bounded_depth_surface_development_v1_2026-09-06.md).
Runner: `scripts/probe_go2_bounded_depth_surface_development_v1.py`.
Output: `.generated/go2_bounded_depth_surface_development_v1_attempt_001`.

## Recorded comparison

The same initial/terminal saved development observations were tested nominally
and with both signed independent 0.1-mm patterns from multi-pixel V1. All six
surfaces were available under the declared 0.1-mm non-quantization range bound,
float32 half-bin allowance, 1-mm tube and .001 up error. This is six finite
members from two reused observations, not independent experimental trials or a
validated error population. The seed selection remains noise-sensitive.

All three initial-view members cover all 27 physical footprints. Their maximum
patch vertex residual bounds are 0.0376–0.0769 mm, below the fixed 1-mm tube.
All three terminal-view members cover zero underbody footprints: a bounded surface
does not make unseen floor observable. Every query matched the reference rectangle
enumeration; there were twelve queries total, separating zero point-error surface
isolation from reuse of the exact original per-shape/per-view point allowances.

Initial-view lower-gap bounds with original point errors:

| Member | Front-left lower calf | Front-right lower calf |
| --- | ---: | ---: |
| Nominal | -1.380 mm | +0.0108 mm |
| Positive noise pattern | -1.597 mm | -0.1442 mm |
| Negative noise pattern | -1.124 mm | +0.1909 mm |

The original point allowances remain 28.2429 mm and 28.2494 mm, respectively.
Without those allowances, all 23 non-foot shapes are separated in these conditional
queries and four feet remain ambiguous candidates. That isolation is not an
alternative permission path. With the original allowances, the nominal result
has 22 separated shapes, the positive-noise member 21, and the negative member
22. The front-right nominal margin is not robust to even this small fixed
perturbation, so it must not be promoted to a resolved clearance result.

No covered definite penetration was found in these six diagnostic queries, but
the failed mission, uncertain calves and missing future-gait evidence remain.
No new non-floor evidence or joint sensor-state replay was performed here.

## Verification and identities

Twenty-one new tests cover noisy analytic floor, missing/step patch and footprint
data, quantization, unchanged point-error identities, penetration, unseen floor,
dependent interior range/up samples, degenerate triangles and interpolated tube
enclosure. Ninety-five focused tests passed. The full regression passed
**2,063 tests in 165 explicit files**, 176.16 s.

A separate auditor recomputed interval endpoint geometry using extended precision
on all 699,264 admitted cells across the six members, checking 11,188,224 triangle
endpoint configurations. All vertex endpoints remained within the 1-mm tube and
all oriented endpoint margins stayed positive. Maximum admitted residuals ranged
from 0.9853 to 0.9990 mm; the smallest orientation margin was approximately
2.43e-6 m². Those near-tube-boundary cells were checked, not omitted from the audit.
Surface selection and perturbation generation are shared with the runner;
endpoint arithmetic is independent. This is not independent physical replication.

The launch binds 491 sources and 6,635 inputs plus native/OpenCV identities; the
endpoint audit separately binds its source and exact outputs. All bindings were
verified unchanged after execution. The six-member comparison took about 1.957 s
offline; that is not a deployment-cycle timing measurement.

- Launch SHA-256: `b9f3d5f1db71e53b92036c3c687597c65919eae630970d2b9c87bc90f2ed1cd5`.
- Result SHA-256: `f4d239b524e62f4c69fb7194fcc2d84353b866a65ed39a94e171d12ff35f980f`.
- Queries SHA-256: `e800e98330a893342e188cac2a533123729ee04d157aab12693705641b87b431`.
- Endpoint audit SHA-256: `488853ba08c9e315a63ec0530f320f54d45899156317e0bceceb0f0f7381283c`.

## Next: joint finite-error motion validation, not another clearance multiplier

Use the separately recorded, already audited three-arm shadow-motion trials as
an independent development trajectory relative to the failed maze. They contain
weak-depth intervals and a retained nominal estimator-budget failure; unlike this
maze trajectory, they can exercise RGB complement and inertial fallback. They
are not unseen final-test data or three independent layouts. Preserve the original
neutral-arm failure and do not restart a failed estimator on later frames.

Build a finite-member replay of the actual RGB-D owner with persistent range
gain/offset, timestamp-consistent gyro/force biases, initial-velocity error,
independent represented depth noise and explicit RGB rejection conditions. Record
each member's validity/rank/match/budget failure independently, retaining terminal
states instead of losing the whole ensemble on the first failed derivative pair.
Check nominal replay against original recorded states before using native motion
solely for independent scoring. Evaluate relative physical surface/body queries,
not just global position norms. A finite population cannot certify an unsampled
continuous error set, and missing camera/kinematic error sources remain explicit.

After justified joint bounds, integrate the bounded surface into a new common
floor/non-floor consumer without falsely exempting obstacles. Then validate actual
prospective gait/braking and latency, complete discovery/backtracking/return, and
run matched JEPA versus supervised/geometric, genuine multistep rollout and memory
comparisons on independent layouts/seeds and robustness shifts. Real-sensor and
bounded hardware evidence remain required. The ultimate goal is unchanged and active.
