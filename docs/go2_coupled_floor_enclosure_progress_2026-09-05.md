# Coupled finite-error floor footprints: coverage improves, height gate fails

The preceding user-facing status turn was no progress. This continuation adds
an unlaunched finite-error geometric enclosure, negative tests, and a read-only
diagnostic that changes the next action. No physical controller, pose radius,
ground threshold, frozen source or original result was changed.

## Construction and scope

`lewm/coupled_floor_enclosure_development.py` uses the relationship between point
position and its height above a plane, rather than independently combining point
and height extrema. For nominal unit plane normal n, observed anchor a and unit
up u, write the possible planes as n'·(q−a)=b'. The caller supplies deterministic
allowances ||n'−n||≤en, ||u'−u||≤eu and |b'|≤eb. These are not inferred from a
covariance, fitted to these four negatives, or claimed to be calibrated.

Nominal h0(q)=n·(q−a)/(n·u), so q−u*h0(q) is affine in q. Its coordinate extrema
over a box occur at the eight vertices. With d0=n·u, D=en+eu+en*eu,
L=max||q−a|| and H=max|h0|, the height-error allowance is

    dh = (en*L + eb + H*D)/(d0−D).

The denominator must remain strictly positive. The per-coordinate footprint
allowance is |u_j|*dh + eu*(H+dh). The implementation uses actual floating-point
normal/up norms in D to account for accepted unit-vector rounding and adds an
outward arithmetic allowance. Arbitrary dependencies between point, normal,
offset and up errors are allowed within these supplied sets.

The observation wrapper seeds its nominal plane from the first triangle of the
located centre-footprint patch; it does not search for a favourable fitted plane.
It checks every cell in the enclosing projected rectangle using the existing
missing-data, normal, height and 3-mm planarity gates. Both measured triangles
of every eligible cell must also belong to the supplied anchored plane family.
All possible signed heights must remain within the existing ±60-mm band; there
is no clipping to that band. A missing seed, camera coverage, interior return,
plane-family agreement or height condition leaves the query unresolved.

This is conditional geometry on an interpolated measured mesh. It does not
validate the assumed point/plane/up bounds, continuous physical surface,
uncertain ground classification, camera calibration, contact mechanics, future
gait or navigation. All support/navigation approval flags remain false. In
particular, compatibility of measured triangles with the plane family is
necessary evidence, not proof that the real surface lies in that family.

## Recorded four-region diagnostic

The read-only probe reconstructs the first 181 original north packets with the
saved nominal depth records and the unchanged fusion/ray-memory consumer. It
then queries the previously unresolved turn-volume samples 217, 218, 226 and
227 across the same 30 retained/current views. All four original ray queries
remain blocked. No physical experiment is repeated or rescored.

The illustrative allowances are en=0.002, eu=0.001 (Euclidean vector differences,
not angles) and eb=1 mm. They are the synthetic positive-test hypotheses, not
validated sensor bounds. First-view point radii remain 20.946–20.962 mm.

| Sample | Views with measured-cell coverage and plane-family agreement | First-view cells | Nominal-plane upper height over point box | Upper height including allowances |
| --- | ---: | ---: | ---: | ---: |
| 217 | 14 | 85 | 62.559 mm | 63.895 mm |
| 218 | 16 | 80 | 62.598 mm | 63.932 mm |
| 226 | 16 | 80 | 62.608 mm | 63.940 mm |
| 227 | 13 | 85 | 62.594 mm | 63.930 mm |

The first-view rectangles now contain zero non-ground cells, while their
predecessor independent point/height rectangles contained 2, 2, 12 and 78.
This follows from the derived footprint relation, not deleting negative returns.
Measured normal deviations are at most 0.000249 and anchored offsets at most
3.71 micrometres in those first-view regions.

However, **none of the four passes the full height condition in any view**.
The first-view height lower bounds including allowances are positive,
17.22–17.24 mm. Even the nominal-plane upper heights over the unchanged point
boxes exceed 60 mm; the new allowances are not the sole cause of rejection.
The diagnostic does not establish collision, safe support, or mission success.

Probe 31047 completed the initial diagnostic. Probe 1377 completed the same
read-only diagnostic with additional nominal-plane height reporting and the
floating-point denominator refinement. Both verified all 333 predecessor source
bindings and bound inputs/artifacts before and after processing. The latter
four-point/all-view diagnostic took about 1.24 s including its ending binding
verification. This deliberately straightforward implementation rebuilds a floor
index per view; it is not a full-loop timing measurement or production path.

## Verification

Focused session 88131 passed 35 tests across the new and predecessor bound tests.
The 19 new tests cover affine coupling, 60,000 exact perturbed intersections
(including box vertices and extreme error magnitudes), growing error sets,
nonrepresentable/singular inputs, observed seeds, missing interior cells,
non-ground discontinuities, quantized-mesh rejection of a zero-error plane
family, and height-band rejection without clipping. Random sampling tests the
implementation; the finite-set coverage claim relies on the derivation, not on
sampling alone.

Full session 34271 passed 1,396 tests across 123 explicitly enumerated files in
90.09 s before the additional height reporting and denominator refinement.
Final full session 83042 passed all 1,396 tests across 123 files in 90.42 s after
those refinements, with no concurrent source edits. All test/probe handles are
terminal. The tracked worktree remains unchanged.

## Next action: physical floor semantics, then execution

The source of `nominal_turn_volume` is important: these points are samples of
gravity-aligned, 40-mm-padded yaw envelopes of measured postures, not measured
foot-contact locations. A sample receives a ground role if every active
primitive is calf/foot and the band is below −150 mm in body coordinates.
That role alone does not establish support contact or future motion safety.

1. Separate three questions in a new, explicit geometric/contact interface:
   observed floor coverage, non-support-body clearance, and permitted foot
   contact. Derive how geometry padding and pose error enter each. A symmetric
   proximity band is not, by itself, a physical nonpenetration constraint:
   moving a point farther above a floor can fail proximity without implying a
   collision. Do not simply raise 60 mm or relabel these negatives as safe.
2. Test the distinction with above-floor, below-floor, missing-floor,
   wall/step and mixed-link negatives. Retain all-view obstacle contradictions;
   floor evidence must not clear a wall or unknown support area. Bound the
   model's domain explicitly, including interpolated surface and future gait.
3. Validate the declared sensor/registration error assumptions for that domain,
   reuse shared per-frame geometry, and measure the complete loop. The previous
   139–157-ms processing subset still misses the 100-ms target; this diagnostic
   has not remedied it.
4. Integrate explicit fused speed and observation/repositioning actions in a
   separately named navigation successor. Execute complete discovery/return
   missions, then matched supervised/JEPA action, memory and genuinely
   multistep-rollout comparisons on independent layouts and seeds. Infrastructure
   tests and repeated diagnostics of these two mazes are not that evidence.

Whole-task success remains 0/2. Learned JEPA navigation, independent-maze
generalization and hardware evidence remain unproved. The full goal is active.
