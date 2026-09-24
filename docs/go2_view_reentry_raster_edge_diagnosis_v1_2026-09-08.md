# Completed view-recovery raster edge diagnosis V1

Bind the completed view-recovery native result and readout, their complete
artifacts and frozen sources. Reconstruct the original sampled visibility score
for every failed primary frame; this completed input has exactly frame 909 and
one failing sampled ray. Read its captured subpixel/depth precision, raw depth,
camera transform and scene specification as evaluator-only inputs.

Record the native and independently expected 3x3 depths around the failing
pixel. Project every box edge, including foreground silhouettes and hidden
edges, using the existing analytic camera convention. Record the five nearest
projected edges, their exact screen coordinates and signed pixel-line distances.
Compare with a hypothetical nearest-grid rounding of the projected endpoints at
the captured subpixel bit depth. Report side changes without claiming that this
is the native driver's rounding rule, rasterizer, exact shader arithmetic or a
proven sensor precision bound. No model is loaded and no scene is executed.

Preserve both the original strict failure and the existing conservative
pixel-footprint diagnostic. Do not certify boundary pixels, repair policy depth,
change public masks, discard a failing ray, or promote the old result. The
geometry is evaluator-only and cannot feed a policy correction. This is evidence
for designing a prospective precision/sensor investigation, not qualification.

Two focused tests cover signed edge distance and a hypothetical subpixel side
change, degenerate edges, input bounds, order invariance and non-certification.
Require 8 GiB available RAM and 128 MiB above the 40 GiB artifact reserve. One
CPU readout may run alongside the separately owned recorded-sensor prefix when
resources permit; no concurrent native scenes or training are introduced.
Check source and input hashes before and after. Preserve partial failure.
Exclusive output: `go2_view_reentry_raster_edge_diagnosis_v1_attempt_001`.
