# Observed geometry refinement: completed recorded-data diagnosis

Result `aa2162f7bbe98213a5c2e16188961b183e6ed808695583f959770d2c06a2a804`
in `go2_observed_geometry_refinement_v1_attempt_001` completed with 948 bound
source files. All 367 available map receipts and both final floor/occupied map
identities reproduce exactly; 21 unavailable observations remain unavailable.
The nine analytic geometry/classification tests passed. The 40-frame-per-case
benchmark was exact with one and two threads: 8.676 versus 6.915 seconds; two
threads were selected. Post-launch work took 59.406 seconds.

The unchanged 0.45-m continuous nominal disk admits connectors at six of the
18 original blocked-start selections: layout 052 ticks 38, 43, 53, 58, 63 and
layout 039 tick 43. At 052 tick 38, the observed base has 0.461160 m clearance
from occupied squares; a connector to [0.425, 0.775] has 0.451784 m minimum
clearance. Its 1.784-mm margin is nominal and uncalibrated, not a safety bound.
The remaining twelve blocked-start states actually violate the nominal radius;
continuous geometry cannot repair those. Earlier additional-view states also
admit continuous connectors: 052 tick 18 and 039 ticks 23 and 28. Overall nine
of the original additional-view decisions admit an entry using this calculation.
These are alternative recorded-state proposals, not executed outcomes.

All 87 original stride-four samples in the 17 unique first-witness voxels behind
the 213 candidate shape-conflict records on layout 039 pass the measured floor
patch test. Every sample has four adjacent ground-normal/planarity-qualified
pixel quads and nine valid pixel heights within 10 mm of the fixed measured
floor hypothesis. Maximum sampled-point deviation from that hypothesis is
2.758 mm. Exact pixel/point coordinates, first RGB/depth witnesses and all
candidate occurrences are saved. Layout 052 has no such shape-conflict records.

This establishes the measured local floor character of those first returns.
It does not classify later returns in the same voxel, certify support or future
gait, remove a surface veto, or prove that every possible voxel intersection is
physically harmless. A future floor-aware collision representation must retain
mixed/unknown returns and distinguish measured terrain from obstacle geometry.

The immediate prospective change is continuous entry-connector geometry with
the same radius, conservative floor-grid BFS, observer, weights and articulated
surface vetoes. The original runs and failures remain unchanged. No new native
navigation outcome is produced by this diagnostic.

Bindings: launch `37bb23ab795cba07c153605732f66f997cdc97c7088afe6bc462f044c0be6e31`;
workload `8b6f5bf16ab0ee195aec552bf6ae37513624fee99e05ff49d3e43904fc157a7b`;
052 `fb0a5c931c6c8c6dc02667d591adfea51ae91e5795955db0b7404d1b57442e28`;
039 `effad400dc9c3f16b0dd9c3d1ca786f1a3c871e9b48cf7eb814d01bfb7bd3e22`.
