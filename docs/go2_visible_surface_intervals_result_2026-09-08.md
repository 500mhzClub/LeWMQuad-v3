# Completed conditional visible-surface interval diagnosis

Session 62601 completed, output `go2_visible_surface_intervals_v1_attempt_001`.
Launch SHA-256: `59cfbd66934617440a55a8c772d3aed82c5cf8cc373049c0ecde0d8f217381fd`.
Result SHA-256: `7131f396b608007b07d767982a7eb3f7e36e5e8d55a66bb4b3fd7d9b3ad8c632`.
The 1,470-source closure and completed input artifacts were verified before and
after the diagnosis. Eleven tests passed in 0.94 s after the prelaunch clipping
fix recorded in the protocol. No scene or model was loaded.

At recorded frame 909 pixel [260,428], an assumed 1/256-pixel radius yields
visible foreground depth 1.0753882227–1.0753919149 m and background depth
2.5441687396–2.5441776286 m. The native return, 1.0753831863 m, is within the
unchanged 1 mm metric tolerance of the foreground interval. A fabricated 1.5 m
depth has no supporting interval. The same conclusions hold at radius 0.5 pixel,
with appropriately wider separate intervals. Two coincident foreground faces
report duplicate areas; these must not be summed as union coverage.

The original strict score reconstructs exactly and remains false. Only this
recorded pixel was evaluated by the new interval helper. No full-image or
all-frame qualification is inferred.

Synthetic checks distinguish assumptions rather than hiding them. At radius
1/256 pixel, the missed thin-post background return is rejected because the
foreground covers the region. At radius 0.5 pixel, the background is visible in
part of the region, so it is supported as an ambiguous outcome. Both reject the
fabricated empty-gap depth. The physical 0.004 m near occluder rejects the
background under both radii; native near clipping never removes it from the
geometry calculation.

The radius remains a supplied hypothesis, not a native precision bound.
Floating-point enclosures and zero-area ties are also unproved. The current
controller has no corresponding public uncertainty representation. This is a
conditional evaluator component with useful counterexamples, not a replacement
navigation gate, sensor repair, or success result.
