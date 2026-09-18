# Completed actual-mesh subpixel coverage diagnosis

Session 10218 completed. Output: `go2_mesh_subpixel_coverage_v1_attempt_001`.
Launch SHA-256: `4bda1e1e91c6347408b03a2b44816a82a899ea60a22155024d19a9d7e5c1a2c0`.
Result SHA-256: `8f0eaccafbdde8406665fd7758ee4ccbd0ab22ea1fe78112f076bf778e49788a`.
All 1,469 source bindings and completed native/edge-diagnosis artifacts verified
before and after execution. Seven focused tests pass in 0.14 s.

The saved PLY reconstructs exactly the native float32 wall-position array:
28,600 vertices, positions SHA-256
`5977284f74911d62a9ac1560d38a6e0e56fe49c82fb94abd52a7663a0b768fe8`.
The mesh contains 14,300 triangles. This diagnostic tests 6,901 whose three
vertices are beyond the near plane and explicitly excludes the other 7,399;
it does not reproduce clipping or claim complete coverage.

At frame 909 pixel [260,428], exact projection finds two covering triangles.
The nearest is triangle 5774, with plane depth 2.544169239098972 m. Hypothetical
nearest/even 1/256-pixel snapping finds four covering triangles; the nearest
becomes foreground triangle 10277, which did not cover the original sample.
Its original plane extrapolates to 1.0753915404768208 m, only
0.00000835413648880845 m above the native return of 1.075383186340332 m.

This extends the earlier coarse box-edge hypothesis to the actual subdivided
float32 mesh input. The [upstream source review](go2_llvmpipe_subpixel_source_review_2026-09-08.md)
independently confirms an eight-bit fixed grid in Mesa 25.2.8, but installed
Ubuntu binary equivalence, rounding mode, shader arithmetic, clipping, culling,
tie ownership and framebuffer depth interpolation remain unproved here.
The reported plane value is an extrapolation, not an exact native depth replay.

The original strict visibility score reconstructs exactly and remains false.
No sensor pixel, policy, model, native scene or old outcome was changed.
This is stronger mechanism evidence, not a complete sensor error bound or
qualified navigation result.
