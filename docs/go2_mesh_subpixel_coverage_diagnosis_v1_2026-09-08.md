# Actual mesh input subpixel coverage diagnosis V1

Bind the completed view-reentry native result and completed raster edge diagnosis.
Load only the saved wall-union PLY, camera/readback and depth evidence under those
bindings. Require the float32 vertex array to match the actual raster-order
positions SHA-256, not merely a rounded geometric identity.

At frame 909 pixel [260,428], project all triangles whose three optical vertices
are beyond the unchanged native 0.005 m near plane. Report excluded triangles
explicitly; this does not implement clipping. Compare ideal double-precision
projection of the actual float32 mesh inputs with hypothetical nearest/even
rounding to the captured 1/256-pixel grid. Use barycentric containment to list
the nearest five candidate planes for each representation. Report the original
plane's extrapolated depth where snapped coverage moves outside its exact
triangle. These plane values do not reproduce framebuffer depth interpolation.

The native return and original strict score must reconstruct exactly against the
completed diagnosis; strict failure remains unchanged. Shader arithmetic, clip
generation, culling, interpolation, tie ownership and installed rounding mode
are not simulated or certified. The upstream implementation evidence is recorded
in docs/go2_llvmpipe_subpixel_source_review_2026-09-08.md; it grants no runtime
modification or passing sensor gate.

Seven synthetic tests pass: coverage switch, unchanged float32 input, explicit
clipping exclusions, degenerate triangle handling and malformed evidence.
One bounded CPU diagnosis may accompany the native scene; require 8 GiB RAM and
128 MiB above the 40 GiB reserve. Verify all completed inputs and source bindings
before and after. Exclusive output: go2_mesh_subpixel_coverage_v1_attempt_001.
No native scene, policy input, model, command or prior result is changed.
