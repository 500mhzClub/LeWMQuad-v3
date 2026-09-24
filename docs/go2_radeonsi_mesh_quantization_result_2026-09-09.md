# All source-listed quantization hypotheses retained

Diagnosis6675 completed, result SHA-256
a3b158cfc9312a0c5c02b8d5f218561dcc03cd31597bc7cafe1748be4b6a7522,
root go2_radeonsi_mesh_quantization_diagnosis_v1_attempt_001. Launch
6065242364de203e51b54f7318a65c59e4f59815092fd4ed006636794b6c39aa binds1555
sources and the completed native/edge/mesh/provenance artifacts. Inputs and
sources verified before/after. No model, GPU context, render, scene or physics.

The source rationale and exact four upstream file identities are in
go2_radeonsi_mesh_quantization_diagnosis_v1_2026-09-09.md. The archive hash
matches the [official release notes](https://docs.mesa3d.org/relnotes/25.2.8.html).
The [Mesa source](https://archive.mesa3d.org/mesa-25.2.8.tar.xz) reports8-bit
subpixel capabilities but selects8/10/12-bit viewport-dependent quantization.
The ordinary640x480 size branch suggests12 bits; no active register was captured.

All three hypotheses were evaluated on exactly the same original float32
mesh, frame909 transform and native pixel[260,428]. The8-bit report reproduces
the earlier completed mesh diagnosis exactly. Native depth remains
1.075383186340332m; the strict physical visibility score remains false.

| Fractional bits | Covering triangles | Nearest triangle | Original plane depth (m) | Plane minus native (m) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 10277 | 1.0753915404768208 | 0.00000835413648880845 |
| 10 | 2 | 5774 | 2.544169239098972 | 1.4687860527586398 |
| 12 | 4 | 10277 | 1.0753915404768208 | 0.00000835413648880845 |

Triangle10277 did not cover the sample under the original exact projection;
its reported plane value extrapolates outside that triangle. Triangle5774 did
cover it. The same small discrepancy under8 and12 bits shows that this pixel
cannot distinguish those modes. No mode was selected because it fits.

This preserves the earlier helper's explicit limitations:6901 wholly
near-plane-front triangles examined,7399 excluded, no clipping, culling,
tie ownership, shader arithmetic or framebuffer depth interpolation replay.
Radeonsi upstream source is not proved identical to the patched installed
binary. Neither the observed GL_SUBPIXEL_BITS value nor matching one returned
depth proves a complete sensor error bound. Query the actual camera context
and capture its effective render state in a separately named future acquisition
before treating driver-specific reconstruction as qualified sensor evidence.
Policy-side point/pose uncertainty remains unresolved; no old score was changed.
