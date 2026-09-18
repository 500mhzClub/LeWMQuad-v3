# Radeonsi source and saved-mesh quantization diagnosis V1

The completed provenance diagnosis found a fresh default radeonsi context and
live maze-worker graphics activity on the same PCI device. The earlier empty
llvmpipe context is insufficient provenance for maze depth captures. Keep all
historical scores and hypotheses unchanged.

The official Mesa25.2.8 archive SHA-256 is
097842f3e49d996868b38688db87b006f7d4541e93ce86d2f341d8b3e7be7c93,
verified against the [official release notes](https://docs.mesa3d.org/relnotes/25.2.8.html).
Four exact upstream files were read from the
[source archive](https://archive.mesa3d.org/mesa-25.2.8.tar.xz); no build or install.
The installation's patched-binary equivalence is not proved.

| Source path inside mesa-25.2.8 | Bytes | SHA-256 |
| --- | ---: | --- |
| src/gallium/drivers/radeonsi/si_state.c | 223847 | 4ec5c81ed6ade46a39823e917c9e1dc84b2ee8dbd0c951597700708d35cdf57c |
| src/gallium/drivers/radeonsi/si_state_viewport.c | 31064 | e9cc5d598a118ea5b312503ced23dcd9f66ba9050e4b8b523367293c5fd3da22 |
| src/amd/common/ac_gpu_info.c | 106481 | 013cbe742d7594976083675d444b8a7d6c1ddf61b17b662fe8c03bad417e5c8d |
| src/gallium/drivers/radeonsi/si_get.c | 55955 | 1d90c032e56043f748a1f0fb745cfb95c14ff1a3901b72ccdb30d1ab828e368d |

si_get.c1308–1312 reports viewport/rasterizer subpixel capabilities of8.
si_state_viewport.c353–356 selects round-to-even and a quantization mode.
Lines479–484 choose12 fractional bits for max viewport corner<=1024,
10 for<=4096 and8 otherwise. Other source branches include family/binning
exceptions, shader viewport-index unions and disabled clipping/viewport paths.
For an ordinary640x480 viewport the size branch suggests12 bits; this is an
inference from source, not a captured hardware register or per-frame proof.
The stored GL_SUBPIXEL_BITS=8 therefore cannot alone fix the active grid mode.

Exclusive output go2_radeonsi_mesh_quantization_diagnosis_v1_attempt_001. Reuse
the exact completed view-reentry native mesh/frame909 and the completed mesh
diagnosis, binding original native/edge/mesh/provenance results and artifacts.
Reconstruct all three source-listed grid hypotheses8/10/12 with the existing
diagnose_mesh helper. Require the8-bit report to equal the completed earlier
diagnosis exactly. Report every hypothesis, nearest covering triangle and
original-plane depth difference from the unchanged native pixel[260,428].
No mode selection, fit, altered pixel, filter, tolerance or visibility outcome.

This retains the existing helper's limits: only triangles wholly beyond the
near plane, hypothetical rounded projected vertices, no clipping/culling/tie
ownership/shader arithmetic/native depth interpolation reconstruction. The
plane value may extrapolate outside the unsnapped triangle. No sensor error
bound, policy-side uncertainty or navigation qualification follows.

One small CPU geometry process beside the existing single scene; one numerical
thread. Inspect hardware/resources, require8GiB RAM and128MiB above40GiB reserve,
and bind all sources/inputs before and after. No GPU context, render, scene,
physics, model load or training. Preserve every failure and frozen source.
