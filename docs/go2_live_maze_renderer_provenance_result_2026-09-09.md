# Renderer provenance gap confirmed

Bound diagnosis47326 completed. Result SHA-256
76a592ec2bfa31927acb07cfdc6bb056de78df1693ab4066477f20a15159150a,
root go2_live_maze_renderer_provenance_v1_attempt_001. Launch
bcef998cccc5acccf0fe0c7fe4537fe3b4aee6210b6438719f35826de1048cac binds1544
sources, the ninth native launch, prior context result and additional native
runtime identities. No scene, physics step, draw call or rendered frame was
created. The active worker was observed through public kernel metadata only.

Across3.177259500s, worker2398270 advanced from579 to582 completed timing rows.
Its single amdgpu DRM client19955 at0000:7b:00.0 had52,437,270ns additional
graphics-engine work and268,672ns compute-engine work. Descriptors15/16/17
refer to that same client and were deduplicated. PCI vendor/device1002:13c0,
subsystem1043:8877. The local PCI description identifies Granite Ridge Radeon
Graphics. This is graphics-device activity, not a change to CPU simulation.

The new, separately queried default EGL context reports:

- Vendor AMD; renderer AMD Ryzen9 9950X3D16-Core Processor, radeonsi,
  raphael_mendocino, LLVM20.1.2, DRM3.64, kernel7.0.11-76070011-generic.
- OpenGL4.6 core, Mesa25.2.8-0ubuntu0.24.04.2; GLSL4.60; software flag false.
- EGL device/dev/dri/card0, resolving to the same PCI address0000:7b:00.0.

The older empty-context probe reported llvmpipe LLVM20.1.2 and OpenGL4.5 core.
Neither probe queried the ninth worker's actual camera context. Matching current
PCI and loaded library identities do not prove historical per-frame GL strings
or the reason for the earlier context-selection difference. No process memory,
debugger, environment or running graphics state was accessed or modified.

Both the live worker and new query context mapped the bound installed libraries:

| Library | SHA-256 |
| --- | --- |
| libEGL_mesa.so.0.0.0 | ee0de80dc8521bbccf83c90961c53a1cef6a418eabed560a9c618ab7387581b4 |
| libgallium-25.2.8-0ubuntu0.24.04.2.so | e37486422ecafdad74fb05bbcdcfc35300241a5f46e72b63ca442d4e0c0cca8b |
| libLLVM.so.20.1 | 4b93481bd1c9d42c0951275dd275d4a9d30db78c309ae2cc0405669bc6dbbf8a |

The exact installed EGL selection source has SHA-256
44d9c6439a919e78afcf1ec087dac01c3f8d09d2d8dc4f39d9692cd5ffe7dbb4.
It tries available devices until one creates a context unless an explicit
EGL_DEVICE_ID is supplied. The measured selected context is authoritative for
this new probe only. Source/library bindings were checked before and after.

The earlier llvmpipe source review remains valid for that implementation; it
does not establish the maze renderer's rounding algorithm. Actual-mesh snapping
remains a hypothetical mechanism diagnosis and strict visibility remains failed.
Do not build a sensor qualification argument solely on llvmpipe fixed-point
source or CPU physics backend selection. Future acquisition provenance must
query its actual camera context. The separate helper
lewm_genesis/lewm_genesis/camera_renderer_identity_development.py implements
that readback and passed seven synthetic checks; it is not installed into any
frozen collector. Renderer arithmetic, clipping/interpolation and consistent
policy-side measurement bounds remain unresolved. No outcome was relabeled.
