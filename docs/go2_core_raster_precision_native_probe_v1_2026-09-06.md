# Native core-profile framebuffer query probe

The first ordered moving-sensor pilot is terminal with an invalid-enumerant
OpenGL query error before any episode commit. Its terminal audit reports one
attempted/uncommitted run, seven unattempted and no available comparisons. No
restart or reinterpretation of that result is allowed.

The old precision helper queries GL_DEPTH_BITS using glGetIntegerv. The proposed
correction queries the bound framebuffer's GL_DEPTH_ATTACHMENT using
glGetFramebufferAttachmentParameteriv and GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE.
The [Khronos reference](https://wikis.khronos.org/opengl/GLAPI/glGetFramebufferAttachmentParameter)
specifies this query for attachment depth-component bit counts. This source
supports the correction; the exact native failure mechanism still needs testing.

Allocate the frozen native renderer's ordinary 640x480 single-sample and
multisample framebuffer targets in one software EGL context, with no scene,
robot, render call, dataset or physics step. Record native renderer/version/core
profile, invoke the old helper once and retain its full exception including the
query arguments, then invoke the corrected helper once. Record all corrected
precision and RGB sample positions, framebuffer restoration and clean GL error
state. Require old invalid-enumerant reproduction and successful corrected query.
An unexpected outcome is retained, not repeated until favorable.

Bind failed pilot launch/failure/audit identities, inherited native/source
closure, corrected helper, tests, probe and this protocol before allocation.
Exclusive external output `go2_core_raster_precision_native_probe_v1_attempt_001`;
32MiB metadata budget, serialized launch at most16MiB,40GiB free reserve. Release
all allocated GL resources. No installed library or frozen source is modified.
This probe neither retries the pilot nor qualifies moving sensors or navigation;
any subsequent corrected pilot needs distinct sources, protocol and output.
