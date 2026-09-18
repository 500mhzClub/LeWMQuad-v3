# Renderer identity helper works on an actual rendered Camera

Integration probe36972 completed with exit0 in21.619726510s after launch. The
existing helper queried the actual context and depth framebuffer belonging to
a real Genesis Camera after rendering. The scene contained a plane and fixed
box, no robot, and executed zero physics steps. All three RGB/depth capture
archives were byte-identical. Context identity, sampling/precision readbacks
and camera pose also remained identical across captures.

Output:go2_camera_renderer_identity_v1_attempt_001.
Result:c1636416aa33295641358bec601a6bb6202e664410104f6afb9dbcd35103a8f5.
Launch:214974735107a19abc495985048fd0a7fa658d47aa2e5dc4392530c563ac1983.
Each capture archive:
2061c24611a7065a33c9e2b97c7f8b72569baf4dc9006c7b8ec69640ab406291.
1546 source bindings and explicit installed camera/rasterizer/offscreen/
renderer sources plus inherited graphics libraries passed validation before
and after. Preflight18011 passed. The probe ran after timing63970 completed,
with only the independent settling CPU replay alongside it.

Actual camera context reports AMD Ryzen9 9950X3D, radeonsi raphael_mendocino,
LLVM20.1.2, OpenGL4.6 core Mesa25.2.8-0ubuntu0.24.04.2, GLSL4.60, EGL device
/dev/dri/card0 and software flag false. The depth framebuffer is single-sample,
24-bit depth, pixel scale1; RGB uses four samples. GL_SUBPIXEL_BITS reports8.
The last value is a queried implementation property, not proof of the actual
viewport quantization mode or an arithmetic/rasterization error bound.

This closes the helper's actual-Camera API integration gap. It establishes
repeated render stability for this new simple scene, not any historical maze
context identity or maze-renderer equivalence. The helper remains unchanged
and is not inserted into the frozen settling controller/native definition.
Future acquisition must record its own actual camera provenance. Visibility
failure909, measurement/pose uncertainty and navigation qualifications remain
unchanged. No old output was relabeled.
