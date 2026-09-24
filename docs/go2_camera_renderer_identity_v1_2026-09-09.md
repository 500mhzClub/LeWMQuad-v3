# Actual camera-context identity integration probe

The prior provenance analysis queried a separate empty context, not a live
Genesis camera. Its renderer_identity_readback helper has synthetic coverage
but has not been exercised on an actual Camera after RGB/depth rendering.
This probe addresses that integration gap without revising historical output.

Use the existing helper unchanged. Create one fresh CPU Genesis scene with
a plane, one fixed box and one640x480 rasterizer camera. No robot, learned
model, physics steps, navigation commands or dataset access. Scene, seed,
camera and three repeated RGB/depth pairs are fixed in the new script. After
each depth render, read identity, sampling and precision from that camera's
existing context/target. Persist all three RGB/depth pairs and readbacks.
Require zero scene steps, exact unchanged camera pose, stable context identity,
and byte-identical RGB and depth across the repeated readbacks/renders.
GL precision queries do not by themselves prove a rasterization error bound.

Bind the completed prior provenance result and launch, the existing helper,
all inherited source/runtime identities and explicit installed camera,
rasterizer, offscreen and renderer sources. Record only the relevant rendering
environment selectors. Validate before and after; preserve a failed attempt
without modifying or retrying it. Destination:
go2_camera_renderer_identity_v1_attempt_001.

Run --preflight-only, inspect resources, then launch after the paired controller
timing experiment finishes so this scene does not add graphics/compilation
competition to that measurement. One CPU process/scene and one numerical
thread beside the independent settling CPU replay. Require16GiB available RAM
and64MiB output above40GiB reserve. These are capacity admissions, not OS limits.
No parallel native navigation scene, training, hardware motion or frozen
collector modification. Preflight creates no output and no graphics context.

A passing result establishes helper integration and repeated render stability
for this new simple scene only. It does not identify any historical camera
context, prove equivalence to a maze rendering, resolve frame909 visibility,
certify sensor/pose uncertainty, or qualify navigation/hardware. Future
acquisitions must still capture their own actual context provenance.
