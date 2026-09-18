# Corrected JEPA floor visibility result

The missing nominal foot tiles were never wholly inside any retained camera
frustum. No failed tile had an in-frustum rectangle rejected by the measured
floor-pixel test. This isolates a viewing-geometry limitation in the recorded
prefix, rather than supporting a floor-classification threshold change.

All 108 recorded decisions replayed exactly, and every candidate's centre and
covered-tile count matched the preceding diagnosis. At tick 97 the 98 retained
views left 29, 13, 10, 24, 20 and 41 of 64 tiles without any complete in-frustum
view for hold, forward, left arc, right arc, left turn and right turn,
respectively. Every one of the 64 tiles for every candidate was outside the
current camera frustum and had a corner inside its 0.2-m optical near limit
or behind the optical plane. These are nominal geometric facts, not a claim
that terrain was observed or physically unsafe.

The next candidate is a separate depth view pitched downward 30 degrees, with
a proposed optical centre at body coordinates [0.35, 0, 0.08] m, preserving the
original RGB/model stream. Its initial characterization must remain geometric:
robot self-occlusion, mount feasibility, raw sensor timing, depth reconstruction,
observed obstacle retention and native execution have not been checked. The
candidate cannot supply floor evidence until actual depth pixels are acquired
and audited. The old controller, sensor and failures remain unchanged.

Artifact root: `go2_training_bias_floor_visibility_v1_attempt_001` under the
fixed navigation development artifact root.

- Launch: `81a57c9166897ed6c9486b106c56b3d6b3fd4f58a61fe50378d394b9e6620af2`
- Visibility: `b0b4f80eb52b36750d30087e97f75b4f01ee0975a1da871a3019d9b665611e40`
- Result: `a9880fb3a6dd8ae760348a6ff27430448b88373c508a8444e3ac127ea5e71741`

The complete input, source, model, correction and predecessor-diagnostic
revalidation passed. Replay plus analysis took 47.853755159070715 s after
admission. Two synthetic tests passed in 0.32 s, distinguishing visible valid
floor, visible rejected pixels and out-of-view tiles, and preserving good
retained observations. No native scene or optimizer was launched. There are
still zero verified goal arrivals and zero independent novel-maze evaluations.
