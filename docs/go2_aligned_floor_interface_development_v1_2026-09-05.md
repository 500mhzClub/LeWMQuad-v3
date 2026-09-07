# Aligned floor interface V1: fixed development assay

This is a new scene-construction assay, not a repeat or correction of an old
navigation experiment. Existing native planes, recordings, identities and the
0/2 whole-mission result remain unchanged. No learned checkpoint is loaded.

The alternative consists of one native collision-only plane at z=0 and one
native visual-only plane whose entity translation is +5 mm. The installed
native visual mesh is offset -5 mm locally; actual post-build world vertices
must independently coincide with the physical plane within 1 nm (readback
roundoff, not a sensor-accuracy claim). Requested morph settings alone are not
evidence. Native renderer/mesh source hashes are checked before and after.

One fresh CPU scene, dt=2 ms, gravity -9.81 m/s², seed 2026090501. A sphere of
22-mm radius starts at (-1,0,0.10) m behind the camera. Record 500 physics steps
including position and native collision-plane contact forces. The last 100
sphere heights must be within 1 mm of the radius and contain plane contacts.
This tests the collision surface, not Go2 foot dynamics or contact estimation.

Four fixed cameras at (0,0,0.35) m: yaw/pitch (0,0), (0,-0.15), (+0.4,-0.15),
(-0.4,-0.15) radians. Same 640x480, focal length, near/far as existing RGBD
contract. Separate RGB and single-sample optical-depth renders at unchanged
physics step and camera transform. Independently intersect pixel-centre rays
with the physical z=0 plane, stride 8, and check >1000 finite interior rays in
0.22..4.98 m and maximum depth error <=1 mm per camera. Record raw RGB/depth,
camera geometry, framebuffer readback, trace, and identity for external audit.
Do not shift recorded depth or supply world geometry/contact data to a policy.

One-shot output: `.generated/go2_aligned_floor_interface_development_v1_attempt_001`.
Bind inherited frozen source/input/artifact identities and the new exact source,
test and protocol paths before execution. An existing directory prevents a
second launch; exceptions produce terminal failure, no automatic retry/resume.
All artifact hashes are saved with the result. A fresh read-only audit must
recompute alignment, ray errors and contact summary from raw saved artifacts.

Passing this assay establishes only this scene interface. A new Go2 scene
builder/session and its contact-identity auditor must explicitly adopt the pair;
current frozen controllers and old identity schemas remain untouched. Full-loop
timing, observed ground hypotheses, contact admissibility, prospective motion,
complete missions, JEPA/memory/multistep comparisons, novel layouts and hardware
evidence remain required for the full scientific goal.
