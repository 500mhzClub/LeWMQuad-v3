# Complementary fusion and persistent ray memory implemented

The new RGB-D/inertial state and persistent ray-memory path pass synthetic
contracts. A fresh robot-scene adapter also passes native collision/visual
identity checks against the frozen primitive builder. No new walking trial,
empirical fusion result, complete mission or JEPA advantage is claimed.

## Implemented state and evidence ownership

`lewm/rgbd_inertial_fusion_development.py` owns one DepthRelativeState and thus
one fast-gyro integration. The frozen point-tracking kernel consumes that same
attitude. Tests reproduce the frozen standalone RGB-D observer's complete point
output exactly while integrating only50gyro intervals per100ms transition.

Raw plane and point observations remain separate. With weak plane basis W,
P=W.T@W, the effective displacement is the original measured plane projection
plus P times an accepted RGB-D translation. Their jointly observed components
must agree under an explicit supplied development hypothesis; disagreement
latches a conflict instead of selecting whichever measurement enables motion.
Original depth rank, null full-depth displacement and sensor provenance remain
unchanged. Rejected RGB follows the original inertial weak-space path; tests
reproduce the original SetupVelocityIntegrator's relevant outputs exactly.

Accumulated position error never resets on local point success. Initial-velocity
sensitivity uses the effective remaining weak projector, retaining its already
accumulated position contribution. Point-error contributions add linearly,
without covariance-independence assumptions, and point-derived velocity error
continues into later inertial dropouts. The inherited depth/inertial proxy is
still explicitly uncalibrated. Two positive point hypotheses must be supplied;
there are no runtime defaults. Synthetic2mm step/10mm agreement examples are
not deployment error estimates, selected action-admission limits, or claims that
calibration/aliasing errors are covered. The code explicitly reports otherwise.

`lewm/rgbd_inertial_ray_memory_development.py` passes that fused state once to
the original ray-memory storage/query implementation. It does not integrate
again or mutate frozen globals. A synchronous single-use adapter binds the
same input objects, copies the fusion result and rejects replacement or reuse.
The original budget gate prevents storage of an unadmitted view. State/memory
faults latch together; stale queries are rejected. Compiled/reference geometric
query classifications agree in the integration fixture. This is local ray
memory, not verified place recognition, a trusted topological edge or return.

## Fresh robot-scene construction

`lewm_genesis/lewm_genesis/rgbd_motion_scene_development.py` exposes an independent
appearance seed and separates visual-only surfaces from collision Plane/Box/URDF
entities. Original-seed tests exactly reproduce the frozen appearance generator;
new seeds change colors without changing geometry. Unsupported rolled/pitched
boxes and unreviewed native modes are rejected explicitly.

Native preflight used a new6m enclosure, an offset angled partition, spawn
(-.25,-.2,.375), heading0.27rad, physics/topology seed2026090606 and appearance
seed2026090607. Each of neutral/repeated/distinctive has27actual robot collision
shapes and6environment shapes. All physical shape data, link names, world poses,
friction and solver values match a same-definition reference made with the
frozen primitive builder. Only global geometry/link indices are excluded from
reference comparison because visual-entity order changes them; full native
robot and environment identities, including those indices, match across the
three new arms. Native visual triangle identities match across arms as well.

An independent saved-witness audit verifies all28artifacts, regenerates every
surface's vertices/faces/colors, checks native triangle identities and repeats
the physical/reference comparisons. Zero physics steps and zero rendered frames
were taken. No gait checkpoint, actuator gains or contact dynamics were tested.
Inherited COM and qpos0-limit warnings remain unresolved; native construction
success is not hardware or dynamics qualification.

## Verification

Focused34226 passed34fusion tests; expanded40391 passed40. Combined62468 passed
52fusion/scene tests. Full47042 passed1902tests across153explicit files in153.92s.
No tested source changed while these ran. Pack/source preflight43947 passed
445sources/4773inputs plus inherited native and exact OpenCV bindings. After
that and full tests terminated, the new native runner alone gained an actual
geometry manifest digest and explicit cross-arm full robot-ID witnesses.
Native32547 and independent saved-witness audit8463 both ended exit0.

Output `.generated/go2_rgbd_motion_scene_preflight_v1_attempt_001`:

- Launch: `76fd6be7675427fd4bbb275a5b0de5dba116f613948aff09eb98c36888c86744`.
- Result: `d2a088a415a59f3cf3b7963d1bc62aa7941a547f9be30017d5711af50011ed30`.
- Audit: `cb3af02c4872b5663d6084595b7d10721c739e242650fa9eb859502928007cbf`.
- Auditor source: `78c5b4c267cf076c535af42f9af84ae4c075e076c43e4655b171a93875b75b3c`.

All launched sources/inputs/protocol/output are frozen. No predecessor changes,
sealed access, GPU training, physical commands or hardware actuation occurred.
All handles above are terminal.

## Remaining work and next action

Build the fresh physical initializer and capture session around the new builder.
The old BoundedRGBDSession is not directly compatible: its floor checker expects
a two-triangle visual Plane, and its contact-identity loop excludes only the
visual floor, not the new visual-only walls. A new adapter must check tessellated
surface identities and exclude **all** visual-only links from physical contact
attribution. Keep actual camera/body mounting, explicit optical conversion,
single-sample capture, unchanged gait/sensor contracts and independent geometry
versus depth checks. Do not modify the old floor checker or native sources.

Then preregister and execute fresh bounded bidirectional walking/turning/braking
with shadow fusion and exact raw-tape audit. Validate empirically before using
fusion for action admission, and separately validate any error model. A shadow
estimator failure must remain terminal even if an independently supervised data
collector continues its declared tape. Preserve matched blank/repeated cases.

Long missions need defensible relative-pose transport and eventually validated
loop closure/place evidence. Repeatedly adding absolute pose envelopes can
exhaust a local clearance budget despite useful relative observations. Do not
erase global error or enlarge the budget to hide this; explicitly model shared
anchor correlations and validate relative transport and false loop closures.
The full-task and contribution sequence remains in the
[next-execution plan](go2_rgbd_fusion_and_fresh_motion_next_steps_2026-09-06.md).
Full-maze0/2, no demonstrated JEPA advantage and no hardware evidence remain.
