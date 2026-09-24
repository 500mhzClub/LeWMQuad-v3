# New native stop: floor-plane extent admission

The eleventh native collection at
go2_dual_camera_settled_maze_pilot_v1_attempt_001 has closed its1915 observations,
1914 completed commands and96450 physical samples with ten terminal zero
commands. Case result SHA256
dc5d54e1372b67ae88c783cfb06034777c3e50eccdba40f2aeb1dc5bb3f4fdc5.
Root native audit/result remain pending; session67399 is still active.
There was no physical or acquisition stop. First controller failure is1904,
after the last mission/registered pose1903. Preserve the entire failed return.

The new controller used the auxiliary pose at1872 and continued its return
turn instead of stopping at the predecessor's tracking failure. A bounded
live receipt read19407 observed current auxiliary pose and active RETURN at1872,
then the terminal error by1909. These were evaluator progress snapshots, not
independent physical arrival/return verification.

Read-only reconstruction15541 CLOSED pass on frames0,1903,1904. It verified
all1616 frozen source bindings and19 explicit closed input bindings before and
after: collection result, decision stream, public body/depth/fast manifests and
histories, auxiliary acquisition audit, and the four current RGB/depth files
for each of those three frames. It restored only the declared JSON identity,
validated each saved current dual-camera visual witness against its current
public packets, reconstructed all unchanged measured floor candidates, and
called the original fit/composition/admission functions. No model inference,
scene, command, threshold change, point trimming or native pose input was used.
The full raw visual replay remains the running native audit's responsibility.

| Frame | Primary candidates | Auxiliary candidates | Joint-plane result |
| --- | ---: | ---: | --- |
| 0 | 6783 | 19145 | Available; exactly matches saved admitted plane |
| 1903 | 0 | 6780 | Available; exactly matches saved admitted plane |
| 1904 | 0 | 6514 | Rejected: insufficient_combined_two_axis_extent |

At1903 the auxiliary maximum residual is4.058999262e-6m and RMS1.110891883e-6m;
its current visual camera is auxiliary. At1904 the current auxiliary visual
pose still passes its accessor. Original compose_plane and fit_joint_plane
have no differing composition fields; both mark the plane unavailable. The
rejection precedes residual evaluation. The fixed second-eigenvalue gate is
0.05**2m². No exact eigenvalue or calibrated uncertainty conclusion is claimed
by this preliminary readout.

The generic validator message is “admitted combined measured moments must
reconstruct exactly”. Here it is triggered by the unavailable extent result,
not an observed disagreement between recomputed moment fields. This diagnosis
does not justify lowering the extent gate or inventing a current floor plane.
Next, bind a complete diagnosis to the final audited collection and inspect the
actual extent/geometry. Investigate an explicitly typed continuation based on
current visual measurements and an admitted floor reference when current plane
observability is unavailable; keep qualified conflicts terminal and preserve
all old failures. No such continuation is implemented or adopted yet.
