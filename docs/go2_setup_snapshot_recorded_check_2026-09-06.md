# Fixed recorded setup-snapshot check

Evaluate the first sensor epoch of the existing bounded-floor Go2 interface
tape only: episode (0,0,0), time 1.5 s. This is post-hoc development validation
of a proposed setup condition, not a new mission or independent physical sample.

The proposed initial-body velocity ball has zero mean and radius 0.02 m/s.
The non-floor-clear starting prism is [-1,-0.75,-0.5] to [1,0.75,0.6] metres
in the initial body frame, expiring at 3.5 s. Its bounds are fixed before this
checker executes and must not be resized to pass. No region prior is injected
into the already recorded controller or its sensor packets.

Check the velocity ball against the actual initial reference twist. Check the
entire oriented region against the complete physical non-floor object inventory
enumerated independently in the recorded contact topology; reject missing,
duplicate, moving, disabled or non-box objects. Preserve actual native BOX
dimensions, reserved slots, poses and quaternion orientations. Use all fifteen
separating-axis families and retain touching as non-clear. Check that the 27
measured-posture URDF primitives, with 4-cm padding, fit in the proposed region.

Verify the actual aligned native collision plane at z=0 separately. Reconstruct
the first sensor epoch's native contact rows with the existing contact reader.
Report positive upward forces on all four calf support groups, other loaded
contacts, and nominal non-foot separation from the plane. This is an
instantaneous native support witness, not identification of individual foot
collision geoms, a calibrated compliance/friction model, inferred deployment
contact, or permission for a future gait. Do not choose a penetration tolerance
from this recording.

Evaluator world poses, velocities, environment geometry and contacts remain
outside sensor packets and policy consumers. Output is evaluator-only diagnostic
evidence. Verify all existing recorded source/input/artifact/reader bindings and
the explicitly declared new checker/test/protocol bindings before and after.
Preserve the existing no-prior frame-1 stop, prior-conditioned frame-12 budget
stop, and 0/2 complete-maze results. No physical run, training or artifact
replacement is part of this check.
