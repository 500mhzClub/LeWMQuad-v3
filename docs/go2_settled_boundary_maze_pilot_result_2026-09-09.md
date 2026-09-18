# Tenth native episode: physically settled arrival, failed return

Native result: `a7a02db120b4b662cd66efee01f10784edb6f2ed6984bdc420007773a1b1b6fb`.
Native launch: `eee7ddac0d0a5410806c7e16f6d5abc71ee2cf9ffb3772ed4f910333868de188`.
Audit: `0c5debfa4eb4c8986c5b661f7902ddcf36158e71131e13ced02536ad3f5c76f3`.
Prefix: `9673a7e60427c7a2153a544b343f4f56ec56f4e9574fe00958570334fab946a1`.
Readout result: `a5523b729310d39a81baf9ccf9cf94fa6fb8a82077e322d4fbe394a45748a70d`.
Readout launch: `caa8777e40aeddd78b70bda38ab64b9a67a628c55a3a171894658f7b1e7de2ab`.
Roots: `go2_settled_boundary_maze_pilot_v2_attempt_001` and
`go2_settled_boundary_maze_readout_v2_attempt_001` in the development navigation
artifact store. Native1573 and readout1575 source/input final checks pass.

The controller declared arrival at frame1868 after ten complete measured quiet
intervals. The unchanged independent native criterion passes: over the full
one-second physical window, maximum distance to the goal is37.619mm and maximum
3D speed is0.039016m/s, below the0.05m/s limit, with the required zero-command
window. This is the first physically settled outbound arrival among the ten
completed policy episodes. It is not a qualified round trip or independent-
layout result.

The episode travelled8.946839m in188.2s simulated time. Minimum goal distance
was21.496mm; terminal distance38.715mm. It recorded1883 paired observations,
1882 commands and94850 physical samples. The loop-erased outbound path crosses
six open maze edges. No invalid crossings occurred. Return has no edge
crossings, the route is not physically retraced, and terminal native quiet
fails. Visual pose admission fails at1872, after last accepted pose1871; the
controller then preserves its terminal stop and ten zero-command drain.

All raw sensor reconstruction, complete model/controller replay, actual
command audit and model-state checks pass. The1867-observation physical/public
prefix exactly reproduces the prospective settling replay, including the
declared mission-state change1866 and unchanged compared commands. Strict
visibility still fails; hard-measurement failure frames are empty. Verified
round trips remain zero. The physically settled arrival does not override the
visibility failure or promote navigation qualification.

Registered XYZ error mean4.235mm/max8.582mm; XY mean4.233mm/max8.578mm on the
executed admitted trajectory. These are postfit errors, not uncertainty bounds.
The readout identifies86 completed later-floor-resolution command intervals
and1765 completed waypoint intervals, including1714 local reranks. These are
executed interventions, not matched proof of prediction or memory benefit.

Receipt-inclusive median iteration is1231.597ms, maximum1936.498ms; all1883
observations exceed100ms. Physics remains paused during computation. Native
wall time4828.639s is not real-time or hardware qualification.

The separate dual-camera observer and registered replay have tracked all1881
frames of the ninth recorded episode, including its old failure/drain. A full
dual-camera controller prefix is running on this tenth collection. All11331
of its collected-input bindings match this final native artifact map. Complete
that prefix before a prospective dual-camera episode; preserve this failed
return. Only maze0 has executed. Independent layouts, matched JEPA/planning/
memory comparisons, strict sensing/uncertainty/timing and bounded hardware
evidence remain unfinished; the full goal stays active.
