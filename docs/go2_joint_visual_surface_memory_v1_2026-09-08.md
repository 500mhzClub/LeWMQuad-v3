# Joint visual surface memory V1 — source and replay protocol

Build a persistent observation layer for subsequent route planning, using only
public RGB-aligned depth, the current joint visual-pose witness, measured joints
and the fixed robot URDF. This development version retains surface evidence and
chronological visited poses. It does not yet infer traversable floor/free space,
choose exploration frontiers, grant motion, or execute a backtrack.

The map consumes each exact 10-Hz frame once, requires matching RGB/depth digests
in the accepted joint pose, and latches failure on a missing/stale/mismatched pose.
Every fourth pixel in both dimensions supplies at most 19,200 public measured
returns. Unknown depth supplies no point. Transform returns with the measured
joint pose and retain 25-mm voxel enclosures with their first observation witness.
Retain all evidence up to 250,000 voxels / 4,096 poses / 50-m coordinate bounds;
exhaustion stops admission rather than erasing contradictory evidence. No privileged
geometry, native pose, ray carving, command integration or loop-reset is used.

Query the exact current-joint URDF primitive AABBs at a supplied discrete predicted
base displacement/yaw. Any voxel intersection is a POSSIBLE surface conflict;
absence is UNKNOWN. These are enclosing boxes of sampled returns and instantaneous
nominal primitives, not precise collisions, support classification, future gait,
swept-volume coverage, calibrated pose errors or safe clearance. Include every
primitive, including feet; do not erase floor contacts with a guessed floor label.
Record current-frame and persistent-memory answers separately. A reverse route
proposal includes every actually visited pose in reverse order without shortcuts;
it requires renewed clearance checks and physical execution.

First validation: four synthetic tests cover unknown space, negative/boundary
voxels, monotone witness retention, invalid-input atomicity, route ordering and
stale/failed query rejection. Then exclusively create
`go2_joint_visual_surface_memory_v1_attempt_001` for complete replay of both cases
in `go2_family_transition_goal_probe_v1_attempt_001`, launch SHA-256
`5d708568e8e1fb8d518d741d471172d0f718d7d158d130e98fe1fd6af0a55608`, result
`eb70bb999f78f063031575b2773e2871651f095cb583e0bd6fb4b94ec84bfa4e`.
Verify every source/input/artifact binding before and after replay. Reconstruct
the unchanged observer and require exact equality with recorded evidence. After
the original visual failure, require memory queries to remain unavailable; count
the entire trace rather than silently dropping failed/drain frames.

At all 15 recorded model selections, query all six saved candidate predictions
at the first half-second and four-second horizons. Do not refit/reselect models
or actions. Require every current-frame conflict also to appear in persistent
memory. Report possible intersections by shape with original surface witnesses,
including the contact case's final selected action. These are retrospective
diagnostics; original contact, goal, tracking, depth and timing failures remain.
Time map updates separately; no full-controller speed claim follows. This small
CPU replay needs no new native simulation or training concurrency allocation.
