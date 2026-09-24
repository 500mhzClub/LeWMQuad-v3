# Joint visual surface memory replay result — 2026-09-08

The new persistent surface memory completed both original native traces and
exposed possible robot/surface intersections that the current depth frame alone
missed. In the contact case, memory flagged the front-left hip in the final
selected half-second prediction using surface evidence from frame 0. This is a
retrospective geometric warning, not a demonstrated collision-prevention result.

The new source [joint_visual_surface_memory_development.py](../lewm/joint_visual_surface_memory_development.py)
accepts current public RGB-D and its bound joint visual pose, retains sampled
surface voxel enclosures and every observed pose, and queries nominal robot
primitive boxes. It supplies a reverse chronological route proposal with no
shortcuts. Unknown/no-hit answers never establish free space or permit motion.
The fixed [source/replay protocol](go2_joint_visual_surface_memory_v1_2026-09-08.md)
was frozen before executing this replay. No prior frozen source was edited.

All 84 frames were accounted for: 29 accepted poses in case 052 and 44 in case 039
were indexed; the other 11 frames in 052 recorded visual failure or subsequent
unavailable memory. Raw corner-observer replay matched every original active
evidence record exactly. Queries remained unavailable after failure. Both old
goal failures, the panel contact and the frame-36 hard depth failure remain as
recorded in the [native result](go2_family_transition_goal_probe_result_2026-09-08.md).

At all 15 recorded selections, the readout queried all six candidates at 0.5 s and
4 s, giving 180 candidate/horizon queries, each with current-frame and persistent
memory variants. Every current-frame conflict was retained by persistent memory.
All 27 instantaneous URDF primitives participated, including feet; no guessed
floor label removed a return or conflict.

| Case | Queries | Current-frame conflict queries | Persistent conflict queries | Queries with additional shape conflicts from memory |
|---|---:|---:|---:|---:|
| `family_episode_052` | 72 | 4 | 17 | 13 |
| `family_episode_039` | 108 | 5 | 52 | 49 |

These counts are possible AABB/voxel intersections, not true-collision rates.
Voxel and primitive boxes enclose their geometry, depth is sampled every fourth
pixel, pose uncertainty is uncalibrated, and future predictions hold joints at
the currently measured posture. Discrete horizons do not cover a swept gait.
Historical returns assume a static scene. No-hit queries remain unknown.

For case 039 at tick 43, the selected left arc's half-second prediction had no
current-frame intersections. Persistent memory reported `base:0` (42 intersecting
voxels; first witness frame 2), `FL_hip:0` (30; frame 0), `FL_thigh:0` (20; frame 0)
and `FL_calf:0` (one; frame 10). The same selected half-second query flagged the hip
at ticks 33 and 38 using retained evidence. At the current measured posture at
tick 43, memory also flagged `FL_hip:0` using frame-0 evidence. The independent
native trace records hip/panel contact 80 ms after tick 43; the map did not consume
that future contact or native pose. No alternative action was executed here.

For the uninterrupted case 039, the route proposal contains all 43 preceding
poses in reverse order (frames 42 through 0), even if a shorter straight path
could be imagined. This tests history ordering and retention, not physical
backtracking. Case 052's failed observer leaves its route query unavailable.

The memory retained 16,229 / 15,067 voxels, below its fixed 250,000 bound. Median
map-update time was 36.628 / 39.617 ms, maximum 41.986 / 41.724 ms. These times
exclude image loading, observer computation, footprint queries and native command
execution. They add work to the existing pipeline and do not establish a 100-ms
control deadline. The entire replay and verification took 17.531 s.

Four focused synthetic tests passed (0.15 s), covering unknown space, closed
negative/boundary voxel intersection, original witness retention, invalid-insert
atomicity, no-shortcut reverse history, and stale/fault rejection. The complete
raw replay additionally exercised real sensor/pose digest admission, uninterrupted
frame numbering, model-prediction shapes, persistent/current conflict inclusion
and the original visual failure. No optimizer or native simulation ran.

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_joint_visual_surface_memory_v1_attempt_001`.
The launch freezes 895 source paths; all inherited native/input bindings and the
original native artifacts were checked before and after, as was the fixed URDF.

| Identity | SHA-256 |
|---|---|
| Launch | `21b63a30376c25fcc20dbfd86dc035eb4d411d576a20b10a9869b078d6b298c0` |
| Result | `d19b8254779c7aafb7ef4d0bee612a0d266da5236f1bea831f376d775c79dcfe` |
| Case 052 memory report | `eff33d5c4b1851bfb76965450147498599581f3fc58f14a1ce660b042e17bb31` |
| Case 039 memory report | `7c84db9168acef6e87084867476b4254255ed74f63dc945afb1105a865f7d3b0` |

The next controller experiment should compare current-frame and persistent
surface-intersection handling while keeping the checkpoint, goal, candidate bank,
cost, observer and command budget matched. Freeze how a conflict changes selection
before new native outcomes; retain all candidate evidence and failures. It must
test actual goal progress and contact, not merely count vetoes or stops. A route
proposal still needs observed floor/free-space support, frontier selection and
execution toward remembered waypoints. The full objective remains independent
maze navigation with predictive/reactive, JEPA and memory comparisons, physical
backtracking, realistic timing and eventual bounded hardware evidence.
