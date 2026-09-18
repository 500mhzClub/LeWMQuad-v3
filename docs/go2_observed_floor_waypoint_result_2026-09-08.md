# Observed floor/waypoint replay result — 2026-09-08

The new floor map extracted measured floor coverage, but none of the 73 accepted
poses in the two recorded native traces supported an entry route under the fixed
planner. Every proposal requested an additional view. This identifies a missing
active-view step before approaching the panel; it does not demonstrate waypoint
execution, exploration, recovery or goal-reaching.

The new [floor mapper](../lewm/joint_visual_floor_map_development.py) consumes
public RGB-D, quiet initial specific force and the bound current joint visual
pose. It infers a candidate flat-floor height, then independently checks every
image cell in each projected 5-cm floor square against measured ground-mesh and
height predicates. A fitted plane alone never supplies observed coverage.
Sampled elevated surfaces populate a separate persistent obstacle map. The
[waypoint proposer](../lewm/observed_floor_waypoint_development.py) uses a nominal
450-mm obstacle inflation, observed-floor graph routes and explicit unknown
initial connectors. It cannot grant motion or full robot clearance.

The [protocol](go2_observed_floor_waypoint_v1_2026-09-08.md) was frozen before
replay. All 84 original frames were accounted for. The corner observer was
reconstructed exactly. Case 052 admitted 29 floor-map updates, then failed at
frame 29 because the original visual pose was unavailable; its ten following
records remained unavailable. Case 039 admitted all 44 updates. No additional
floor-map admission failure occurred while a current pose was available. The
original contact, visual and frame-36 near-depth failures remain unchanged.

| Quantity | Case 052 | Case 039 |
|---|---:|---:|
| Initial observed floor squares | 493 | 367 |
| Final retained floor squares | 1,452 | 1,318 |
| Final retained obstacle squares | 93 | 87 |
| Initial floor height in gravity-aligned map | -0.319917 m | -0.319585 m |
| Accepted poses requesting another view | 29 / 29 | 44 / 44 |
| Proposed entry routes | 0 | 0 |

A post-replay inspection of first-frame witnesses found 259 / 200 floor squares
outside inflated obstacles. The nearest such square was 1.018 / 1.064 m away;
32 / 22 were within the fixed 1.25-m entry radius. The origin itself was outside
the closed inflated obstacle cells. Every entry connector nevertheless crossed
an inflated obstacle. The failure therefore was not simply absence of any floor
coverage, an out-of-range nearest square, or an occupied origin. The camera's
observed floor did not supply a connection around the panel under this nominal
footprint. Do not reduce inflation or enlarge the entry radius merely to make
this same trace produce a route.

The floor hypothesis is conditional on quiet initial force, a static flat scene,
and uncalibrated visual transforms. Obstacle height filtering and sampled returns
do not establish continuous volume clearance, and a route through floor squares
would still require robot-footprint and execution checks. Even an obstacle-free
unknown connector would remain explicitly unobserved rather than being written
into the floor map. No held-out data, native pose, native scene geometry or future
contact entered the mapper or proposer.

Five tests passed in 1.33 s. They covered a genuine graph detour around an
inflated wall, four-neighbour/no-corner-cut paths, unknown connector retention,
rejection of connectors through observed obstacles, closed line/corner coverage,
and rejection of missing depth, a wrong plane and out-of-view floor squares.
The full native-data replay then tested actual packet/pose admission, coverage,
proposal generation and fault latching without executing a new command.

| Recorded CPU stage | Case 052 median / max (ms) | Case 039 median / max (ms) |
|---|---:|---:|
| Map update | 89.825 / 118.442 | 93.143 / 116.113 |
| Waypoint proposal | 70.802 / 98.830 | 59.936 / 99.963 |

These are separate stage timings excluding acquisition, the visual observer and
native physics. They add substantial work to an already over-budget controller;
no real-time claim is made. Complete replay/verification took 25.538 s on one
CPU process. No GPU, optimizer or native simulation ran.

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_observed_floor_waypoint_v1_attempt_001`.
The launch freezes 900 source paths; the two detailed map reports total 366,797
bytes. Source/input, predecessor surface-memory and original native artifact
bindings were verified before and after replay.

| Identity | SHA-256 |
|---|---|
| Launch | `ff9159b290db08aa3e636058cb9bcf22b8be1b13fc4baacc6cc6a5198da88cf7` |
| Result | `1c482b9a875092c9866d8085832426e95c802f60ca094bc4b52c4261428f1182` |
| Case 052 floor report | `008c87039eb4f300de966270d4e667405061f0029905711672aa441a71d16ad2` |
| Case 039 floor report | `56d2b05f99321adbc210ba1461d985bd4c6343f0da9ef94006e27ae1763ad1fa` |

Next, integrate an explicit view-acquisition phase with the learned candidate
planner: use measured heading and image-derived map evidence to request a side
view while still away from the panel; regenerate waypoint proposals after that
observation; retain unknown connector and clearance limitations. Test actual
closed-loop view acquisition and onward goal progress in a fresh frozen native
experiment. Preserve this no-route result. Recovery/backward motion requires its
own declared command and training support, as established in the
[matched native result](go2_surface_memory_goal_probe_result_2026-09-08.md).
Independent-maze navigation, physical backtracking, matched ablations, realistic
timing and bounded hardware evidence remain unfulfilled parts of the active goal.
