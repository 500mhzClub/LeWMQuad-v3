# Shared grid correction and refined containment

The `v0_startup_c2` screen finished its ten fixed assignments unchanged. Its
navigation outcomes are diagnostic only and cannot pass the gate. It consumes
one harness version, as directed. No further startup defect was found: all ten
episodes initialized at frame zero (five primary, five existing auxiliary).

| Episode | Diagnostic outcome | Simulated seconds |
|---|---|---:|
| 00/0 | Round trip | 130.42 |
| 01/0 | Pose loss | 341.00 |
| 02/0 | Pose loss | 58.50 |
| 03/0 | Round trip | 463.12 |
| 04/0 | Round trip | 110.92 |
| 05/0 | Timeout | 480.00 |
| 06/0 | Timeout | 480.00 |
| 07/0 | Pose loss | 249.00 |
| 08/0 | Timeout | 480.00 |
| 09/0 | Timeout | 480.00 |

All ten had zero disallowed contacts, confirmed hard-clearance violations and
unresolved sampled hard clearance. The cohort took 9,290.87 wall seconds including
readers. Original results and failures are preserved.

The known route defect uses a stale `+100` array origin after the cost grid's
origin moved to 160. It changes 04/0's first command at 84.3 s and 05/0's at
82.7 s. For 05/0 the preceding plan at frame 824 changes its original waypoint
from [-1.575, 3.625] to [-1.575, 3.725], then its lookahead selects [-1.925, 3.625].
For 04/0 the plan at frame 840 changes [3.475, 0.925] to [3.475, 1.025]. These
are defects, not improvements. The original 00/0, 03/0 and 07/0 trajectories
were reproduced exactly by C2.

## Structural counts carried forward

| Set | Episodes | Old cue coverage passes | Old whole-maze coverage passes | Expanded coverage passes | Estimated floor-insufficient starts |
|---|---:|---:|---:|---:|---:|
| Dev-tune | 20 | 18 | 4 | 20 | 9 |
| Validation | 40 | 38 | 7 | 40 | 17 |
| Sealed test | 120 | 114 | 29 | 120 | 60 |

These are the completed structural checks, not new sealed access. The floor
estimate is geometric and omits settling, noise and self-occlusion. The 01/0
first-frame replay verified the actual cause: 91 primary floor quads versus
the unchanged requirement of 100; the existing auxiliary camera supplied 615.

## Correction and containment

The successor routes map resolution, extent and offsets through shared
generator-derived constants. Camera range, optical pixel coordinates, local
obstacle crop, safety margins, tracker estimation, sensors and model weights
remain unchanged. A boundary contract also found that target cell 158 has a
centre at 7.925 m. Internal route-cell centres now use the ±8-m storage bound;
mission cues still use ±7.9 m. Five of nine terminal-corner cases failed before
that correction.

The source audit covers 33 loaded implementation files and 47 function entries,
with body-relative obstacle crops explicitly separated from global map coordinates.
The final combined contract has 99 passing cases; six containment cases were
then rerun with an additional native-dtype equality check (all six passed).
An earlier functional suite passed 99 cases. Thirteen historical September
source-identity cases were excluded from that functional rerun; six fail their
old source hashes after these authorised edits. Those original manifests and
failures are preserved, not rewritten.

First-exposure command replays completed on exactly the five original
planning episodes. They reproduce native states and consumed sensor hashes
bitwise, regenerate frames transiently, and stop at the first actual old-bound
exposure or original endpoint. No controller is retuned or rerun by this check.
00/0 first clipped a measured floor cell at frame 312 (31.2 s). Complete replays
of 03/0, 04/0 and 05/0 found no clipping, so their corrected missions must reproduce
the whole original recording. 07/0 first clipped floor and obstacle observations at frame 1816 (181.6 s).
All five checked prefixes reproduced native arrays and consumed sensor hashes bitwise.

The corrected screen is frozen for commit and launch of all ten C1 assignments
in fresh roots. Configuration SHA-256: `a869a3ec8ff4b79d189218c48efc9dfc74d166fcd8883d3d0c8c742ab41f9d54`.
Harness SHA-256: `8ec038a87878f5d4c4da2ba5ab841f8b452563420c4389ac5946c1db4eacf6c9`.
The measured projection is 76.94/160 wall-hours including 15% contingency,
with 46.88 GiB additional output against 99.17 GiB usable after reserves. It
includes this new ten-episode screen; another iteration will require a new
projection. Any divergence before the applicable cutoff stops execution and
further changes. Passing containment makes this successor a correctness
version with no additional harness-version charge. Only after a clean screen
will regenerated-frame diagnosis guide a single recovery change; changes to
tracker estimation require approval.
