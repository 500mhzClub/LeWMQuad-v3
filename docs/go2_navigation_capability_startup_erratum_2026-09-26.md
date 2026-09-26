# Startup correctness erratum — 26 September 2026

Authority: the user's post-screen adjustment. The original ten-episode C1 screen
finished unchanged, with 3/10 round trips and no disallowed contacts or hard
clearance violations. Startup failures were 01/0, 02/0, 06/0, 08/0 and 09/0;
05/0 exhausted its mission budget and 07/0 lost visual pose after 249 s.
Those outcomes remain preserved under `v0_task_c1`.

| Set | Episodes | Old beacon/home coverage passes | Old whole-maze coverage passes | New coverage passes | Estimated initial floor insufficient |
|---|---:|---:|---:|---:|---:|
| Dev-tune | 20 | 18 | 4 | 20 | 9 |
| Validation | 40 | 38 | 7 | 40 | 17 |
| Sealed test | 120 | 114 | 29 | 120 | 60 |

The structural check accessed only exact, hash-bound paths of the newly
registered sets, with sealed results retained in aggregate. No rendering or
physics was used. Floor visibility is an analytic pinhole/ground/wall estimate
at a level 0.32-m body height, applying the deployed floor-quad classifier. It
omits settling, noise and self-occlusion and is not a measured qualification.

The map bound was an arbitrary ±4.9-m point limit within ±5-m storage. The
fixed generator envelope is 5.5 × 5.5 m. Its diagonal is 7.778174593 m; rounding
that plus the existing 0.1-m guard upwards gives ±8-m storage and ±7.9-m point
validation. This derives from maximum generator dimensions, not selected
episodes. Shared routing, coverage and coarse/fine geometry bounds use the
same constants; camera range and resolutions are unchanged. The direct
allocation/point/clearance contract failed before the change and passes after.
All 180 registered start transforms fit the derived domain.

The authorised first-frame regeneration of 01/0 reproduced every consumed
RGB-D hash field bitwise. The primary camera supplied 91 qualifying floor
quads, below the unchanged 100 requirement; the already-deployed downward
camera supplied 615. The paired floor plane was available. No mission command
was executed during this diagnostic, and no regenerated frames were retained.

`v0_startup_c2` preserves primary initialisation when it passes. Otherwise it
uses the existing auxiliary depth with its actual calibration and the same
100-quad requirement. If both fail, it preserves the measured frame-zero
origin, continues acquiring, waits one second, and may request the existing
left-turn command through the existing current-depth obstacle guard. Recovery
is bounded by 24 s and 2π accumulated absolute gyro yaw; translation is zero.
The mission's 480-s budget includes this time. After recovery it holds for
400 ms before ordinary planning. Other sensor-contract defects still fail.
No true pose or geometry enters this recovery.

The full ten-episode rerun is also the containment screen. Conservatively,
**all five original episodes that reached planning** (00/0, 03/0, 04/0, 05/0,
07/0) must reproduce native arrays, requests and consumed sensor hashes exactly,
even though some could have encountered the truncated map extent. Any divergence
will consume a harness version and be diagnosed; it will not be silently
classified as harmless. Five original pre-planning failures are excluded from
that equality requirement, not from the success denominator.

The coarse grid grows from 40,000 to 102,400 cells. Its coordinate array grows
from 640,000 to 1,638,400 bytes (+998,400 bytes). Episode peak RSS, measured
planning latency and wall time will be compared after the containment screen;
no measured runtime-equivalence or latency claim is made in advance. The
software/model environment pin, task transform, model weights, sensors,
candidate bank, physical safety criteria and retention policy remain fixed.

Original reports and the pre-fix checker failure are preserved. The existing
clearance-preference test initially failed only because it asserted the old
200×200 allocation; its expected shape was updated to 320×320. The other 24
focused checks passed, including recovery bounds and preservation of the
obstacle veto. The corrected allocation check also passed.

Final prelaunch binding: `go2_navigation_capability_harness_v0_startup_c2_final_2026-09-26.json`. The first preflight manifest is preserved; source review caught and restored the original map-publication clock hook before any C2 cohort execution. Three recovery tests passed again. The software environment pin passed unchanged. Prelaunch projection is 73.52 hours; applying 2.56× source wall cost to every controller gives a conservative 119.68-hour sensitivity, both below 160 hours. Storage projection is 46.88 GiB against 100.62 GiB usable.
