# Confirmed-floor collection: provisional terminal diagnosis

The native collection finished, but session 53952 is still performing its fresh
raw audit and final prefix comparison. The completed experiment result does not
yet exist. Do not treat this note as an audited result or launch a successor
from it. The preserved collection-level result has SHA-256
`8241e3753fe952701bee956632257eeca95fd40b4b0b50126ec18325c665829a`;
the complete decision stream has SHA-256
`cfbd2ef3396eff6a3c548a32a55dd4b6b52ee9b49eed000b26cfe06538c352a4`.
Both were rechecked unchanged after each bounded read-only inspection.

Collection reports 1,547 completed commands, 1,548 paired observations and
78,100 physics samples. Terminal frame 1537 is
NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS, followed by ten
zero commands. No physical/acquisition stop or observed arrival is reported.
Observed outbound goal distance is 1.1449574973376988 m; it is not a native
verified distance. The changed command at frame 1482 is the prescribed left arc.

All six final eight-segment nominal paths pass, with current nominal clearance
0.48794527908004415 m. All primary surface checks pass. Every candidate is
blocked by auxiliary FL_foot:0 sample bounds, now at cells [152,-10,-10],
[152,-11,-10] or [151,-10,-10], whose latest contributing frames are 1469,
1474 and 1466 respectively. The current primary confirmation has zero seeds,
adds no floor classifications, and the original current auxiliary floor count
is also zero. The additional partition retains 29,529,600 returns, including
3,300,577 other/unknown returns.

Read-only inspection 97938 completed; subsequent plane inspection 71161 also
completed. It uses the same measured ground-quad geometry as the earlier
descriptive diagnosis, without applying the old absolute-height seed band.
No native pose, segmentation, model fit or classification change is involved.

| Frame | Primary candidate points | Primary plane RMS | Auxiliary candidate points | Auxiliary plane RMS |
| --- | ---: | ---: | ---: | ---: |
| 1466 | 68892 | 11.319 micrometres | 266395 | 5.497 micrometres |
| 1469 | 64411 | 17.084 micrometres | 260339 | 5.996 micrometres |
| 1474 | 62204 | 14.199 micrometres | 258234 | 5.425 micrometres |
| 1537 | 57442 | 10.186 micrometres | 254370 | 4.926 micrometres |

The retained initial floor height is -0.3197604032098499 m. At terminal frame,
primary candidate heights are 10.628–16.185 mm above it, and auxiliary heights
10.402–16.067 mm above it. Thus the original 10 mm band excludes the current
plane despite abundant geometrically coherent measured patches. The two fitted
terminal normals are approximately [0.0023880,0.0029507,0.9999928] and
[0.0023859,0.0029484,0.9999928], with offsets 0.3000123 and 0.3000201 m.

The first floor correction addressed retained classifications but still relies
on the initial-height primary seed rule; its availability fails as observed
plane registration drifts. A next candidate should address observation-grounded
floor/pose registration explicitly, with a new pose-consumer contract if poses
change. It must not silently overwrite the frozen joint-pose witness or just
declare all low returns to be floor. First complete the current raw audit and
readout, then bind any prospective change. Current native and old strict failures
remain preserved; the full navigation goal remains unachieved.
