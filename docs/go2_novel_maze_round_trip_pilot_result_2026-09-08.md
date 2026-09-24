# First prospective maze native result

**0/1 verified round trips; no observed arrivals.** The unchanged JEPA controller
crossed two valid edges of the first fixed prospective maze before stopping at
decision 414 with no phase-admissible action satisfying the original surface and
nominal-path constraints. Collection, raw sensor reconstruction, complete model
and command replay, model-state integrity, and physical visibility audits passed.
There were no physical or acquisition stops and no hard measurement failure
frames. All 425 auxiliary frames had zero robot-occluded pixels.

Artifact base:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.

| Artifact | SHA-256 |
| --- | --- |
| `go2_novel_maze_round_trip_pilot_v1_attempt_001/launch.json` | `98f656a5bd8c4d7afdde6cda6a094afcb2c2b33c0cccb8efa30f6585947e7985` |
| Same native root, `result.json` | `5874c1fea08b40d90e676b69acb7d3a103d96a6281e0911801bf6bd6e1ab1570` |
| `go2_novel_maze_round_trip_readout_v1_attempt_001/launch.json` | `9255a6a8e3d9ce86c6942f19e566992449d630fac9b05f11c375d9584a81f893` |
| Same readout root, `result.json` | `fe2b0fbdadba8f1ae29e67d8897d2ea3bdfe36ec5caa8e4a4a4200a349087bc4` |

The native launch bound 1,401 source files and completed collection plus audit
in 721.619 seconds. The readout bound 1,404 sources. The attempt contains 424
completed commands, 425 paired observations, 21,950 physics samples and ten
terminal zero commands. The 3,000-interval mission budget was not exhausted.

## Physical and controller outcome

The native cell sequence was `[-1,0] -> [0,0] -> [0,-1]`, with crossings at
physics samples 4,552 and 9,878. Both were declared open edges, all native XY
positions remained in the maze, and maximum 2 ms displacement was
0.000483603 m. There was no return phase or backtracking claim. Actual XY path
length after initial observation was 2.407213306 m. Minimum distance to the
outbound instructed goal over the whole trajectory was 2.768958301 m; terminal
distance was 3.010469432 m at initial-frame XY
`[0.8930771628498098, -1.1539135463442127]` m. Maximum observed/native pose XY
discrepancy was 5.11103 mm.

There were 400 waypoint and twelve view-acquisition selections. Every waypoint
proposal targeted an observed frontier. Selected actions were 14 forward,
54 left arcs, 78 right arcs, 93 right turns, sixteen left turns and 133 holds;
24 selections were infeasible, including the terminal eleven-frame wait sequence.

At terminal decision 414 all six original surface checks were clear. The map
requested an additional view. The observed position was
`[0.889016715481914, -1.1522774725300697, -0.026807366383923687]` m. The nearest
recorded occupied cell was `[13,-15]`; hold/turn first-segment minimum distance
was 0.44472697212062623 m, inside the unchanged 0.45 m nominal radius. Every
candidate's original eight-step path therefore failed. This is a nominal-policy
failure, not an observed robot collision or a sensor integrity failure.

The stored final forecasts suggest a possible explicit recovery hypothesis:
from frames 407–414, the phase-allowed, surface-clear left-turn path never gets
closer to recorded occupied cells than the current starting clearance, although
it remains inside the nominal radius. This is forecast evidence only; those
turn commands were not executed, and neither actual recovery nor calibrated
model error is established. Any new policy must preserve this failure and get
its own source, replay and prospective native attempt identities.

## Timing and limits

Median observation-plus-control time was 750.580 ms, median iteration with command
was 782.439 ms, and median iteration including lossless receipt writing was
787.680 ms. Every recorded iteration exceeded 100 ms. Receipt writing itself had
median 5.588 ms and maximum 15.402 ms. These measurements are from one native CPU
worker. Physics paused during computation and ideal sensor/zero acquisition
latency assumptions remain; no real-time or hardware qualification follows.

One new development maze has now been executed and independently raw-audited,
unsuccessfully. Matched direct, reactive/non-predictive controls, other layouts,
and RGB/planning/memory attribution remain outstanding. Neither this result nor
a future local recovery alone meets the full active goal.
