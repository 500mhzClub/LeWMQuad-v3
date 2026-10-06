# Stage-2 strip stalls: is the obstacle veto real or a phantom? (5 October 2026)

Andrew asked for a strict one-hour diagnosis, with no harness edits. On the stalled strips, is `CURRENT_OBSERVED_OBSTACLE_VETO` firing on a real wall within the veto distance, or on a phantom from slip-induced pose or map error?

**Answer: a real wall.** The robot's centre is genuinely within 0.45 m of a wall at every veto. The veto measures that distance to within about 2 cm, and the tracker is accurate. Once inside the 0.45-m disc, the dispatch veto blocks every command, so the robot holds until tracking fails.

PRELIMINARY. Runs:
- `s2smoke_mup030` (v2 strips, maze 33);
- `s2smoke3_marked` (mazes 33 and 47);
- `s2smoke3_unmarked` (maze 33).
All are C1, μ_p = 0.3, harness `reserve_exit_v2`.

## How the veto works (`lewm/fresh_obstacle_dispatch_development.py`)

- **Input:** current paired depth from both cameras.
- **Obstacles:** returns 3–65 cm above the current measured floor plane, binned into 1-cm cells.
- **Test:** each dispatch tick, a 0.45-m disc swept from the current position to the commanded endpoint (requested speed times the remaining window) must be clear of those cells.
- **For an in-place turn** the endpoint is the current position, so any return within 0.45 m of the centre vetoes it.
- **Robot position and obstacles come from the same current pose and depth.** Pose or map drift therefore cannot by itself create a phantom here. Only a floor-plane misfit could, by turning floor points into obstacles.

## Evidence

**At every veto tick:**

| Run | Veto ticks (on strip) | True centre-to-wall (median, max) | Veto-reported nearest return minus true (median; 95% within) | Tracker position error (median, max) |
|---|---|---|---|---|
| maze 33 unmarked (v3) | 461 (100%) | 0.393 m, 0.457 m | +0.021 m; 0.042 m | 0.017 m, 0.019 m |
| maze 33 marked (v3) | 476 (100%) | 0.394 m, 0.444 m | +0.020 m; 0.041 m | 0.018 m, 0.021 m |
| maze 47 marked (v3) | 560 (100%) | 0.393 m, 0.422 m | +0.025 m; 0.045 m | 0.014 m, 0.015 m |
| maze 33 (v2 smoke) | 461 (100%) | 0.393 m, 0.457 m | +0.021 m; 0.042 m | 0.017 m, 0.019 m |

How each column was measured:
- **True distance:** from the physics base position to the nearest wall-box surface (`specification.json` wall boxes).
- **Reported nearest return:** `nominal_connector.minimum_observed_cell_distance_m`.
- **Tracker error:** the registered pose against physics truth at the veto's observation time, in the initial body frame.

On the strips, 100% of veto ticks have a true distance under 0.45 m (two borderline ticks at 0.457 m). The reported distance runs about 2 cm long, consistent with the 1-cm cells and the wall surface. No phantom returns: every veto has a real wall inside the disc.

**Where the robot sits:**

| Run | Centre to nearest wall, on strip | Off strip |
|---|---|---|
| maze 33 unmarked | 0.40 m (min 0.35) | 0.56 m (min 0.45) |
| maze 33 marked | 0.40 m (min 0.35) | 0.58 m (min 0.48) |
| maze 47 marked | 0.40 m (min 0.35) | 0.59 m (min 0.46) |

**How it gets inside the disc** (first time the centre comes within 0.45 m of a wall, and the 3 s before):
- **Marked mazes 33 and 47:** 0.20 m and 0.19 m inside the strip, while **turning in place** (99–100% turn commands). Over those 3 s the centre slid from 0.58 m and 0.57 m to 0.45 m from the wall, with lateral body velocity up to 0.10 m/s. Low friction under the turning legs lets the body slide sideways.
- **Unmarked maze 33:** 0.14 m before the strip, while moving forward (93% forward or arc commands), from 0.51 m to 0.45 m.

**Then it is trapped.** After first entry the centre stays within 0.45 m for 87% (maze 33 unmarked), 99% (maze 33 marked) and 100% (maze 47, never leaves) of the remaining mission.
- Every command is vetoed, because an in-place turn checks the disc around the current position, and a translation starting there also overlaps the wall.
- The veto latches for the command window (`COMMAND_WINDOW_VETO_LATCHED`, 75% of ticks in maze 33).
- The robot holds and the planner keeps selecting scan turns. The cameras face the near wall, visual support falls, and the mission ends on "measured visual pose unavailable" (0 contacts).

## Conclusion

- **Real, not phantom.** The dispatch veto correctly sees a real wall closer than its 0.45-m disc.
- **The cause is slip plus a missing exit.** Slip during in-place turns on a strip carries the centre about 12 cm toward a side wall: 0.57 m from the wall is only 0.12 m of margin. The dispatch veto, unlike the forecast check's v1/v2 exits, has no exit inside the disc.
- **Same family as stage 1's turn-exit contacts:** in-place turning plus slip near walls. Here the dispatch veto stops the motion before contact, at the cost of a stall.
- **No harness change made,** as instructed. Recordings proceed with the 60-s stall stop. Evaluation classifies these stalls ("strip stall: dispatch veto inside the disc").
