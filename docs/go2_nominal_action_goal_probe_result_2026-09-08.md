# Nominal action constraints: consistent forecasts, still zero verified goals

The new constraint now makes every selected first-half-second forecast pass
both the route planner's 0.45-m observed-obstacle test and the existing
articulated-surface filter. All 27 focused tests and all four fresh native raw
replays passed. This improves a specific controller inconsistency but does not
establish navigation: verified goals remain 0/4.

| Model / layout | Terminal reason | Terminal / minimum goal distance |
|---|---|---:|
| RGB direct / 039 | All candidates surface-vetoed at tick 83 | 1.444182 / 1.115987 m |
| RGB JEPA / 039 | Mission budget exhausted | 1.129313 / 1.129313 m |
| RGB JEPA / 052 | Nominal start infeasible at tick 28 | 1.068954 / 1.068859 m |
| RGB direct / 052 | Every predicted candidate nominally infeasible at tick 28 | 1.083074 / 1.083011 m |

There were no physical stops, disallowed contacts or visual-estimation failures
in these bounded attempts. All 426 RGB-D frames passed strict visibility and
hard measurement checks. The three deliberate no-candidate stops retained their
ten zero-command drain intervals. Avoiding later visual failures by stopping
earlier does not repair the observer or count as successful navigation.

Direct 039 provides useful prospective progress. At ticks 28, 33 and 38 the
constraint substitutes right turns for the original proposed right arc; at 43
it substitutes a right arc for forward. It then retains WAYPOINT mode through
the rest of the run, instead of repeatedly losing its route and scanning.
Terminal initial-frame displacement is [-0.059699, -0.706272] m: an unfinished
detour, not progress toward verified arrival. At tick 83 its current nominal
clearance is 0.840638 m and every candidate passes nominal geometry, but all six
are rejected by front-foot surface intersections. Hold hits both front feet;
left turn hits the right front foot, and right turn hits the left front foot.
The inherited surface veto therefore remains a distinct blocker.

The two 052 cases expose different predictive limits. JEPA's last accepted
forward forecast still predicts a clear chord, but its measured endpoint enters
the nominal disk exclusion region (0.439757 m clearance). Direct's measured
start remains just outside it at 0.451828 m, but every predicted continuation,
including hold and both turns, falls inside it. The controller explicitly stops
in both cases; it does not erase observations or waive the radius to escape.
The zero-command drain still permits physical settling, so terminal positions
are not identical to the stopping-decision positions.

JEPA 039 makes two waypoint choices, then returns to view acquisition at tick
33 and mostly holds (39 hold selections). The filter changes 42 selections in
that case. At tick 238 the observed start clearance is 0.477862 m; hold predicts
only 0.451828 m minimum clearance, while forward, both arcs and right turn fail
nominal geometry. No surface veto is involved there. Incorrect or conservative
forecasts can stall exploration even when the consistency rule is implemented.

Across the four cases there are 77 forecasts and 74 selected actions; every
selected forecast passes both constraints. Four direct-039 choices, 42 JEPA-039
choices and the two 052 stopping choices differ from pre-filter selection.
The retrospective one-square check finds zero selected predicted conflicts,
but one measured endpoint-chord conflict (JEPA 052). This remains evidence of
unbounded prediction error, not a calibrated safety margin or continuous-path
certificate. Maximum accepted visual XY errors are 4.780 mm, 2.532 mm,
0.877 mm and 0.877 mm in the table's case order.

The first changed command is tick 28 for direct 039, 23 for JEPA 039 and 28 for
JEPA 052. Through each pre-command observation, RGB, complete observer/map
evidence, model forecasts and all native/public array prefixes exactly match
the continuous-connector predecessor. Direct 052 instead first changes its
terminal state at tick 28 while both controllers still request zero; the first
different requested command is tick 33. The readout correctly reports evidence
and array differences over that longer command-only prefix because the new
controller has already entered terminal drain. A separate exact comparison
through the terminal intervention (29 frames; native samples 0 through 2149)
matches every physics/body/gyro array field, observer/map evidence and forecast.
Do not mistake terminal-phase bookkeeping after that intervention for a failed
raw replay or claim equality over the readout's longer prefix.

Next, refine the surface representation using the original observed returns:
test tight within-voxel bounds and primitive geometry while retaining every
sample, mixed/unknown evidence and all first-observation identities. The earlier
floor-patch diagnosis does not justify deleting floor voxels or waiving foot
collisions wholesale. Separately, moving-state prediction/braking uncertainty
must be addressed before relying on millimetre nominal margins. These are
prospective changes; none alters this attempt or its zero-goal result.

Preflight observed 16 physical/32 logical CPUs, 0.3% CPU activity, 82.22 GB
available RAM, 88.44 GB artifact free space and idle GPUs. Four fresh single-
thread CPU processes used the previously verified native scaling configuration.
Post-launch work took 184.073 seconds. Native artifacts bind 966 sources and
1,834 files totaling 687,620,399 bytes, with 24,100 physical samples and 426
RGB-D frames. The readout binds 968 sources. All 422 command iterations exceed
100 ms; case median full-iteration times range from 299.919 to 345.692 ms.
These are concurrent timings with paused simulation during compute, not
real-time or uncontended performance.

| Artifact under the development base | SHA-256 |
|---|---|
| `go2_nominal_action_goal_probe_v1_attempt_001/launch.json` | `2cf7994a176e64c5d74318bdcb65330f4626a38a2caa25ccd8936b06e03d3145` |
| Native `result.json` | `f9708a794b678bec4c61c5aba13dc3a3886ec836de7127f5440178cd5737288b` |
| `go2_nominal_action_goal_readout_v1_attempt_001/launch.json` | `dbb73497869cc7a86d304ad0c42bdcd89c95469875d5eacafeda9658e6c71053` |
| Readout `result.json` | `c4f22e57ad82e3e50a03692fc288705c106b0c070df942f0d6c2f55a76931219` |

The complete novel-maze goal remains active and unfulfilled. These known-layout
development runs establish neither physical backtracking, independent-maze
generalization, JEPA advantage, calibrated safety nor hardware readiness.
