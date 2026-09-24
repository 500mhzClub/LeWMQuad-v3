# Continuous connector probe: zero verified goals in four fresh cases

The continuous entry-connector implementation passed all 22 focused tests and
all four fresh native cases passed exact sensor-to-command replay. It did not
produce goal-reaching. Three cases eventually lost visual registration; the
fourth exhausted its mission budget. All 598 camera frames passed strict
physical visibility and hard measurement checks. There were no physical stops
or disallowed contacts. Fixed direct/JEPA snapshots remained unchanged.

| Model / layout | Terminal | Terminal goal distance | Closest goal distance |
|---|---|---:|---:|
| RGB direct / 039 | Visual failure, tick 118 | 1.086760 m | 1.061885 m |
| RGB JEPA / 039 | Tick budget | 1.144473 m | 1.144473 m |
| RGB JEPA / 052 | Visual failure, tick 92 | 1.035081 m | 1.032475 m |
| RGB direct / 052 | Visual failure, tick 101 | 1.069851 m | 1.066355 m |

The change admits waypoint pursuit earlier: tick 23 on both 039 cases and tick
18 on both 052 cases. At precisely those ticks the first requested command
differs from its commitment-pose predecessor. All RGB pixels, complete visual
observer/map evidence, model forecasts, native physics, and public body/gyro
arrays match exactly through the observation before that differing command
executes. This supports attributing the initial intervention to connector
geometry; it does not compare unexecuted outcomes after divergence.

The route then disappears again. Direct 039 pursues waypoints at ticks 23, 28
and 38, but is back in view acquisition at 33 and from 43 onward. Both 052
cases return to view acquisition at tick 28. JEPA 039 makes one waypoint
selection at tick 23, then returns to view acquisition and mostly holds (42
hold selections overall). More accurate entry geometry alone therefore does
not resolve this controller's route-following failure.

The recorded selected forecasts expose a missing consistency constraint:
waypoint utility and the articulated-surface filter do not enforce the route
planner's nominal disk geometry. Direct 039 at tick 38 selects a right arc whose
predicted half-second straight chord comes within 0.422349 m of an already
observed square, violating the same 0.45-m nominal radius. The subsequent
measured endpoint chord comes within 0.432599 m. JEPA 039's tick-23 selected
forecast also violates the nominal radius, although its actual endpoint does
not. These one-square witnesses establish nominal conflicts; they do not prove
articulated collision or certify the unexamined squares/intermediate motion.

Prediction error is a separate issue. JEPA 052 at tick 23 predicts 0.493148 m
clearance from the recorded nearest square, while its measured endpoint chord
has only 0.439757 m clearance; endpoint error is 62.195 mm. Direct 052's tick-23
prediction has only 0.000384 mm nominal margin over 0.45 m, with 4.228 mm endpoint
error. Its measured endpoint still clears that square, but such margins are
not robust guarantees. The next implementation must connect candidate action
feasibility to the observed route geometry and address moving-state prediction
error; merely adding another clear entry or increasing the scan budget cannot
establish reliable navigation. Check all occupied squares, keep uncertainty
explicit, and handle an already infeasible nominal start without pretending
that additional views erase persistent observations.

All three visual failures retain both anchor and previous-frame registration
failure: rigid consensus fraction, grid support or displacement was rejected.
Their maximum accepted visual XY errors are 4.444 mm (direct 039), 2.336 mm
(JEPA 052), and 3.177 mm (direct 052); JEPA 039's maximum is 2.532 mm. These
accepted-frame errors do not validate extrapolation through failed frames.
The [floor-return diagnostic](go2_observed_geometry_refinement_result_2026-09-08.md)
is preserved separately; this native revision removed no floor/foot/calf veto.

Hardware preflight recorded 16 physical/32 logical CPUs, 0.3% CPU use, 82.03 GB
available RAM and 89.50 GB artifact free space, with both GPUs idle. Four fresh
CPU processes were selected using the bound native scaling benchmark. Collection
and replay plus final verification took 196.786 seconds after launch. The native
result binds 956 source files and 2,522 artifacts totaling 1,045,379,005 bytes;
32,700 physics samples and 598 RGB-D frames were retained. The readout binds
958 source files. The inherited per-episode `collection.status` still uses the
predecessor's terminal-audit tag; new controller labels, roots, source bindings
and top-level statuses identify this continuous-connector attempt unambiguously.

Every one of the 594 recorded command iterations exceeds 100 ms. Median complete
iteration times range from 335.738 to 345.490 ms under concurrent load. Physics
pauses during compute. These are not uncontended timings, real-time evidence,
independent-maze results or hardware qualification.

Artifacts under the development base:

| Root / artifact | SHA-256 |
|---|---|
| `go2_continuous_connector_goal_probe_v1_attempt_001/launch.json` | `a6ee9729c8943d3d20b485f7ca5d941f29ac675f2bd74b99833d65612fd47e4f` |
| Native `result.json` | `ee58cdda5586b0558665000d24fbc14c3858adad6b0efd052fbce466739c9af9` |
| `go2_continuous_connector_goal_readout_v1_attempt_001/launch.json` | `dce953ca6a60ad18e04e6950c730569eaa8124a83ca77987628e878296655854` |
| Readout `result.json` | `fa994152ac29dd12f63ee526498758f957677094c896fe4d9c7f739cdd5ee257` |

No attempt or frozen source was overwritten. The full novel-maze navigation
goal remains active and unfulfilled.
