# Residual planner memory comparator: causal boundary verified

Original replay 82947/PID2649125 exited 0. Result SHA-256:
`f802f1c14263878520e4d9221128bd037442c2ed9fdae62de2ba589eb1efaf49`.
Root `go2_residual_current_observation_planning_prefix_v1_attempt_001`, launch
`7441802f63b5ee8ebf86cadf9972cb036f5c65aee663ed9dd99c91bc71690416`, decision stream
`7bca35ed3f4c876e9b4cf4559fcfdfb46a9e36b639928d4140355e44d11a6743`.
The completed result binds 1755 sources and both outputs. Reported phase wall
275.4552524499595s includes the final original-input authentication and excludes
the initial authentication before launch. Do not relaunch this completed process.

The replay consumed observations 0 through 11 of the completed original
residual-anchored maze2 episode. The first changed request is at observation 11:
right turn `[0,0,-0.45]` becomes left turn `[0,0,0.45]`. Both controllers remain
active at this boundary. All eleven earlier requests match completed original
commands. No following observation was consumed on the changed trajectory and
no candidate command was physically executed.

Both fresh model instances retained exact original corrected state
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`, without
parameter gradients. All nine common raw six-action forecast banks, frames
3 through 11, match exactly. Complete original decisions reproduced. Public
input bytes, accumulated observation cells, observed mission state, executed
residual history and retained contact-state fingerprints match. The full prior
pending forecast was checked before each observation; only the current choice
may differ at the stopping boundary.

| Boundary field | Persistent planning cells | Current paired observation |
| --- | ---: | ---: |
| Floor cells available to planning | 1264 | 906 |
| Occupied cells available to planning | 75 | 60 |
| Route cells | 40 | 34 |
| Waypoint X (m) | 0.47500000000000003 | 0.525 |
| Waypoint Y (m) | -0.025 | 0.07500000000000001 |
| Selected action | right turn | left turn |

Both proposals are `OBSERVED_FLOOR_ROUTE_TO_FRONTIER` and both selectors remain
in waypoint mode. Current-view cell hashes:
floor `92011726aa6a8f3fe2a43cac2b5eba1a30d8fcd90ef3ee6f62452c7d6b041cee`,
occupied `abe4b575cbe3d1d3b82a48c37deda35944e0142d6872dcdc5449bfc1bc01f35c`.
Retained contact evidence, tracking/floor anchors, model temporal history,
residuals, mission and scan state remain. This is not a fully memoryless policy.

Independent saved-boundary verification 90257 exited 0. It reconstructed all
12 public packet fingerprints, recomputed every saved comparison against the
original decisions and actual tape, checked all nine forecast-bank equalities,
and verified input/source/stream bindings before and after. Its receipt is
`go2_residual_current_observation_planning_boundary_verification_2026-09-10.json`,
SHA-256 `df9bd78800e1aad7a31e3f39a542ebc59452e875297d798a4a105cb21ba5c294`.
That independent pass did not rerun neural inference; the original runner
reproduced the complete original controller decisions with two freshly loaded
models and performed the retained contact-state comparisons.

Completion check 26564 exited 0 after verifying the final result's sources,
outputs, original input bindings and exact report fields. The original full
completion verifier passed both before and after replay. Each call executed
the original verification conditions across 137894 unique files, hashing
64656470245 bytes initially and again at the end, using no persistent digest
cache or module-global mutation. All 1890 live-queue source bindings remain
unchanged.

This proves that planning-grid persistence causally changes the selected action
under the current residual planner. It does not prove either turn is better,
a memory advantage, an arrival, physical backtracking or a round trip. A fresh
same-model physical comparison would need the 1300-sample / 12-observation
common prefix through observation 11; a study using different newly fitted
models must bind those models and its own matched startup evidence.

The original maze3 raw audit and six-model maze2 waiter remain live. They retain
ownership of the next native execution. This result adds no native episode:
30 completed audited episodes, zero verified round trips. The goal remains
active. Independent-layout execution, matched method/training/memory results,
realistic sensing, timing feasibility and bounded hardware evidence remain open.
