# Verified tiled-controller profile and prepared freeze-only successor

The full-history profile V2 and its automatic completion checker both completed
successfully. All 1,428 input/decision identities and 1,425 forecasts matched the
completed tiled-controller replay. All three saved pstats files reconstructed
their complete JSON summaries. Sensing failure 1173 remains and there is no
navigation, real-time or hardware qualification.

- Profile root: `go2_tiled_density_progressive_floor_late_history_profile_v2_attempt_001`.
- Result SHA-256: `3ba7d2de82ed55fb4fca54a438d3e481993c3563cad1513753dfdc8ea9cbebab`.
- Completion: `docs/go2_tiled_density_progressive_floor_profile_v2_completion_verification_2026-09-11.json`,
  SHA-256 `1c3351b8647566f2d9745502aa8d43bb050101d27dfb2991536e25189cd522cc`,
  2,352 sources. Sessions 29134 and 80413 both exited zero; owners 2866966 and
  2867206 ended. Do not repeat the completed checker.
- Completion-watch result: `e0a489b0fb492c156714c439da6f16fbaff5697ec2184252d830a98c42c6b0fe`.

The late ten-frame profile has 10.903944191 s total exclusive profiled time.
Freezing original footprint receipts involved 1,923,168 visitor calls,
1.219649787 s visitor self time and 1.324683984 s total freeze cumulative time.
The fused-footprint, standard-copy, receipt-copy and frozen-receipt modules
contributed approximately 1.5121, 0.9046, 0.8590 and 0.2939 s exclusive time.
These measurements include profiler overhead and shared-host interference.
Cumulative times overlap; they must not be added. They motivate investigating
receipt handling, not a predicted whole-controller speedup or deadline claim.

## Component candidate and preserved mixed result

`lewm/atomic_leaf_fused_footprint_development.py` implements primitive-leaf
freezing and a separate primitive-leaf cached clone. Its 44 tests passed in
2.10 s, session 74422, covering graph validation, aliases, cycles, readonly and
independent ownership, cache keys/capacity/errors and original recovery functions.

The predefined synthetic component benchmark completed in session 77803,
with 2,357 source bindings. Its document is
`docs/go2_atomic_leaf_fused_footprint_component_benchmark_2026-09-11.json`,
SHA-256 `4d9bfd6302f01b82ad1da9c4bb116b2fb1447ce51dd13769418c082196799083`.
Thirty paired repetitions per operation alternate order and retain all timings.

| Workload | Freeze total-time reduction | Cached clone total-time reduction |
| --- | ---: | ---: |
| Primitive-heavy list | 52.5391% | 23.6893% |
| Nested geometry-shaped graph | 24.2938% | −0.2314% |
| Shared DAG | 10.3263% | −32.7533% |
| Late unsupported leaf, original-input fallback | 52.1394% | Not applicable |

These are synthetic component timings, not controller timings. The clone
regressions are preserved. The full controller candidate adopts only the new
freezer and keeps the original `fused_scoped_footprint_development._clone_cached_receipt`.

## Controller composition prepared, not replayed

`lewm/atomic_leaf_freeze_tiled_controller_development.py` provides
`AtomicLeafFreezeTiledController`, derived from the completed
`TiledDensityProgressiveFloorController`. It retains the existing copied-memory
scope constructor and privately binds only the freezer in the original cache
query body. Its selector reuses the original selector body with that scope.
Fresh selector fields are copied without replacing residual, map, mission or
history objects. Observation and advance methods are unchanged. The only public
metadata additions are controller identity and `atomic_leaf_footprint_freeze_enabled`;
`normalize_to_tiled` removes only that declared difference.

Six integration tests passed in 4.68 s, session 56867. They verify private
bindings and the unchanged cached clone; initial selector fields and retained
state; original sensor-failure behavior; and actual public packets, registration,
map/cache state, robot footprint queries and separately owned public receipts.
This is 50 passing component/integration tests. No full-history forecast or
controller timing comparison has yet been executed for this successor.

Next, prepare its paired replay against the completed tiled-controller result
`9f83c8e10db48c31918d386ba4867818a3e143625dde191e565aa6b796fb7307`,
launch `8ad6b00ce598b26b2058c9c904a51a908f2f63f4398b6eb5b46e8484b6ca45dc`,
completion `543ff2e72fc8f31b1f241f5f905212ca8d952348c3216ca6f73af7666b564556`.
Require all 1,428 original public observations/decisions, 1,425 model forecasts,
the original retained-state witnesses, complete raw/model input admission,
alternating paired order, complete timing populations and unchanged negative
sensing scope. Do not adopt the regressing clone candidate or infer savings by
adding component percentages. The full CPU replay slot is now free.

The original extended-budget native worker 2867880 / creation 1789139673.31
remains live under launcher 2843773. Latest checked worker CPU time was 833.30 s
and physical reads 488,829,800,448 bytes. Its episode directory is still absent:
worker input verification is advancing, but no simulation samples or new outcome
have been established. Preserve the original native queue and its frozen sources.
The full unseen-maze navigation goal remains active.
