# Row-tiled dense floor geometry, prepared for controller comparison

The completed receipt-copy profile places dense floor-index construction among
the substantial remaining CPU costs. Its ten late observations included 40
index calls with 2.2346 s cumulative profiled time; these timings predate density
routing and progressive patch batching and include profiler overhead. Do not
add overlapping cumulative times or infer current whole-controller gains.

`lewm/tiled_dense_floor_cell_index_development.py` keeps the original full-grid
validator, camera transform, per-corner height gates, three triangle formulas,
thresholds, prefix construction, output ownership and immutable result mapping.
It evaluates triangle arithmetic in 32-row blocks with a final partial block.
No cells or triangles are skipped and no numerical tolerance is introduced.
`lewm/tiled_density_routed_floor_cell_index_development.py` privately binds that
kernel only to the existing router's dense branch. The sparse branch and its
original ambiguous-boundary fallback remain unchanged.

The recorded probe used the original primary and auxiliary depth and recorded
registration orientations at frames 0, 100, 400 and 800. It compared original
dense, current density-routed and tiled-density kernels with 12 repetitions per
camera, balancing all six execution orders. Every returned array matched the
original byte-for-byte, including dtype and shape; inputs were unchanged.
Original raw-worker input bindings were verified before and after the probe.

| Frame | Camera | Current routed median ms | Tiled median ms | Total time reduction |
| --- | --- | ---: | ---: | ---: |
| 0 | primary | 31.732 | 29.804 | 17.22% |
| 0 | auxiliary | 33.616 | 25.624 | 31.03% |
| 100 | primary | 19.761 | 19.923 | 0.21% |
| 100 | auxiliary | 32.075 | 25.672 | 20.56% |
| 400 | primary | 19.082 | 19.023 | -1.98% |
| 400 | auxiliary | 48.865 | 26.911 | 48.57% |
| 800 | primary | 21.495 | 21.239 | -0.10% |
| 800 | auxiliary | 49.647 | 27.076 | 57.45% |

The primary path is literally unchanged in this candidate. Its timing variation,
especially the first frame, demonstrates shared-host and cache noise. The
auxiliary improvement is promising component evidence, not a whole-controller
or real-time result. Preserve all timings, including the small regressions.

Probe: `go2_tiled_density_floor_index_recorded_workload_probe_2026-09-11.json`,
SHA-256 `5fc6065017cced7b28a4f43456a91490490b2eaedb3afa94af627d189939ba44`.
It binds 2,283 source paths; session 95306 exited zero. The 40 component tests
passed in 4.22 s (session 47811), including tile-boundary/partial-tile cases,
alignment and planarity neighbours, output ownership and input validation.

`TiledDensityProgressiveFloorController` is prepared as a separately named
successor to `ProgressiveBatchedFloorController`. It replaces only fresh
registration and map-cache consumers with private bindings to tiled-density
indexing, retaining progressive patch stores, memory aliases, selector, model,
observer, mission and original observe/update bodies. Registration and controller
integration tests passed: 13 tests in 5.38 s, session 57733. No live controller
or queued navigation source was changed. No full tiled-controller replay has
been launched and no tiled-controller timing gain has been established.

Next steps:

1. Complete the currently running progressive replay: session 69904, PID
   2856701, creation time 1789134105.22, artifact root
   `go2_progressive_batched_floor_late_history_v1_attempt_001`. Preserve any
   failure and run its prepared completion checker only after its owner ends.
2. Use that completed result to prepare an exact full-controller comparison
   against the tiled successor. Keep full decision equality, raw/model
   admission, alternating timing, independent models, all 1,428 observations,
   1,425 forecasts and seven original retained-state witnesses. Do not launch
   a second full CPU replay while the current one remains active.
3. Continue monitoring the unchanged native queue. The extended-budget launcher
   is active in repeated nested input verification and has not yet created a
   native scene. Later sustained-turn, contact/flow and chained-anchor waiters
   remain queued behind it. No new round trip or independent-maze result exists.

The last verified whole-controller median remains 598.603 ms against a 100 ms
target. Reliable round trips, independent-layout causal comparisons, useful
memory/backtracking and bounded hardware evidence remain outstanding.
