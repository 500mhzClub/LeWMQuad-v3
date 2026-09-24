# Progressive retained-floor batching: complete controller replay

Compare the completed DensityRoutedFloorController with a fresh
ProgressiveBatchedFloorController on the same 1,428-observation full-RGB JEPA
development prefix. Both use the original assigned model, density-routed floor
registration and mapping, and the existing packed/fused/scoped footprint queries.
The candidate changes only the two initially empty retained patch stores: the
first projection batch remains 32 frames and subsequent batches contain up to
128 frames. Chronological earliest witnesses, complete coverage receipts,
exception behavior, retained arrays and append behavior are unchanged.

The completed wide-128 synthetic benchmark exposed a 146.93% immediate-witness
time regression; preserve that result. The progressive successor measured
22.72%, 20.25% and 5.78% total-time reductions for invisible, sparse-visible and
visible-uncovered histories, with a 0.31% immediate-witness regression. These
are shared-host synthetic microbenchmarks, not complete-controller performance.

The completed density-routed predecessor reduced paired whole-controller total
time by 10.29997658524292% across 1,425 navigation observations. Its median was
598.603 ms; all 1,425 observations exceeded 100 ms. Completion witness:
`6cc04190aea13972ff2a6236dc510170d2cc458672d07df26345a743170ff370`.
It preserves the original strict sensing failure at frame 1173 and proves no
new navigation outcome.

Privately bind the unchanged original full-history replay function. Preserve
all 1,428 input and command-tape checks, alternating paired execution, complete
normalized decisions, 1,425 model forecasts, independent tensor storage,
unchanged model digests and gradients. Compare all seven original retained
memory/map/residual/history witnesses at frames 3, 12, 395, 404, 1173, 1418 and
1427. Normalize only the existing ten patch/index type paths, and the new
controller name and implementation flag. Require every baseline decision hash
to match the completed density-routed candidate hash. These seven state checks
do not separately cover every registration or mapper internal field each frame.

Authenticate the completed predecessor by reconstructing its full report,
timing population and raw/model input bindings before and after execution.
Use two independently stored fresh copies of the same assigned model. Require
the predecessor owner ended, at least 64 GiB available RAM, 41 GiB artifact
space and four physical CPUs. Use one exclusive CPU replay attempt and preserve
any failure without retry or resume. Timings include the entire observe call,
excluding sensor acquisition, input reconstruction and equivalence checks.

Run no physics or hardware and consume no observation 1428. The queued native
navigation diagnostics retain their existing controllers. Prospective native
execution, independent-maze comparisons and realistic timing remain necessary.
