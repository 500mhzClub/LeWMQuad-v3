# Frame-cache implementation equivalence completed

All960 complete controller decisions matched the saved common-floor candidate
prefix exactly, including registration, original visual witnesses, maps,
mission state, forecasts, contact checks, nominal constraints and commands.
No changed-command outcome was read. Model state remained
`4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.
Source and input bindings were verified before and after the replay.

Root `go2_frame_cached_floor_prefix_v1_attempt_001`; session88913 completed.
Result SHA-256:
`633a16730506480011d9a00a4a76c63daa9c62c19eaf39695a3376abfc97e14b`.
Launch `6e16b5963432fc4f0ed4ab131c1e8d21d0e5ae8e93f3557a9cb66b44fe61db6a`;
timings/equality stream
`8209f8e24afef6bce02d332ac4a2e90c41f88c6dfc3b35fba91abead18c5264e`.
1508 bound sources. Wall time762.990808s. Median non-warmup controller call
734.115640ms. This is an unpaired replay timing, not a controlled full-loop
speedup. The original native experiment and diagnostics competed during part
of the replay; no native timing qualification follows from this result.

All960 observations had exactly two cache misses and no capacity fallback.
946 had seven hits,13 had six hits and the initial observation had eight hits.
The original pure floor-index computation therefore ran twice per observation
inside the map, preserving every decision on this tested prefix. Registration's
own plane computations were not cached. No cross-observation result survived
the explicit cache scope. The separate record-copy candidate was not installed.

This establishes implementation equivalence only on the bound960 observations.
It is available for a separately frozen future controller experiment, while
the running eighth native attempt keeps its original1509-source implementation.
It does not repair that attempt's navigation failure, its known predecessor
visibility failure, or the remaining independent-layout/baseline/hardware goals.
