# Combined mesh-reuse replay completed and independently checked

The combined frozen-footprint and mesh-reuse controller preserved all 405
complete normalized decisions, 402 fresh model forecasts and four retained-state
checks on the original JEPA prefix. Its total controller time was 9.50% lower
than the original controller in this paired run. Every post-warmup call still
exceeded 100 ms. This adds no native episode or navigation success.

The raw replay completed normally in session 52230. The independent checker and
its waiter completed normally in session 70769. All 2,017 original source
bindings, two output artifacts and 2,020 verification source bindings were
checked unchanged.

Root: `go2_reused_floor_mesh_prefix_v1_attempt_001` under the existing external
navigation development artifact directory.

- Result: `1a7909f8ae1ab90a5e6281cfa187495140d029c384c7ea9d37422743854f543e`.
- Launch: `15db8432621bbbc33fb098ce2a71bfa63b9b82fdf9e62421603508f8e879d6b0`.
- Comparison stream: `600666c150d489dc9f852c31dfd17ceb932f07399f70f7e9aae6d82395581946`.
- Independent verification:
  `docs/go2_reused_floor_mesh_prefix_verification_2026-09-10.json`, SHA-256
  `f9e9e615dbbb54ac98fc9b6563f8ac2c6b1f6ea8e1700a5c0763b83818f90820`.
- Verification waiter terminal: SHA-256
  `4727dcd13a39c18faa93bc3921bd9768c1ec110ca2274a65b2c1af97a36637fc`.

| Window | Calls | Original total | Candidate total | Original median | Candidate median |
| --- | --- | --- | --- | --- | --- |
| Observations 3–404 | 402 | 336.089509 s | 304.162161 s | 788.836 ms | 728.716 ms |
| Early navigation, 3–12 | 10 | 7.035249 s | 6.638938 s | 697.175 ms | 642.110 ms |
| Repeated hold, 395–404 | 10 | 16.388428 s | 12.168225 s | 1643.940 ms | 1190.936 ms |

The total-time reductions are 9.50%, 5.63% and 25.75% respectively. The
post-warmup median reduction is 7.62%. Both controllers exceeded 100 ms on all
402 post-warmup calls. Order alternated by observation; these are shared-host
controller-observe timings, with sensor acquisition excluded. The recorded
1688.9305194180924-second wall time starts after initial admission and includes
the final full input audit; it is not controller latency.

Two separately stored copies of model
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`
were unchanged and had no gradients. Public arrays and original completed
command endpoints matched. Observations 3, 12, 395 and 404 reproduced the same
retained-state hashes as the completed frozen-footprint reference. The raw state
population contains memory, floor and occupied dictionaries, residual and
history; it uses no type or value normalization. No observation 405 was read.

The independent checker reconstructs expected original and candidate decision
hashes from saved original rows, checks command endpoints and reference public
hashes, recomputes timing windows and compares the reported retained-state
hashes. It does not rerun inference, reload sensor arrays or independently
reconstruct hidden state; the original paired replay owns those checks.

This run does not establish an improvement over the earlier frozen-footprint-only
candidate. That separate paired replay reported 13.50% lower total time and a
663.82 ms candidate median. The runs were not a direct head-to-head comparison,
so their difference does not isolate mesh reuse or prove an incremental
regression. The isolated helper improvement did not translate into demonstrated
overall superiority. Keep mesh reuse unpromoted; do not replace a queued native
controller or claim real-time behavior from this result.

The six-case native batch remains in progress. Navigation failures, independent
layout comparisons and realistic sensing/timing validation remain unresolved.
