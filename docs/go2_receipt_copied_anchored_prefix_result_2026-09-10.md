# Completed paired receipt-copy replay

The paired raw replay completed with 405 original/candidate observations and
402 forecast comparisons. Every original decision was reconstructed; candidate
decisions matched after only the declared controller identity and copy-provider
flag changes. Both fresh models retained state
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.
Complete retained observed state matched at observations 3, 12, 395 and 404.
Original full input verification passed before and after replay.

Result: `c3363fe9626fc36b8feb723346bef0344b7becccd3d9f68297d5f99ac3f7b0c0`.
Launch: `9ecb3f03a74ea5b2b8ca2f533f9bf2a744614e33ff06257e48c9702df923bc1d`.
Comparison stream:
`ab76d3075fc030991a02b5e9e4f624d003c04155a3af12030d3dbaa47d90fb4d`.
Output root: `go2_receipt_copied_anchored_prefix_v1_attempt_001` under recovery
storage's navigation artifact root. The original process exited successfully.

Controller execution order alternated by observation. Calls were timed without
profiling; packet loading, hashing, serialization and state comparisons were
outside the timed intervals. Other workstation processes remained active.

| Window | Calls | Original median s | Candidate median s | Original total s | Candidate total s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early navigation, 3–12 | 10 | 0.674186 | 0.672521 | 6.778174 | 7.020461 |
| Post-warmup prefix, 3–404 | 402 | 0.793502 | 0.778877 | 338.975874 | 331.976292 |
| Repeated hold, 395–404 | 10 | 1.609512 | 1.501643 | 15.765399 | 14.633581 |

The hold-window median was about 6.7% lower and full-prefix total controller
time about 2.1% lower. Early-window total time was slightly higher. All 402
post-warmup calls exceeded 100 ms for both controllers. This small improvement
does not resolve the execution-time requirement. These are measurements from
one alternating-order replay, not an isolated benchmark, statistical advantage
claim, full sensing/control speedup or native-navigation result.

Independent checks reconstructed all 405 original/candidate decision hashes
and public input fingerprints, verified the original command endpoints, all
1,972 frozen sources and both outputs, and recomputed all final timing windows.
The original completed replay performed neural inference and the four hidden
state comparisons; the independent checks did not rerun those computations.
Verification:
`docs/go2_receipt_copied_anchored_prefix_verification_2026-09-10.json`.

No frozen native experiment adopted this implementation. Further performance
work must address costs beyond these copy calls. The floor-index cache is
already integrated; retained historical patch queries are a separate measured
cost. Navigation, independent comparisons and hardware qualification remain
unachieved by this performance result.
