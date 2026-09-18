# Completed frozen-footprint receipt replay

The original paired raw replay completed successfully, including full input
verification before and after replay. Its scheduled independent recorded-
evidence checker also completed. Session 59832 and verification session 58639
both exited zero. All 405 complete normalized decisions and 402 fresh-model
forecasts matched. The complete retained observed-state checks at observations
3, 12, 395 and 404 matched without any state-type or value normalization.
Model states remained unchanged, gradients were absent, and public input arrays
were unchanged. Observation 405 was not consumed.

| Fixed window | Calls | Original median | Candidate median | Original total | Candidate total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Post-warmup, 3–404 | 402 | 0.772860 s | 0.663818 s | 332.620860 s | 287.714975 s |
| Early navigation, 3–12 | 10 | 0.684536 s | 0.605047 s | 7.150183 s | 6.209752 s |
| Repeated hold, 395–404 | 10 | 1.608513 s | 1.218144 s | 15.816992 s | 12.451360 s |

Total post-warmup controller time decreased by 13.50%, with a 14.11% lower
median. Early-window total time decreased by 13.15%. Repeated-hold total time
decreased by 21.28%, with a 24.27% lower median. Every one of the 402
post-warmup controller calls in both implementations exceeded 100 ms.

These measurements cover controller observe only, exclude sensor acquisition,
and alternate original/candidate execution order on the shared workstation.
The original direct-model native collection was concurrent. This is a single
recorded prefix, not an isolated benchmark or independent-maze population.
The synthetic eight-copy workload's much larger reduction did not carry through
to the whole controller. The measured improvement supports retaining this
candidate for further evaluation, but does not resolve the timing gap or
qualify a real-time controller.

The intervention places read-only footprint receipts behind a temporary
selector-only mapper view. Every footprint query still invokes the original
memory and complete geometry calculation. Original selection functions perform
their original deepcopy calls; frozen internal receipts can be shared during
that invocation. Before returning a selection, ordinary mutable public data is
restored. No proxy or frozen receipt becomes controller/map state. The model,
scores, feasibility gates, retained observations and forecast horizons remain
unchanged. Earlier custom-copy, batched-query and recovery-sharing interventions
were not combined with this candidate, and no native implementation was changed.

Independent verification authenticated the exact launch witnessed while the
original owner was live, the completed result and predecessor chain, and both
replay output artifacts. It reconstructed all 405 original decision hashes and
metadata-only candidate hashes, all 405 completed original command endpoints,
and all timing windows. Public-input hashes matched the completed profile and
copy replay. All four reported state hashes matched the completed reference.
The checker did not rerun neural inference, independently reconstruct hidden
state or reload raw sensor packets; the original paired replay owns those
calculations. The original decision stream and command tape were authenticated
before and after reading.

The replay binds 2,000 source files and two output artifacts. The checker binds
2,003 source files, verified before and after its checks. The candidate and
replay passed 40 focused tests, and the checker passed 38 tests. Original replay
wall time after initial admission was 1683.344198 seconds; this includes the
final input verification and is not controller latency. Initial admission
occurred before that timer.

Exact identities:

- Root: `go2_frozen_footprint_anchored_prefix_v1_attempt_001` under the
  recovery-storage navigation artifact directory.
- Assigned JEPA model:
  `35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.
- Launch:
  `6a95ec47e25a08467346a56e449bf99c9a2994ecd082d8d2a8d47c15cce1cacb`.
- Result:
  `4fb9363938101682473115563f4aca2f48a68a8a6d8e9ead2cf3a8ea2ddae772`.
- Comparisons:
  `cc349e16c13c0fadcb355435cea4ee342ba73fb7f684029ebef6ddfd08644d44`.
- Independent verification:
  `go2_frozen_footprint_anchored_prefix_verification_2026-09-10.json`,
  `4b574cf6a46943b7135ec11c472f7c1c73a01af48411178a9a4917e0dd5681c8`.

Original replay PID 2715071 and checker waiter PID 2715834 have ended. Original
direct-model native worker 2709978 remains active under batch parent 2659758.
The frontier, hold-reorientation and contact-score native waiters retain their
existing order. This result adds no navigation episode, arrival, backtracking,
independent-layout outcome or hardware evidence. The verified development count
remains 39 audited episodes and zero verified round trips. The full goal remains
unachieved.
