# Completed scoped support-cache paired benchmark

Session 98475 exited 0. Result
93a7962cef275f170d51bcfd0387b6c195ceaf8e36f30376fd849179bf2082d9
in go2_scoped_support_cache_benchmark_v1_attempt_001. All 1,709 source bindings
and three output bindings were independently rechecked after completion
(38551 exit 0). The runner completed its original ancestry verifiers as well.

All 514 original and candidate complete decisions are exact; public arrays and
both model states unchanged. Assigned model
4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.
There are 500 active observations, 3 warmup and 11 terminal observations.

| Active 500 | Single-pass receipt-copy baseline | With scoped support cache |
| --- | ---: | ---: |
| Median controller ms | 577.181104 | 565.780799 |
| Mean controller ms | 616.266122148 | 599.044402970 |
| Observations over 100 ms | 500 | 500 |

Median paired reduction: 18.1989945 ms. Alternating order groups each contain
250 observations; median paired reductions are 20.1265175 ms when baseline
runs first and 14.9270805 ms when cached controller runs first. The difference
of population medians is not the median paired reduction. Do not add this
result to savings from separately timed prior experiments.

The bounded cache saves repeated identical support computations within one
footprint and retains every geometry/receipt result. This measured benefit is
modest relative to the remaining timing gap. No acquisition or receipt I/O is
included; concurrent CPU jobs were possible; no profiling instrumentation was
used. This is not whole-loop 100 ms operation, a native navigation result or a
deployment qualification. No native controller has adopted this optimization.

Bindings:

- launch.json: 2f1b337ed04947b649b7c6b4f06fc1579083c230cccf8bac63d0b7a72e8bb5f7
- paired_timings.jsonl: 0bd45a70cd8200fd0baa10d53703e845eb4162bb5e1351f66c87db352b8ab931
- resource_monitor.jsonl: 66d34788b9dca03fbc1260337d7cfb05f012a04338c3e328c1f4744f7da9fd90

Wall time after admission including replay and final verification:
1,575.655748239 s. No new native episode or round trip; aggregate remains
22 completed/raw-audited episodes and zero verified round trips.
