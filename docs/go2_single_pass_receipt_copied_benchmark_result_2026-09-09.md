# Additional index optimization benefit verified on complete current maze2 replay

75686 exited0. Result SHA-256
d688f2ed9d30177d2e55fb98e9c9f25d2b035f2258d86449ac8e86615cd13c72,
launch5ae02642ad918e2b271b48e92e0f0adb9c392c7b2598412d03b36656e2b81e18.
Root:go2_single_pass_receipt_copied_benchmark_v1_attempt_001. All1,696
source/input/predecessor bindings verified before and after execution. Result
and output artifact hashes were checked again when reading these findings.
Wall time1,059.9303622s after admission, including final verification.

All514 complete decisions from receipt-copy alone and the combined index/
receipt-copy candidate exactly equal the saved original maze2 decisions. Public
inputs and both separately assigned corrected JEPA model states remain unchanged;
gradients are absent. No decision metadata normalization or model/policy change.

For500active observations, excluding3warmup and11terminal observations:

| Measurement | Receipt-copy alone | Plus packed-owned/single-pass indices |
| --- | ---: | ---: |
| Median controller time | 661.482598ms | 567.2465515ms |
| Mean controller time | 698.2383198ms | 599.8590145ms |
| Observations exceeding100ms | 500 | 500 |

Median paired reduction97.1747875ms. The250-observation subgroup with receipt-
copy first had98.333615ms median paired reduction; the opposite-order subgroup
had97.0875525ms. Both execution-order groups show an additional benefit from
the index substitution on this episode.

This measures additional index benefit over the receipt-copy controller. Do not
add reductions from different benchmark runs or treat their absolute times as
controlled comparisons across runs. Acquisition, reconstruction, receipt I/O,
comparison and physics remain outside the timer. No profiler hooks were used;
other native/audit work shared the machine. Every active optimized cycle still
misses100ms. No full-loop real-time result, new navigation outcome or native
controller adoption follows from this benchmark.

Output bindings:

- paired_timings.jsonl:
  bea9688f43e7b77a7e6ac116056a82f3d8fe09fdc3fb186825f4605f32478411
- resource_monitor.jsonl:
  86d9ae7a4549d79737d52ceb2dc978af295ff39834892282358d24dcf8b4660d

The failed initial preparation24468 and its original verifier-schema correction
remain recorded in docs/go2_single_pass_receipt_copied_preparation_2026-09-09.md.
That submission ended before creating output or loading models. Do not rerun
the completed benchmark or modify its frozen sources.
