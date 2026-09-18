# Auxiliary packet construction: a modest latency opportunity

On four previously recorded maze02 frames (0, 3062, 4003, 4013), all three
constructions produced bitwise identical complete auxiliary RGB/depth packets.
Twelve warmed, rotating-order repetitions per frame gave:

| Construction | Median ms | p95 ms |
| --- | ---: | ---: |
| Existing two depth-archive reads | 23.542 | 26.065 |
| Prepared single-read composition | 16.003 | 17.690 |
| Already acquired RGB/depth arrays | 8.094 | 8.546 |

The measured median differences are about 7.5 ms and 15.4 ms. These are small
beside the previously recorded approximately 787 ms median acquisition/control
loop. Avoiding these reads alone cannot make that loop meet 100 ms. The next
latency investigation should focus on controller computation and capture,
before investing in a larger persistence rewrite.

This is a warm-cache component experiment on a shared host while the current
trial's raw audit was running. It excludes rendering, archive writes, primary
packet construction, controller execution and physical advancement. Individual
outliers reached 57.5 ms for duplicate reads and 168.2 ms for single reads;
the small sample does not establish tail-latency guarantees. Equality was
checked outside the timers. The array variant retained pixel identity and
public packet validation; it did not read evaluator segmentation or pose.

The production acquisition path and both queued experiments are unchanged.
No whole-loop speedup, physical sensor latency or continuous-execution result
is claimed. The original completed development input result is identified in
the JSON, and all timings, selected pixel identities and scope flags are in
`go2_auxiliary_packet_reconstruction_timing_2026-09-13.json`.

Reproduction source:
`scripts/measure_auxiliary_packet_reconstruction_development.py`.
Its first execution completed successfully in session 68662. No model was
loaded and no scene was created.
