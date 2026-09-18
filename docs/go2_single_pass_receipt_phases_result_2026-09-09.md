# Remaining controller costs after the combined optimization

The phase diagnosis completed with all 514 full decisions exactly equal to the
original learned maze-2 run. All public arrays and assigned model weights remain
unchanged, gradients are absent, and model timing hooks were removed. Exclusive
phase durations partition every controller call exactly. These measurements
identify current costs; instrumentation overhead is included and this is not
a controlled speed comparison or full acquisition/command/I/O timing.

Root: `go2_single_pass_receipt_phases_v1_attempt_001` under the external
development artifact root. Session 10216 exited 0.

- Result SHA: `6ad04101046cdcb9e855a100a446bb0206397354614608f89a03c38d80ecdf76`.
- Launch SHA: `e4132d9e76e7225924dee4208523c1f76adc38eed00668690309ae93590b54d7`.
- 1,701 bound sources; all source and output bindings were rechecked after
  completion. Wall time after admission: 697.932405110 seconds.
- Assigned unchanged model:
  `4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6`.

Of the 514 observations, 500 were active rather than warmup or terminal. Their
instrumented controller median was 574.5553725 ms, mean 607.412527142 ms;
all 500 exceeded 100 ms. Do not compare the instrumented median directly with
the uninstrumented paired benchmark to infer an additional speed change.

| Phase | Mean exclusive time per active observation |
| --- | ---: |
| Selector work outside its timed children | 118.846 ms |
| Memory contact queries, six calls per observation | 111.354 ms |
| Original auxiliary integration | 81.396 ms |
| Floor registration | 73.500 ms |
| Primary map coverage outside insertion | 58.092 ms |
| Visual motion | 52.093 ms |
| Primary memory insertion | 39.218 ms |
| Primary classification | 31.794 ms |
| Auxiliary confirmation outside its original integration | 18.989 ms |
| Model forward | 7.393 ms |
| Remaining controller advance | 6.159 ms |
| Map waypoint query | 4.755 ms |
| Remaining controller observe | 2.949 ms |
| Remaining map observe | 0.699 ms |
| Controller result | 0.176 ms |

The table uses mutually exclusive durations; rounded entries need not sum
exactly. Inclusive selector time was 242.348 ms and inclusive map time was
230.188 ms. Those overlap the child rows and must not be added to them.

The existing receipt-copy and single-pass index implementations were both
present. The older floor cache is already inherited; this result does not call
for reimplementing it. Next optimization should inspect the remaining selector,
contact and repeated floor-processing operations, preserving complete decisions
on the actual trajectory. Faster model inference alone cannot account for the
current gap. Floor registration plus visual motion already average about 126 ms
before the remaining controller work, so reaching 100 ms requires addressing
multiple measured costs and then timing actual full-loop execution.

This repeated development-episode diagnosis changes no navigation outcome.
No real-time, navigation, generalization or hardware qualification is established.
