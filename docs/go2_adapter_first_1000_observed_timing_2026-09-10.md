# Observed timing of the first corrected JEPA development case

The first 1,000 preserved observations show that the current paused-physics
development loop does not meet its 100-ms command interval in wall time.
Across observations 3–999, all 997 post-warmup observation/control operations
exceeded 100 ms. The median was 1.809 seconds, the 95th percentile 2.358 seconds,
and even the fastest was 831 ms. These are workstation observations from that
run, not a real-platform latency qualification or an isolated benchmark.

| Recorded component | Minimum | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Sensor packet acquisition | 158 ms | 241 ms | 258 ms | 282 ms |
| Controller observation/decision | 605 ms | 1,570 ms | 2,115 ms | 2,361 ms |
| Acquisition plus controller | 831 ms | 1,809 ms | 2,358 ms | 2,640 ms |
| Iteration through command execution | 865 ms | 1,843 ms | 2,392 ms | 2,673 ms |

The measured iteration intervals sum to 1,662.37 seconds for 997 nominal 100-ms
request slots (99.7 seconds). This sum excludes work outside those recorded
intervals, such as final persistence/audit and some receipt writing. It is not
the full job duration. Physics pauses during computation, so this run does not
measure the effects of stale observations or delayed commands on continuously
advancing robot dynamics.

The 595 saved hold decisions account for 1,064.08 of the 1,392.16 seconds of
controller time. Their mean is 1,788 ms; the means for individual moving action
groups range from 723–857 ms (the precise forward and arc means are retained in
the JSON report). This association does not establish that the hold action
itself causes the additional cost: holds occur later in the trajectory, and map
growth, observation processing and feasibility recovery may all contribute.
Inference time was not separately recorded in the saved selector receipt, so
these numbers cannot be attributed entirely to the learned model.

Provenance: fixed snapshot `go2_adapter_hold_prefix_v1_attempt_001`, result
`1a9b9627a9496b40aaf5e20aacad00172fbc892c827bda3e4d830df5b4843884`.
Tool session 52330 exited zero after checking all 1,932 source bindings and
three artifact hashes, reading exactly 1,000 consecutive rows, and reproducing
canonical decoded-prefix SHA-256
`c6ff5bd00ccc2320c7487ecd3de6891504bcaee7f6b57461412dec8b11bcbc2a`.
Exclude only rows 0–2; retain the one later no-action selection. Require all
included rows to be nonterminal and every duration finite/nonnegative. Verify
acquisition plus controller equals the recorded combined duration per row.
Summaries use NumPy minimum, median, default linear 95th percentile, maximum,
mean and sum. Exact numeric results are in
`go2_adapter_first_1000_observed_timing_2026-09-10.json`.

The completed episode's raw audit was still running when this diagnosis was
made. No native trace, new raw model inference, candidate outcome, changed
command or new layout was consumed. Existing and queued experiments remain
unchanged. Navigation correctness still needs the scheduled prospective tests.
Before any deployment claim, use a separately controlled component profile to
identify the costs, preserve the original decision evidence when optimizing,
and evaluate sensing/command delays with continuously advancing dynamics and
bounded real-platform evidence. This timing diagnosis does not establish a
speedup, a hardware rate or a model-planning advantage.
