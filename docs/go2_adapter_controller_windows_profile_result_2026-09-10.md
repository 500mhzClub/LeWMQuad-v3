# Completed controller timing profile

The unchanged expanded-model JEPA controller completed its fixed 405-observation
replay, reproducing every original decision and 402 model forecasts. Both full
original input checks passed, and the process exited successfully. Result:
`8be3553ba54c67827790a281aaf3a08bd2facbc8aff39537dd7f353a2b3b3fb0` in
`go2_adapter_controller_windows_profile_v1_attempt_001` under the recovery
storage navigation artifact root.

Ten observations were profiled in each of two fixed windows:

| Window | Observations | Exclusive profiled time | Time inside deepcopy, cumulative |
| --- | --- | ---: | ---: |
| Early navigation | 3–12 | 16.911 s | 9.194 s |
| Repeated hold | 395–404 | 41.503 s | 28.545 s |

Cumulative times overlap other function times; do not add them together.
These measurements include profiler overhead and concurrent workstation work.
They profile the controller, not live sensor acquisition, and do not establish
unprofiled latency, a speedup, or real-time qualification.

Recursive copying is a major measured cost. During the hold window, direct
`deepcopy` calls from `plan` accounted for 7.017 cumulative seconds, and calls
from `constrain` for 6.822 seconds. These caller counts include smaller copies
as well as whole selections. In three inspected original selections (3, 395,
404), detailed surface-check evidence accounts for more than 98% of the counted
tree nodes; each complete selection occupies about 2.8–3.0 MB as compact JSON.

The next performance experiment should reduce the cost of copying this
evidence while preserving all checks, predictions, decisions and complete
receipts. Candidate rows are mutated by some stages, so blanket shallow copying
has not been shown safe. Establish ownership and isolation, replay the complete
controller against the fixed original data, then measure without profiling.
No optimized controller has been implemented or executed by this profile.

Independent verification checked all 1,963 frozen source bindings and six
outputs, reconstructed all 405 public packet fingerprints and original saved
decision hashes, checked original command timing endpoints, reconstructed both
JSON summaries from the binary profiles, and matched the final report windows.
This independent check did not rerun neural inference; the original completed
profiler performed the controller replay. The machine-readable verification is
`docs/go2_adapter_controller_windows_profile_verification_2026-09-10.json`.

This is completed performance diagnosis, not another navigation episode.
It adds no goal arrival, round trip, independent-layout result, or hardware
evidence. The supervised navigation comparison and hold-reorientation raw
replay continue separately; their registered native successors retain their
original queue order.
