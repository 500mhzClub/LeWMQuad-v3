# Complete paired chained-controller replay adapter prepared

The native chained-controller attempt is still collecting. Its original worker
PID 2994743, creation time 1789194027.81, was observed running through flushed
decision tick 1133 during this preparation, with no terminal result or failure.
Its launch SHA-256 remains
`0f2963636abc4f1fc738e9323d2a74db02888ca7062dc1afa21023f8c443b8ff`.

`scripts/measured_plane_chained_full_history_replay_development.py` creates
private copies of the existing complete paired replay and saved-output checker.
The function bodies are unchanged. Explicit globals select the chained native
case and input root, its assigned model, the exact chained baseline and
single-pass controller classes, the new chained comparison helper, and the
caller's output directory. No original module globals are changed.

The retained loop reconstructs public packets, checks physical command
endpoints, alternates controller execution order, compares complete original
and normalized decisions, counts actual model forwards, checks model state and
gradients, compares the existing complete retained-state fingerprints at fixed
and final checkpoints, monitors resources, and bounds comparison output. The
retained checker reconstructs every packet and original decision, requires the
complete saved population and recomputes the report. Timing remains controller
observation time and carries no isolated-benchmark or real-time claim.

Fourteen focused tests passed in 2.35 seconds. They check private function-body
reuse and actual chained bindings without input reads or output creation;
complete five-observation synthetic replay including a terminal observation;
decision, public-input, model-call, retained-state and model-weight faults;
forward-hook cleanup; and missing, extra or altered saved rows, packets, clocks,
original-decision bindings, reports and final-state evidence. Synthetic
controllers isolate orchestration here. Actual image-to-action chained
equivalence is covered by the separately recorded composition tests; neither
test set is a completed native-history replay.

| File | SHA-256 |
| --- | --- |
| `scripts/measured_plane_chained_full_history_replay_development.py` | `edc77290c30c1cb4eb79fe2db52272949ac3f681cfd71e1e149ffa536835e5fc` |
| `lewm/tests/test_measured_plane_chained_full_history_replay_development.py` | `b7f24fcbf48c6cf51fdcc2b6cd8aed48a2dbf875962025264328b95835e17ae9` |

All 2,627 sources bound to the active native launch were independently
rehashed and remain unchanged. The new adapter and test are outside that live
binding. No real replay, profiler, launcher or queue was started. Completed
native input admission, source/protocol binding, exclusive output creation,
resource scheduling and final artifact authentication still belong in the
future runner. The adapter alone does not perform those operations. No new
navigation, speed, real-time, hardware or deployment result is established.
