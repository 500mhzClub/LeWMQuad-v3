# Frozen footprint receipts replay: execution started

Latest status: the original replay and scheduled independent verification have
completed successfully; sessions 59832 and 58639 both exited zero. All 405
normalized decisions, 402 forecasts and four retained-state checks matched.
Post-warmup total controller time decreased by 13.50%; median decreased from
772.86 ms to 663.82 ms. All 402 post-warmup calls still exceeded 100 ms. See
`go2_frozen_footprint_anchored_prefix_result_2026-09-10.md` for the complete
verified result and limitations. Earlier progress observations below are
historical; no replay or verification remains pending.

Latest provisional observation at 2026-09-10 16:33:55 UTC: all 405 comparison
rows, indexed 0 through 404, have been written in order. Every row reports exact
original and normalized candidate decisions and unchanged public input arrays.
No terminal failure or final result is present. The original replay remains
live in its final input audit, and the independent checker is waiting for that
exact original process to complete. Timing and retained-state conclusions
remain pending the completed result and independent verification. Native
worker 2709978 remains active. Earlier observations below are historical.

The original replay has now completed initial input admission and begun paired
controller execution. Its actual launch SHA-256 is
`6a95ec47e25a08467346a56e449bf99c9a2994ecd082d8d2a8d47c15cce1cacb`.
All 2,000 launch source bindings match the pre-execution record and were
rechecked. The independent waiter observed and bound this launch while the exact
original owner was live; its launch-identity record SHA-256 is
`eec31e53101fa34b921136e914069ed4ba89d5fba87ccc3a487d531fbe5928ab`.

A provisional inspection found complete comparison rows 0 through 24, all
reporting exact original and normalized candidate decisions, with no terminal
failure. This includes forecast-bearing observations but is not the completed
405-observation result, the final retained-state check, independent result
verification, or a timing conclusion. The original process and native queue
remain active. The initial execution details below are retained as history.

The candidate passed 40 focused receipt, controller-boundary and replay tests
in 2.34 seconds (session 7723, exit zero). Its source-only preflight passed with
2,000 bindings (session 70905, exit zero). The existing contact-native waiter's
1,988 source bindings were independently rechecked and remain unchanged.

One CPU paired replay was started under original PID 2715071, creation epoch
1789056341.94, tool session 59832. The command is:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/replay_go2_frozen_footprint_anchored_prefix_v1.py
```

Execution record: `go2_frozen_footprint_anchored_execution_2026-09-10.json`,
SHA-256 `a0fb7db9722f2f88cac208e25134d44cd7ed35a0e46d1adfd03bb0d33227c36f`.
It binds the exact process, all 2,000 prepared sources, resource inspection,
test results and the completed predecessor. Preserve these sources while the
original run is live. Do not start another copy if observation of the tool
session times out or if its output root has not appeared during input admission.

The process has reported `FROZEN_FOOTPRINT_ANCHORED_INPUT_ADMISSION_STARTED`.
Its exclusive root is `go2_frozen_footprint_anchored_prefix_v1_attempt_001`
under recovery-storage navigation artifacts. No completed replay result or
controller timing improvement is claimed at this stage. The synthetic copying
workload in the component protocol is not a whole-controller benchmark.

Prelaunch inspection found 70.59 GiB available RAM, 597.47 GiB free artifact
storage and 16 physical CPUs. Original direct-model native worker PID 2709978,
creation epoch 1789053472.31, remained active under original batch parent
2659758. The new process constructs no native scene; the six-case batch,
frontier, hold-reorientation and contact-score native ordering is unchanged.

On completion, inspect the original result or retained failure, authenticate
all source and artifact bindings, and independently reconstruct the recorded
comparisons and timing summaries. Require exact whole-controller decisions,
402 forecasts and four retained-state checks before any performance conclusion.
The replay adds no navigation episode; the current verified count remains 39
audited development episodes and zero verified round trips.
