# Batched floor-query controller execution

Completed update, 2026-09-10 14:04 UTC: the original replay has completed,
including final input verification, and its process has ended. Result SHA-256
is `650fe137b41c91f8860ce59987c876d1c2591329ee2fc63ad056e63059e47419`.
An independent recorded-decision, hash and timing check is complete. The
405-observation comparison preserves all normalized decisions; total
post-warmup controller time is 4.95% lower and repeated-hold median time is
17.3% lower. All 402 post-warmup calls still exceed 100 ms. See
[the completed result](go2_batched_patch_anchored_prefix_result_2026-09-10.md)
and its bound verification JSON for scope and identities. The entries below
are retained historical observations, not current process status.

Latest check, 2026-09-10 13:33:36 UTC: all 405 comparison rows are present.
The same replay process remains live in final input verification; no result or
failure exists. Recomputing the fixed timing windows from these rows gives
344.83435924886726 s original versus 327.7592413465027 s candidate total for
402 post-warmup calls (about 4.95% less time). Their medians are 0.8004403539234772
and 0.7566889928421006 s. The repeated-hold medians are 1.6443458255380392 and
1.3597600674256682 s (about 17.3% less time); early-window total time is slightly
higher for the candidate. Every post-warmup call in both controllers exceeds
100 ms. These timings are provisional until the original replay's complete
final admission/result is available; they do not establish real-time operation.

Earlier check, 2026-09-10 13:22:17 UTC: input admission has completed and the
paired replay has written 232 comparisons, with no terminal result or failure.
Launch SHA-256 is
`38a474e183ca8d0972d2e10b1f3afb18efb191aca0e6fd0113e3928964477c30`;
the same session 27563 / PID 2695783 owns this execution. All 405 comparisons,
four retained-state checkpoints and final input verification are required
before a completed equivalence or timing claim. Brief source tests and a
four-observation saved-score check ran concurrently; this is not an isolated
timing benchmark.

Earlier check, 2026-09-10 13:08:19 UTC: replay PID 2695783 remained live,
with 462.84 CPU seconds and 614,245,543,138 logical bytes read. The output
root is still absent and input admission remains in progress. Do not restart
the process. The supervised worker has finished collection and remains live
in its pending audit; its collection exhausted the mission budget with no
arrival. See `go2_all_phase_adapter_full_supervised_maze02_provisional_result_2026-09-10.md`.
The batch parent and both original native waiters remain live. Neither queued
native pilot has started. The audited episode count remains 38, with zero
verified round trips.

At 2026-09-10 12:59 UTC the separately named paired replay process is live in
full original input admission. No replay output root or timing result has yet
been reported. Missing output during admission is not a failed process.

Implemented only fresh primary/auxiliary patch-store replacement in the
original measured floor memory. The original mapper, memory class, selector,
observation and mission implementations remain unchanged. The paired runner
uses the completed receipt-copy replay's exact code object in a private
namespace, with only its candidate constructor, strict normalizers, output and
progress provider replaced. Receipt-copy optimization is not enabled.

Validation before execution:

- Session 67834: 66 component, integration and replay tests passed in 2.89 s.
  Coverage includes both camera stores, inherited failure decisions, complete
  retained-state comparison, rejection of partial/changed implementations,
  detection of changed witnesses/poses/map/residual/history, and exact original
  replay code with unchanged undeclared global bindings.
- Session 46645: source-only preflight passed 1,982 source bindings.
- Session 12766: immediately preceding resource assessment passed: 16 physical
  CPUs, 32 CPUs in affinity, 76,684,730,368 available RAM bytes and
  644,959,485,952 free artifact bytes. The native supervised worker and the
  existing frontier/hold waiters were present; no other diagnostic replay was
  live. The new replay uses one numerical thread and creates no native scene.

Frozen new source identities:

| File | SHA-256 |
| --- | --- |
| lewm/batched_patch_anchored_controller_development.py | 11df7d2d66ae917095d1f06d5db221d2f423dfcfee030ff405fe00e7b6c1223e |
| lewm/tests/test_batched_patch_anchored_controller_development.py | b22d022f9cced4a468da52c476816b29188482131baeeb64568e6b1c061050dc |
| scripts/replay_go2_batched_patch_anchored_prefix_v1.py | 4a9755aeb5c2ffd74f605f6f5aa1f0c8d88122f8a3e48383221e61043efa51c3 |
| lewm/tests/test_batched_patch_anchored_replay_development.py | 3b639413d555071f20e8433acb2c74fb6c1a571e4680c79774c03eba914b0ece |
| docs/go2_batched_patch_anchored_prefix_v1_2026-09-10.md | e36da8131e90c42d8b81121dfdceee2d600539c1084dfe46781781f454c2047d |

Process: tool session **27563**, PID **2695783**, creation epoch
**1789045142.75**. Command:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/replay_go2_batched_patch_anchored_prefix_v1.py
```

At 12:59:20 UTC its CPU usage had reached 13.36 seconds, with
20,431,860,340 logical bytes read and 4,391,780,352 storage bytes read. The
session emitted `BATCHED_PATCH_ANCHORED_INPUT_ADMISSION_STARTED`. This is live
input verification, not a completed timing or equivalence result. The output
is reserved as `go2_batched_patch_anchored_prefix_v1_attempt_001` under the
existing recovery-storage navigation artifact root. Do not launch a duplicate
or edit the bound sources while this process remains active.

The native comparison remains owned by parent PID 2659758. At 12:59:20 UTC
supervised worker PID 2672443 was live with 2,966 observations and no worker
terminal result. Frontier waiter PID 2663938 and hold waiter PID 2671835 retain
their original ordering after the six-case batch. The verified completed
episode count remains 38 with zero verified round trips until a new audited
worker terminal is available. This execution note is not a source input to
the batched replay.
