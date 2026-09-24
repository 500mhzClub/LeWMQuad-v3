# Paired receipt-copy replay registered

## Completed and independently verified

The original session 90540 exited successfully with result
`c3363fe9626fc36b8feb723346bef0344b7becccd3d9f68297d5f99ac3f7b0c0`.
Independent session 91509 verified all source/output bindings, the final
model/state-check report and every timing window against the previously
verified comparison stream. The final result and verification are recorded in
`docs/go2_receipt_copied_anchored_prefix_result_2026-09-10.md` and
`docs/go2_receipt_copied_anchored_prefix_verification_2026-09-10.json`.
The modest improvement is now a completed measurement; it does not meet the
100-ms requirement. Pending-state sections below are historical snapshots.

## Full comparison stream written; final verification pending

The original process has written all 405 paired comparison rows. Independent
session 36312 verified all 1,972 sources, the original worker terminal and
bound original stream/tape/acquisition metadata, then reconstructed all 405
original and candidate decision hashes and public packet fingerprints.
Candidate hashes were reconstructed from each original complete decision with
only the two declared implementation metadata changes. All 402 forecast-bearing
decisions and original command endpoints matched. The comparison stream stayed
unchanged throughout, SHA-256
`ab76d3075fc030991a02b5e9e4f624d003c04155a3af12030d3dbaa47d90fb4d`.
This independent check did not rerun neural inference or reconstruct hidden
observed state. The final runner result was still absent.

The complete saved timings give the following provisional controller-only
measurements; they include no profiler instrumentation:

| Window | Observations | Original median s | Candidate median s | Original total s | Candidate total s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early navigation, 3–12 | 10 | 0.674186 | 0.672521 | 6.778174 | 7.020461 |
| Post-warmup prefix, 3–404 | 402 | 0.793502 | 0.778877 | 338.975874 | 331.976292 |
| Repeated hold, 395–404 | 10 | 1.609512 | 1.501643 | 15.765399 | 14.633581 |

The hold median is about 6.7% lower; full-prefix total controller time is about
2.1% lower. Early-window total time is slightly higher. Every one of the 402
post-warmup calls exceeded 100 ms for both controllers. This is a modest
controller improvement on one alternating-order replay, not a solution to the
100-ms sensing/control requirement. Sensor acquisition is excluded and other
workstation processes were active. Do not infer an isolated benchmark,
statistical significance, full-loop speedup or native-navigation benefit.
Final original input verification must finish before this becomes a completed
timing result. No current native policy was changed or newly qualified.

## Input admission completed; paired replay started

The original process PID 2688118 has completed full original input admission
and created launch
`9ecb3f03a74ea5b2b8ca2f533f9bf2a744614e33ff06257e48c9702df923bc1d`
in `go2_receipt_copied_anchored_prefix_v1_attempt_001`. An independent read
verified this exact launch and all 1,972 source bindings. It retains the fixed
405 frames and observed-state checkpoints 3, 12, 395 and 404. At the first
post-launch observation of the process, eight complete comparison rows were
written. This is a started correctness/timing replay, not a completed speedup.
The registration and admission snapshots below are historical.

The completed controller profile identified repeated copying as a major cost.
An existing exact-container copy helper already had alias, cycle, custom-copy
and NumPy tests and an earlier component benchmark. The new candidate reuses
that helper only within the anchored hold-reconsideration calculation, through
six private function namespaces. Original function code objects, imported
module globals and upstream selection remain unchanged.

Validation completed:

- Session 30049: 26 tests passed in 2.35 seconds, comprising the existing copy
  semantics tests and the new isolated-selection tests. Checks include shared
  mutable objects, cycles, source isolation, original nominal-policy gates,
  exact exceptions, code identity and inherited observation/failure behavior.
- Session 54058: twelve paired-runner tests passed in 2.19 seconds, covering
  metadata normalization, fixed execution order, timing populations and
  rejection of malformed timings. Total focused tests: 38.
- Session 81149: source-only preflight passed 1,972 bindings and the resource
  envelope; no runtime root was created.
- Session 73316: independently reverified all 1,972 current bindings after
  starting the process and confirmed its exact command and creation time.

The separately executed existing semantics-test file
`lewm/tests/test_receipt_copy_development.py` has SHA-256
`3c0c17308b6f9a07c1cd897c677254955ece9d6633ed71457c5bdbc59367abfa`.
It is recorded here as test-run provenance, rather than represented as one of
the nine new runtime source bindings below.

The running paired replay is PID **2688118**, created **1789041120.63**, tool
session **90540**. Command:
`.generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/replay_go2_receipt_copied_anchored_prefix_v1.py`.
Use deterministic hash seed, PYTHONDONTWRITEBYTECODE, PYTHONPATH
`.:lewm_genesis:lewm_worlds`, and one OMP/MKL/OpenBLAS thread. The process has
started full original input admission. Its intended exclusive root,
`go2_receipt_copied_anchored_prefix_v1_attempt_001`, does not exist yet. Quiet
admission is not a terminal failure; do not restart this original process.

The initial source set includes nine paths additional to the completed profile:

| Path | SHA-256 |
| --- | --- |
| docs/go2_adapter_controller_windows_profile_result_2026-09-10.md | 1611a5ed7901f271c3348b500d7323b55b015ce723ad1d26de1425804828cfa5 |
| docs/go2_adapter_controller_windows_profile_verification_2026-09-10.json | f512a04ffc0ef7f8e1461176c4f497dd9ceb7d1567d79101004c5ae8dddffc23 |
| docs/go2_receipt_copied_anchored_v1_2026-09-10.md | 2e768e824c1b89dc2d6bd093b82ec6e6bccb5a37e556522803e7b22b30126b8e |
| lewm/receipt_copied_anchored_controller_development.py | c75b27fab391754f820b399e66d18d192ebe582ec5bdb98085d697a63d520284 |
| lewm/receipt_copied_anchored_selection_development.py | 1dd186b3626ad1d4dde25093cd1393832adf67a6e9bf0da05e2a43dd944ba48a |
| lewm/receipt_copy_development.py | 6237bc01257e2eacf3b2ef2effa6c3aa24af6eb902eb5ce6f40a757db232c6ea |
| lewm/tests/test_receipt_copied_anchored_development.py | 55cf6383bb93977a1f621bc60dcec20b374d213a395b0b7df341c971121b33ae |
| lewm/tests/test_receipt_copied_anchored_replay_development.py | d26ab000214ec29a05ee14063873f45c412951cdfe41f114070a70edb74ec88b |
| scripts/replay_go2_receipt_copied_anchored_prefix_v1.py | 0fdc4ac3493ce8c4744cab4a7270c3ff8b41163fe9575988a2111fbbe5bb1a7d |

Do not edit those sources, predecessor bindings or the completed-profile result
documents while this attempt runs. This execution note is outside that source
set. The fixed trial uses two fresh controllers and models, alternates execution
order across 405 raw observations, compares all normalized decisions/public
inputs, checks complete retained observed state at four fixed observations,
and measures controller calls without profiling. It performs the original full
input admission again before declaring completion. There is no measured
controller speedup yet.

At launch the host had approximately 70 GiB available RAM and 603 GiB free
artifact space, exceeding the 48 GiB/four-physical-CPU/41 GiB requirements.
The existing supervised native worker and hold-reorientation replay remain
live; no additional native scene is launched. The frontier and hold-native
waiters retain their original order. This optimization adds no navigation
episode, goal arrival, independent-layout outcome or hardware qualification.

At 11:53:54 UTC the original hold-reorientation replay log reported
`HOLD_REORIENTATION_RAW_FRAME 405 changed True`, following unchanged checkpoints
through 400. That runner compares every intervening frame, so it reached the
fixed intended boundary; its final full input verification remains pending.
This is not completed native execution of the changed command.

## Verification resource sample

Read-only session 98767 sampled the same process identities from 12:04:48 to
12:05:28 UTC (40.00035 seconds). Linux process I/O counters distinguish logical
read bytes from storage read bytes; they are not controller latency samples.

| Process | CPU cores used over interval | Storage reads MiB/s | Logical reads MiB/s |
| --- | ---: | ---: | ---: |
| Supervised native worker 2672443 | 0.995 | 0.003 | 0.708 |
| Hold replay final verification 2672447 | 0.622 | 92.692 | 833.257 |
| Copy replay initial verification 2688118 | 0.623 | 108.534 | 833.279 |

The artifact filesystem is ext4 on `/dev/mapper/data-root`, corresponding to
`dm-1`. Its counters showed 200.866 MiB/s reads, 0.631 MiB/s writes, a reported
I/O-busy fraction of 0.424 and weighted average queue of 0.855 over this sample.
These logical-device counters do not prove physical-device saturation. The
verifiers alternated between running and uninterruptible I/O wait; the native
worker continued using approximately one CPU core. This documents substantial
verification I/O and does not establish that serial admission would be faster.
No scheduling priority, concurrency, source or active process was changed.

At 12:06:07 UTC both the hold final verifier and copy initial verifier had
reported successful completion of their first two original verification stages.
Their third stages remained active, with no final hold result, copy launch
root or failure yet observed.

## Earlier floor cache is already integrated

Read-only session 78625 verified the current source bindings and instantiated
both original and receipt-copy controller constructors without executing any
sensor observations. Both use `MeasuredFloorTransportMap` and
`MeasuredFloorTransportMemory`, inheriting `FrameCachedFloorMap` and
`FrameCachedFloorMemory`. Their map observation method is
`LaterResolvedFloorMap.observe`, which creates and closes the per-observation
floor-index cache. Both primary and auxiliary retained patches still use
`RetainedFloorPatches.coverage`.

The completed current profile also records 90 `FrameFloorIndexCache.index`
calls and ten cache closures in each ten-observation window. Thus this is not
an unused predecessor optimization to add again. Retained-patch coverage
queries are a distinct operation from computing the current observation's
floor index. The old cache completion identity
`633a16730506480011d9a00a4a76c63daa9c62c19eaf39695a3376abfc97e14b`
and its 1,508 source bindings were reverified for chronology, without rerunning
that older experiment or treating its unpaired timing as a current speedup.
No current cache implementation or frozen source was changed.
