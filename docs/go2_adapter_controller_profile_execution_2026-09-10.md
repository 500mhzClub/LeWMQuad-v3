# Controller timing profile: execution registered

## Completed

The original profiler session 27421 exited successfully with result
`8be3553ba54c67827790a281aaf3a08bd2facbc8aff39537dd7f353a2b3b3fb0`.
Independent session 92968 verified all 1,963 sources, six outputs, and both
completed report windows against reconstructed binary profile summaries.
The result and verification are recorded in
`docs/go2_adapter_controller_windows_profile_result_2026-09-10.md` and
`docs/go2_adapter_controller_windows_profile_verification_2026-09-10.json`.
The pending-state sections below are historical execution snapshots.

## Replay outputs written; final verification still pending

Independent session 46328 subsequently reconstructed the public packets for
all 405 observations, including auxiliary RGB/depth, and matched every saved
input fingerprint. It also matched all original decision hashes and completed
command endpoints, verified the original worker terminal and its bound
decision stream, acquisition metadata and command tape, and rechecked all
1,963 sources before and after. The unchanged comparison stream SHA-256 was
`1bb486ca28a2b6c1ee7bd3b7a76cd6640e55a71f357447424e4de0a6dc2a4995`.
This was a packet/receipt reconstruction, not a neural inference rerun.
The final profiler result was still absent when this check completed.

Further read-only attribution found that the repeated-hold profile has 100
direct calls to `deepcopy` from `plan` (7.017 seconds cumulative) and 50 from
`constrain` (6.822 seconds cumulative). These counts include both whole
selection copies and any smaller copies issued directly by the same caller;
they are not counts of controller observations. Exact source inspection found
that `constrain` replaces top-level fields, whereas `plan` also changes candidate
rows and explicitly preserves first-step candidate evidence. Therefore simply
substituting shallow copies across this chain has not been established safe.

Session 93157 inspected only original saved decisions 3, 395 and 404, within
the already fixed profiler prefix. Node counts include container keys and
values, with the top-level selection keys omitted consistently. JSON byte
counts use compact separators without a trailing newline:

| Observation | Action | Selection nodes | Surface-check nodes | Selection JSON bytes |
| --- | --- | ---: | ---: | ---: |
| 3 | left_arc | 201,143 | 198,451 | 2,783,742 |
| 395 | hold | 206,664 | 203,807 | 2,918,803 |
| 404 | hold | 210,740 | 207,883 | 2,993,784 |

The surface checks account for more than 98% of the counted nodes in all three
receipts. Repeated copying of detailed evidence is consequently a specific
performance investigation. These saved JSON trees do not establish live
object aliasing or mutation ownership. Any optimization must preserve the
full evidence and selection checks, establish input/output isolation where
required, reproduce complete controller decisions, and measure execution
without profiler overhead. No source change or speedup is claimed here.

The same live profiler process has written both window JSON reports and the
405-row comparison stream. No final result or failure existed when checked.
Independent session 9872 verified the original worker terminal hash and its
bound original compressed decision stream, then matched every saved profile
decision hash to the original decision for observations 0–404. Both windows
contain exactly ten observations; the other 385 rows are unprofiled. All
reported controller times were finite and nonnegative. Public input hashes
were not independently recomputed in this check, and this was not another
neural/controller replay.

The provisional early window contains 16.911158428 seconds of exclusive
profiled time; the hold window contains 41.502755868 seconds. The function and
module exclusive-time sums agree. The `deepcopy` cumulative times are 9.194
and 28.545 seconds respectively, with 13,029,201 and 38,905,128 calls including
recursive calls. These cumulative times overlap other function times and
must not be added to them. Profiler overhead remains included.

The provisional hold-window report identifies
`reconsider_anchored_continuation` (24.440 seconds cumulative) and
`filter_selection` (13.895 seconds cumulative) among the costly callers.
Exact source inspection confirms whole-selection deep copies in the surface
filter and corrected continuation path. This identifies an optimization
investigation; it does not establish a measured speedup, safe copy removal,
or a completed profile. Wait for final input verification before promoting
these outputs to a completed timing result. No policy source was edited.

## Input admission completed; raw controller replay started

The existing process PID 2673728 has completed all three original input
verification stages and started replaying the unchanged controller. Launch
`b2b9b4f1fda09de3493289b6c60332c7f69cb9a0e4f181723fb4dba428843a51`
now exists in `go2_adapter_controller_windows_profile_v1_attempt_001`.
An independent read verified this exact launch hash and all 1,963 source
bindings. The launch retains model
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`,
405 observations, and the fixed windows 3–12 and 395–404. The live session
reported frame 0; this is a started profile, not a completed timing result.

At the same check, the original supervised worker PID 2672443 had written
1,021 observation timing rows and the original hold-prefix child PID 2672447
had logged successful comparisons through frame 150. All three PIDs retained
their original process start ticks and increasing CPU time. Available memory
was approximately 72 GiB. Existing frontier and hold-native waiters keep their
original execution order. No policy source or running process was changed.

This execution note was confirmed outside all six active launch source maps
before updating it. The registration snapshot below records the earlier state.

The previous turn established median observation-plus-control latency of
1.809 seconds against the 100-ms command interval. This turn implemented a
separate fixed-window profiler to identify controller costs while preserving
every original raw decision. It does not alter the running or queued policies.

Profile process: PID **2673728**, created **1789037950.04**, tool session
**27421**. Command:
`.generated/venvs/genesis_rocm_0_4_6_v1/bin/python scripts/profile_go2_adapter_controller_windows_v1.py`.
Environment: deterministic Python hash seed, PYTHONDONTWRITEBYTECODE, PYTHONPATH
`.:lewm_genesis:lewm_worlds`, one OMP/MKL/OpenBLAS thread. The process is live
performing full original input admission. Its exclusive output
`go2_adapter_controller_windows_profile_v1_attempt_001` does not yet exist;
there is no completed profile or measured speedup to report.

The fixed windows are observations 3–12 and 395–404 of the completed original
JEPA case. Reconstruct all observations 0–404 with the unchanged controller and
model, profiling only controller.observe in those twenty frames. Require exact
complete original decisions, public input preservation and unchanged model
state. Save function calls, exclusive and cumulative times; cumulative times
overlap and must not be summed. The profiler neither consumes the intervention
at 405 nor profiles live camera acquisition. It does not remove profiling
overhead or isolate the workload from other workstation processes.

Admission requires 48 GiB available RAM, at least four physical CPUs and the
40+1 GiB artifact envelope. The host has 16 physical CPUs, with capacity for this
single-thread CPU diagnostic alongside the existing native worker and raw
replay. Recheck resources after full input admission. No native scene or GPU
training is launched by this profile.

Validation completed:

- Session 19568: ten tests passed in 2.17 seconds, covering exclusive versus
  overlapping cumulative time, invalid statistics, protected path names and
  the resource envelope.
- Session 15755: source-only preflight passed 1,963 bindings and created no
  runtime output.
- Session 19794: all 1,963 bindings were reverified unchanged. New source hashes:
  profiler `8c140fac2e9defeaf6eed76dfce4aa7d3a0f46a33fa131b193dc282c4b94e450`,
  protocol `e9dba7dafe1b68417b2919908f88fc44a70463b0cf6e4ca67eac6d8d3fd62e8f`,
  tests `6326b88dbf01d2916458ebce8343f917bea213fbbb64786621e21ef107c15ef5`.

Meanwhile, the existing hold-reorientation raw replay completed its initial
full input admission and created launch
`f3cdd9b0d97cb56328adc6f60edac2f5a62161a81bc91bf882117f7d1d5207f2`
with 1,943 frozen sources in
`go2_hold_reorientation_maze02_prefix_v1_attempt_001`. Its original child PID
2672447 remains live. A bounded inspection observed 41 complete rows (0–40),
with no changed request and exact recorded comparisons. That is progress toward
the required 406-row prefix, not a completed intervention or navigation result.

The full supervised native case continues in worker PID 2672443. Existing
frontier waiter PID 2663938 and later hold-native waiter PID 2671835 retain
their scheduled order. Do not restart any of these processes because an
admission or raw replay is quiet. The goal remains active: 38 completed audited
development episodes, zero verified round trips, no independent-layout or
real-platform qualification from this work.
