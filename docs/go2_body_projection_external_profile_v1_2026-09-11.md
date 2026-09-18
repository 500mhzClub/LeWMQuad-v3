# Prospective full-history external profiling diagnostic

This diagnostic samples the completed BodyProjectedTiledController on its
original full 1428-observation development history. It changes no policy,
model, sensor inputs, history, state normalization or native navigation job.
Its fixed output is `go2_body_projected_external_sampling_v1_attempt_001` under
the original navigation-development artifact root. There is one attempt, no
automatic retry/resume and no alternate data selection. Preserve every failure.

## Admission and ownership

The pinned original reference is completion
`47c6e00d30f9b7e8c31b19dbeed511699d7565145e2b34273fe56e3c96eb25ab`,
result `548c8afe30d87c5f2d503c820a77a03c75d7c764dd6d045529d1a258fad94eea`.
The actual read-only admission probe and owned-child marker smoke must already
be complete. The previous CPU replay, completion watcher and admission probe
must be ended under their original boot/PID/creation/command identities.
Existing resource gates require 64 GiB available RAM, 41 GiB artifact free
space and four physical CPUs. At most one full CPU replay runs alongside the
existing native scene/audit queue.

The parent freezes sources, hardware, environment and its own identity in an
exclusive launch before creating one Python child. That launch explicitly says
input admission is pending. The child authenticates its parent and launch,
executes the unchanged original completion checker with private in-memory
output, and requires the entire original witness to reconstruct except UTC.
No controller executes before this admission succeeds.

The child grants Linux PR_SET_PTRACER only to its parent and descendants. No
system ptrace setting changes. The parent starts the pinned py-spy 0.4.2
binary in --pid mode against this newly created child only. Both readiness
handshakes and exact parent/child/profiler identities precede controller work.
The launcher accepts no external target PID. The parent owns and reaps both
subprocesses. It does not restart work because observation is slow.

Sampling uses speedscope, 100 Hz, idle threads, thread identifiers and full
filenames. No GIL-only, nonblocking, native-stack or subprocess sampling mode
is enabled. The existing running native worker is never attached to a profiler.

## Full replay and profile requirements

The frozen replay derivative reconstructs every original decision, public input
and command endpoint over all 1428 observations, with 1425 model forecasts and
seven original retained-state witnesses. All input arrays, model state and
gradient checks remain. It samples the original ten-observation windows
3–12, 395–404 and 1418–1427 through 30 distinct Python stack markers.
Sensor acquisition, normalization and post-replay admission are also sampled
when they occur during the attached interval; they lie outside the marked
controller subtrees and are excluded from the controller summary.

After replay, the child repeats full original input admission, compares all
returned evidence, rechecks sources/tool/owner liveness, writes its result and
exits. Both child and profiler must exit zero. The parent reauthenticates the
reference again after both exit and reconstructs the marked-stack summary.
All 30 markers require descendant samples on the exact original main thread.
Every profile row, including unmarked samples, must parse under the fixed
bounded schema. Reported sample totals must match the trace and reported
sampling errors must be zero. Failure rejects this attempt; no partial profile
is called complete.

The independent completion checker runs only after the parent has also ended.
It requires actual result and launch hashes, authenticates all 12 artifacts,
reconstructs the summary from the completed trace, checks every replay row,
the seven state witnesses, model identity, exact scope flags and all timing
window assignments, and repeats original raw/model admission. State-size
snapshots are descriptive and are not independently reconstructed.

## Interpretation

The profile establishes sampled Python stack occupancy during marked controller
observations. Nominal sample weights are not measured durations, CPU self times
or native call counts. No profiler overhead or sampling bias has been removed.
Profiled elapsed times must not be presented as an unprofiled speed comparison.

The original strict sensing failure at frame 1173 remains. This is not a new
navigation run, independent-layout evidence, a JEPA causal comparison, useful
real-time operation or hardware qualification. The broader goal remains active.
