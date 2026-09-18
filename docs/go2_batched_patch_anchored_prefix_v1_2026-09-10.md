# Batched historical floor queries: paired controller replay V1

The current controller exceeds its 100-ms interval. The completed receipt-copy
replay reduced full-prefix controller time by about 2.1%, and retained patch
coverage remains a separately measured cost. This experiment measures batching
alone against the original anchored controller. It does not adopt receipt-copy
optimization or either queued navigation policy intervention.

`BatchedPatchAnchoredController` inherits the original observation, mission,
selection, residual, model and failure calculations. Construction replaces only
the fresh primary and auxiliary `RetainedFloorPatches` objects inside the
unchanged `MeasuredFloorTransportMemory`. Their append implementation and
storage are inherited. Both camera stores use the separately checked
`BatchedRetainedFloorPatches.coverage`, with batches of at most 32 small
projection arrays and chronological original image-prefix queries. The map,
memory class, selector, gates, uncertainty margins, witnesses and targets remain
original. No active or queued native controller is modified.

The separately named runner is
`scripts/replay_go2_batched_patch_anchored_prefix_v1.py`, using exclusive output
`go2_batched_patch_anchored_prefix_v1_attempt_001` in the existing recovery
storage navigation artifact root. Bind completed predecessor result
`c3363fe9626fc36b8feb723346bef0344b7becccd3d9f68297d5f99ac3f7b0c0`,
its source/output identities and its completed original controller profile.
Verify full original worker inputs before and after execution. Retain failures;
no replacement or resume within this attempt.

The runner specializes the preceding paired `replay` function in a private
namespace with its exact unchanged code object. Only the candidate constructor,
decision normalizer, state-tree normalizer, output root and progress-print
provider differ. All other bindings retain identity. No imported module globals
are changed. Original controller and model calculations use their existing code.

Replay exactly 405 original observations, 0–404, from the completed expanded
JEPA maze-02 case. Use two fresh independently stored models with state
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.
Alternate controller order by frame parity. Verify public sensor input arrays
after each call and reconstruct every full original saved decision. Require
candidate decisions to equal those decisions after replacing only controller
identity and removing `batched_retained_floor_queries_enabled=True`.
All 402 post-warmup forecasts, model state and absence of gradients must match.
No observation 405 or changed command execution occurs.

At observations 3, 12, 395 and 404 compare complete retained memory, floor and
occupied cells, residual state and model input history. The state normalizer
requires the original memory class and two matched original or batched patch
objects with only their `frames` field. It replaces only these structural tags:
`memory.fields.patches.type` and `memory.fields.auxiliary_patches.type`.
Every field, array and witness is preserved. It does not rewrite class-name
strings inside evidence. These are four explicit complete state checks, not
checks at every intermediate observation.

Time only each `controller.observe` without profiling. Exclude admission,
packet loading, fingerprinting, decision serialization and state comparisons.
Report observations 3–404 plus fixed ten-observation windows 3–12 and 395–404,
including median/total times and counts exceeding 100 ms. Concurrent workstation
work remains possible; this is one paired replay, not a statistically replicated
isolated benchmark. A speedup does not establish navigation or real-time success.

Before launch and again after admission require at least 48 GiB available RAM,
41 GiB artifact storage and four physical CPUs. Use one numerical thread and
no new native scene. Source-only preflight performs no controller replay.
Native trials, independent-layout comparisons and real-platform evidence remain
separate requirements of the full goal.
