# Isolated receipt copying for anchored selection V1

The completed controller profile
`8be3553ba54c67827790a281aaf3a08bd2facbc8aff39537dd7f353a2b3b3fb0`
identifies recursive copying as a major cost, particularly while reconsidering
holds. This candidate changes the copy provider for that calculation only.

Reuse `lewm/receipt_copy_development.py`, retaining its standard fallback for
non-builtin objects, shared memo, aliases, cycles and source lifetime rules.
Do not replace deep copies with shallow copies or remove detailed receipts.
Six explicit functions retain their exact original code objects in private
global namespaces: `constrain`, `plan`, `filter_selection`,
`reconsider_hold_feasibility`, `_reconsider_anchored_hold`, and
`reconsider_anchored_continuation`. Their existing copy calls use `copy_receipt`,
and references between these six functions use their private versions.
Imported module globals and all unlisted dependencies remain unchanged.

`ReceiptCopiedAnchoredSelector` keeps the original selector choose body and
upstream `ResidualFirstIntervalSelector`. The controller inherits observation,
mission, model, map, residual and command execution behavior. Its output changes
only controller identity and the explicit
`anchored_selection_receipt_copy_enabled` implementation flag. No hold timeout,
frontier retirement, forecast correction, utility, gate or threshold is added.
This performance candidate is separate from the two queued policy interventions.

Required validation before any native use:

1. Bind unchanged predecessor files and require exact function code plus only
   the declared namespace substitutions. Check input ownership, aliases,
   cycles, rejection gates, exceptions and inherited failure behavior.
2. Freeze a separately named raw replay. Reconstruct the full original fixed
   prefix 0–404 from the completed expanded-model JEPA case, compare complete
   original and candidate decisions after removing only the two declared
   implementation metadata differences, verify public inputs, observed state
   and model state, and retain every failure. No observation 405 is needed.
3. Measure both controllers without profiling, using alternating execution
   order and separate fresh mutable state. Report early and hold windows plus
   the full prefix, concurrent workload and exact timing scope. No full-loop
   speedup follows from the older component-copy benchmark alone.

The paired runner is `scripts/replay_go2_receipt_copied_anchored_prefix_v1.py`,
with exclusive output `go2_receipt_copied_anchored_prefix_v1_attempt_001`.
Require the completed profile, unchanged original full JEPA worker, and full
original input verification before and after execution. Bind the same model
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`
in two fresh controllers with independent model storage. Alternate which
controller executes first by observation parity. Preserve and check public
input arrays after each call. Reconstruct complete retained contact memory,
planning cells, residual state and model input history at observations
3, 12, 395 and 404, including the final complete retained state. These are
four explicit state comparisons, not a claim of checking hidden state at
every intermediate observation.

Time only `controller.observe`; packet loading, fingerprint checks, decision
serialization, state comparison and admission are outside those intervals.
Report all post-warmup observations 3–404 and the two fixed ten-observation
windows 3–12 and 395–404. Record counts exceeding 100 ms even when a ratio
improves. A minimum of 48 GiB available RAM, four physical CPUs and 41 GiB
artifact space is required, rechecked after admission. Use one numerical
thread and no new native scene. Any failure ends this exclusive attempt.

No raw replay, native trial, independent-layout comparison or hardware action
is launched merely by importing these sources. Frozen running and queued experiments
retain their original sources. Even a verified speedup here would not by
itself establish 100-ms sensing/control, maze success or goal completion.
