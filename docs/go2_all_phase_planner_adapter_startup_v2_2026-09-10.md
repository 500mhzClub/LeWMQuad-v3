# Expanded-model interface startup replay V2

Retain the exact models, adapter, four-observation stopping boundary, complete
original raw-decision reproduction and forecast comparisons from V1. V1 exited
with a diagnostic comparison failure after the first case's observation3:
`observed state changed outside model interface: observed_goal_distance_m`.
Its directory, source, launch, partial stream and failure receipt remain intact.

The original controller exception path returns `_result(..., None)`, which
sets its top-level observed_goal_distance_m display to None. The corrected
controller returns the already unchanged mission receipt's observed distance.
That reporting-field difference is a consequence of removing the interface
failure, not a change in measured pose or mission state.

V2 requires exact evidence, original visual evidence, map/contact receipt,
mission receipt and floor partitions. The three complete warmup decisions must
still match in full. At observation3 it explicitly requires the old recorded
wrapper failure, old null display, candidate nonterminal state and candidate
distance equal to the unchanged measured mission receipt. It records both
display values rather than requiring their equality. Malicious/mistaken changes
to pose, mission distance, reported distance, terminal or failure are tested
as rejections. This is the only replay-comparison correction.

The adapter itself passed 11 actual-selector/forward tests in 2.32s, handle
60186, exit 0. It copies the exact expanded model state, preserves buffer names
and forward arithmetic, and provides the existing planner's required interface.
No existing source, controller, model weights, correction coefficients, camera,
clearance gate, mission, native collector or original failed worker is changed.

The original six-model parent may still be completing its top-level audit.
This read-only replay authenticates its six completed worker/audit/artifact
sets and preserves that process. It consumes no observation after the first
terminal divergence, executes no native command and does not establish physical
navigation. Full original input admission remains required by any subsequent
newly named prospective native launcher.
