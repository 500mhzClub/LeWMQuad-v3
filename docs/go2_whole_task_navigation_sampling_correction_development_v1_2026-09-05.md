# Whole-task native-sampling correction V1

The original V1 stopped before its first recorded settling sample with the exact
[legacy region-logger failure](go2_whole_task_navigation_development_v1_result_2026-09-05.md).
This separate root permits one corrected four-trial execution, not a restart of
that terminal root. The original launch/result and all original source bindings
are verified before execution. All four original scene specs, seeds, controller,
marker detector, memory arms, budgets and success criteria remain byte-identical.
The [original protocol](go2_whole_task_navigation_development_v1_2026-09-05.md)
remains the scientific specification. No layout or controller adaptation follows
the failure because no navigation outcome was observed.

The only physics-session change replaces native `_sample` region annotations
with its otherwise identical state/contact readback. No dummy membership labels
or hidden teacher geometry are supplied. Settling execution is identical except
its returned trace aggregates the actual recorded fields, not a legacy fixed
region-field inventory. C3 method resolution inserts this leaf below the existing
contact-stop, ordinary-sensor, fast-sensor and phase wrappers; their methods,
gait commands, physics stepping, reset and timing remain unchanged.

Required checks: exact AST equivalence after the enumerated annotation/aggregation
removals, wrapper order, actual mock native readback, unchanged common fields
against the inherited logger with a separate test-only legacy geometry, native
contact/body-stop preservation, ordinary/fast sensor updates on terminal rows,
and collector/auditor body identity except the session import/output binding.
Mock checks are not physical success. The four actual trials and full raw audit
remain necessary; preserve every outcome, no retry, geometry fitting, controller
change or metric relaxation within this correction attempt. The fixed-forward
integration comparison does not establish JEPA, independent-maze or hardware
qualification.
