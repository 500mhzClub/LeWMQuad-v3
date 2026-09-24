# Footprint receipt-copy candidate prepared

The separately named `ReceiptCopiedFootprintController` composes the existing
receipt copier with the completed packed/fused controller. It changes copying
in two pure contact-evidence paths through an invocation-local memory view.
The original memory type, persistent fields, observation path, failure checks,
geometry guard and scoped cache remain. Source functions retain their original
code objects, closures and defaults; imported module globals are unchanged.

The first component suite passed 28 tests in 7.52 seconds (session 73443,
exit zero). The final combined suite passed 35 tests in 7.32 seconds (session
47676, exit zero), covering real public-packet observation and articulated
footprints, owned cache results, failure propagation, copier aliases/cycles and
fallbacks, and the complete synthetic paired replay with corruption rejection.

Source-only preflight passed for 2,197 paths (session 91378, exit zero), with
80,958,324,736 bytes available RAM, 617,862,889,472 artifact bytes available and
16 physical CPUs. The completed profile owner was absent. Preparation rechecked
resources, owner termination and output absence without loading a model or
performing full input admission.

[Preparation record](go2_receipt_copied_footprint_preparation_2026-09-11.json):
`d8173c070ab81da0b768872df7a4d34dcc51541269a459497054e87553dda8ee`
(session 38751, exit zero).

The actual 1,428-observation paired replay must complete and its outputs must be
verified before claiming an incremental speedup. No new sensor collection,
native controller adoption, model training, or navigation qualification follows
from these component tests. See the [fixed replay protocol](go2_receipt_copied_footprint_late_history_v1_2026-09-11.md).
