# Scoped reuse plus batched retained-floor queries

The completed late-history profile spends 20.546 seconds cumulatively in
retained patch coverage over ten observations, within 36.852 total profiled
seconds. These inclusive times overlap other functions. Existing batched patch
queries preserved the complete earlier 405-frame controller prefix and reduced
total controller time by 4.95%; early navigation regressed slightly. The currently
running scoped-reuse replay tests a different optimization of repeated queries.

This source-only combination uses the existing batched controller constructor
and existing scoped selector. The constructor replaces only fresh primary and
auxiliary retained-patch stores; the selector preserves the original anchored
policy while reusing exact footprint queries within one synchronous selection.
The memory type remains unchanged so the scoped selector uses its supported
branch. No projection kernel, observation update, mission, learned forecast,
feasibility criterion, history limit or model correction is changed.

Require complete synthetic observed-memory and actual articulated Go2 footprint
receipts to match the scoped-only baseline, including independent public receipt
ownership and cache cleanup. Retained state may normalize only the two patch
object type tags, using the already reviewed batched replay normalizer. Preserve
all frame arrays, witness order and bytes. Complete decision normalization removes
only the declared combination and batching flags and restores the scoped-only
controller identity; it must not hide nested evidence or other changes.

Before any full raw execution, finish and authenticate the current scoped-only
late-history replay, review its measured benefit and retained-state checks, and
prepare a separately named incremental paired replay against that exact baseline.
It must reconstruct all 1,428 observations and 1,425 forecasts, check the seven
fixed retained-state frames, preserve original commands and public packets, and
repeat original input admission before and after execution. Alternate execution
order and time complete controller observation calls without a profiler.

Do not install this class in the live or queued simulator studies. Component
equivalence does not establish the combination's raw-replay equivalence, speedup,
100-ms execution, qualified sensing, navigation or hardware readiness. Preserve
the original visibility failure and zero-round-trip outcome. The current CPU
replay owns its slot until its exact process ends.
