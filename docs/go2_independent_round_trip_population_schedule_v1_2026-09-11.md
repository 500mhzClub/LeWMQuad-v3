# Fixed bounded collection and audit schedule

The scheduling state machine fixes all 32 cases in the existing layout-major
cyclic arm order. It permits one collector and one auditor concurrently, with
at most two unaccepted cases. A completed collection occupies the single
pending slot until the active audit finishes. Pending audit starts take
precedence over a new collection.

Within each layout, the first arm's parent-accepted audit must finish before
another arm can collect. Later arms carry that exact first worker terminal
SHA. Merely finishing first-arm collection does not release the barrier.
The next layout's first collection may overlap the preceding layout's final
audit. No case, arm or layout is replaced based on outcome.

The future process driver must call collection lifecycle confirmation and
audit parent acceptance before reporting completion to this state machine.
The scheduler checks case, order, reference, collection identity and completion
claims but does not authenticate files or processes itself. Its source-only
snapshot grants no execution permission.

An ended failed collector or auditor stops all new dispatches. The other
already-running stage is allowed to finish and preserve its evidence. If
collection finishes after an audit failure, its confirmed data remain pending
and no new audit starts. A resource or admission failure likewise stops new
work without cancelling running workers. Every failure remains recorded;
there is no resume or retry transition.

Before collection, require the original conservative remaining-population
disk allowance, at least 64 GiB currently available RAM, and four physical
CPU cores. Already collected but unaccepted cases remain in the disk budget.
These are conservative admission thresholds, not measured overlap capacity
or OS limits. CPU-only audit qualification and role-aware native-idle checks
are still required by the future driver.

Tests drive the complete fixed population through differing collection/audit
durations, all worker-failure positions, first-arm barriers, identity changes,
resource rejection and failure draining. They run no scene, process or model.
A process driver with start barriers and source-bound final admission remains
to be integrated with the already prepared lifecycle and audit workers.
