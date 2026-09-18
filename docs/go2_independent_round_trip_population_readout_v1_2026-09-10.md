# Complete independent population and paired readout V1

This adds completion/readout and matched-startup helpers for the previously
fixed 32-case, eight-layout, four-arm comparison. It does not create a native
launcher or consume an independent layout. The existing six-model maze2 waiter
retains ownership of the next native run.

`scripts/independent_round_trip_paired_startup_development.py` selects the first
fixed roster case in each layout as its startup reference, before observing
outcomes. It reuses the original physical/public startup reader: 900 physics
samples, four paired public observations and three actually completed zero
warmups. First navigation commands, terminals, forecasts and subsequent
controller states may legitimately differ across arms. The reader never
compares later physical outcomes as though they shared a command prefix.

An audited physical/acquisition stop with fewer samples, observations or
completed warmups produces `UNAVAILABLE_EARLY_STOP`. It carries actual counts
and makes no matched-startup claim. A short record without a recorded stop is
rejected. No alternative reference is chosen after seeing an early stop. If
the fixed reference lacks a startup, all comparisons using it are explicitly
unmatched; the scientific failures stay in the planned population.

`lewm/independent_round_trip_population_readout_development.py` requires all
32 ordered original workers, raw audits and complete physics-contact arrays.
Each worker must bind the exact case/model/treatment, collection receipt,
arm-specific raw replay, startup and compact readout. Compact readouts are
reconstructed from original audit arrays, collection counts and physics
contacts. Any missing, reordered, substituted or incompletely audited case
prevents a complete-population claim. A terminal scientific failure is retained.

Success requires the existing joint native and strict visibility outcome.
Additional consistency checks reject contact or acquisition/physical stops,
missing outbound/home dwell evidence, missing return traversal, missing terminal
quiet, the wrong observed mission terminal, or an incomplete zero drain. These
receipt checks do not replace the original raw audits or artifact authentication.
The future launcher must execute both.

Readouts include observed/native arrivals, distinct and repeated crossings,
return traversal, contacts, failure reasons, strict visibility and decision
timings. A fully audited acquisition stop before any decision retains zero
timing samples with undefined median/p95/maximum. It is not dropped or assigned
zero latency. The native evaluator requires at least the 750 settling samples;
earlier incomplete collections remain preserved infrastructure failures and
cannot pass this population-completion helper.

The three fixed paired comparisons are JEPA versus supervised training, the
predictive method versus reactive control, and persistent versus current-pair
JEPA planning grids. Each reports all eight layout pairs, discordant successes,
ties and descriptive success-rate differences. All eight layouts remain in
each arm's denominator, including failures and unavailable startup matches.
Treatment repetitions do not become additional independent layout units.
These summaries do not by themselves establish reliability, JEPA advantage,
planning/memory attribution, real-time behavior or hardware qualification.

Focused tests: process 45148, exit 0, 39 passed in 1.99s across the new population
tests and original startup tests. Synthetic cases cover all-negative complete
populations, paired positive/negative counts, corrupted or excluded records,
incomplete startups, zero-observation timing, distinct-edge accounting and
false success despite failed physical/visibility gates. The startup tests also
check exact pre-command evidence while allowing subsequent policy differences.
No native scene or model inference was performed by these tests.

Before execution, still required: complete original inventory, expanded fits,
correction and completed six-case development-result admission; a frozen
launcher enforcing fresh processes and one native scene; original raw artifacts
and startup/readout persistence; refreshed full-population resources; and review
of queued expanded-model failures using existing development evidence. The full
goal continues to require reliable end-to-end navigation, physical backtracking,
realistic sensing/timing and bounded real-platform evidence.
