# Staged fixed-population process driver

This source connects the bounded 32-case schedule to fresh spawned collection
and audit workers. It has no CLI or final launcher. Initial and final full
population admission, source-bound final policy/queue/input verification, and
a separate source-bound overlap verifier are mandatory. The overlap verifier
must authenticate actual CPU-only audit qualification against the exact frozen
auditor, controller/model factory and driver sources; a flag alone is not
qualification. No such qualification record or final verifier is created here.

Each child sends its actual identity and waits on a start barrier. Its parent
observes that identity, registers the exact owned handle and only then releases
the worker. Before working, the child checks its on-disk registration against
its own identity, same original parent, boot, case, reference, source map and
launch. Parent failure during registration cancels the unreleased start. A
running collection is not cancelled because a concurrent audit fails.

Collection closes its log and publishes the original complete handoff. The
parent confirms its owned zero exit. A different fresh worker runs the fixed
original raw audit; its parent accepts the exact registered zero-exit handle
and full case evidence. Stage entry failures, native collection failures and
raw audit failures preserve evidence. No stage is retried or resumed.

Before a collection starts, the driver checks resources and the native process
inventory. Only its exact registered audit process can be excluded from that
inventory, after the source-bound overlap verifier has returned successfully.
Other native runners and unregistered spawn workers remain competitors. This
is an observed process admission check, not an OS-wide exclusive lock.

The scheduler enforces one collector, one auditor, two unaccepted cases, fixed
order, and first-arm parent-acceptance barriers. On failure, it stops dispatch
and waits for already-started workers to end, authenticating any completed
collection and retaining it without starting another audit. Resource records
include live owners and schedule state. Full final admission and complete
population authentication precede the final result. Navigation failures stay
in the population and do not become successful round trips by acceptance.

Tests use synthetic collection/audit artifacts, controlled process interfaces
and real lightweight spawn start barriers where specified. They establish
driver ordering and failure behavior, not navigation or speedup evidence.
Actual CPU-only audit qualification, final policy review, final source-bound
verifiers and joined input/extended-queue admission still precede execution.
