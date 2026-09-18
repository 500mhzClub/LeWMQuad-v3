# Planning-map comparator for residual-anchored continuation

Reuse the completed current-observation planning map rather than creating a
second representation of current cells. Its earlier maze0 prefix, native pilot
and paired readout are already recorded in the 2026-09-09 current-observation
planning result documents. That pilot failed on tracking and strict visibility;
it does not establish a planning-memory advantage. The earlier memory scope
audit predates and is superseded by those implementation/execution records.

This successor changes only the selector base: the complete residual-anchored
continuation chain now receives the existing immutable current paired-camera
planning view. All proposals, waypoint targets, nominal connector/path checks,
first-interval feasibility and anchored-continuation checks must use that one
view. The map's original measured cells include reobserved old cells and retain
the exact primary/auxiliary arithmetic. Do not filter first-witness timestamps
or fall back to accumulated planning cells.

The full accumulated contact map, primary/auxiliary indices and partitions,
floor patches and later-contact evidence remain. Tracking, floor registration,
model temporal history, executed residuals, scan state, mission and settling
state remain. The current controller's model, action set, horizon, task budget
and arrival rule are unchanged. This compares planning-grid persistence; it is
not a fully memoryless controller. The inherited persistent=False interface is
not used and remains unsupported.

Before native use, check a causal prefix against the completed original
residual-anchored maze2 episode. Authenticate its exact result, source, model,
raw sensor and command bindings with the original completion verifier. At each
observation compare unchanged public inputs, observation/contact/mission state,
executed residual history and all common raw forecasts. Stop immediately at the
first changed requested command or terminal; consume no following observation.
Changes to the current pending forecast are allowed only at that boundary.
No unexecuted physical outcome or improvement may be inferred from this replay.

The original maze3 audit and six-model maze2 waiter retain ownership of their
processes and frozen sources. This source/prefix work launches no native scene,
changes no queued assignment, and makes no independent-layout navigation claim.
A later matched experiment must bind the completed prefix and fresh model and
controller assignments, include complete raw audits and preserve every failure.
