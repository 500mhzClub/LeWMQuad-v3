# Parent-owned collection process boundary

This source preparation supports the future independent eight-layout, four-arm
study. It launches no study episode and changes no running worker or queue.

The future coordinator must create a fresh spawn child and keep it alive while
the parent registers its actual handle, PID, creation time, command, parent
identity, boot, case, launch, frozen sources and completed first-arm reference.
Registration is exclusive. A caller-supplied PID or an on-disk receipt cannot
replace the original in-memory handle or resume an interrupted coordinator.

After collection publishes its existing closed-log and complete-data handoff,
the parent can confirm only its registered handle's normal zero exit. The
confirmation rechecks the full handoff, original reference, source identities
and collection bytes. Child replacement, nonzero exit, live child, altered
bindings, failure markers, duplicate registration and consumed tickets fail.
The coordinator must record any failure and preserve all collection artifacts;
these helpers never retry, kill, or restart a child.

The resulting record proves an owned collector ended normally and its bound
collection bytes passed admission. It does not prove a raw audit completed,
scientific success, global simulator idleness, or permission to execute an
audit. Those claims remain explicitly false. The serial runtime's existing
same-process execution receipt cannot be used for this future split lifecycle.

Tests use real lightweight spawn children and synthetic collection evidence.
No native scene, new independent-layout observation, trained-model execution,
or timing-speedup evidence is produced. A role-aware single-scene coordinator,
separate CPU audit worker and acceptor, final policy review and joined input
admission remain required before executing the study.
