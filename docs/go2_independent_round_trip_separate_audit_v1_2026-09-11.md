# Separate original raw audit worker

This module prepares a separate audit process for the fixed independent study.
It launches neither a collection nor a process. Before any raw audit, it
authenticates the collection parent's zero-exit confirmation, original spawn
registration, complete data and closed collection log, original launch and
sources, and completed first-arm reference. The auditor must be a fresh spawn
child of the same live parent, distinct from the collector.

The fixed existing multiarm auditor reconstructs the public sensor/controller
stream and checks commands, physical state, visibility, startup and readout.
It receives the same assigned correction and a fresh robot geometry. All
original collection bindings and parent/child identities are checked again
after audit. Failure preserves the log, raw files, known bindings and report,
and writes the existing worker-failure marker. There is no retry or resume.

The generic case `_worker.log` contains this raw audit's output; the collection
log remains separately bound by its handoff. Generic audited-case evidence is
persisted through the existing strict writer. Execution is recorded under
`_separate_audit_execution.json`, explicitly declaring collection and audit
occurred in different processes. The original same-process receipt is never
written or accepted as a substitute.

The saved-evidence reader requires the original auditor to have ended and
authenticates complete case records plus collection process evidence. It does
not prove the audit process's zero exit or accept the case for the population.
The future coordinator must verify the exact audit handle it started, preserve
first-arm reference barriers and admit one native scene with bounded queued
audits. CPU-only qualification, role-aware native-idle checks, overlap timing
and memory measurement, final policy review and joined input admission remain
required. The worker makes no CPU-only or speedup qualification claim.

Synthetic integration tests substitute raw auditor/model execution and process
admission while retaining actual file hashing and case-evidence persistence.
Process-boundary tests separately exercise the parent's real spawned handle.
No independent-layout raw data or navigation evidence is produced by tests.
