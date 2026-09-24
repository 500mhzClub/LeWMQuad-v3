# Owned audit process and parent case acceptance

This helper connects a future coordinator's exact spawn handle to the
separate raw auditor's completed results. The child must wait before starting
its audit until parent registration returns. Registration records its actual
PID, birth time, command, parent, boot, assigned case, completed collection
confirmation and frozen launch sources. It requires the same parent that
registered collection and a distinct live audit process.

Parent acceptance requires the original in-memory ticket and handle, normal
zero exit, no failure marker, and all original saved audit and collection
evidence. Audit identity, source map, launch, reference and collection
confirmation must equal registration. The parent rechecks all artifact and
source bindings and writes one exclusive `_parent_completion.json`.
An ended process or successful receipt alone is insufficient. Duplicate
registration, substituted processes, nonzero exits, changed evidence and
reused tickets cannot yield an accepted case.

Acceptance means the assigned episode was collected and audited. A failed
navigation outcome remains a completed experimental observation; scientific
success is not required for acceptance. Verified round-trip outcome is taken
from the original authenticated raw-audit result. The helper grants no global
simulator-idle, real-time, hardware or goal-completion claim.

Tests pair real lightweight spawned process lifetimes and exit status with
synthetic collection/audit outputs. The original case-evidence persistence and
hash checks run; no new-layout sensor data, trained model or simulator runs.

No process is started, stopped or retried by this module. A bounded coordinator,
collection/audit start barriers, role-aware single-scene admission, CPU-only
qualification and overlap resource measurements remain required. The first
arm's parent acceptance must gate the other arms of that layout. Final policy
review and joined input/queue admission still precede study execution.
