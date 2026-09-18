# Monitored independent-study audit integration

The successor worker calls the unchanged independent multiarm raw auditor
inside AuditCPUMonitor. Its surrounding separate-worker code is reused with
an isolated function-global dictionary containing only the wrapped audit
call. Original imported module globals and frozen predecessors remain intact.
The original registration barrier, collection validation, model assignment,
raw audit, post-audit validation, persistence and failure handling remain.
Robot geometry is constructed immediately before the monitored call by the
original worker; model loading inside the independent auditor is monitored.

Require assertions, hash seed zero, one BLAS/OpenCV/Torch thread and
OPENCV_OPENCL_RUNTIME=disabled before import. Launch binds the monitor,
worker, parent checker, successor driver, tests, protocol and exact monitor
policy. Any scope violation, including a caught device error or removed
profile hook, prevents original audit completion. A closed failure monitor
is retained alongside the original worker failure. A monitor that cannot
enter retains a failed receipt with no completed scope.

The separate CPU receipt binds the launch, collection confirmation, reference,
source manifest, boot, actual audit owner and complete returned audit report.
Parent acceptance verifies that receipt against the original ended audit
execution and saved report, along with the original owned normal-zero-exit
process check and full artifact bindings. A failed navigation episode can be
accepted as a scientifically completed episode, but missing, substituted or
violating monitoring cannot. No positive tensor count is required for the
model-free reactive arm. Counts and exact scope flags must be consistent.

The successor staged driver preserves original one-collector/one-auditor,
two-unaccepted-case bounds, first-arm acceptance barriers and failure drain.
Its spawn entry remains a top-level pickleable function. Only the audit
worker, parent acceptance and CPU receipt added to polling differ. Collection
and scheduling bodies are unchanged. Parent completion and final population
artifact maps include every accepted CPU receipt. The old separate-execution
record continues to make no CPU qualification claim; additional enforcement
is represented by its separately checked and bound monitor evidence.

The final source-bound population and overlap verifiers are still required
by the original driver and are not provided by this integration. This module
does not launch native collection or certify new layout results. The prior
four-arm monitored startup and complete old short raw-audit replay are
bounded runtime evidence, not execution of the entire independent study.
Each actual future independent audit is monitored as it runs. Monitoring is
not an OS device sandbox; native CPU library threads are not individually
profiled. No speedup, real-time performance, navigation success or hardware
qualification is claimed.

Tests use synthetic collections/reports and actual parent-owned spawn handles;
they do not use new independent-layout sensor data or native scenes. Existing
jobs keep their original source bindings and queue order.
