# Fixed tiny outside-keeper fit and group-OOM probes V1

Prospective protocol. No result is assumed. This authorizes source preparation
and two tiny new development service probes, not native challenge execution,
training, checkpoint/data access, original collector changes, retries, or a
claim that an 8 GiB native workload fits.

## Fixed attempts and resources

Run exactly one `fit` and, only after that same-source fit verifies, one
`overflow` attempt. Never reuse the earlier `lewm-tracking-memory-*-20260907-v1`
probe units. The new units are:

- `lewm-tracking-keeper-fit-20260907-v1.service`;
- `lewm-tracking-keeper-overflow-20260907-v1.service`.

Their respective fresh roots under the established owned development artifact
base are `go2_tracking_keeper_memory_fit_v1_attempt_001` and
`go2_tracking_keeper_memory_overflow_v1_attempt_001`. Neither the real challenge
unit nor its challenge/supervisor root is consumed. Existing units/roots cause
rejection, never replacement, reset, resume or retry.

Each tiny unit has 64 MiB charged-memory ceiling, zero swap, group OOM handling,
16 tasks, 15-second service runtime, no restart, control-group termination and
30-second stop timeout. The parent and its leaf both verify their actual kernel
controls and identity using the same standard-library-only admission function
as the prospective native challenge. The same service-prefix constructor is
used, with the explicitly different fixed limits and tiny program. The tiny
program verifies its own and the kernel helper's exact source hashes before
starting the leaf or allocating. NumPy, Torch and Genesis imports are forbidden
inside this probe, and no scientific files are read or written there.

The parent flushes the leaf-start identity before sending the exact one-shot
`ALLOCATE_ONCE` newline-terminated token through the child's stdin pipe. The
leaf must receive that grant after scope/source admission and before allocating;
EOF or any other token fails. This prevents scheduler order from losing the
parent's identity before an intended group OOM.

The `fit` leaf allocates 8 MiB and touches each page, reports allocation return,
then exits zero; the parent waits and reports the same normal exit. The
`overflow` leaf requests 128 MiB inside the already admitted 64 MiB group.
Expected outcome is local group OOM, killing the tiny parent and leaf before
either allocation/parent-return event can be emitted. Import failure, timeout,
an arbitrary signal, setup failure or exit code alone is **not** that outcome.
No real collector process is moved into this group or deliberately terminated.

## Outside evidence and preregistration

Before launch, compute the exact source/config definition hash with the new
runner's read-only `definition(mode)` and `identity` functions. Its explicit
import closure inherits and verifies the unchanged 786-source original learning
definition. Freeze those hashes for both modes before the first launch; do not
change sources between the two attempts to make the second one pass.

The outside process requires at least 2 GiB actual available memory and 40 GiB
free storage plus its 96 MiB evidence allowance. It verifies the exact new unit
and root are absent. It writes the bound definition, exact command and outside
identity into `request.json`, then uses the new production `relay_process` and
`EvidenceStore` code, not a replacement recorder. Each of the three evidence
files (`request.json`, `unit.log`, `terminal.json`) is capped at 32 MiB; the
complete 96 MiB allowance is per probe. Preserve all bytes of these small logs,
with no omitted diagnostic bytes accepted. Source hashes verify again afterward.

Both logs must show admitted parent and leaf identities in the same correctly
limited owned cgroup, a matching child-start record, and the intended allocation
request after leaf admission. For fit require command exit zero and both normal
return events. For overflow require a nonzero command exit, no return events,
and the manager's explicit `Finished with result: oom-kill` diagnostic in the
captured log. Use C locale and disable diagnostic colors. Preserve any different
observed failure unchanged; do not rerun a consumed attempt or call it OOM.

Persist the actual handle return code and bound log/request even if the probe's
parent cannot report. An outside observation error remains child-state-unverified;
it is not proof of workload termination or authority to restart. Actual outside
SIGKILL, host or filesystem failure cannot guarantee a new terminal receipt.

## Interpretation and follow-through

The tests address the real outside relay's ability to retain evidence across a
small process-tree OOM, and the shared kernel-admission/service-construction
path. They are not a workload-scale native test, calibrated peak-RSS proof,
global-OOM guarantee, full independent challenge qualification, or navigation
result. Record current/peak diagnostics without treating them as a calibrated
native-memory forecast. Inspect actual parent/leaf process and unit state after
termination and authenticate the saved artifacts independently.

Keep the existing learning collector running. After these probes, independently
review recording ceilings, native contact overflow checking, service controls
and outside failure recording. A genuine source-bound resource review may only
state supported claims; native workload fit remains unproved. The full fixed
eight-trial/eleven-stress challenge still waits for the original twelve-layout
collection and 36-fit matched learning study, source freeze and resource admission.
