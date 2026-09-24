# External profile V1 failed in tracing; nonblocking smoke passed

The V1 full external profile did not complete. Its last complete comparison row
is frame 1274. The Python child stopped making CPU and I/O progress while both
of its threads remained in `ptrace_stop`, owned by profiler thread 2885639.
The profiler continued consuming CPU without new reads or output. This is
evidence of a tracing-tool stall, not a controller computation bottleneck.

Two recorded snapshots ten seconds apart retain identical child CPU and I/O
counters and the stopped thread states. Earlier observations also showed the
same counters and replay frame. A graceful SIGINT did not release the tracee;
the exact authenticated profiler was then terminated with SIGTERM. Its owning
parent detected the premature profiler exit, terminated and reaped its own
Python child, and recorded terminal failure. The watcher preserved that failure
without invoking the completion checker. No native job was signalled.

All four profile-related owners are now confirmed ended. Sessions 88315 and
34376 exited 1. The original sources and artifacts remain unchanged. No complete
profile or full replay result was written; there is no partial-profile success.
The full CPU replay slot is free again.

- Original root: `go2_body_projected_external_sampling_v1_attempt_001`.
- Original launch SHA-256:
  `89192077c33ab33bd1eb147fa4877f6e5d2f801c22633071be1897fd2b35d207`.
- Root failure SHA-256:
  `7fcb85edd00e0d32500f9bd401e5788f88a88e75904bad7123c73f8c386fe825`.
- Execution SHA-256:
  `38285f61c0310e72e9b737f805320c4b1fe26601ccf3772822bc289c8af8e41a`.
- Failure audit: `docs/go2_external_profile_v1_tracer_lock_failure_audit_2026-09-11.json`,
  SHA-256 `c0ec1fd70fec1ea2e2848e5bb6eeab407c664e4e607e16234036af3d982c20cb`.

The audit binds all eleven preserved root files, the process observations and
both signal records, and the watcher execution/failure records. It rechecks
all 2391 original source bindings. The observation receipt SHA-256 is
`d6105050d16b3f21f740fcbdb5b3fe08857c37c422f48e8ec726883207a62351`.
The watcher failure SHA-256 is
`9dee69fab689adcd81e3577f67056528bd745978cc1f3439b29b33dba795f071`.

An [upstream issue](https://github.com/benfred/py-spy/issues/732) and
[draft locking change](https://github.com/benfred/py-spy/pull/802) describe a
class of process-lock hangs. They do not prove the exact cause of our stall.
No draft upstream patch was installed and no system ptrace setting changed.

## Separate nonblocking synthetic probe

`scripts/run_go2_external_marker_nonblocking_smoke_v3.py` privately reuses the
previous owned-child smoke body, changing only its output/source bindings,
result identity and the added `--nonblocking` profiler option. It uses the
same original synthetic marker child and preserves all 30-marker coverage,
original-owner, exit-status and profile-format checks. No previous file is
modified and no existing process is attached to the new profiler.

Six command-scope tests passed in 0.13 seconds. The actual probe completed in
session 39854, exit zero, with 286 marked samples and every marker represented.
Both child and profiler exited zero and were reaped.

- Smoke root:
  `.generated/tools/go2_py_spy_0_4_2_v1/marker_nonblocking_owned_child_smoke_v3`.
- Result SHA-256:
  `341eb89e317ba7d99ea530c9933df1b8e158ed70a9173f60c6062c64c92c7369`.

Nonblocking sampling avoids pausing the child but cannot guarantee a consistent
instantaneous stack snapshot. The result explicitly retains that limitation.
This short probe establishes availability and marker coverage, not full-run
reliability, unbiased attribution or profiler overhead qualification.

## Next action

Prepare a separate V2 full attempt with a new root and source/protocol/checker
identities. Preserve the unchanged controller, full 1428 observations, 1425
forecasts, seven state witnesses, original before/after input admissions and
negative sensing scope. Require the failed V1 parent, child, profiler and watcher
to be ended and the failure audit to remain authenticated. Add only the explicit
nonblocking sampling change and retain its consistency limitation; do not mutate
or resume V1. The existing marker parser and complete-row checker remain useful.
No V2 full launcher or controller run has yet been created or executed.

The extended-budget native worker remained live in post-collection auditing at
18:05:48 UTC. Its preliminary return-leg sensor/model stop remains pending final
audit. The original native queue and broad navigation goal remain active.
