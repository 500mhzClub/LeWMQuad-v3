# Outside failure evidence survives a real tiny process-tree OOM

Both fixed probes completed without retries or changes between attempts. The
fit parent and leaf exited normally; the overflow parent and leaf were killed
after the admitted allocation request. The outside keeper preserved complete
request, diagnostic log and terminal receipt in both cases. **This verifies a
small failure-recording mechanism, not native workload fit or navigation.**

| Observation | Fit | Overflow |
| --- | --- | --- |
| Definition SHA-256 | `33becb80b46902afb3a6004f0f69c4cb082e0ec2111c98ba55bb7b4625e499e7` | `104acc231a20d7fc8791106b8923b8aeee4aa62d7988a7ecec21d27e080846a0` |
| Parent / leaf PID | 2119023 / 2119024 | 2119255 / 2119256 |
| Outside keeper PID | 2118955 | 2119187 |
| Kernel controls, independently reported by both processes | 64 MiB memory, zero swap, group OOM 1, 16 tasks | Same |
| Leaf allocation request | 8 MiB | 128 MiB |
| Allocation / parent return events | Both present | Both absent |
| Service-manager result | success; exited/status=0 | oom-kill; killed/status=KILL |
| `systemd-run` return code | 0 | 1 |
| Complete retained log | 1,678 bytes, zero omitted | 1,433 bytes, zero omitted |
| Service runtime reported by manager | 42 ms | 45 ms |

Both used the same source-bound 792-file closure, the shared lightweight kernel
admission and service-prefix code, and the actual new `EvidenceStore` and
`relay_process` implementation. The parent flushes leaf identity before its
one-shot stdin grant releases allocation. Actual logs independently confirm
that the leaf-start record precedes allocation. The 15-second runtime and
30-second stop timeout were not reached.

The fit's in-process kernel current/peak reading was 25,677,824 bytes, while the
manager summary reported 2.7M. The overflow summary also reported 2.7M. **This
discrepancy remains unresolved; neither summary is used as a calibrated peak
measurement or a native-memory forecast.** The demonstrated facts are the
admitted limits, requested allocation, manager OOM classification, retained
evidence and terminal process state—not a measured 64 MiB overflow peak.

After completion, all four parent/leaf PIDs were absent. Both exact new units
were `not-found` and inactive after automatic collection. The original learning
supervisor 2063013 and `l07` worker 2113431 remained live and unchanged. No old
probe unit, real challenge root/unit, checkpoint, GPU, native simulation or
scientific dataset was consumed by these probes.

## Artifact and verification identities

Roots under the established owned development artifact base:
`go2_tracking_keeper_memory_fit_v1_attempt_001` and
`go2_tracking_keeper_memory_overflow_v1_attempt_001`.

| Artifact | Fit SHA-256 | Overflow SHA-256 |
| --- | --- | --- |
| `request.json` | `430f052fdbc9dbf772a3809f1f354e496680437f83a3694167bd8a45f83fa05a` | `814786512aadf4eb8cfd07d96be7eb06ed3de94b9c33bb431b09a9bd358379f2` |
| `unit.log` | `5b42a3875acc0a7af000c61b7b98c90271b471e1b02e2501684c61cae944b524` | `64e88ae2bd152faf5cfc7451110767d939476c9f3bc44532beebaf1784d4eb55` |
| `terminal.json` | `581a16cb696f3214827c07c2b2e6883a945dc72059450a0896a8ba89ba75c8e9` | `8e2953e524d481dbae27b9b06af21fd9156d137a282ceb3ce9e5696431b6bad5` |

57582 terminated exit 0 with `TINY_KEEPER_FIT_VERIFIED`; 50966 terminated exit 0
with `TINY_KEEPER_GROUP_OOM_EVIDENCE_VERIFIED`. The latter is the outside probe
runner's success at observing its expected failure; its actual service command
returned 1. It is not an experiment retry or a relabelled native failure.

Independent read-only verification 2670 (dc2281, terminal exit 0) did not call
the runner's log inspector. It authenticated both terminal/request/log sets,
recomputed canonical definition hashes, rechecked all 792 source bindings for
each attempt, checked the same-source predecessor fit link, rebuilt role/PID/
group/control/allocation joins and event order, checked complete logs against
saved event arrays, and verified process absence and false scientific claims.
An earlier independent read expected numeric `status=9`; the actual manager
uses symbolic `status=KILL`. Its format-only assertion failed after the fit
verified. The corrected reader maps the observed signal name through Python's
signal constants to SIGKILL. No probe, frozen source, log, result or
preregistered OOM criterion changed; the original criterion already required
the explicit manager `oom-kill` result, not a numeric signal spelling.

## Tests and supported resource-review scope

The adjacent 19-file regression passed 495 tests in 299.36 seconds before the
probe-only handoff change. JUnit SHA:
`4753309577531d77f377ddaa4a0eba381b9d4a787891165f417d3092d508356f`.
After that change, 107 focused tests passed in 6.06 seconds, including five new
grant tests. Focused JUnit SHA:
`b4001f3086692a2304a5d03b0de8800d9702892bfc3ec12dfb4f4504158e087b`.
This is not a single 500-test run. Both exact definitions were frozen before
the first actual attempt in the accompanying frozen-definitions JSON.

The shared production profile remains 8 GiB, zero swap, group OOM, 512 tasks and
48 hours; direct unbounded native parent/worker entry is rejected. The probes
support the shared mechanism and outside recorder, not a guarantee that the
larger workload fits or that the keeper survives global OOM, host failure or
storage failure. Kernel charged-memory limits are not strict instantaneous RSS
bounds. The service-manager peak discrepancy is not silently resolved.

Together with source-checked recording ceilings and per-sample/terminal native
contact checking, this now supplies evidence for a narrowly stated resource
review. It does not release the original twelve-layout/36-fit scheduling gate,
freeze the final challenge definition, qualify perception/control, or establish
JEPA/rollout/memory benefits. Novel-maze navigation remains the unmet endpoint.
