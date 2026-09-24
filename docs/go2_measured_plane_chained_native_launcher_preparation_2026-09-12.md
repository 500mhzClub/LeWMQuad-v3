# Chained native launcher prepared, execution still awaiting replay

`scripts/run_go2_measured_plane_chained_maze02_v1.py` and its prospective
protocol are prepared. No native attempt has been created or queued by this
preparation. The existing timing and chained-controller replay owners remain
live; the actual completed chained waiter result is still required.

The launcher composes the original native worker with the prepared chained
candidate pipeline, completed-input admission, dynamic physical-prefix checker
and raw-result validator. The original worker's collection, complete raw audit,
artifact hashing, model checks, readout and failure preservation remain in
place. A public wrapper remains picklable for a fresh spawned worker. The
launcher retains the same physical/public execution configuration as the
original learned pilot and does not adopt the separate single-pass optimization.

All 13 focused launcher tests passed in 2.30 seconds. These check actual worker
and candidate pipeline composition, the fixed model and budget, source-only
isolation, required completed-result identity, rejection of unfinished or
failed admission before model/output creation, exclusive output roots,
dynamic-report forwarding, and revalidation of the matched launch fields.
The separate native-result checker previously passed 21 focused tests.

The actual source preflight verified 2,627 bound sources and available hardware.
It loaded no model and started no worker or scene. An actual runtime preflight
using an intentionally invalid placeholder result SHA was then rejected by
the live-owner gate, before result lookup, with:

> both original queue owners must finish and end first

The exclusive root `go2_measured_plane_chained_maze02_v1_attempt_001` remained
absent. This rejection is the expected pending dependency, not a failed native
attempt and not a reason to create a replacement attempt. Memory and artifact
space passed the launcher's existing 32 GiB and 55 GiB admission floors.

| Prepared file | SHA-256 |
| --- | --- |
| `scripts/run_go2_measured_plane_chained_maze02_v1.py` | `e42ce2a98671726d798a8c8d46d6c130dc8f312ef38f477bdc4965054df738af` |
| `docs/go2_measured_plane_chained_maze02_v1_2026-09-12.md` | `d701a310ecdf16d155447896baef9c5f48f884e99c2683090c45247e479acaa8` |
| `lewm/tests/test_measured_plane_chained_native_launcher_development.py` | `60cfa425e8f6a625abed97a2092b324409628a9b9cd51066de1c96616eacb550` |

Next: inspect the completed timing and chained-controller replay results when
their original owners end. If the chained replay establishes an admitted first
intervention, pass the actual completed chained waiter result SHA to this
launcher's `--preflight-only` mode under the existing deterministic environment.
After that passes, invoke the same launcher and exact SHA without preflight;
the launcher rechecks completed inputs, native serialization and resources
before freezing its source hashes in the launch and executing one worker.
If the replay is negative or fails, preserve it and diagnose the actual
evidence; this launcher cannot bypass it. The full navigation goal remains
incomplete regardless of source preflight or synthetic test success.
