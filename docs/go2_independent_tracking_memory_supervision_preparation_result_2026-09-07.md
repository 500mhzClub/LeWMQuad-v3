# Whole-challenge memory supervision implemented; 467 tests pass

The unfrozen independent tracking challenge now requires a fixed, memory-limited
user service and an outside evidence keeper. This removes the earlier unsupported
claim that source inspection alone proved an 8 GiB native worker memory peak.
It does **not** demonstrate that the workload fits, launch native trials, or
qualify tracking/navigation. The genuine resource-review JSON remains absent.

## Implemented execution boundary

The outside keeper performs the existing source/metadata-only admission: original
twelve-layout collection and 36-fit matched study complete, unchanged definitions,
genuine resource review, exact CPU environment, scheduling priority and storage.
Both new attempt roots and the exact service must be absent. It writes an
exclusive request before starting the service. No whole-tree export, old source
change, replacement service, retry, shorter population or unbounded fallback is
provided.

Inside the service, the challenge parent handles all eight fresh native workers,
base replay, all 88 stress replays, raw scoring and predecessor comparison. Its
kernel cgroup must have 8 GiB `memory.max`, zero swap, group OOM handling and
512 tasks. The service command also sets a 48-hour runtime, no restart and
control-group termination. Entry checks reject a different/unlimited group,
changed controls, foreign user, unreviewed child group or OOM-exempt process.
The parent and workers authenticate the outside request and a live owned keeper
outside the workload group; checks recur through the main workload phases.

These are prospective charged-memory and finite-attempt controls, not a proof
of workload fit or strict instantaneous RSS. Kernel memory limits can temporarily
overshoot; OOM grouping concerns the workload group, not immunity of the outside
keeper to global memory pressure or its own failure.
[Kernel control-group documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory)
describes those distinctions. The fixed service uses `OOMPolicy=kill`, whose
[systemd 255 definition](https://github.com/systemd/systemd/blob/v255/man/systemd.service.xml)
sets group OOM handling; 255 is the installed systemd major version.

## Failure evidence outside the workload

The keeper drains the exact `systemd-run --wait --pipe --collect` child and
persists its return code and bounded diagnostics. The
[systemd 255 command documentation](https://github.com/systemd/systemd/blob/v255/man/systemd-run.xml)
specifies service waiting, reported exit information and post-completion unit
collection. Consequently the keeper captures the live handle's output instead
of assuming a completed unit remains queryable afterward.

The new outside root has exactly `request.json`, `unit.log`, `terminal.json`,
each at most 32 MiB. The log retains its first and last 16 MiB when oversized,
records omitted-byte counts, and rejects supervised completion if anything was
omitted. This is diagnostic evidence only: scientific tapes/streams are not
truncated or repaired to satisfy this limit. The terminal tail can retain the
manager's failure report when the inner program cannot write its own result.
A nonzero return code is not automatically labelled OOM, nor does a killed
`systemd-run` handle alone prove the workload service terminated.

An interrupted keeper reports child-state-unverified, rather than waiting
silently for the workload or claiming its termination. The keeper cannot promise
a new durable receipt after its own SIGKILL, filesystem failure or host failure;
the exclusive root/request and whatever log prefix reached storage remain.
Actual unit/process state must be inspected before further action. No restart is
authorized by a missing receipt or observation error.

Exit zero is insufficient. Successful supervision additionally authenticates the
complete inner result and its output bindings, a bound matching launch, the
original completed learning receipt and the same outside request. All scientific
qualification flags remain false, even after successful supervised execution.

The original inner 240-path roster/238-binding successful terminal result stays
unchanged. Adding 96 MiB of outside evidence gives a two-root worst-case envelope
of 55,809,409,024 bytes (51.9765625 GiB), below the existing 52 GiB allowance with
24 MiB headroom. The 40 GiB free-storage reserve remains. The terminal writer
reserves its own file allowance, not all three allowances again after two files
already exist. This is byte accounting and writer admission, not a filesystem
quota against unrelated processes.

## Tests and actual checks

- Initial focused run: 67 passed, one failed because a test expected a 17-byte
  marker in a 16-byte retained tail. The implementation retained the correct
  last 16 bytes. The corrected test checks exact prefix/tail slices and the
  actual omitted-byte count.
- Subsequent focused run: 108 passed in 16.28 seconds across supervision,
  launcher and native-contact guard tests. Three additional review-rejection
  cases were then included in the final regression.
- Final regression session 12942 terminated exit 0: **467 passed in 298.75
  seconds**, across 18 explicit files. Durable JUnit reports 467 tests, zero
  failures/errors/skips, and 298.560 seconds of suite time. This is not a native
  experiment or full-repository qualification.
- The new suite includes 48 tests: effective-kernel fixtures, exact command and
  request admission, exclusive/bounded evidence, interruption semantics,
  successful/failed orchestration, corrupted/missing inner evidence and no
  retry. Two tiny real ordinary subprocesses exercise pipe/exit recording:
  normal exit and self-SIGKILL. **They do not induce OOM or create user units.**
- Read-only actual checks reject the current ordinary process from the challenge
  scope, verify the prospective unit is absent, and verify challenge, supervisor
  and matched-study roots remain absent. The seven native contact-source
  bindings and the frozen 701/771/786 source identities remain unchanged.
- Source-only traversal from the six current implementation/test/protocol paths
  plus the inherited 786-source learning definition yields 815 paths. This is
  not the final reviewed challenge definition: no resource review is invented.

JUnit:
`.generated/navigation-development-staging.m6MDz1/independent_tracking_memory_supervision_adjacent_v1.xml`
SHA-256 `6e8ec44c459b7bb5848f305c2fe6be0996ff6c4ec5d87f2e581044f685e98049`.

Current source identities:

| Path | SHA-256 |
| --- | --- |
| `scripts/independent_tracking_memory_supervision_development.py` | `21dcbc8bcf04b5a9693ae2ecb635f93274094e0c1ab7b7432e4497ace460e5f9` |
| `scripts/supervise_go2_independent_tracking_challenge_v1.py` | `3c99d4e18544775522500fbdf1d828332c3e6d1ce302f33279ff6d79ee658921` |
| `scripts/run_go2_independent_tracking_challenge_v1.py` | `90935be2131c5344bb805f8c788f01a42125dcf0ea4732c41dd2bc63ae030251` |
| `lewm/tests/test_independent_tracking_memory_supervision_development.py` | `c1e1862eb0c3e25fc9c83829cf662cabf3004269e0ea1d523538ac83f86b16dc` |
| `lewm/tests/test_independent_tracking_launcher_development.py` | `4e0bc4d27a6a728d6d619a2ab307521dfd06de18123181631e5593eeee5b8fb6` |
| `docs/go2_independent_tracking_challenge_v1_2026-09-07.md` | `fd602443271acfae9cabc8512d5f42438eddbb3169d385a7ff13b4453632f867` |

## Remaining work and live learning collection

Run a separately scoped tiny end-to-end fit/OOM service test that exercises the
new durable relay without consuming the real challenge unit/root. Reuse neither
of the already completed old probe units. The new memory module currently
imports the existing artifact guard, which imports NumPy/Torch transitively;
a 64 MiB child cannot be assumed to import that graph. Inspect/isolate the
lightweight kernel admission before choosing the tiny probe's resource contract.
Do not treat import-time OOM as a successfully admitted allocation test.

Then independently review the recording ceilings, native error handling,
service controls and outside failure evidence. Only issue a source-bound resource
review if those claims are actually supported; workload fit remains unproved.
Preserve all eight trials and eleven stress conditions. After the original
collection/study completes and admission passes, freeze/run the fixed challenge,
independently verify it, and use its outcome to decide a fresh closed-loop test.

Learning collection remains live under supervisor PID 2063013, current `l07`
worker PID 2113431. The completed `l06` launch/audit bindings authenticate 120
eligible trials: 115 full schedules and five physical-terminal recordings.
Seven layouts / 840 eligible trials are verified overall; these are not
navigation successes. The planned 36-fit study has not launched. Disk space must
be rechecked after collection and training: current free space is not a promise
that the later full 92 GiB launch requirement will still be met.

The scientific endpoint is unchanged: reliable local physical execution,
deployment-valid real-time sensing, useful predictive JEPA and online-rollout
contributions against matched baselines, online memory/backtracking, independent
novel-maze completion, and bounded real-platform evidence when hardware permits.
The latest actual simulated room-return result remains 0/3. This preparation
does not achieve those requirements or justify marking the goal complete.
