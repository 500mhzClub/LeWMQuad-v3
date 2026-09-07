# Recording ceilings enforced; isolated memory-controller smoke tests complete

This advances the unfrozen tracking challenge's resource preparation. It does
not launch the challenge or learning study, adopt a tracker, change a frozen
collector, or establish maze navigation. The original learning collector is
still running on layout seven, with six layouts / 720 eligible trials complete.

## Recording changes and evidence

`lewm/independent_tracking_recording_budget_development.py` defines the exact
numeric archive layouts and conservative per-archive serialization ceilings.
The full fixed population requires at most 1,019,246,279 reserved bytes across
the six numeric archives. The contact archive contributes 1,000,239,314 of
those bytes. This is reservation arithmetic, not an observed compression ratio
or proof that every contact-capacity row is physically reachable.

The exclusive episode writer now enforces each numeric archive's ceiling before
compression and again on serialized writes. Its 23 static JSON paths each have
a 32 MiB ceiling. JSON encoding stops once accumulated output would exceed the
limit, without writing a truncated record or assembling the whole oversized
document. The raw-byte writer also rejects a bypass of these file ceilings.
The total static serialization envelope is therefore **1,790,998,215 bytes**,
below the 2 GiB deferred allowance (2,147,483,648 bytes). Existing small setup
and per-frame external-write reservations remain separately checked.

These limits do not certify an arbitrary native renderer's writes or a valid
complete tape. A rejected oversized archive remains a recording failure, not
an absent measurement that can be silently filled. Existing partial-evidence
and secondary-failure handling remain in place. The writer does not silently
cast or drop fields to satisfy the expected layouts. The new exact shape/dtype
checker is a separate checker, not a replacement for raw sensor/physics audits.

Tests invoke the actual inherited physics sampling, contact attribution, ideal
50 Hz body sensing, 500 Hz gyro sensing, causal history construction, RGB-D
observation capture and snapshot methods. Only native getters and RGB
rasterization are synthetic. At 850 samples and three observations, all saved
members/shapes/dtypes match the independently declared layouts. A separate
physical-stop case retains sample 851, including its contact and gyro record.
Both masked contacts and self-contact rows survive serialization. No Genesis
scene is constructed, and no synthetic measurement is claimed to be native.

Additional tests retain the empty-recording case, reject excess counts and
shape/dtype/member drift, stop oversized arrays before compression/file
creation, compare bounded JSON byte-for-byte with the original encoder, and
check exact-limit, early-termination and raw-write-bypass cases. Initial new
recording tests passed 16/16. Recording, memory-admission, collection and launcher
focused tests then passed 84/84 in 28.30 seconds. The final adjacent regression
passed **379 tests across 16 explicit files in 287.50 seconds**, with zero
failures, errors or skips. This is the preceding 354-test scope plus 16 numeric
recording tests and nine memory-scope admission tests. All five new/changed
source and test files stayed unchanged during the final run. The JUnit file is
`.generated/navigation-development-staging.m6MDz1/independent_tracking_recording_budget_adjacent_v1.xml`,
SHA-256 `6a70d42ae7d12316e5fa2b35f4c60515c2c1db5ecfff507a2b5d3b65b9ca647b`.

## Native contact-capacity caveat found during review

Installed Genesis defaults are nondifferentiable float32/int32, at most 150
collision pairs, `box_box_detection=False`, and five contact slots per pair.
The coarse 750-contact-per-sample envelope is consistent with the inspected
allocation path. `func_add_contact` sets an overflow error bit when capacity is
exceeded; the contact sorting path clamps the count to allocated capacity.
The rigid solver's `check_errno()` raises for that bit instead of resizing.

However, `Simulator.step()` checks errors at the beginning of a step only when
the global substep count is divisible by `RATE_CHECK_ERRNO = 10`. A capacity
error is therefore not guaranteed to be raised in the same sample callback,
especially at a final stopping point. The prospective challenge must explicitly
check effective options and native error state during acquisition and at its
terminal boundary, without dropping/clamping additional data or modifying the
frozen collector. A size-compliant contact tape alone does not prove that the
native simulator omitted no contacts. No overflow was observed or alleged in
the live collection; this is a newly identified prospective validity gap.

Relevant inspected native sources include `engine/solvers/rigid/rigid_solver.py`
(SHA-256 `4f16b91c2cb417916e2cebb9e2ba32cffbb6f1f096b6eac32ce08d4df3dea76e`),
`engine/simulator.py`, and `engine/solvers/rigid/collider/contact.py` in the
installed environment identified by the preceding resource analysis.

## Actual isolated memory-controller evidence

Read-only inspection found a unified cgroup v2 hierarchy with memory control
available to the existing systemd user manager. The active collector remains
in its original `tmux-spawn-872c8a7f-4ac1-445c-8f6a-2b1e4bd43b98.scope`, whose
`MemoryMax` and `MemoryHigh` remain unlimited. No existing scope was changed.
At inspection, the worker's virtual size was about 14 GiB while its resident
memory was about 2.5 GiB. An 8 GiB virtual-address-space limit would therefore
not be an appropriate substitute for resident-memory containment.

Two previously absent, fresh user service units ran the standard-library-only
`scripts/check_go2_tracking_memory_scope_development.py`. Before allocation the
probe checked its own exact unit identity and kernel files: `memory.max=67108864`,
`memory.swap.max=0`, `memory.oom.group=1`, `pids.max=16`, and no OOM exemption.
Each unit also had a 15-second maximum runtime. These settings apply only to
the tiny probes and must not be copied blindly to a multithreaded native job.

- `lewm-tracking-memory-fit-20260907-v1.service`, PID 2110192: requested 8 MiB,
  returned successfully, and exited 0. Its in-process cgroup current/peak read
  was 15,728,640 bytes. The transient service's later summary reported 3.0 MiB;
  this accounting discrepancy was not resolved, so it is not an exact peak
  calibration result.
- `lewm-tracking-memory-overflow-20260907-v1.service`, PID 2110265: admitted the
  exact same controls, requested 128 MiB, and was killed before allocation
  returned. `systemd-run` reported `oom-kill`, `code=killed/status=KILL`, 64.0 MiB
  peak and zero swap, with exit 1. This is the expected synthetic failure,
  not a failed native experiment or a reason to retry it.

Both transient units were automatically collected and subsequently reported
`LoadState=not-found`, `ActiveState=inactive`. The collector's PID, scope and
unlimited settings were rechecked afterward; it remained live. No scientific
artifact or user data was deleted, and no live process was moved into a new
scope. Nine synthetic admission tests reject wrong units/users, legacy cgroups,
missing limits/group policy, unlimited task counts and OOM-exempt processes.
An additional actual invocation in the ordinary, unlimited session was rejected
on scope identity before any allocation request; it did not create a service
or run an overflow workload.

The kernel documents `memory.max` as a charged-memory limit with possible
temporary overshoot; if reclaim cannot resolve pressure, OOM killing is scoped
to that cgroup. This is useful containment, not proof that a workload will fit,
that its instantaneous RSS is strictly bounded, or that every memory category
is accounted identically. [Linux cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory).

## Next executable work

1. Add the new challenge's explicit per-sample/terminal native overflow checks
   and effective-capacity witnesses, with tests retaining prior evidence and
   rejecting an overflowed terminal tape. Do not change live/frozen methods.
2. Integrate a reviewed isolated memory scope for the entire challenge workload
   (including the replay/scoring parent and native child processes), with an
   outside supervisor that retains unit exit/OOM evidence even when the inner
   process cannot write a failure report. Test both ordinary and killed-process
   paths using synthetic workloads before native execution. No silent unbounded
   fallback, retry, population reduction or contact truncation.
3. Replace the draft's unsupported peak-memory assumption with an honest
   admission/containment/failure contract and justified headroom. A kernel limit
   is a failure boundary, not an assertion that the fixed scientific workload
   will finish below it. Preserve all original motion and sensor-stress aims.
4. Once the original twelve-layout collection and 36-fit study are genuinely
   complete, review/freeze the exact revised challenge source/configuration and
   resource evidence, then launch. The successful resource-review file and
   final challenge definition SHA are still absent; neither smoke test grants
   native execution authority.

The latest actual room-return result remains 0/3, and earlier JEPA predictors
have not beaten the simple empirical baseline. Independent challenge results,
fresh closed-loop execution, low-friction and real-time control, matched JEPA/
rollout/memory benefits and deployment-valid hardware evidence remain required.
