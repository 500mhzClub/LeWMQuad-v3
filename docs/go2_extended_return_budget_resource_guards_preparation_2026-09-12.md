# Longer native lifecycle resource guards and closed checks prepared

The prospective longer pipeline now has sampled resource guards around
acquisition, controller execution and the raw audit, plus a checker that
reconstructs their closed receipts. No native trial has adopted or executed
these wrappers. The fresh native launcher, artifact binding and physical-prefix
audit still need integration before dispatch.

## Evidence informing the envelope

The completed chained native result was reauthenticated at SHA-256
`163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849`.
Its worker terminal record reports **14,611,058,688 bytes maximum RSS**
(about 13.61 GiB), measured with `getrusage(RUSAGE_SELF).ru_maxrss` after
collection, full raw audit and prefix verification. This is a worker-process
peak across that lifecycle, distinct from the sampled system-availability
range in `docs/go2_extended_return_budget_resource_observations_2026-09-12.md`.
It does not establish which stage peaked or a maximum for twice the duration.

The bound native contact archive was independently rehashed to
`97be8d01843c71bd330c76bda65d39b61cf6ae642e1f7119d7e8357a6272021e`.
Its array headers record 651,751 contact entries over 201,400 physics samples,
with **37,766,491 bytes** of uncompressed NumPy members. Together with the
previously inspected 76,332,008-byte physics-array archive, this helps distinguish
persisted numeric data from Python-object, raster/history and audit memory.
Contact population and compressed decision sizes can vary; neither archive
provides a worst-case bound for a longer execution.

The chosen prospective startup requirement is **64 GiB available RAM and
84 GiB artifact space**. Disk allowance is allocated as 28 GiB collection,
8 GiB persistence and 8 GiB audit, retaining a 40 GiB reserve. Runtime samples
require at least 16 GiB available RAM and at most 48 GiB worker RSS. These are
conservative development guard thresholds, informed by the completed shorter
run rather than claimed mathematical bounds on the longer one.

## Source identities

| File | SHA-256 |
| --- | --- |
| `scripts/extended_return_budget_resource_guard_development.py` | `8ede2c2c453426c5251fd8e0af35748f6a378c56be950a30d0859329544c1fe2` |
| `scripts/resource_guarded_extended_return_maze_development.py` | `49b8e82d0d80a62b9d9d36515eb5557416777a5be5382cb1535c19f518c0bfd1` |
| `lewm/tests/test_extended_return_budget_resource_guard_development.py` | `8a9d771c6ce452e5e2091a23d718da1fd540bca8e5752f68f23a9bdd4d3de9f8` |
| `scripts/extended_return_budget_resource_audit_development.py` | `9fc10b8638c5e08132e95e9ecfbed986d8b6a78f7591ed788b0961a9c3bfe0d7` |
| `lewm/tests/test_extended_return_budget_resource_audit_development.py` | `97c273ccd88bfcff94cb31f794ccea7bf90e713ec2e27107f12278a8f2f0dc52` |

All 2,639 source files bound by the live chained/single-pass comparison were
independently rehashed afterward and matched. The five files above are outside
that roster. The already prepared recorded-prefix runner remains unchanged.

## Behavior and limits

Collection requires startup admission and records resource samples before and
after each packet and controller call. The original controller constructs all
state and produces the complete unchanged decision. A wrapper delegates to it;
resource measurements are not model inputs and add no decision fields.
The existing collector retains its own 28 GiB collection stop rule. The new
guard additionally checks the combined collection/persistence allowance and
space retained for later stages.

The audit checks resources before/after its raw sensor audit and each controller
call. Both phases record a completed sample after their original function
returns. Each phase has an exclusive JSON-lines resource stream and final
receipt. The future worker must bind these four root-level artifacts in
addition to the original per-case artifact roster.

A breached threshold latches a `ResourceLimitError` and forbids another
controller call even if availability later recovers. It propagates outside
the collector's packet-contract exception handler. Original `finally` blocks
still attempt raw persistence and scene cleanup. The wrapper records whether
the phase completed, the first breached sample and any original exception.
A resource abort is an operational failure, not a new physical-contact label
or a successful navigation result.

The closed checker requires the recorded collection's exact decision count,
ordered resource cycles, startup admission, all sampled limits, disk deltas,
resource extrema and exact completed phase receipts. A partial trailing
acquisition cycle is allowed only when the actual collection records an
acquisition or physical stop. Audit cycles must be complete for every recorded
decision. Artifact hash authentication remains the future worker's responsibility.

These checks are at execution boundaries. They do **not** enforce an OS memory
limit, bound an allocation peak inside a call, reserve RAM against other
processes or guarantee cleanup after severe resource exhaustion. Persistence
is checked after the collector returns; the 8 GiB allowance is headroom,
not an instrumentation claim about each individual write. Resource thresholds
and failures must remain visible in the eventual native result.

## Focused validation

The guard/wrapper suite passed **17 tests in 2.15 seconds**, tool session 22621.
The closed-resource checker suite passed **18 tests in 1.89 seconds**, tool
session 79914. Both passed on their first observed invocation using the
original deterministic single-thread environment and
`pytest -q -p no:cacheprovider`.

The tests generate a complete 8,014-observation resource population with
32,058 collection checks, exercise exact admission thresholds and all four
runtime breach reasons, verify failure latching, phase-specific disk allowance,
bounded clocks/frames, exclusive paths and preservation of original errors.
Synthetic delegated collection verifies unchanged decisions, original cleanup
and no command execution after an after-controller breach. Synthetic raw audit
stops before model execution when the sensor-audit check fails.

Closed-check tests reconstruct complete and valid partially stopped collection
cycles, then reject missing/reordered/extra rows, altered frames or clocks,
changed RSS/RAM/disk values, fabricated disk deltas or extrema, mismatched
decision counts, oversized rows and failed startup admission. They establish
resource instrumentation and receipt semantics, not native physics, rendering,
actual memory consumption or the new physical prefix.

These short tests overlapped the already declared non-isolated timing
comparison. No isolated timing result, full return, independent-maze result,
planning/memory advantage, real-time qualification or hardware qualification
is established here.
