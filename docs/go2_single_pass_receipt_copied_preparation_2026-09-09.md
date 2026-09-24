# Compose prior map-index and receipt-copy optimizations on the current controller

The completed receipt-copy benchmark preserved all514 current maze2 decisions
but still measured695.732ms median controller time. Earlier single-pass map
indices preserved1873 dual-camera decisions on an older trajectory. Their
combined benefit on the current measured-floor controller was not established.

Implemented a separate SinglePassReceiptCopiedController. It inherits the
receipt-copy selector and substitutes eight independently owned, empty primary,
auxiliary and partition indices with the unchanged packed-owned/single-pass
index class. It preserves current measured-floor observe/advance methods,
registration, mission, residual owners and decision metadata. No policy,
tracking fallback, model, memory retention or geometry rule changes.

The new paired benchmark compares receipt-copy alone against the combined
candidate on the complete514-observation original maze2 episode. Both decisions
must match the saved unoptimized controller exactly, without normalization.
Alternate arm order each observation, stop immediately on a mismatch, check
public input immutability and both unchanged model states/absent gradients.
Report additional index benefit over receipt-copy for active observations and
each order subgroup, including100ms deadline misses. Acquisition, receipt I/O,
physics and comparison are outside the timer. No whole-loop speedup claim.

The runner authenticates both completed optimization predecessors and original
learned cohort before and after execution, merging all source bindings without
conflicts. The earlier success of separate components does not admit their
composition as equivalent; the actual complete paired replay must establish it.

14 tests PASS in3.28s,81533exit0. They verify independent empty indices, current
state-owner/method types, sensor-driven paired floor mapping, exact stored bounds
and query state, unchanged missing-RGB stop behavior, paired order/timing scope,
complete-decision/input mismatch rejection and no following observation after
a changed candidate. Both synthetic replay models must be independent.

Five final source SHA-256 bindings for the corrected benchmark submission:

| Path | SHA-256 |
| --- | --- |
| lewm/single_pass_receipt_copied_controller_development.py | 8a10d305569b279135030c3667763ed8a141ecffb3964a0667a056b71ccbaeab |
| scripts/benchmark_go2_single_pass_receipt_copied_v1.py | a8444f8ff05280496fc8210d165c62285b283d986982e4957c35610f30f5d7e4 |
| lewm/tests/test_single_pass_receipt_copied_controller_development.py | 8be601da0fd7e58edecde1e3abea8e9f433a64de936df5c71af4f59271a674e3 |
| lewm/tests/test_single_pass_receipt_copied_benchmark_development.py | 4facb5e3a7f9ef04d9f6b0dd5100ff492358944ced37dc98f8af2e2401993ac9 |
| docs/go2_single_pass_receipt_copied_benchmark_v1_2026-09-09.md | 59b5f4879be4e3c84bab6743e2d80ce80df7ef06793f30836d5dc52f59c76036 |

Hardware91653exit0:72,407,638,016bytes available RAM,78,112,964,608bytes artifact
free,21,359,149,056bytes workspace free,16physical/32logical CPUs,all32affinity,
CPU3.4%,GPUsidle. Sole native worker2485788 uses10,758,172,672bytes RSS. One
16GiB CPU paired replay fits beside the existing32GiB native allowance. This is
capacity evidence, not measured runtime performance. Runner refreshes resources
before output creation and periodically during the replay.

Submitted24468 with the established single-thread Python environment:

```text
scripts/benchmark_go2_single_pass_receipt_copied_v1.py
```

Exclusive root:go2_single_pass_receipt_copied_benchmark_v1_attempt_001. Input
authentication is the first stage; no result or speed claim at submission.
Preserve the same handle and five frozen sources. No native collector adopted
either optimization, and no new navigation outcome is produced by this job.

## Authentication-schema correction before experiment admission

24468 exited1 before output creation, model load or controller replay. Its
generic ordered-launch verifier raised KeyError('input_sha256') on the older
single-pass dual-camera launch, which uses separate native/prefix/performance
artifact bindings instead. The old runner SHA was
dd0abdce5517cb14fd3e99771344db199a2042f441c46428a182b6bb718d56d6;
old benchmark-test SHA was
d363368a438e16e0294d2e424c886a30e77d5b4e6b3b1238405e5386592a5bed.
The output root was confirmed absent. This is a failed preparation submission,
not a failed or completed physical/controller experiment; no result was replaced.

Corrected the admission to call each predecessor's exact original verifier:
the receipt-copy inherited verify_inputs and single-pass verify_all. Neither
source, artifact nor environment verification was weakened. Three regression
cases check distinct schema dispatch, source-conflict rejection and propagation
of predecessor verification failures. Benchmark tests now14PASS2.03s21689;
with the unchanged three controller tests,17tests pass in total.

Resubmitted as75686 after hardware95623exit0:72,388,341,760bytes available RAM,
78,112,837,632bytes artifact free,21,359,140,864bytes workspace free,CPU3.3%,
GPUsidle,all32affinity. Same sole native scene and16GiB CPU replay allowance.
Follow75686, not closed24468; preserve the final five sources listed above.

## Actual paired replay admitted

75686 passed input/source admission and began execution. Launch SHA-256
5ae02642ad918e2b271b48e92e0f0adb9c392c7b2598412d03b36656e2b81e18
binds1,696 sources. Launch hardware71,706,488,832bytes available RAM and
78,116,331,520bytes artifact free,CPU3.4%,GPUsidle,all32affinity. First61 complete
decision pairs are exact. This is partial progress, not complete equivalence
or a speed estimate. Keep the same running handle and wait for final checks.
