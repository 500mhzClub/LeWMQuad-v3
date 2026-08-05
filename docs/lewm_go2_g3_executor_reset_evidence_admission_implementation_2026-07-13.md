# G3 Executor/Reset Admission Fail-Closed Remediation

Date: 2026-07-13

Status: candidate source remediation after definitive independent `BLOCK`.
This is not a G3 PASS and does not authorize executor/reset evidence, physical
promotion, exact-equivalence execution, or learned G3 output.

## Withdrawn Candidate

The first executor/reset adapter was not an authority boundary. Its public
`bind` accepted caller-selected producer, protocol, and geometry hashes, and
its public issue methods accepted raw reset stances and pose sequences. A
caller could therefore make the adapter endorse facts the runner had never
produced. Importable capability globals, mutable issuance tables, copyable
objects, unconstrained timing/yaw, and a privileged replay path compounded the
problem.

Independent probes produced a custom 5 m body/4 m step contract, a synthetic
501-pose 10 m traversal, reset at cell `(1000,1000)`, and a self-consistent
100 m by 100 m traversal that would have certified 1,000,000 physical cells.
The earlier 102 passing tests did not test the authority that created those
inputs. That candidate is withdrawn.

## Structural Remediation

There is no reviewed canonical runner outcome issuer in the repository, and
the deployment geometry still reports `physical_promotion_ready=False`.
Consequently the promoted executor/reset path is now structurally unavailable:

- `PromotedExecutorResetEvidenceAdapterV1` cannot be instantiated;
- there is no public `bind`, raw pose/reset issue, transaction build, or fuse
  API;
- promoted `RevisionedPhysicalMemory` unconditionally rejects
  `EXECUTOR_OUTCOME` and `RESET_CLEARANCE` authorities;
- it independently rejects every caller traversal polygon or execution block;
- exact and direct learned promoted-runtime paths remain locked; and
- runner/reset producer and outcome-protocol identities are hard-unset `None`,
  which is a disable state rather than a wildcard.

This intentionally provides less functionality than the blocked candidate.
It does not substitute hash-shaped caller claims for a missing runner.

## Frozen Future Contract

The validation-only canonical contract reopens fixed repository sources and
requires their exact hashes:

- deployment geometry identity:
  `e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca`;
- deployment geometry file:
  `e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52`;
- directional body policy:
  `c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc`;
- primitive registry:
  `cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8`;
- platform manifest:
  `5ac4a08b17cfaa3552f3c3ccd45930b8a929ac5ca31eb1f9440923f037c78189`.

The fixed geometry is directional actual-yaw support with forward
`0.3700000000000001 m`, rear `0.43210313102250314 m`, half-width
`0.2668059073252429 m`, planning/reset radius `0.47 m`, exact registered
89-cell reset support, physical lattice `0.10 m`, and maximum translation
substep `0.025 m`.

Future runner outcomes are additionally bounded to an 11-pose maximum,
50,000,000 ns pose cadence, 500,000,000 ns maximum duration, 0.025 rad maximum
wrapped yaw substep, five executed commands, and 100,000,000 ns command
cadence. The validation helper checks only this geometry/timing shape and never
creates evidence authority.

A successor may enable admission only after a separately reviewed runner owns
typed executor outcomes and reset certificates. Those objects must reopen
canonical outcome bytes/events and bind the actual requested and executed
command sequence, sequence IDs, timestamps, cadence, duration, complete swept
pose sequence, map frame, live memory instance and revision, terminal pose,
and failure event. The controller must only consume those issued objects; it
must not construct them from raw poses.

## Capability, Copy, And Replay Closure

All module-global admission, adapter-binding, and replay capability objects
were removed. The memory has no execution binding, admission registry,
reserved-receipt set, or consumed-receipt set to mutate.

Memory and the unavailable adapter use slots, reject `copy.copy`,
`copy.deepcopy`, and pickle reduction, and have no instance `__dict__` that can
be copied into `object.__new__` clones. An object-new adapter has no build,
issue, or fuse surface. An object-new memory lacks initialized physical state
and cannot receive the removed issuance tables.

Canonical deserialization has no authority token or alternate admission hook.
It reopens typed transaction bytes through the same public fail-closed checks.
Therefore a serialized blocked-candidate execution transaction cannot be
changed to `promoted_runtime=True` and replayed into promoted memory. Empty
promoted state and privileged-tainted exact development state still round-trip
canonically.

## Permanent Regressions

The focused suite permanently checks:

- exact source hashes, dimensions, policy, 89-cell reset morphology, timing
  limits, unset runner identities, and hardware-promotion false;
- rejection of custom 5 m body, 4 m step, and promotion-ready contracts;
- absence of bind/issue/build/fuse APIs;
- rejection of 501-pose 10 m, four-metre step, overlong, huge timestamp-gap,
  and instant-pi-yaw sequences;
- atomic rejection of reset at cell 1000 and a self-consistent 1,000,000-cell
  traversal forgery;
- absence of importable capability globals and withdrawn memory hooks;
- issuance-table injection, copy/deepcopy, object-new, and transaction-transfer
  attacks;
- rejection of serialized execution history promoted by config tampering; and
- unchanged learned/exact promoted locks and exact development taint replay.

No protected dataset, scene manifest, RGB, model, checkpoint, simulator output,
GPU, or iGPU was opened or used. Verification is CPU-only with numerical
thread counts capped at one.

## Candidate Verification

Focused G3 command:

```text
env PYTHONPATH=.:lewm_worlds OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python3 -m pytest -q \
  lewm/tests/test_revisioned_physical_configuration_memory.py \
  lewm/tests/test_promoted_executor_reset_evidence_adapter.py
```

Observed focused result: `48 passed in 0.20s` (`16` permanent remediation tests
and `32` existing G3 tests). The combined G3/G4/G5 dependency run reported
`102 passed in 29.31s`. Python compilation also passed.

Candidate hashes:

```text
13fccc662784c0a7eed75965a9d4154369666f26e804173482b461c55b8b9add  lewm/planning/revisioned_physical_configuration_memory.py
7a6388353cab7b25064d29a29f906864ce29b680d9402e9dbe1e6687c7e56ca6  lewm/planning/promoted_executor_reset_evidence_adapter_v1.py
d3fb231358b8a23532917048090680e953e5dd6b3d3b5d6a0f55be5f7f6b7776  lewm/tests/test_promoted_executor_reset_evidence_adapter.py
```

The exact development adapter was unchanged:

```text
2dc1629750a6487740187a1464c3d65f42d9fa78e491e8470a0f0cbfbf5cacad  lewm/planning/zero_inflation_exact_physical_adapter_v1.py
```

Independent review is required. No PASS claim is made here.
