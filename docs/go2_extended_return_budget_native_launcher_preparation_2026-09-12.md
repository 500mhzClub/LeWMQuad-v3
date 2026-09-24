# Longer native worker and launcher prepared

The fresh 8,000-step native worker, parent launcher and prospective protocol
are implemented and have passed focused synthetic tests and source-only
preflight. No longer-budget native execution or prospective budget-prefix
replay has started. Both exclusive attempt paths were confirmed absent.

The original chained/single-pass complete-history comparison remains live:
PID 3015121, creation time 1789209876.83, launch SHA-256
`b8a67fc80aebb2f39968092067cea00b94304c449bd7c86fdff30458c58d6260`.
The observed progress was frame 2599, or 2,600 of 4,014 observations, with
matching original and normalized candidate decisions and no result or failure
file. This is an intermediate observation, not a completed comparison.
All 2,639 source bindings in that actual launch were independently rehashed
and unchanged after the new preparation. The custody instructions and ignore
rules also retain their previously recorded identities.

## Implemented behavior

Initial native input admission reconstructs the completed prospective
prefix's public output and authenticates its entire artifact roster plus the
original native artifacts, raw audit, unchanged model and source ancestry.
Later verification rehashes those same rosters without recursive execution
of predecessor controller or physical-prefix analyses. The actual new native
prefix is reconstructed by the worker, then independently by the parent
after the worker ends. Early negative native episodes cannot claim an
unexecuted paired prefix or successful round trip.

The worker integrates the tested 8,000-step collection/audit pipeline and
phase resource guards. A separate complete-worker ledger observes peak RSS,
available RAM and cumulative disk use through terminal-record writing.
Failure after a terminal record has been written still raises to the parent
and prevents completion. Partial raw evidence, failure records and logs are
retained. The parent authenticates the saved worker terminal against the
returned record, independently verifies the actual physical prefix, checks
all closed artifacts and preserves final serialization headroom. Neither
layer automatically retries or resumes an existing attempt.

The exact scientific scope, resource assumptions and limitations are in
`docs/go2_extended_return_budget_maze02_v1_2026-09-12.md`. This is preparation
for a reused-layout experiment, not evidence of a completed return,
independent navigation, real-time execution or hardware performance.

## Observed verification

All invocations used the existing Genesis environment, deterministic
single-thread CPU environment, `PYTHONDONTWRITEBYTECODE=1`, Python `-B`, and
pytest `-q -p no:cacheprovider` with only the named test files.

| Focused invocation | Observed result | Session |
| --- | --- | --- |
| Native inputs | 18 passed in 2.35 s | 91919 |
| Whole-worker resources and native result acceptance | 34 passed in 3.45 s | 14975 |
| Actual worker orchestration with synthetic collection/audit | 9 passed in 2.77 s | 25311 |
| Parent launcher and final acceptance | 23 passed in 2.29 s | 46014 |

These were the first observed invocations for these new test groups and
all exited zero. No failed invocation was discarded or retried. The tests
use synthetic evidence and injected failures; they do not execute a native
scene. Result-acceptance tests perform the actual raw-prefix comparison on
synthetic archives and a real synthetic prefix-replay receipt, with raster
decoding replaced by the existing synthetic fixture. Phase resource checks
are independently covered by their previously recorded tests. Worker and
parent tests replace expensive collection/audit or process execution to
exercise ordering, artifact closure, late errors, outcome flags and dispatch
gates. They do not establish actual long-run memory or disk consumption.

Source-only preflight, session 50535, exited zero and printed
`EXTENDED_RETURN_NATIVE_SOURCE_PREFLIGHT 2675`. It validated the complete
recursive source/test ancestry without input admission, model construction,
native scene execution or output creation. Available RAM at that snapshot
was 66,847,571,968 bytes and artifact storage 568,374,423,552 bytes. Source
preflight is not resource admission; actual dispatch still requires 64 GiB
available RAM, the declared disk headroom and idle predecessor processes.
Short tests overlapped the original live replay; its timing remains
nonisolated development timing.

## Source identities

| Exact path | SHA-256 |
| --- | --- |
| `scripts/extended_return_budget_native_inputs_development.py` | `6b1f76825b4c175a1322673079a9ed823e1dbe8ccde13bd34451c652545469b0` |
| `scripts/extended_return_budget_worker_resources_development.py` | `98881df3a2b894529683a5b1d2360b5061df771afefffb1bb6d8af12435dabe4` |
| `scripts/extended_return_budget_native_result_development.py` | `c57e3835339cf73c7af84debe870d5fe610f91795d811212723da21ad950a7b3` |
| `scripts/run_go2_extended_return_budget_maze02_v1.py` | `c89a87bee5b389d46facecc8393c8c75e1b688e3a8f7afa516efd5f66b7747e1` |
| `lewm/tests/test_extended_return_budget_native_inputs_development.py` | `b54fc9db99463e580e9094d182a6879158483f454737fd50ea7e73c6dd1d5b6f` |
| `lewm/tests/test_extended_return_budget_worker_resources_development.py` | `62ea85f32c3d5203fcaa915437f9aa40e62270b02c59b650149dbe241498da2e` |
| `lewm/tests/test_extended_return_budget_native_result_development.py` | `2be672415f61eb723059bf0459f7f2a007688b520c884b2fecf1ca9d67b1adbe` |
| `lewm/tests/test_extended_return_budget_native_worker_development.py` | `409e3c85921d338720b8d32ccd424d0a8c2e4ab77805f82906cfb6f0d64f7f78` |
| `lewm/tests/test_extended_return_budget_native_launcher_development.py` | `f4e5791b1399372ae73847ecf091cd39dbe3554145371c2c7a085b938bbd2aa2` |
| `docs/go2_extended_return_budget_maze02_v1_2026-09-12.md` | `5cb55c1e1d14be203611ad9bffbed2ec676afe24e161a3433357574fdc79175f` |

All ten paths are outside the active replay's frozen 2,639-source roster.

## Next actual execution

Continue observing the exact live comparison owner. Once it has ended,
authenticate its actual completed result and run the already prepared
prospective budget-prefix launcher with that result SHA. Preserve any
negative or terminal result without bypass. Only a completed positive
budget-prefix proof admits this fresh native run:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONPATH=.:lewm_genesis:lewm_worlds OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled .generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B scripts/run_go2_extended_return_budget_maze02_v1.py --controller-prefix-result-sha256 ACTUAL_COMPLETED_PREFIX_RESULT_SHA256
```

The uppercase argument documents a future value, not a supplied result or
queued command. The larger goal remains active and incomplete.
