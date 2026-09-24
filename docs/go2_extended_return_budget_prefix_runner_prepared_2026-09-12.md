# Longer-budget prefix runner prepared and source preflight passed

The prospective recorded-prefix runner, closed-output checker, input admission
and exclusive launcher are implemented. They have not executed an actual
recorded prefix. The source-only preflight passed with **2,656 bound paths**,
and the prospective output does not exist. The original chained/single-pass
comparison must finish and yield an authenticated result before dispatch.

## Reviewed source identities

| File | SHA-256 |
| --- | --- |
| `scripts/extended_return_budget_prefix_replay_development.py` | `e663499baeb832526552ecb2f0518bfcb9d797e30c07fb3ced3f18ec99fee52b` |
| `lewm/tests/test_extended_return_budget_prefix_replay_development.py` | `8c6539043767b3bf58000123a7331130a5bc01989e8bdc70a9963688f0c9c38e` |
| `scripts/extended_return_budget_prefix_inputs_development.py` | `53a286c637cbaf94a99e9286135f3abffb9129bff807cdb5e846d7239026f877` |
| `lewm/tests/test_extended_return_budget_prefix_inputs_development.py` | `8d60bd3de369c47b5fb26550b4009c207db2a4d02330cb3cb1a980eb61cdd01a` |
| `scripts/replay_go2_extended_return_budget_controller_prefix_v1.py` | `03b27b1c8394bbeba351c48faf1c61081e7b1bdc34730c09418d319a5f949fc2` |
| `lewm/tests/test_extended_return_budget_prefix_launcher_development.py` | `c258da736e9370fbb9fa3ab5be1563600baa8682d4cdd4eacba997cc05a82917` |
| `docs/go2_extended_return_budget_controller_prefix_v1_2026-09-12.md` | `fb97e4c8d0102633ed187c333c6891eaa2c5a8926ccc9a96bfa200b33b01b8b4` |

The active comparison's **2,639** original source bindings were independently
rehashed afterward and matched. All seven listed files are outside that live
roster. The new prefix closure retains that ancestry and adds the longer
controller, its comparison rules and the required tests/protocol.

## Validation evidence

The replay/checker suite passed **26 tests in 3.37 seconds**, tool session 1826.
The input-admission and launcher suites passed **24 tests in 3.40 seconds**,
tool session 76431. These were their first observed invocations and had no
failures. All used the original deterministic single-thread environment and
`pytest -q -p no:cacheprovider`.

The orchestration fixtures use synthetic models, controllers and packets.
They shorten the prospective intervention to five observations while retaining
the real comparator, stream, checkpoint and output-checker behavior. They
prove that the stopping row is retained and the following packet is never
decoded, including during closed-output reconstruction. They do not prove
actual recorded-history equivalence. The separate real synthetic image-to-action
and 4,004-observation mission/comparator tests are documented in
`docs/go2_extended_return_budget_prefix_comparison_preparation_2026-09-12.md`.

The runner tests also reject shared model storage, changed public inputs,
model calls, retained state, weights and articulated geometry; remove hooks
on failure; preserve an early decision intervention as a negative comparison;
reject incomplete/extra saved rows, altered packet/clock/endpoint/decision
bindings, missing checkpoints/identities/resource records and changed reports;
enforce exclusive artifacts and the output-size limit; and reject a wrong
admission before creating models or reading packets.

Admission tests use complete synthetic 4,014-row timing accounting with actual
file hashes. They reject a live owner, terminal failure, wrong result/scope,
missing or changed artifacts, source-ancestry mismatch, incomplete population
or state checkpoints, and a different native admission. Launcher tests cover
source-only behavior, resource/CPU/admission rejection before output creation,
complete positive and negative result binding, input reauthentication,
terminal failure preservation and rejection of a second attempt at the same
output. The actual CPU-idle/resource functions are reused from the already
tested chained timing launcher.

The source preflight, tool session 43534, exited successfully and printed
`EXTENDED_RETURN_PREFIX_SOURCE_PREFLIGHT 2656`. It reported 16 physical and
32 logical CPUs, 72,501,346,304 bytes available RAM, 568,350,056,448 bytes free
on artifact storage and 21,209,579,520 bytes free in the workspace. These are
a preflight hardware snapshot, not runtime admission. The active comparison
was visible as PID 3015121, so no second full CPU replay was dispatched.

These short tests and source preflight overlapped the already declared
non-isolated timing comparison. No isolated timing result is claimed.

## Execution requirements and next work

The fixed prospective output is
`go2_extended_return_budget_controller_prefix_v1_attempt_001` under the existing
owned artifact root. Dispatch uses the actual completed chained timing result
SHA via `--completed-chained-timing-result-sha256`. There is no queued process,
automatic waiter or placeholder result binding.

Admission reconstructs the completed timing result's full scalar accounting
and authenticates its bound packet-verification result. It then reauthenticates
the original native artifacts and reconstructs the original physical prefix.
The new replay independently reconstructs its consumed public packets and
reproduces the complete original decisions. Its closed-output checker repeats
packet reconstruction through the stopping row. Both original admissions are
reauthenticated afterward without rerunning the prior timing controller pair.

The protocol specifies a 4,000-step baseline and fresh 8,000-step candidate,
two independent unchanged assigned models, initial/fixed/stopping state checks,
and stopping on any normalized decision difference through frame 4003. A
successful prefix comparison still cannot establish the physical prefix of
a new execution or the outcome of the changed command.

Next, finish and authenticate the active timing comparison, then execute this
prepared prefix. Separately finish the longer native resource assessment and
prepare its fresh protocol/launcher and actual physical-prefix audit. No new
native trial, completed return, independent-maze reliability, learned planning
or memory advantage, real-time result or hardware qualification is established
by this preparation.
