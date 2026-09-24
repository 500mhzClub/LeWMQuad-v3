# Receipt-copy optimization and full controller replay

Implemented `lewm/deferred_atomic_memo_copy_development.py`. It skips atomic
memo lookups while ordinary copying owns an unexposed private memo. Once any
fallback passes that memo to standard deepcopy, all atomic lookups remain
enabled, preserving custom memo overrides and value-before-key evaluation.
Aliasing, cycles, immutable leaf identities, NumPy/custom fallback and exceptions
retain the original behavior. The original copier source is unchanged.

The paired component benchmark used the two fixed serialized boundary decisions
from measured-plane controller result
`8385e643b776865a44d9271404e8c05a8acc46b37e7ff9bc4b8bf48396e93047`.
Both are approximately 3.2 MB of serialized evidence. Seven rounds alternated
arm order, with ten copies per arm per round and GC disabled for both timed
arms. Every copied serialized field matched and the source was unchanged.

| Saved receipt | Baseline median | Candidate median |
| --- | ---: | ---: |
| Original boundary decision | 12.4046385 ms | 7.7647524 ms |
| Measured-plane boundary decision | 12.3296161 ms | 7.8190487 ms |

Output is
`docs/go2_deferred_atomic_memo_copy_component_benchmark_2026-09-11.json`,
SHA-256 `a54320b92c7272a90b6c2fe406098dfc12b9e958a0083a2024e84f439e5ae405`.
Session 33967 exited 0. These are copier timings on serialized inputs, which do
not retain the original runtime alias graph. They establish no controller-level
speedup, real-time behavior or navigation improvement.

Composed the new copier into only the two existing pure footprint-copy paths
using `lewm/deferred_memo_single_pass_controller_development.py`. Memory,
registration, map, indices, observation and controller logic remain those of
the completed SinglePassBodyProjectedController. Scope/cache behavior and
independent public footprint receipts are preserved. No queued native job has
adopted the optimization.

Focused checks all exited 0:

- Copier graph and custom-memo semantics: 15 passed, 0.17 s.
- Controller composition and actual public geometry/footprints: 10 passed,
  4.66 s, session 91797.
- Complete synthetic 1,428-frame harness and corruption rejections: 16 passed,
  4.71 s, session 80850.
- Runner report/scope checks: 7 passed, 2.12 s, session 28757.

The trained full-history run is
`scripts/run_go2_deferred_memo_single_pass_late_history_v1.py` under
`docs/go2_deferred_memo_single_pass_late_history_v1_2026-09-11.md`.
It compares SinglePassBodyProjectedController against the separate
DeferredMemoSinglePassController across all 1,428 original frames, with 1,425
forecasts, complete decision equality and seven full retained-state comparisons.
It preserves the original sensing-failure limitation and times only controller
observation. No profiler is attached.

Source preflight passed with 2,419 source bindings, session 41098, exit 0.
It observed 81,226,928,128 available RAM bytes, 593,384,292,352 artifact-volume
free bytes, 21,211,721,728 workspace free bytes, CPU 3.4% busy and idle GPUs.
The previous full CPU replay owners were confirmed ended.

Input admission finished and the full paired replay is active in session
79122, PID 2910940, creation time 1789159905.04. Its argv is the original
environment interpreter, `-B`, and the runner source path. The exclusive root
is `go2_deferred_memo_single_pass_late_history_v1_attempt_001` under the fixed
artifact volume. Launch SHA-256:
`f7b6d7a7e49373f236019731454622fb2f8a473b29c0aeb826e0e0bdcbcdd8fb`.
The launch binds 2,419 sources. Last inspection confirmed frame 321 completed
with required decision equality and about 3.52 GB RSS. No terminal result has
been claimed. Do not edit bound source or restart the process.

The completion checker is
`scripts/verify_go2_deferred_memo_single_pass_completion_v1.py`. Its 13 tests
passed in 2.31 seconds, session 70758, exit 0. Tests cover complete populations,
truncation, input/decision/state corruption, invalid times/order, negative scope
and ended-owner/boot identity. A serialized synthetic payload was also checked
against the actual launch field types. The checker reconstructs all rows and
timings and reauthenticates the original raw/model inputs; it does not rerun
the observer or model.

An automatic completion waiter is registered:

- Source `scripts/await_go2_deferred_memo_completion_v1.py`, protocol
  `docs/go2_deferred_memo_completion_wait_v1_2026-09-11.md`.
- Root `go2_deferred_memo_completion_wait_v1_attempt_001`.
- Preflight session 68950 exited 0, 2,423 source bindings, exact original owner
  confirmed live.
- Launch SHA-256
  `0beb4e02fcdbc1c86b07773c88f21176c99a6ce1a21ee237f073da634c25da26`.
- PID 2911825, creation time 1789160334.29, session 55188; confirmed sleeping
  while the original replay remains live.

The waiter invokes the checker once after the exact original owner ends. Keep
the full CPU replay slot occupied until both processes end, preserve any
failure, and do not invoke a second checker in parallel. The measured-plane
fresh-navigation waiter remains separately queued behind the original native
chain. No controller-level speed result or navigation improvement is claimed
before complete verification. The full goal is active and incomplete.
