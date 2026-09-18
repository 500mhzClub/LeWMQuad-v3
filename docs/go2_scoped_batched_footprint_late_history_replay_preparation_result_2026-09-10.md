# Full incremental scoped/batched replay prepared

The new harness compares scoped-only and scoped-plus-batched controllers over
the same 1,428 original observations. It reuses the existing paired loop's exact
code in an isolated namespace, without modifying imported modules. It requires
all 1,425 forecasts, original command endpoints and complete normalized decisions,
unchanged public inputs and model states, independent model storage, and seven
retained-state checkpoints matching both controllers and the completed scoped
predecessor. Only two patch-store type tags may be normalized.

The final focused run passed 52 tests in 9.30 seconds (session 4033, exit zero).
These cover full-length synthetic comparisons and their failure paths, exact
loop/global isolation, completed predecessor identities and timing reconstruction,
live or reused owners, source-only execution boundaries, final input-change
failure persistence and no second attempt. The included controller composition
tests use actual Go2 collision geometry with synthetic observations. The suite
does not replay an original trained model or audit a native episode.

Actual source preflight passed with 2,153 bindings (session 35990, exit zero).
The prepared source map was reverified. Available RAM, disk and physical CPUs
met the inherited pair envelope. The actual predecessor owner remained live and
the runtime gate correctly rejected it with `original scoped replay remains live`.
No new artifact root, model replay or simulator scene was created.

[Preparation record](go2_scoped_batched_footprint_late_history_replay_preparation_2026-09-10.json)
has SHA-256
`36b447e2ed4684e8a53b3980defb5fc3cfe7bb79eaee68297f3b0c1cfd1838fb`.

The source-only harness can now wait for the scoped replay to finish. Authenticate
its terminal result and review its full timing, decision and state evidence before
deciding to execute the combination. The command requires that exact result SHA:

```text
scripts/replay_go2_scoped_batched_footprint_late_history_v1.py --scoped-result-sha256 <completed-scoped-result-sha256>
```

Use the established single-thread environment and exact repository Python.
Never pass a launch SHA in place of the completed result. The pending root is
`go2_scoped_batched_footprint_late_history_v1_attempt_001`; no retry or resume is
permitted. Full original input admission precedes output creation and is repeated
after replay. It is not a new simulator policy or a navigation qualification.

At the concurrent status check, the scoped replay reached frame 525, beyond its
395/404 state checkpoints. The fifth simulator case reached observation 2908.
Both original processes remained live and had no terminal result or failure.
