# Preparation of optimized-controller phase diagnosis

The combined controller's completed paired benchmark preserves all 514 original
maze-2 decisions but reports a 567.2465515 ms active median, above the 100 ms
target. A separate CPU-only replay now measures remaining controller phases.
The older phase diagnosis measured a predecessor controller and trajectory;
its cost breakdown cannot establish the current optimized controller's costs.

The new timer classes inherit the verified combined controller, map and memory.
They retain all eight single-pass indices and the existing receipt-copy selector.
Explicit nested scopes delegate the original methods and preserve return values.
The runner requires every complete decision to equal the saved original, exact
public input fingerprints, all 514 observations and actual prior commands,
unchanged assigned model weights and absent gradients. It removes model hooks
on success and failure. It checks that exclusive phase durations partition each
full controller call, and reports warmup/terminal observations separately.

Validation: 27 tests passed in 4.98 seconds, session 89130, exit 0. Tests cover
real dual-camera mapping, the missing-RGB stop, index ownership, nested timing,
hook cleanup, complete decision and input mutation rejection, endpoint mismatch,
truncation, premature command failure, model mutation, completed-benchmark
admission and timing aggregation. The tracked diff whitespace check passed.

Frozen new source identities at submission:

- `lewm/single_pass_receipt_phase_timing_development.py`:
  `28d9ae083105117786e816b2de7a509881dd7df3cdd27975bf897d935f2bf4fd`.
- `scripts/diagnose_go2_single_pass_receipt_phases_v1.py`:
  `3461ffd198a9eef156c49f6ea9f7b7cb5b77957dd0e979b10eec22811230df82`.
- `lewm/tests/test_single_pass_receipt_phase_timing_development.py`:
  `2e1d52e3f0b5f244322adef4ef89a57b82e90dec70afe0f57e02c34e68a7a9b8`.
- `docs/go2_single_pass_receipt_phases_v1_2026-09-09.md`:
  `5675df1b9e6cacb3c0f8706e556e737b62309026566f3e4bb489ef66800ba7a0`.

Fresh hardware assessment, session 80838, exit 0: 16 physical/32 logical CPUs,
all 32 in affinity, 6.5% CPU busy, 76,757,487,616 bytes available RAM,
76,054,351,872 bytes free on the artifact volume and 21,359,009,792 on the
workspace volume. Both GPUs idle; discrete VRAM 34,208,743,424 total and
1,398,722,560 used. Live competing jobs were the single tracking scene and
the hold-reconsideration replay. Conservative planned memory allowances of
32 GiB native + 8 GiB hold + 8 GiB timing fit. The new run refreshes resources
after full input admission, before creating output, then every 64 frames.

Submitted the original new runner as session 10216 with single numerical threads,
fixed hash seed, no bytecode writes and the existing Genesis environment.
Output is exclusively `go2_single_pass_receipt_phases_v1_attempt_001` under the
external development artifact root. Submission is not completed admission;
launch and result identities remain to be recorded. No benchmark source was
edited and neither running job was restarted. This run makes no navigation,
controlled-speedup, whole-loop timing or real-time claim.
