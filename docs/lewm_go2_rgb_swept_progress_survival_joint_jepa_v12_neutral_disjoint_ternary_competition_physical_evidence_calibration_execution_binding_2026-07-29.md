# V12 Neutral Disjoint Ternary Competition Physical Calibration — Execution Binding

- Date: 2026-07-29.
- Status: authorizes exactly one inference-only physical-calibration attempt.
  Reserving `attempt_v1` consumes the attempt; no retry, threshold relaxation,
  refit, alternate calibration, or replacement is authorized.
- Physical preregistration / terminal-result / frozen source commits:
  `c63e98162a1b03a33225e6e0a04b67a357c7ed89` /
  `c25b27cea61baf8ec2625f5995b59ce6d15e1dcb` /
  `3a7f1ab0d002b159b9e57fa143faba324b81f278`.
- Physical preregistration file SHA-256:
  `c23ad76797e6449f27e64ab27c912b09bc60dc992cc19c4bfaf5c4f6475ce5f1`.

## Frozen implementation

| File | Bytes | SHA-256 |
|---|---:|---|
| `lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence_adapter.py` | 38,695 | `96060ad821050e9958a9cef8383b0cc3f44206b4d53874f93989ace4ce057171` |
| `scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence.py` | 35,382 | `c9e3dbfc4e8ef99706e170431733f0e9254cc17d2629054f9752f795a6903e22` |
| `lewm/tests/test_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence_adapter.py` | 13,311 | `c7ac61f59731101c1c7a5b0fbb28b32202007f9faa2765a8dafdca4a5522cc5f` |
| `lewm/tests/test_calibrate_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence.py` | 20,019 | `960854fb6c78d3631152ff0359c23fdd30c831edbef5f50bdaeb95bcfd686a8e` |

- The runner validates all 12 frozen non-self sources, including the V12 model,
  final adapter hash, and exact inherited V4 source closure, before candidate
  or development-payload access.
- The adapter performs one CPU `weights_only=True` load, strict bit-exact V12
  reconstruction, full freeze, exact counters, disjoint height-role geometry,
  neutral ternary algebra, and branch/semantic activity validation.
- V4 data-boundary, role collection, one calibrator fit, 2,016-tuple threshold
  selection, metrics, gates, serialization, and access procedures are direct
  object aliases. All 64x64 cells remain in the unchanged populations.

## Verification and review

- New adapter/runner tests passed `14/14`; the combined frozen V4/V9/V10/V12
  physical-calibration and V10/V11/V12 model/training/executor regression set
  passed `101/101`.
- Independent adapter/runner review returned PASS after one required source-
  only correction: branch activity requires 14 tensors active from update 1,
  while the registered semantic receipt correctly has 8 active at update 1
  and all 12 by update 2. The final adapter and synthetic receipt now bind
  those exact values.
- Review checked result-before-checkpoint admission, one checkpoint read/load,
  exact V12 state and neutral algebra, direct V4 procedure aliases, 415/495
  rows, 2,016 tuples, write-once behavior, terminal failure receipts, and
  authority limited to physical calibration and possible G2 preparation.
- No model-state deserialization, training trace, calibration/selection
  payload, accelerator, G2, navigation, held-out, or sealed material was
  opened during implementation or review.

## Candidate and exact attempt

- The runner first admits only the exact 74,226-byte V12 `result.json` with
  file/content SHA-256
  `8268cabd23b57c66597c8ffd0f0b18b3eb296e9887acbc81363a666b70ff6ab6` /
  `6a6a4ef0d8545b1510f9830cb35ebf67ea3e8cdff25006b889b2ef6d0511feff`.
  Only after complete validation may it read and deserialize the
  29,676,571-byte checkpoint with SHA-256
  `8212925759c0f496b0b6b1690168391d497c13688ba3cbb47b57640d173fe33f`.
  The training trace remains closed.
- Fresh output root, confirmed absent and not a symlink:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence_calibration/attempt_v1`.
- Interpreter resolves to `/usr/bin/python3.12`; invoked runtime-link SHA-256
  is `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`.
- Execute exactly once from repository root:

```text
/home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition_physical_evidence.py
```

- Exit zero means all four physical gates passed and opens only preparation of
  a separately bound G2 run. Exit two is a valid scientific failure and closes
  V12. Any other exit is operational and requires receipt audit; it does not
  authorize a rerun.
- No training, predictor recomputation, optimizer/backward/EMA, accelerator,
  G2, navigation, held-out, sealed, promotion, deployment, or scientific retry
  is authorized here.
