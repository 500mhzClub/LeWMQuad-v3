# V10 Projective Cell-Volume Lift Physical Calibration — Execution Binding

- Status: authorizes exactly one inference-only physical-calibration attempt.
  Reserving `attempt_v1` consumes the attempt; no retry, threshold relaxation,
  refit, or alternate calibration is authorized.
- Physical preregistration / terminal-result / frozen source commits:
  `6bc4dca93daf0e220bbaa4fc524470addb880e21` /
  `7ccb9cc88f1ddfa687a6d9b5cef847bbb3f11cfe` /
  `861f8377539742ce28591e64b4bdae6f430cd939`.
- Physical preregistration file SHA-256:
  `47b2844ef989a1a0ee8fe4f9061301c83d4d1ffef1056f37611c173321ec0e72`.

## Frozen implementation

| File | Bytes | SHA-256 |
|---|---:|---|
| `lewm/benchmarks/go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence_adapter.py` | 35,118 | `03b6d8e0e69e31adb1a9bf4b8227769b4440de8f589d1b066efb36a40d45b414` |
| `scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence.py` | 27,139 | `2c044593ddb8c38bc2ca673dc9c18d6d8ada5fe5ed5fd10c09c6045446858505` |
| `lewm/tests/test_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence_adapter.py` | 9,715 | `e67e619ad909107eef32a908a7ebaf9561cad79c240d6fc8f4f4c562061f0b7b` |
| `lewm/tests/test_calibrate_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence.py` | 16,543 | `ac59a20891b9dc6e03a220e6f47d0fa57574e2e6564e3160a1612134d78cd135` |

- The runner validates all 12 frozen non-self sources and the exact candidate
  result before its sole checkpoint read. The adapter performs one CPU
  `weights_only=True` load, strict bit-exact state reconstruction, full freeze,
  and V10 `cell_valid_mask` semantic validation.
- V4 data-boundary, role-collection, calibrator, threshold selection, metrics,
  gates, serialization, and access procedures are direct object aliases. All
  64x64 cells remain in the unchanged populations.

## Verification and review

- New adapter/runner tests passed `12/12`; the combined frozen V4/V9/V10
  calibration and V10 model/executor regression set passed `50/50`.
- Independent adapter review returned PASS, including one strict checkpoint
  load, byte-exact state, CPU/frozen inference, mirrored geometry, and the
  crucial volume-valid/ground-centre-hidden semantic route.
- Separate runner/custody review returned PASS, including result-before-
  checkpoint admission, 12-source closure, 415/495 rows, 2,016 tuples,
  write-once behavior, and operational-failure receipts.
- No generated candidate, checkpoint, dataset payload, accelerator,
  calibration output, G2, navigation, held-out, or sealed material was opened
  during implementation or review.

## Candidate and exact attempt

- The runner first admits only the exact 70,550-byte V10 `result.json` with
  file/content SHA-256
  `f62fa6c908fe8cfb4ae838878d40b615e14ad343d5f123c1dd24e16f274bbb70` /
  `01ce5f55d3b2cc264b21a9924d27e64568873dfaf2a2364e1448991adda0b6b6`.
  Only after complete validation may it read the 29,741,203-byte checkpoint
  with SHA-256
  `f63a037868de1e4db465fb4f85af2b8e6eba9883880c19d908216db20d82faa0`.
- Fresh output root, confirmed absent and not a symlink:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence_calibration/attempt_v1`.
- Interpreter resolves to `/usr/bin/python3.12`; invoked runtime-link SHA-256
  is `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`.
- Execute exactly once from repository root:

```text
/home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift_physical_evidence.py
```

- Exit zero means all four physical gates passed and opens only preparation of
  a separately bound G2 run. Exit two is a valid scientific failure and closes
  V10. Any other exit is operational and requires receipt audit; it does not
  authorize a rerun.
- No training, predictor recomputation, optimizer/backward/EMA, accelerator,
  G2, navigation, held-out, sealed, promotion, deployment, or scientific retry
  is authorized here.
