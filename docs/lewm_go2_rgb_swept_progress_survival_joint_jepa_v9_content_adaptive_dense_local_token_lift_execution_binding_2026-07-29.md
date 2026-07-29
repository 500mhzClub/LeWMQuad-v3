# RGB Swept-Progress Survival Joint-JEPA V9 Content-Adaptive Dense Local Token Lift — Execution Binding

- Date: 2026-07-29.
- Status: authorizes exactly one fresh development attempt at the bound output
  root and command below. The scientific attempt is consumed when the command
  first reserves `attempt_v1`. No retry, resume, second seed, extension,
  checkpoint selection, or replacement is authorized for a scientific
  failure.

## Frozen authority and source

- Preregistration commit/file SHA-256:
  `47043472466e7a258ad0f0be854c05393e233db8` /
  `166c1f85f3aa1e3248ed5787a9f110cabd4c45ed0986ce1ab39373cc0afafe38`.
- Preimplementation amendment commit/file SHA-256:
  `04db6b26d46875297e3aa515fdf1d688bee2b755` /
  `0583e626b3d1a5f4385e19b641a5c29a00e64d7652fc900cc11cd616e8b68dd3`.
- Frozen V9 source-and-test commit:
  `5c70884c108fe8c6b4051249cb614a31c442f0fd`.

| Frozen file | Bytes | SHA-256 |
|---|---:|---|
| `lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift.py` | 18,096 | `eb5ac85cfe1394b946eddd5f56167066085bfa6598aaa364e15cf432c2228d0c` |
| `scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift.py` | 13,008 | `6a076e5185b1477de80eef9ce140ce38e4e0943865168bd26d533bf1ec13eb3f` |
| `scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift.py` | 40,451 | `173aa8bb55a7fc0a53b302ef5dcf3e85c31ca09acd0de2360649369be810b7fa` |
| `lewm/tests/test_geometry_anchored_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift.py` | 12,925 | `2155be2b03de4469f5bc83bd4944c3cd23e7050218240aede91bcfa53699ab33` |
| `lewm/tests/test_run_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift.py` | 9,710 | `2672432b092505bf5b7f435d322b8bc9feaa9d0e2d3ab74c8ae1104a06569920` |
| `lewm/tests/test_execute_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift.py` | 8,854 | `a584be50bcefcefce6992400f4a74ab0e2b0d8a301b393300279545af0343d68` |

## Verification and reviews

- Focused V9 model/runner/executor tests passed `19/19`.
- The complete swept-progress V1--V9 model/runner/executor regression set
  passed `157/157`.
- Independent model/runner review returned PASS with zero blockers. It checked
  5x5 grid ordering and packing, attention math/masking/scale, null semantics,
  clean-V4 inheritance, isolated RNG, target freeze/EMA, the exact seven-tensor
  16,576-parameter inventory, optimizer ownership, gradient receipts, and
  unchanged V3/V4 training delegation.
- A different independent executor review returned PASS with zero blockers.
  It checked source/authority binding, clean-V4 audit before device transfer,
  one-shot reservation, checkpoint/trace ordering, failure receipts, the
  unchanged 24-check gate, and conditional calibration authority.
- One source-only random B=4 lift forward/backward on GPU0 produced finite
  `[4,64,64,64]` output in `0.560014 s` with peak allocated memory
  `582,782,976` bytes. It opened no dataset, RGB, label, checkpoint, generated
  runtime, navigation, G2, held-out, or sealed material and wrote no file.

## Hardware and runtime preflight

- `HIP_VISIBLE_DEVICES=0` exposes exactly one accelerator.
- Device: `AMD Radeon AI PRO R9700`; total memory `34,208,743,424` bytes.
- Runtime: PyTorch `2.14.0.dev20260726+rocm7.1`, HIP `7.1.52802`.
- Interpreter path resolves to `/usr/bin/python3.12`; invoked runtime-link
  SHA-256 is
  `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`.
- The exact V9 `attempt_v1` path was absent and was not a symlink at preflight.

## Exact attempt

- Output root:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift/attempt_v1`.
- Exact command from repository root:

```text
HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift.py
```

- Hard cap: 1,000 optimizer updates, 1,000 EMA updates, 4,000 B=4
  microbatch graphs/backwards/predictor objectives, and 16,000 ordered
  presentations.
- The executor must write the terminal checkpoint and complete training trace
  before unchanged development scoring. The terminal update-1000 checkpoint
  is the only decision state.
- Exit zero means all 24 development checks passed and licenses only a
  separately frozen use of the existing development physical-calibration
  gate. Exit two is a valid scientific gate failure and closes V9. Any other
  exit is an operational failure whose receipts must be audited before any
  further decision; this binding itself grants no replacement.
- No outcome here authorizes G2, navigation, held-out, sealed, production,
  deployment, or promotion access.
