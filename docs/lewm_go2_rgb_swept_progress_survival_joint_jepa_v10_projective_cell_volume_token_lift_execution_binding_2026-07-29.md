# RGB Swept-Progress Survival Joint-JEPA V10 Projective Cell-Volume Token Lift — Execution Binding

- Date: 2026-07-29.
- Status: authorizes exactly one fresh development attempt at the bound output
  root and command below. Reserving `attempt_v1` consumes the attempt. No
  retry, resume, second seed, extension, or replacement is authorized by this
  binding.

## Frozen authority and source

- Preregistration commit/file SHA-256:
  `b9eaae6560c42e588c86fb8bf949cc95bd9e29e9` /
  `78533b69ecf3d6ae38b8b2448a6c73ce38a05f6b79bdf33d2ee5975f7ac476eb`.
- Frozen V10 source-and-test commit:
  `8a239d2c9a7d602533cd76545b32a9672d187b48`.

| Frozen file | Bytes | SHA-256 |
|---|---:|---|
| `lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift.py` | 19,203 | `68ade72ef4293bd23136ad739c269af360a962c901216c996c2247d494a88196` |
| `scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift.py` | 37,750 | `008af6a1956ae13ce85a37d854f62a2969fecec785d4be83ebf4bab26ad88d7d` |
| `lewm/tests/test_geometry_anchored_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift.py` | 16,483 | `94b0f3d68c014a40350cb15532bc3aaee7c8f0459aa58f592983e181fc70814f` |
| `lewm/tests/test_execute_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift.py` | 9,373 | `327ba21879f95f8587d30cbd660d6a6ffd5689b2e44822fa580a4514ae53e877` |

## Verification and review

- Focused V9 regression plus V10 model/executor tests passed `30/30` on CPU
  with accelerators hidden.
- Independent model review returned PASS with no blocker. It checked the exact
  25-point geometry and ordering, mask count/hash, masked mean and attention,
  null/UNKNOWN behavior, isolated initialization RNG, target freeze/EMA, and
  source-only gradient activity.
- A separate independent executor review returned PASS with no blocker. It
  checked fresh N320-to-clean-V4 construction, V9 parameter identity, exact
  unchanged joint objective/data/schedule/gates, the 1,000-update and
  16,000-presentation cap, V10 `cell_valid_mask` routing, write-once failure
  receipts, and that physical calibration is staged metadata only.

## Runtime preflight

- `HIP_VISIBLE_DEVICES=0` exposes one `AMD Radeon AI PRO R9700` with
  `34,208,743,424` bytes VRAM; preflight use was `727,662,592` bytes and 0%.
- Runtime is PyTorch `2.14.0.dev20260726+rocm7.1`, HIP `7.1.52802`.
- Interpreter resolves to `/usr/bin/python3.12`; invoked runtime-link SHA-256
  is `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`.
- The exact V10 `attempt_v1` path was absent and not a symlink.

## Exact attempt

- Output root:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift/attempt_v1`.
- Exact command from the repository root:

```text
HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v10_projective_cell_volume_token_lift.py
```

- Hard cap: 1,000 optimizer/EMA updates, 4,000 B=4 microbatch graphs, and
  16,000 ordered presentations. The update-1000 state is the only decision
  state.
- Exit zero means all 24 development checks passed and licenses only a
  separately frozen physical-calibration attempt. Exit two is a valid
  scientific failure and closes V10. Any other exit is an operational failure
  requiring receipt audit; this binding grants no replacement.
- No outcome here opens G2, navigation, held-out, sealed, deployment, or
  promotion access.
