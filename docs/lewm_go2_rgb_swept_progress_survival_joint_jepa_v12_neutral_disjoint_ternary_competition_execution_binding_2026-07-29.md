# RGB Swept-Progress Survival Joint-JEPA V12 Neutral Disjoint Ternary Competition — Execution Binding

- Date: 2026-07-29.
- Status: authorizes exactly one fresh development attempt at the bound output
  root and command below. Reserving `attempt_v1` consumes the attempt. No
  retry, resume, second seed, extension, or replacement is authorized.

## Frozen authority and source

- Preregistration commit/file SHA-256:
  `ae1568e8f434d715d379eefc3eaf644369154f76` /
  `f558597d37056d12fcce6cd36f8b8911288675d08e6a0030fc31a18ec8507817`.
- Frozen V12 source-and-test commit:
  `1c18fae5325b0ab1dd6b7c4e20fa51fb411f26aa`.
- The inherited tracked dependencies below had no worktree modifications at
  binding time.

| Frozen file | Bytes | SHA-256 |
|---|---:|---|
| `lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition.py` | 8,945 | `6bcdb2b2551f0950d2abe120a9081eb6aeed19dd39207fe648bcc1d18e1c3426` |
| `scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition.py` | 37,755 | `49968c88a9a0db7340e21f9c317d33ff6069ee22aa1f8adbadfc443f6db82c1a` |
| `lewm/tests/test_geometry_anchored_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition.py` | 11,445 | `94223388fc4ff7f95c7316589b435c98c942740b7aba935069f2704fc7c0aa71` |
| `lewm/tests/test_execute_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition.py` | 10,736 | `d1d078c74614733bd47fd71a4f0761e82ab434738b971be04b51647ce6d69e5e` |
| `lewm/models/geometry_anchored_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift.py` | 27,001 | `a4ec329f9da019dd2a6dfa85650c73a74c2ac6528bce8c431ae3cf842e328ab0` |
| `scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift.py` | 27,564 | `bec2412a6e3f08009a455e9ecac2d58357835f66c4905f42768bae1769401637` |
| `scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v11_height_role_factorized_evidence_lift.py` | 48,881 | `091abe64e05b2700872997832888b3155ff91cdc2b49564135528baa94d92103` |
| `scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v3_half_occupied_safety_aux.py` | 5,497 | `7cab73752593b12b638b55710714ff956a2441e92df2fe775902472a7b69a8cb` |
| `scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v2_occupied_safety_aux.py` | 18,988 | `6f76dd5b098ff360a3ada5bbb18f74a13342f3a5212e871da6db8f5f3a5bb1bf` |
| `scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v1.py` | 23,805 | `0d0d5b30c4674ac678de8898c6a55e5103268636cbdff7dc2388461ceb38b662` |

## Verification and independent review

- The V9, V10, V11, and V12 focused model/training/executor regression suite
  passed `67/67` on CPU with accelerators hidden. The V12-only model and
  executor suite passed `11/11`.
- The model review returned PASS. It checked exact reuse of the V11 FREE and
  OCCUPIED axis objects, identical fresh V11/V12 parameter and buffer state,
  caller-RNG restoration, exact neutral `log_softmax([0,f,o])`, finite extreme
  evidence, unchanged lift and predictor outputs, disjoint role routing,
  branch-invalid evidence, exact all-invalid sentinel, and inherited EMA.
- The executor review returned PASS. It checked one call to the reviewed V11
  training helper, exact `4 x B4 / 1,000 / 16,000` accounting, unchanged data,
  loss sources and coefficients, evaluator, controls, and 24-check gate; fresh
  N320-only V12/V11/V10 source witnesses; write-once terminal ordering; and no
  predecessor checkpoint, physical, G2, held-out, or sealed access.
- An independent history and mechanism audit returned PASS after one receipt
  clarification: V12 preserves the `S+P+U+R+O` loss sources and `O=0.5`, while
  the registered semantic algebra intentionally changes their gradient
  surface. The executor records both facts and does not claim identical
  gradients.

## Runtime preflight

- `HIP_VISIBLE_DEVICES=0` exposes exactly one `AMD Radeon AI PRO R9700` with
  `34,208,743,424` bytes VRAM. Preflight allocated and reserved bytes were
  both zero.
- Runtime is PyTorch `2.14.0.dev20260726+rocm7.1`, HIP `7.1.52802`.
- Interpreter resolves to `/usr/bin/python3.12`; invoked runtime-link SHA-256
  is `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`.
- The exact V12 `attempt_v1` path was absent and not a symlink.

## Exact attempt

- Output root:
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition/attempt_v1`.
- Exact command from the repository root:

```text
HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v12_neutral_disjoint_ternary_competition.py
```

- Hard cap: 1,000 optimizer/EMA updates, 4,000 B=4 microbatch graphs, and
  16,000 ordered presentations. Update 1,000 is the only decision state.
- Exit zero means all 24 development checks passed and licenses only a
  separately preregistered physical-calibration attempt. Exit two is a valid
  scientific failure and closes V12. Any other exit is an operational failure
  requiring receipt audit; this binding grants no replacement.
- A passing fresh run supports the registered V12 mechanism as a whole. It is
  not a post-hoc claim that the algebra alone repairs the rejected V11
  checkpoint, which remains unopened.
- No outcome here opens G2, navigation, held-out, sealed, deployment, or
  promotion access.
