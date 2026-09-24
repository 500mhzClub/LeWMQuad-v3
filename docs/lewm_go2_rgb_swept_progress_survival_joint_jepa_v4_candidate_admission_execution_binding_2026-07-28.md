# RGB Swept-Progress Survival Joint-JEPA V4 — Candidate Admission Execution Binding

- Status: frozen before the sole candidate-admission execution and before any V4 runtime-artifact access.
- Candidate-admission preregistration: `b5b4ca50b50257872c9ee12a96b901710e35bac9`.
- Candidate-admission source closure: `fd3348b7c94f8f00617e19bc1b0601ffd92cce1d`.
- Admission script SHA-256: `a367fa198564d6c14b557c8ce12917d763818da27cb8ecef6720227ff34774fa`.
- Admission test SHA-256: `3d64d34f9a9947b74a15d378c70764a1c3eb90df159ee5d187ce507b1d97d1a9`.
- The fresh write-once output root `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_candidate_admission/attempt_v1` was confirmed absent immediately before this binding was written.

## Review and validation closure

- Independent source-only preregistration audit: pass; no result-invalidating blocker or preregistration mismatch found. The reviewer accessed no generated, runtime, checkpoint, data, G2, held-out, or sealed material.
- Independent concrete-bug review: no success-path blocker. Its failure-receipt counter and exact-boolean findings were corrected before source closure; checkpoint reads now use a no-follow descriptor and one in-memory read.
- Focused admission tests: `3 passed`.
- Combined admission, direct-adapter, frozen V4 model, and frozen V4 executor regression: `23 passed`.
- `git diff --check` passed for the frozen admission source and test.

## Sole execution

- Execute exactly once from the repository root with:

  `PYTHONPATH=.:lewm_worlds /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python scripts/admit_go2_rgb_swept_progress_survival_joint_jepa_v4_candidate.py`

- The script may read only the exact V4 result and the single result-bound checkpoint authorized by the preregistration. It must not list or search their runtime root.
- It must verify the result and checkpoint identities before one CPU `weights_only=True` deserialization, strict-load and smoke-test the full jointly trained model, and copy the same verified in-memory bytes into the fresh candidate root.
- No dataset, accelerator, training, optimizer, backward, EMA, calibration, G2, navigation, held-out, sealed, rejected-checkpoint, or production operation is authorized.
- There is no retry or resume. A failure receipt is terminal for this admission attempt and does not authorize another checkpoint or another access to the original V4 runtime root.
- A success grants only `pre_g2_candidate=true`; it grants no G2, navigation, promotion, deployment, training, or resume authority.
