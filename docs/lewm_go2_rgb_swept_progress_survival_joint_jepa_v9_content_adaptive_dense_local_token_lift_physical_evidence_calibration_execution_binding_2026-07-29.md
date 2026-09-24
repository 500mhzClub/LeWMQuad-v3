# V9 Dense Local-Token Lift Physical Calibration — Execution Binding

- Status: frozen after implementation/tests/review and before candidate
  checkpoint or calibration/selection payload access.
- Preregistration / source-closure amendment / frozen source commits:
  `2f561d26f0b6ca154b6f4eab00dba228f8bc8c9e` /
  `b2465b2148b999b216078d53fe9bd556e63703e0` /
  `2f978d3783223f7aed77355f510c5dead27f7627`.
- Candidate terminal-result commit:
  `8a4f335de08884ec4dcc81325234ee69ce164e63`.

## Frozen implementation

- Adapter SHA-256:
  `404d9d912ac38e3a75c45ef188739718ca0e758afedb0eef50722f09b1841c22`.
- Runner SHA-256:
  `d600ebf700b74aa68d766b67d16597bd32f1df76c39c2b440ee0962c7bcc4800`.
- Adapter-test / runner-test SHA-256:
  `37a845ee982c2adb23474ce7a0a8322e802925d60bffc2d9ed8b5c57cce4e130` /
  `53a34205e05302d44bbd0166780addcdeb71577fd1bddaa4ff44f22561a9e450`.
- The runner validates the exact candidate result before its sole checkpoint
  read, binds all 12 non-self transitive sources before candidate access, and
  directly aliases the reviewed V4 data boundary, role collection,
  calibration, threshold selection, gate, and receipt helpers.
- The runner's own non-circular binding is this external source commit and
  SHA-256. Immediately before launch, require the recorded SHA and no diff for
  the four frozen implementation/test files against source commit
  `2f978d3783223f7aed77355f510c5dead27f7627`. This completes amendment
  `b2465b2` without self-hashing machinery.

## Verification and review

- Focused V9 adapter tests: 5/5 passed. Combined new adapter/runner, frozen V4
  calibration, and complete V9 model/runner/executor regressions: 40/40 passed.
- Syntax compilation passed. All 12 embedded dependency hashes matched.
- Two independent implementation reviews passed after adding both the V9 and
  transitive V4 adapters to the pre-candidate source closure. A separate audit
  found no scientific or custody blocker once the runner is externally bound
  here.
- No generated artifact, candidate checkpoint, dataset payload, GPU, G2,
  navigation, held-out, or sealed material was accessed during implementation,
  testing, or review.

## Candidate and output boundary

- The runner may first read only the exact 69,002-byte V9 `result.json` with
  file/content SHA-256
  `698acce34e9221e1660d243133937b621abc6742a5436a859c91b7ffbf55c7e5` /
  `344d10db882314fa3f227597dba4fc7e96747e3fdbe3f6d134e6c7f28c5c2c28`.
  Only after complete receipt validation may it read the bound 25,427,815-byte
  checkpoint with SHA-256
  `5456dc94136503543439e4bf691b8120c63c45a04e692f640c9c246f243c5ffd`.
- Require the fresh output root
  `.generated/go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_physical_evidence_calibration/attempt_v1`
  to be absent before launch.
- On pass or scientific failure, only `calibration.json` and `result.json` are
  terminal outputs. On operational failure, `failure.json` is authoritative;
  no retry or resume is authorized.

## Sole command and stopping rule

- Interpreter:
  `/home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python`,
  resolving to `/usr/bin/python3.12` with SHA-256
  `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`.
- Execute exactly once from repository root:

  `/home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/calibrate_go2_rgb_swept_progress_survival_joint_jepa_v9_content_adaptive_dense_local_token_lift_physical_evidence.py`

- Exit `0` means the physical gate passed and only preparation of a separately
  bound G2 run opens. Exit `2` is a valid scientific failure and closes V9.
  Any other exit is operational and requires receipt audit, not an automatic
  rerun.
- No training, optimizer, backward, EMA, predictor recomputation, accelerator,
  G2, navigation, held-out, sealed, promotion, deployment, or scientific retry
  is authorized by this binding.
