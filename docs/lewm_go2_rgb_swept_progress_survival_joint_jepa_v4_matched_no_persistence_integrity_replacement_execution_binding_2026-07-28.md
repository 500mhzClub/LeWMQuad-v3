# RGB Swept-Progress Survival Joint-JEPA V4 Matched No-Persistence — Integrity Replacement Execution Binding

- Status: frozen before replacement GPU execution.
- Replacement preregistration commit: `d5c25a3b11181aba29a2c96e9954c09c19b8f1ad`.
- Frozen source commit: `222550a4c26c7256b92d3d21ead03850f7b30ce2`.
- Three independent result-invalidating source reviews: PASS; no blocker.
- Full relevant V1/V2/V3/V4/control/replacement suite: 89 passed.
- Focused replacement tests, compatibility tests, whitespace checks, and isolated executor `--help`: PASS.

## Frozen replacement source

- Replacement executor SHA-256: `d2cc1781beae234df0964713b44fc74ce5baeb314a2172ebfb48903f28a9c2e0`.
- Replacement executor-test SHA-256: `abebde301c6540da3474f25f9380a2ae4f9f5f332b114e94fca83e27da581e6e`.
- Mechanical failed-executor-to-replacement delta SHA-256, independently bound in the test: `2fa62ea8a4b70077be6ae10e62c3e23612528a7d0b077ff30ada4a6802e8c261`.
- Byte-identical no-persistence training-core/test SHA-256: `90b66a5e4bdc7e6634db57d6852d9b3c5a187581d67a80ce81bf95fb371c34ab` / `1cb39173c8fa389abe38897ea0409b927ed7717deaa4516412d89ce0d405f647`.
- Frozen failed executor/test SHA-256: `f1e6a74c070d2db018cad120e4dcbc764f5432e4ebff1d88f179db079ad09cfd` / `684a52056fc45cbf6d04e0c9a1ff963e0add0048138acec4d38c9859809f5e69`.
- The sole executable integrity delta is redundant-validator parity: replace absolute error `<=1e-6` with independent `math.isclose(observed, expected, rel_tol=2e-6, abs_tol=2e-6)` checks for both `L_full_diagnostic=S+P_diagnostic+U+R+O` and `L_backward=S+U+R+O`.
- The distinct checkpoint, trace, result, and failure schemas are `lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence_integrity_replacement_checkpoint_v1`, `lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence_integrity_replacement_trace_v1`, `lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence_integrity_replacement_result_v1`, and `lewm_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence_integrity_replacement_failure_v1`.
- Every terminal receipt binds `science_changed=false`, both validator predicates, and the frozen provenance hashes.

## Bound failed attempt

- Original preregistration / clarification: `3dd4ca0680347f0a7f35d42d387781ecf53b1685` / `8cd4486ff8fc5e82dbfb745da1ed8d4b3a4101b1`.
- Failed-attempt source / execution binding: `4d55f6b68ac4edfa8aef93fdb3b2e4c7666f09e2` / `49d281480db196187b20c34f4cb5a61beede264a`.
- Failure-document commit: `8f6b187b52f8d7a47d33392e7ccaa242cb55e072`.
- Failure file/content SHA-256: `b2a99cf0b88c918c80690620f5f9f7ee5c891fb60cde581eabe7118d3f89c6d8` / `86ce444bba577a3744606480fb08803b67ced42e02b86cae5c22c88802d685b9`.
- Attempt V1 completed exactly 1,000 updates / 16,000 presentations but produced no checkpoint, published trace, evaluation, or scientific result. It is terminal and neither its output nor any predecessor runtime artifact is named, opened, loaded, or reused by the replacement.

## Exact scientific match

- Required reconstructed initial-state digest: `181b7cd4eef301a4986a9182940d0819b236ccf28876e471f5c30a62838112fd`.
- Required empty-optimizer digest: `f45a9c253820a4bdab542e34ef07b8975bb799b7cdce2751ba781d905a386d2d`.
- Required exact update-1 pre-step witness: `S=1.313827022910118`, `P_diagnostic=1.0`, `U=0.9792981296777725`, `R=1.0`, `O=1.026371382176876`.
- Backward remains exactly `S+U+R+O`; `P` remains fully computed and traced but absent from backward membership. Trace keys remain exactly `S`, `P_diagnostic`, `U`, `R`, `O`, `L_full_diagnostic`, and `L_backward`.
- Cap and cadence remain exactly 1,000 updates / 16,000 presentations / 4,000 backward calls / 1,000 optimizer steps / 1,000 EMA steps.
- Model/decoder, accepted N320 input, RGB/data/labels, seeds, schedule, optimizer, clipping, masks, losses with `O=0.5`, EMA, evaluator, gates, controls, bootstrap, and thresholds are unchanged.
- The embedded full-V4 family-utility reference SHA-256 remains `8ba8d6126e922f6a36038304e3444d0d21ee69350fef4acd3828265754810e1e`. Positive treatment still requires mean full-minus-control utility `>0`, paired-bootstrap lower bound `>0`, and at least 6/8 positive families.

## Runtime and authority

- Immediately before freezing this binding, GPU 0 was `AMD Radeon AI PRO R9700` with 34,208,743,424 bytes VRAM and no KFD process was running. Execution exposes it alone with `HIP_VISIBLE_DEVICES=0`.
- Fresh write-once root `.generated/go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence/attempt_v2_integrity_replacement` was confirmed absent immediately before this binding.
- Exact command: `HIP_VISIBLE_DEVICES=0 /home/andrewknowles/.local/share/lewmquad-v12-runtime-rocm711/bin/python -I -B scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v4_matched_no_persistence_integrity_replacement.py`.
- Checkpoint and trace must precede evaluation. Any produced checkpoint is development-only, diagnostic-only, unqualified, non-resumable, and must not be opened after the result.
- Execute exactly once. There is no retry, resume, second replacement, tolerance change, repair, extension, or result-conditioned intervention.
- Neither outcome authorizes G2, navigation, sealed, held-out, production, deployment, promotion, or final-evaluation access.
