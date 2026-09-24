# Post-action projective-support selection admissibility census V1 authorization

Date: 2026-07-28

## Purpose

V4 stopped before RGB or training because its original informative-state mask
required every non-HOLD action to have a feasible immediate primitive and blind
bridge. That rejects many useful maze decision points. This authorization permits
one small model-free screen of the scientifically corrected target:

`admissible_prefix(a) = remote_safe_prefix(a)` when both the immediate primitive
and blind bridge for `a` are feasible, otherwise `0`.

A state is proposed-informative when its best non-HOLD admissible prefix is
positive and its eight non-HOLD admissible prefixes contain at least two distinct
values.

## Frozen source

- source commit: `99d93172e5f4ae16b6af62890646362e62df092d`
- diagnostic: `scripts/diagnose_go2_post_action_projective_support_selection_admissibility_v1.py`,
  14,316 bytes, SHA-256
  `52c2f94a2aa528b0c285d388f4ea73691b67e0e40fdabc017de2957946ef5737`
- focused test: `lewm/tests/test_diagnose_go2_post_action_projective_support_selection_admissibility_v1.py`,
  3,885 bytes, SHA-256
  `b63b2d816319a4b04021ecb13e7dc43b9d5aa3831a4bee2573e9927bf15ae741`
- focused test result before authorization: `1 passed`
- exact V4 execution binding:
  `docs/lewm_go2_post_action_projective_support_labels_v4_execution_binding_2026-07-28.json`,
  113,633 bytes, file SHA-256
  `ec767a116cf9d0c231c6f7e5f18d6f6a9c6bb10206eea50e4063604fe707743a`,
  content SHA-256
  `d0870f343e7a379a9627712ab2988a82e68ae35f0f5379f0c1ec753ce6bd1d86`

V4 terminal failure is bound by reservation content SHA-256
`fdc4004e818cca0d12a192b4920cb9b925f4bf33e2cf645ad12af7da3d9916c`,
failure content SHA-256
`6c5986d28f44ee6704905c9a82df6fa582e5b73b62b6dfc90b4c80289fd474c7`,
and preflight-failure content SHA-256
`880b779d4bdd12918313def446f1305bbf1267802d1cc6a3ae1458ad27b7c562`.

## One-shot authority

Authorize exactly one invocation of:

`/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python -I -B scripts/diagnose_go2_post_action_projective_support_selection_admissibility_v1.py`

It may read only the exact V4-bound raw manifest, pairs, endpoints, audit,
geometry contract, directional-footprint policy, primitive registry, and the
render summary, source-frame JSONL, and scene manifest for the exact eight
`checkpoint_selection` scenes. It may use the already reviewed V4 join and
geometry-label functions. Output is one self-hashed aggregate JSON line on
stdout. No diagnostic file, label file, cache, receipt, or other filesystem
output is authorized.

The invocation must not open the schedule, train or calibration scene sources,
RGB/image bytes, V4 label/output roots, models, tensors, checkpoints, GPU/runtime
outputs, navigation, G2, held-out, sealed, or production material. No retry or
resume is authorized by this document.

## Pre-result decision rule

The selection screen passes only with at least 128 proposed-informative states in
total and at least eight in every one of the exact eight registered families.
Otherwise this target stops before RGB/GPU.

A pass does not itself authorize training. It justifies implementing the corrected
target consistently in dense loss, ranking, action choice, and evaluation. Before
GPU use, the full model-free preflight must still prove the existing floors of 512
informative train states, 128 calibration states, 512 informative presentations
in the frozen 16,000-presentation prefix, and at least 32 unequal-prefix ranking
participations for every non-HOLD action.
