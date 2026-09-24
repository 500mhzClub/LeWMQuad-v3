# Swept-progress selection census V1 authorization

Date: 2026-07-28

Authorize one model-free execution of the target frozen at commit
`92920cf6598271af539f586a834088464cc9eb68`.

## Source binding

- implementation commit: `31c6162b87152426c7f2f38eb13d36381908a6be`
- `scripts/diagnose_go2_swept_progress_selection_v1.py`: 14,548 bytes,
  SHA-256 `30d026696c7c55a6df8e67a2482cebbf64e30e4722bbdbbaf1e4d873b7ad9e8e`
- `lewm/tests/test_diagnose_go2_swept_progress_selection_v1.py`: 4,940 bytes,
  SHA-256 `71ed8757d5ed2ec08f5764c6e53c6a4a8d1b35aa2173759f418760b858272bdf`
- focused validation: `3 passed`

## Exact invocation and scope

Run exactly once:

`/home/andrewknowles/.local/share/lewmquad-v12-runtime-torch291-rocm64/bin/python -I -B scripts/diagnose_go2_swept_progress_selection_v1.py`

Allowed reads are the exact V4-bound raw manifest, pairs, endpoints, audit,
geometry contract, directional policy, primitive registry, and render summary,
source-frame JSONL, and scene manifest for only the eight checkpoint-selection
scenes. The program may use the already reviewed join, immediate-primitive, and
footprint-sweep functions. It emits one self-hashed aggregate JSON line on stdout
and writes nothing.

Schedule, train/calibration source records, RGB/image bytes, labels/V4 outputs,
models, tensors, checkpoints, GPU/runtime outputs, navigation, G2, held-out,
sealed, and production material are forbidden. No training or promotion is
authorized. An operational stop consumes this source invocation; any obvious
science-identical integrity correction requires a separately recorded replacement.

The pre-result decision remains PASS only at 128 informative states overall and
8 in every registered family. A STOP closes swept-progress survival before model
work. A PASS permits only the complete model-free target preflight and reviewed
neural source implementation described by the target decision.
