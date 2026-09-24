# Preserve V1 output failure and prepare an identical fresh replay

V1 session 69725 exited 1 with
`ValueError('compressed replay output headroom exceeded')`.
No original policy mismatch or changed command was recorded. The retained
stream has exactly 1,185 decisions, frames 0–1184, each comparison unchanged.
It is 134,243,265 bytes; the frozen stream ceiling was 128 MiB. This incomplete
run is not a full no-intervention result.

All three preserved artifacts were hashed and the complete saved stream checked
in session 98916, exit 0:

- launch.json, 24,047,504 bytes:
  `b344a274f4649943c24d0e6350c5d8cf00a2820a5cb6283ad9ba3e6dd222a725`.
- context_decisions.jsonl.gz, 134,243,265 bytes:
  `b7d386890457e4c29e91660ddf2b19c72f4ac151daf3c7713673786bd753b33d`.
- failure.json, 128 bytes:
  `548bad1eef69df9c2afd51accd3df9d2db17ef7aaf90939c3a45b990cd55e085`.

V2 is a fresh replay from observation zero, using the unchanged V1 controller,
comparator, model assignment, input episode and at-most-3,004-observation stop.
Its replay function is AST-identical to V1. The sole execution-envelope change
is 2 GiB total output allowance with a 1 GiB compressed-stream ceiling. The
complete native predecessor stream is 347,142,791 bytes, well above V1's ceiling
and roughly one third of V2's. Memory remains 8 GiB and reserve remains 40 GiB.
The V2 launcher authenticates V1's exact failed artifacts, original verifier,
all 1,185 unchanged decisions and the inherited frozen sources before admission.
It cannot resume old controller state or discard the failed attempt.

15 tests passed in 3.01 seconds, session 39970, exit 0. Tests verify exact replay
AST, unchanged scientific dependencies, all original first-change/terminal/input/
model/truncation contracts, and rejection of changed or incomplete predecessor
failure evidence. Original V1 source and policy files remain unchanged.

Frozen V2 sources:

- `scripts/replay_go2_residual_hold_prefix_v2.py`:
  `7611e9cfa354af293ddca31704eef2d8c649f966b0aa2e3ae1c1a869b90f732d`.
- `lewm/tests/test_residual_hold_prefix_v2_development.py`:
  `2293b80caf49b9e401fb35c19f09157bfeb0433cc80c85f1c8f594ef87ed2102`.
- `docs/go2_residual_hold_prefix_v2_2026-09-09.md`:
  `b7aa4e3ec22170ae40ea363e528db402ece71e6aa744106e83dd6f79d03e412e`.

Fresh hardware 78334: 81,062,461,440 RAM bytes available, 75,988,107,264 artifact
bytes free, 21,358,866,432 workspace bytes free, all 32 CPUs in affinity, 6.3%
CPU busy and both GPUs idle. Tracking native has ended; the phase diagnosis is
the only previous live job. Planned phase 8 GiB + floor diagnostic 8 GiB + V2
replay 8 GiB fits. Submission does not replace each runner's own later admission.

Submitted V2 session 13252, using completed residual native result
`55a7d5071f39337b3c9ea329e5b48320f11c8a5a9ba6e34296926006768ce466`.
It is still authenticating; no final outcome or navigation claim. Do not repoll
the closed V1 handle or restart either attempt. All failed outputs are retained.
