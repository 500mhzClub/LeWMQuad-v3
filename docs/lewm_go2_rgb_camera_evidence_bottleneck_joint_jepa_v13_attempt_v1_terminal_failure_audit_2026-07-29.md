# V13 attempt V1 terminal failure audit

Date: 2026-07-29

Status: **terminal infrastructure failure; no scientific result**.

- The one-shot reservation was created at `2026-07-29T12:12:41Z` and the
  attempt was terminalized at `2026-07-29T12:12:44Z` during
  `post_reservation_runtime_composition`.
- Reservation receipt SHA-256:
  `b1141ada3a0b579e13713f1da5f86953ffa0c0cd2a27235c2b9568bc2ae15690`.
  Failure receipt SHA-256:
  `74c11bd08cbc182f0a342c2545526da3167e8a412d0dbd5eb4d54c07cf09e432`.
  Both receipts are canonical, content-hash-valid, cross-linked, immutable
  mode `0444`, and contained by the mode-`0700` attempt root.
- The exact failure was `FileNotFoundError` for
  `lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py`; the receipt's
  exception-message SHA-256 matches the independently reproduced source-only
  import failure.
- Cause: the recursive source closure followed Python imports but missed a
  module-level `Path.read_bytes()` plus `importlib` edge reached through the
  Direct-BEV compatibility runner. The file was absent from both the 64-path
  manifest and the 74-path clean-export inventory.
- Failure occurred before the narrow label loader, Raw-V13 input loader, N320
  checkpoint loader, hardware query, training controller, or first update.
  Scientific-payload bytes opened, RGB decodes, checkpoints loaded, GPU
  queries, presentations, updates, and produced checkpoints were all zero.
- `attempt_v1` is consumed. Retry and resume remain forbidden. This failure is
  not evidence for or against the V13 camera-evidence-bottleneck hypothesis.

The only admissible continuation is a separately preregistered,
science-identical integrity replacement with a fresh output root, complete
source closure, clean export, independent review, and one-shot authority.
