# Repository custody instructions

These instructions apply to the entire repository.

## Sealed benchmark material

- Never open, print, parse, summarize, index, or recursively search a
  `sealed_test.json`, a `sealed/` directory, or a `sealed_*` directory.
- Treat every legacy sealed role as inaccessible even when it is already
  scientifically invalid. In particular, V4 is development-only and
  permanently ineligible for final evaluation.
- Ordinary source discovery must honor the tracked `.ignore` rules. Do not use
  `rg -u`, `rg --no-ignore`, `grep -R`, `git grep`, IDE-wide indexing, or an
  equivalent bypass across a custody root.
- When a search tool or command does not honor `.ignore`, give it explicit
  exclusions for `**/sealed_test.json`, `**/sealed/**`, and `**/sealed_*/**`.
- Do not use a whole-tree `git archive`, worktree export, checkout copy, source
  package, or equivalent clean-tree materialization while legacy sealed blobs
  remain tracked. Clean source verification must export only (a) the 72
  validated paths in the committed source-closure manifest and (b) the nine
  checker, guard, and synthetic-test paths explicitly enumerated and
  SHA-256-bound in
  `docs/lewm_go2_g2_runner_source_closure_v1_clean_export_certification_2026-07-24.json`.
  Validate every exported path against its binding before copying.
- The RGB multiresolution perception V1 source may additionally be clean
  exported only as (a) the 35 paths in the committed V1 recursive source
  manifest and (b) the ten custody, review, checker, and test paths explicitly
  enumerated and SHA-256-bound in
  `docs/lewm_go2_rgb_multiresolution_perception_v1_clean_export_certification_2026-07-24.json`.
  This exception grants no generated-input, checkpoint, training, GPU,
  qualification, navigation, held-out, production, or promotion authority.
- Filename-only checks may verify that guards exclude protected paths, but
  must not read file contents.
- No future active G8 manifest belongs in the model-facing checkout. It must
  remain in a custodian-owned external root and may be accessed only by its
  reviewed, frozen, one-shot launcher.
- If any command may have read protected bytes, stop immediately and record
  the exact command, path, exposed fields, and downstream recipients. Do not
  infer that an uncommitted incident record is scientifically irrelevant.

Ignore files are defense in depth, not an access-control boundary. Final-test
custody requires operating-system isolation and a fail-closed one-shot
launcher.
