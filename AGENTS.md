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
