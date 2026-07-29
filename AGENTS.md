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
- The RGB multiresolution perception V2 source may additionally be clean
  exported only after its source-and-review commit is frozen, and only as
  (a) the 36 paths in the committed V2 recursive source manifest and (b) the
  23 custody, authority-evidence, review, checker, test, and frozen-V1
  identity-witness paths explicitly enumerated and SHA-256-bound in
  `docs/lewm_go2_rgb_multiresolution_perception_v2_clean_export_certification_2026-07-24.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, or recursive source materialization. The copied
  V1 authorization, terminal audit, and six frozen V1 source witnesses are
  source-only identity evidence and grant no V1 runtime-output access. This
  exception grants no generated-input, checkpoint, tensor, GPU, training,
  qualification, navigation, held-out, production, promotion, deployment,
  retry, resume, or replacement-attempt authority.
- The RGB multiresolution perception V3 source may additionally be clean
  exported only from frozen source-and-review commit
  `d433e0101f96b0c67d59751918d31ed2547d36d3`, and only as (a) the 39
  paths in the committed V3 recursive source manifest and (b) the 27
  source-review, checker, test, frozen-V2 identity-witness, and
  authority/terminal/strict-evidence paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_multiresolution_perception_v3_clean_export_certification_2026-07-24.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V2 witnesses and authority, terminal, and
  strict-failure documents are source-only identity evidence and grant no
  V1, V2, or V3 runtime-output or execution authority. This exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, dataset, RGB,
  GPU, training, qualification, G2, navigation, held-out, production,
  promotion, deployment, retry, resume, replacement-attempt, or compatibility
  run authority.
- The RGB camera-evidence bottleneck joint-JEPA V13 source may additionally be
  clean exported only from frozen source-and-review commit
  `4ae0535f5b3b268250721de09dc869835bada7de`, and only as (a) the 64 paths in
  the committed V13 recursive source manifest and (b) the ten
  preregistration, manifest, source-review, checker-helper, checker, and test
  paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. This is a source-only export exception and grants no
  generated-input, runtime-artifact, checkpoint, tensor, dataset, RGB, GPU,
  training, qualification, G2, navigation, held-out, production, promotion,
  deployment, retry, resume, or replacement-attempt authority.
- The science-identical V13 integrity replacement V1 source may additionally
  be clean exported only from frozen source-and-review commit
  `c0477c7c7955e38e357dde199f60ff97608657c4`, and only as (a) the 80 paths
  in its committed recursive source manifest and (b) the ten preregistration,
  manifest, source-review, checker, and test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v1_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. This source-only exception grants no generated-input,
  runtime-artifact, checkpoint, tensor, dataset, RGB, GPU, training,
  qualification, G2, navigation, held-out, production, promotion, deployment,
  retry, resume, or further replacement-attempt authority.
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
