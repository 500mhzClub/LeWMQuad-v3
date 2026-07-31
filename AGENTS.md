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
- The science-identical V13 integrity replacement V2 source may additionally
  be clean exported only from frozen source-and-review commit
  `d7f88b006ce528d79b5fb9e063b68645693e6222`, and only as (a) the 80 paths
  in its committed recursive source manifest and (b) the eleven
  preregistration, manifest, source-review, checker, and test paths explicitly
  enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v2_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. This source-only exception grants no generated-input,
  runtime-artifact, checkpoint, tensor, dataset, RGB, GPU, training,
  qualification, G2, navigation, held-out, production, promotion, deployment,
  retry, resume, or further replacement-attempt authority.
- The science-identical V13 integrity replacement V3 source may additionally
  be clean exported only from frozen source-and-review commit
  `972dd727f0d84f90cdd90e1c43b1faa46d763fd6`, and only as (a) the 80 paths
  in its committed recursive source manifest and (b) the eleven
  preregistration, manifest, source-review, checker, and test paths explicitly
  enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_camera_evidence_bottleneck_joint_jepa_v13_integrity_replacement_v3_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. This source-only exception grants no generated-input,
  runtime-artifact, checkpoint, tensor, dataset, RGB, GPU, training,
  qualification, G2, navigation, held-out, production, promotion, deployment,
  retry, resume, or further replacement-attempt authority.
- The RGB unified ray-survival joint-JEPA V14 source may additionally be
  clean exported only from frozen source-and-review commit
  `ea9205eb4601e8b7ec6fc1c91cc28b19558476b0`, and only as (a) the 83 paths
  in its committed recursive source manifest and (b) the fifteen
  preregistration, manifest, source-review, checker, and focused V13/V14 test
  paths explicitly enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v14_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. This source-only exception grants no generated-input,
  runtime-artifact, checkpoint, tensor, dataset, RGB, GPU, training,
  qualification, probability-calibration, G2, navigation, held-out,
  production, promotion, deployment, retry, or resume authority.
- The RGB unified ray-survival joint-JEPA V15 extended-horizon source may
  additionally be clean exported only from frozen source-and-review commit
  `7c9ac5a91fdcc3a620e3e9c9bbbaa88141ca2aa5`, and only as (a) the 86 paths
  in its committed recursive source manifest and (b) the fourteen
  preregistration, frozen-V14 source/result identity-witness, manifest,
  source-review, checker, and focused V15 test paths explicitly enumerated
  and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V13/V14 preregistration, V14 clean-export
  certification, and V14 scientific-result witnesses are source-only
  identity evidence and grant no predecessor runtime-output or execution
  authority. The V15 certification itself and later one-shot authority may
  be added to the narrow export only at their exact reviewed paths after
  their respective commits and exact file-SHA-256 validation. This
  source-only exception grants no generated-input, runtime-artifact,
  checkpoint, tensor, schedule, dataset, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, production, promotion,
  deployment, retry, resume, or further-attempt authority.
- The RGB unified ray-survival joint-JEPA V15 extended-horizon integrity
  replacement V1 source may additionally be clean exported only from frozen
  source-and-review commit
  `0a42c1fc582c709375368d365a684ea94e33c40e`, and only as (a) the 86 paths
  in its committed recursive source manifest and (b) the sixteen
  preregistration, predecessor identity-witness, terminal-failure-result,
  manifest, source-review, checker, and focused V15 test paths explicitly
  enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_integrity_replacement_v1_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V13/V14 and original-V15 documents are
  source-only identity evidence and grant no predecessor runtime-output or
  execution authority. The replacement certification itself and later
  one-shot authority may be added to the narrow export only at their exact
  reviewed paths after their respective commits and exact file-SHA-256
  validation. This source-only exception grants no generated-input,
  runtime-artifact, checkpoint, tensor, schedule, dataset, RGB, GPU,
  training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, or further
  replacement-attempt authority.
- The RGB ego-motion-aligned ray-consistency joint-JEPA V16 source may
  additionally be clean exported only from frozen source-and-review commit
  `913ffec009649e347144084a8cb68a3fcc546f29`, and only as (a) the 88 paths
  in its committed recursive source manifest and (b) the fourteen V16
  preregistration, V14/V15 identity-witness, manifest, source-review,
  recursive-checker, and focused V16 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V14 preregistration and terminal V15 result are
  source-only identity evidence and grant no predecessor runtime-output or
  execution authority. The V16 certification itself and later one-shot
  authority may be added to the narrow export only at their exact reviewed
  paths after their respective commits and exact file-SHA-256 validation.
  This source-only exception grants no generated-input, runtime-artifact,
  checkpoint, tensor, schedule, dataset, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, production, promotion,
  deployment, retry, resume, recovery, extension, or further-attempt
  authority.
- The RGB ego-motion-aligned ray-consistency joint-JEPA V16 integrity
  replacement V1 source may additionally be clean exported only from frozen
  source-and-review commit
  `d9828ddffa71517734bf1abd41d2af5bbd401ed8`, and only as (a) the 88 paths
  in its committed recursive source manifest and (b) the sixteen replacement
  preregistration, predecessor identity/failure witnesses, manifest,
  source-review, recursive-checker, and focused V16 test paths explicitly
  enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v1_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V14/V15/V16 predecessor documents are
  source-only identity evidence and grant no predecessor runtime-output or
  execution authority. The replacement certification itself and later
  one-shot authority may be added to the narrow export only at their exact
  reviewed paths after their respective commits and exact file-SHA-256
  validation. This source-only exception grants no generated-input,
  runtime-artifact, checkpoint, tensor, schedule, dataset, RGB, GPU, training,
  qualification, probability-calibration, G2, navigation, held-out,
  production, promotion, deployment, retry, resume, recovery, extension,
  second-integrity-replacement, or further-attempt authority.
- The RGB ego-motion-aligned ray-consistency joint-JEPA V16 integrity
  replacement V2 source may additionally be clean exported only from frozen
  source-and-review commit
  `5a86cb20332d5cd353c0a850f8d81ca002381041`, and only as (a) the 88 paths
  in its committed recursive source manifest and (b) the eighteen V2/V1
  preregistration, predecessor identity/failure witnesses, manifest,
  source-review, recursive-checker, and focused V16 test paths explicitly
  enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_integrity_replacement_v2_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V14/V15/V16/V1 predecessor documents are
  source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, resume, or execution authority. The V2 certification itself and
  later one-shot authority may be added to the narrow export only at their
  exact reviewed paths after their respective commits and exact
  file-SHA-256 validation. This source-only exception grants no
  generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, further-integrity-replacement, or further-attempt authority.
- The RGB delayed-onset ego-motion-aligned ray-consistency joint-JEPA V17
  source may additionally be clean exported only from frozen
  source-and-review commit
  `0047383cac00df81d4292904269293e63c3e4376`, and only as (a) the 88 paths
  in its committed recursive source manifest and (b) the twenty V17 and
  predecessor preregistration/result witnesses, manifest, source-review,
  recursive-checker, and focused V16/V17 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_delayed_onset_ego_motion_aligned_ray_consistency_joint_jepa_v17_clean_export_certification_2026-07-29.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V14/V15/V16 predecessor documents are
  source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, resume, or execution authority. The V17 certification itself
  and later one-shot authority may be added to the narrow export only at their
  exact reviewed paths after their respective commits and exact file-SHA-256
  validation. This source-only exception grants no generated-input,
  runtime-artifact, checkpoint, tensor, schedule, dataset, RGB, GPU, training,
  qualification, probability-calibration, G2, navigation, held-out,
  production, promotion, deployment, retry, resume, recovery, extension,
  alternate-onset, coefficient-search, or further-attempt authority.
- The RGB object-space height-volume joint-JEPA V18 source may additionally
  be clean exported only from frozen source-and-review commit
  `8b348f60d941921ce80ef95786a8e12b915376d9`, and only as (a) the 86 paths
  in its committed recursive source manifest and (b) the seventeen V18 and
  predecessor preregistration/result witnesses, manifest, source-review,
  recursive-checker, and focused V18 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V10/V14/V15/V17 predecessor documents are
  source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, resume, calibration, or execution authority. The V18
  certification itself and later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, architecture variant, or further-attempt authority.
- The RGB object-space height-volume joint-JEPA V18 integrity replacement V1
  source may additionally be clean exported only from frozen
  source-and-review commit
  `f4ea93b5263b9231e051ec307a0b77c5474b3a2d`, and only as (a) the 86 paths
  in its committed recursive source manifest and (b) the nineteen
  replacement/original preregistration and result witnesses, manifest,
  source-review, recursive-checkers, and focused V18 test paths explicitly
  enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v1_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V18 and V10/V14/V15/V17 predecessor
  documents are source-only identity evidence and grant no predecessor
  runtime-output, checkpoint, resume, calibration, or execution authority.
  The replacement certification itself and later one-shot authority may be
  added to the narrow export only at their exact reviewed paths after their
  respective commits and exact file-SHA-256 validation. This source-only
  exception grants no generated-input, runtime-artifact, checkpoint, tensor,
  schedule, dataset, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, production, promotion,
  deployment, retry, resume, recovery, extension, architecture change,
  second-integrity-replacement, or further-attempt authority.
- The RGB object-space height-volume joint-JEPA V18 integrity replacement V2
  source may additionally be clean exported only from frozen
  source-and-review commit
  `4af4e2c6e349355b540a98443e5dcca6b37e7197`, and only as (a) the 86 paths
  in its committed recursive source manifest and (b) the twenty-one V2/V1
  preregistration and terminal-failure witnesses, predecessor identity/result
  witnesses, manifest, source-review, recursive-checkers, and focused V18 test
  paths explicitly enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v2_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V18, V1, and V10/V14/V15/V17
  predecessor documents are source-only identity evidence and grant no
  predecessor runtime-output, checkpoint, resume, calibration, or execution
  authority. The V2 certification itself and later one-shot authority may be
  added to the narrow export only at their exact reviewed paths after their
  respective commits and exact file-SHA-256 validation. This source-only
  exception grants no generated-input, runtime-artifact, checkpoint, tensor,
  schedule, dataset, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, production, promotion,
  deployment, retry, resume, recovery, extension, architecture change,
  further-integrity-replacement, or further-attempt authority.
- The RGB object-space height-volume joint-JEPA V18 command-integrity
  replacement V3 source may additionally be clean exported only from frozen
  source-and-review commit
  `0e771562d6b6ad32f5d7d146c9f7c99bfa0651d3`, and only as (a) the 86 paths
  in its committed recursive source manifest and (b) the twenty-three
  V3/V2/V1 preregistration and terminal-failure witnesses, predecessor
  identity/result witnesses, manifest, source-review, recursive-checkers, and
  focused V18 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_height_volume_joint_jepa_v18_integrity_replacement_v3_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V18, V1, V2, and V10/V14/V15/V17
  predecessor documents are source-only identity evidence and grant no
  predecessor runtime-output, checkpoint, resume, calibration, or execution
  authority. The V3 certification itself and later one-shot authority may be
  added to the narrow export only at their exact reviewed paths after their
  respective commits and exact file-SHA-256 validation. This source-only
  exception grants no generated-input, runtime-artifact, checkpoint, tensor,
  schedule, dataset, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, production, promotion,
  deployment, retry, resume, recovery, extension, architecture change,
  further-integrity-replacement, or further-attempt authority.
- The RGB object-space height-volume executed-successor semantic-grounding
  joint-JEPA V19 source may additionally be clean exported only from frozen
  source-and-review commit
  `0aa8b5670c0f7de22020d4cb290b2013d57bede0`, and only as (a) the 89 paths
  in its committed recursive source manifest and (b) the twenty-five V19 and
  predecessor preregistration/result witnesses, manifest, source-review,
  recursive-checkers, and focused V19 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v19_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V10/V14/V15/V17/V18 predecessor documents are
  source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, resume, calibration, or execution authority. The V19
  certification itself and later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, architecture variant, second attempt, or further-attempt
  authority.
- The science-identical V19 executed-successor semantic-grounding integrity
  replacement V1 source may additionally be clean exported only from frozen
  source-and-review commit
  `b1430e51bb428376bdbe00ef81ce65d971c8b436`, and only as (a) the 89 paths
  in its committed recursive source manifest and (b) the twenty-seven
  replacement/original preregistration and terminal-failure witnesses,
  predecessor identity/result witnesses, manifest, source-review,
  recursive-checkers, and focused V19 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v19_integrity_replacement_v1_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V19 and V10/V14/V15/V17/V18 documents
  are source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, resume, calibration, or execution authority. The replacement
  certification itself and later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, second-integrity-replacement, second attempt, or further-attempt
  authority.
- The RGB object-space height-volume executed-successor semantic-grounding
  joint-JEPA V20 accounting-isolation source may additionally be clean
  exported only from frozen source-and-review commit
  `1692c6029d9e772ad2a7d65447ad70fc634a7afc`, and only as (a) the 89 paths
  in its committed recursive source manifest and (b) the twenty-nine V20 and
  predecessor preregistration, terminal-failure/result, manifest,
  source-review, recursive-checker, and focused V19 test paths explicitly
  enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_grounding_joint_jepa_v20_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V19/V18 and earlier predecessor documents are
  source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, resume, calibration, or execution authority. The V20
  certification itself and later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, further accounting-isolation successor, integrity replacement,
  or further-attempt authority.
- The RGB same-action cross-scene contrastive-innovation joint-JEPA V21 source
  may additionally be clean exported only from frozen source-and-review commit
  `7071a006dda3851280fbdf030e156862c4f19ab3`, and only as (a) the 92 paths
  in its committed recursive source manifest and (b) the thirty-two V21 and
  predecessor preregistration/result identity, manifest, source-review,
  recursive-checker, and focused V21 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V20/V19/V18 and earlier predecessor documents
  are source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, resume, calibration, or execution authority. The V21
  certification itself and later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, architecture variant, integrity replacement, second attempt, or
  further-attempt authority.
- The RGB scene-action contrastive-innovation joint-JEPA V22 source may
  additionally be clean exported only from frozen source-and-review commit
  `e0697a6f2b8498ec64484b216f7366a8d7f199a5`, and only as (a) the 95 paths
  in its committed recursive source manifest and (b) the thirty-seven V22 and
  predecessor preregistration/result identity, manifest, source-review,
  recursive-checker, focused V22 test, and frozen predecessor-fixture test
  paths explicitly enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V21/V20/V19/V18 and earlier predecessor
  documents are source-only identity evidence and grant no predecessor
  runtime-output, checkpoint, resume, calibration, or execution authority.
  The V22 certification itself and later one-shot authority may be added to
  the narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, architecture variant, integrity replacement, second attempt, or
  further-attempt authority.
- The RGB action-prior-residualized wrong-scene survival-output joint-JEPA V23
  source may additionally be clean exported only from frozen source-and-review
  commit `44938145362e5accdf8e12b906bfbaa970d62f25`, and only as (a) the 98
  paths in its committed recursive source manifest and (b) the forty-seven
  V23 and predecessor preregistration/result identity, manifest,
  source-review, recursive-checker, focused V23/V22 test, and frozen
  predecessor-fixture test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V22/V21/V20/V19/V18 and earlier predecessor
  documents are source-only identity evidence and grant no predecessor
  runtime-output, checkpoint, resume, calibration, or execution authority.
  The V23 certification itself and later one-shot authority may be added to
  the narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, local-output-ranking successor, integrity replacement, second
  attempt, or further-attempt authority.
- The RGB predictor-core-protected survival-output joint-JEPA V24 source may
  additionally be clean exported only from frozen source-and-review commit
  `2b6178a4d876dc17c45fb340a4ab03ee302649b0`, and only as (a) the 101
  paths in its committed recursive source manifest and (b) the fifty-six V24
  and predecessor preregistration/result identity, manifest, source-review,
  recursive-checker, focused V24/V23 test, and frozen predecessor-certified
  inventory paths explicitly enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_predictor_core_protected_survival_output_joint_jepa_v24_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V23/V22/V21/V20/V19/V18 and earlier predecessor
  documents are source-only identity evidence and grant no predecessor
  runtime-output, checkpoint, resume, calibration, or execution authority.
  The V24 certification itself and later one-shot authority may be added to
  the narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2, navigation,
  held-out, production, promotion, deployment, retry, resume, recovery,
  extension, coefficient or onset variant, gradient-projection variant,
  further local-output-auxiliary variant, integrity replacement, second
  attempt, or further-attempt authority.
- The RGB per-row persistence-contrastive temporal joint-JEPA V25 source may
  additionally be clean exported only from frozen source-and-review commit
  `43231c689547b66de83f3cafbfac270455a7a234`, and only as (a) the 104
  paths in its committed recursive source manifest and (b) the sixty-five
  V25 and predecessor preregistration/result identity, manifest,
  source-review, recursive-checker, focused V25/V24 test, and frozen
  predecessor-certified inventory paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_v25_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V24/V23/V22/V21/V20/V19/V18 and earlier
  predecessor documents are source-only identity evidence and grant no
  predecessor runtime-output, checkpoint, resume, recovery, calibration, or
  execution authority. The V25 certification itself and later one-shot
  authority may be added to the narrow export only at their exact reviewed
  paths after their respective commits and exact file-SHA-256 validation.
  This source-only exception grants no generated-input, runtime-artifact,
  recovery-state read or execution, checkpoint, tensor, schedule, dataset,
  RGB, GPU, training, qualification, probability-calibration, G2,
  navigation, held-out, sealed, production, promotion, deployment, retry,
  resume, recovery execution, extension, alternate temporal-objective or
  coefficient variant, integrity replacement, second attempt, or further
  attempt authority.
- The RGB explicit-plan discounted-successor-state joint-JEPA V27 source may
  additionally be clean exported only from frozen source-and-review commit
  `e312c4e07b4ae56dd0e6083ef56d2529722b4ba7`, and only as (a) the 117
  paths in its committed recursive source manifest and (b) the ten V27
  preregistration, manifest, source-review, V26 result-identity witness, and
  focused V27 test paths explicitly enumerated and SHA-256-and-byte-count-bound
  in
  `docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The V26 result is source-only identity evidence and grants
  no predecessor checkpoint, runtime-output, retry, resume, or execution
  authority. The V27 certification itself and later one-shot authority may be
  added to the narrow export only at their exact reviewed paths after their
  respective commits and exact file-SHA-256 validation. This source-only
  exception grants no generated-input, runtime-artifact, checkpoint, tensor,
  schedule, dataset-payload, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, sealed, production,
  promotion, deployment, retry, resume, recovery, extension, second attempt,
  or further-attempt authority.
- The science-identical RGB explicit-plan discounted-successor-state
  joint-JEPA V27 integrity replacement V1 source may additionally be clean
  exported only from frozen source-and-review commit
  `c076d4159cad83af5c4660f7d7dbd2ac06dc2414`, and only as (a) the 117
  paths in its committed recursive source manifest and (b) the twelve
  original-V27 preregistration and terminal-infrastructure-failure witnesses,
  replacement preregistration, manifest, source-review, V26 result-identity
  witness, and focused V27 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27_integrity_replacement_v1_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V27 terminal-failure and V26 result
  documents are source-only identity evidence and grant no predecessor
  runtime-output, checkpoint, retry, resume, or execution authority. Do not
  export the original V27 clean-export certification or execution authority.
  The replacement certification itself and later one-shot authority may be
  added to the narrow export only at their exact reviewed paths after their
  respective commits and exact file-SHA-256 validation. This source-only
  exception grants no generated-input, runtime-artifact, checkpoint, tensor,
  schedule, dataset-payload, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, sealed, production,
  promotion, deployment, retry, resume, recovery, extension, second attempt,
  further integrity replacement, or further-attempt authority.
- The science-identical RGB explicit-plan discounted-successor-state
  joint-JEPA V27 integrity replacement V2 source may additionally be clean
  exported only from frozen source-and-review commit
  `84f075b2326cb11e47f5de07191951321ba21001`, and only as (a) the 117
  paths in its committed recursive source manifest and (b) the fourteen
  original-V27 and V1 preregistration/failure witnesses, V2 preregistration,
  manifest, source-review, V26 result-identity witness, and focused V27 test
  paths explicitly enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27_integrity_replacement_v2_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V27/V1 terminal failures and V26 result
  are source-only identity evidence and grant no predecessor runtime-output,
  checkpoint, retry, resume, or execution authority. Do not export either
  predecessor clean-export certification or execution authority. The V2
  certification itself and later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule,
  dataset-payload, RGB, GPU, training, qualification,
  probability-calibration, G2, navigation, held-out, sealed, production,
  promotion, deployment, retry, resume, recovery, extension, third V27
  infrastructure replacement, second V2 attempt, or further-attempt
  authority.
- The RGB object-space explicit-plan terminal-successor-state joint-JEPA V28
  source may additionally be clean exported only from frozen
  source-and-review commit
  `b54a81cf08caae022a442e776521a8b50f4e6645`, and only as (a) the 117
  paths in its committed recursive source manifest and (b) the eight V28
  preregistration, manifest, source-review, V27-V2 terminal-scientific-result
  identity-witness, and focused V28 test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_object_space_explicit_plan_terminal_successor_state_joint_jepa_v28_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V27-V2 result is source-only chronology and
  scientific-identity evidence and grants no predecessor runtime-output,
  checkpoint, retry, resume, or execution authority. Do not export any
  predecessor clean-export certification or execution authority. The V28
  certification itself and any later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule,
  dataset-payload, RGB, GPU, training, qualification, benchmark, G2,
  navigation, held-out, sealed, production, promotion, deployment, retry,
  resume, recovery, extension, endpoint-V2, alternate-endpoint, coefficient,
  gamma, second-attempt, or further-attempt authority.
- The RGB memory-role factorized joint-JEPA V1 source may additionally be
  clean exported only from frozen source-and-review commit
  `2d3934055fdc33e528fcc55e36b35df98fe488f7`, and only as (a) the 114 paths
  in its committed recursive source manifest and (b) the eleven
  preregistration, split-integrity-amendment, source-manifest, source-review,
  V28 terminal-scientific-result identity-witness, and six focused runtime
  test paths explicitly enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V28 result is source-only chronology and
  scientific-identity evidence and grants no predecessor runtime-output,
  checkpoint, retry, resume, or execution authority. Do not export any
  predecessor clean-export certification or execution authority. The V1
  certification itself and any later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule,
  dataset-payload, RGB, GPU, training, qualification, benchmark, probability-
  calibration, G2 payload, navigation, memory integration, held-out, sealed,
  production, promotion, deployment, retry, resume, recovery, extension,
  alternate seed, second attempt, or further-attempt authority.
- The science-identical RGB memory-role factorized joint-JEPA V1 integrity
  replacement V1 source may additionally be clean exported only from frozen
  source-and-review commit
  `57ddeb95f2822fa36c9a71f0741b086fb11f953c`, and only as (a) the 114 paths
  in its committed recursive source manifest and (b) the thirteen replacement
  preregistration, original-preregistration, split-integrity-amendment,
  original terminal-infrastructure-failure result, source-manifest,
  source-review, V28 terminal-scientific-result identity-witness, and six
  focused runtime-test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_integrity_replacement_v1_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V1 and V28 documents are source-only
  chronology and identity evidence and grant no predecessor runtime-output,
  checkpoint, retry, resume, or execution authority. The replacement
  certification itself and any later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule,
  dataset-payload, RGB, GPU, training, qualification, benchmark, probability-
  calibration, G2 payload, navigation, memory integration, held-out, sealed,
  production, promotion, deployment, retry, resume, recovery, extension,
  second integrity replacement, or further-attempt authority.
- The distinct RGB memory-role factorized joint-JEPA V2 source may
  additionally be clean exported only from frozen source-and-review commit
  `6c2ca35edbddb664e7a71ae4ed535fb9a69c49bc`, and only as (a) the 114 paths
  in its committed recursive source manifest and (b) the sixteen V2
  preregistration, retrieval-metadata-preflight, source-manifest,
  source-review, original-V1 preregistration, split-integrity-amendment,
  integrity-replacement preregistration, both predecessor terminal-failure
  results, V28 terminal-scientific-result identity-witness, and six focused
  runtime-test paths explicitly enumerated and SHA-256-and-byte-count-bound in
  `docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v2_clean_export_certification_2026-07-30.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied predecessor and V28 documents are source-only
  chronology and identity evidence and grant no predecessor runtime-output,
  checkpoint, retry, resume, or execution authority. The V2 certification
  itself and any later one-shot authority may be added to the narrow export
  only at their exact reviewed paths after their respective commits and exact
  file-SHA-256 validation. This source-only exception grants no
  generated-input, runtime-artifact, checkpoint, tensor, schedule,
  dataset-payload, RGB, GPU, training, qualification, benchmark, probability-
  calibration, G2 payload, navigation, memory integration, held-out, sealed,
  production, promotion, deployment, retry, resume, recovery, extension,
  second V2 attempt, integrity replacement, or further-attempt authority.
- The V18 spatial-token delay-line causal-convolution joint-JEPA V1 source may
  additionally be clean exported only from frozen source-and-review commit
  `b43cc26cc830b6176d07f6a0c446d7a19cbe89f9`, and only as (a) the 125
  paths in its committed recursive source manifest and (b) the eleven V18
  preregistration, source-manifest, independent-source-review, V5
  scientific-result/source-manifest identity-witness, and six focused V18
  synthetic-test paths explicitly enumerated and SHA-256-and-byte-count-bound
  in
  `docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_clean_export_certification_2026-07-31.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied V5 documents are source-only chronology and
  source-identity evidence and grant no predecessor runtime-output,
  checkpoint, retry, resume, recovery, or execution authority. The V18
  certification itself and later one-shot authority may be added to the
  narrow export only at their exact reviewed paths after their respective
  commits and exact file-SHA-256 validation. This source-only exception grants
  no generated-input, runtime-artifact, checkpoint, tensor, schedule,
  dataset-payload, RGB, GPU, training, qualification, benchmark,
  probability-calibration, G2 payload, navigation, held-out, sealed,
  production, promotion, deployment, retry, scientific resume, recovery
  execution, second attempt, architecture variant, or further-attempt
  authority.
- The science-identical V18 spatial-token delay-line causal-convolution
  joint-JEPA V1 update-zero gate-timing integrity replacement V1 source may
  additionally be clean exported only from frozen source-and-review commit
  `5c1ed4ced698a957bee66544141af9d4750a1dcc`, and only as (a) the 125
  paths in its committed recursive source manifest and (b) the thirteen V5
  scientific-result/source-manifest identity witnesses, original-V18
  preregistration and terminal update-zero scientific-result witnesses,
  replacement preregistration/source-manifest/independent-source-review, and
  six focused V18 synthetic-test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_update_zero_gate_timing_integrity_replacement_v1_clean_export_certification_2026-07-31.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V18 terminal result and V5 documents
  are source-only chronology and scientific-identity evidence and grant no
  predecessor runtime-output, checkpoint, retry, resume, recovery, or
  execution authority. Do not export the original V18 source manifest,
  source review, clean-export certification, or execution authority. The
  replacement certification itself and later one-shot authority may be added
  to the narrow export only at their exact reviewed paths after their
  respective commits and exact file-SHA-256 validation. This source-only
  exception grants no generated-input, runtime-artifact, checkpoint, tensor,
  schedule, dataset-payload, RGB, GPU, training, qualification, benchmark,
  probability-calibration, G2 payload, navigation, held-out, sealed,
  production, promotion, deployment, retry, scientific resume, recovery
  execution, second replacement, architecture variant, or further-attempt
  authority.
- The science-identical V18 spatial-token delay-line causal-convolution
  joint-JEPA V1 batch-schema integrity replacement V2 source may additionally
  be clean exported only from frozen source-and-review commit
  `4fcd8a732fa701c0045ec426fa7db0fdcc7df333`, and only as (a) the 125
  paths in its committed recursive source manifest and (b) the fifteen V5
  scientific-result/source-manifest identity witnesses, original-V18
  preregistration and terminal update-zero scientific-result witnesses, V1
  replacement preregistration and terminal-infrastructure-failure witnesses,
  V2 preregistration/source-manifest/independent-source-review, and six
  focused V18 synthetic-test paths explicitly enumerated and
  SHA-256-and-byte-count-bound in
  `docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_update_zero_gate_timing_integrity_replacement_v2_clean_export_certification_2026-07-31.json`.
  Validate every path against both that certification and the frozen commit
  before copying. Do not use a whole-tree archive, worktree, checkout copy,
  source package, wildcard, recursive copy, or recursive source
  materialization. The copied original-V18, V1, and V5 documents are
  source-only chronology and scientific-identity evidence and grant no
  predecessor runtime-output, checkpoint, retry, resume, recovery, or
  execution authority. Do not export the original V18 or V1 source
  manifests, source reviews, clean-export certifications, or execution
  authorities. The V2 certification itself and later one-shot authority may
  be added to the narrow export only at their exact reviewed paths after
  their respective commits and exact file-SHA-256 validation. This
  source-only exception grants no generated-input, runtime-artifact,
  checkpoint, tensor, schedule, dataset-payload, RGB, GPU, training,
  qualification, benchmark, probability-calibration, G2 payload, navigation,
  held-out, sealed, production, promotion, deployment, retry, scientific
  resume, recovery execution, second V2 attempt, third integrity replacement,
  architecture variant, or further-attempt authority.
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
