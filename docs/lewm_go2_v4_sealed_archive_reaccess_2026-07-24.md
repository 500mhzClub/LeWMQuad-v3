# Go2 V4 sealed archive re-access record

Date: 2026-07-24

Status: recorded and contained; V4 was already permanently invalid for G8

This record is an additive successor to the access table in
`docs/lewm_go2_heldout_maze_authority_correction_2026-07-24.md`. It changes no
promotion or execution authority.

## Incident

After committing the source closure, clean-export verification began with:

```text
git archive --format=tar --output=/tmp/lewm-closure-pPNwth/repo.tar HEAD
mkdir /tmp/lewm-closure-pPNwth/repo
tar -xf /tmp/lewm-closure-pPNwth/repo.tar -C /tmp/lewm-closure-pPNwth/repo
```

The whole-tree archive operation necessarily read and copied every tracked
blob, including already-invalid legacy sealed manifests. No protected content
was printed, parsed, summarized, searched, or supplied to a model, evaluator,
trainer, navigator, dataset builder, or checkpoint selector. Verification
stopped before any command ran inside the extracted tree.

## Containment and consequence

- The exact temporary directory `/tmp/lewm-closure-pPNwth` was deleted after
  explicit approval. The copied archive and extracted tree are not retained.
- V4 was already permanently ineligible for G8; this access does not change
  that status.
- No source-closure conclusion or scientific decision uses an archived sealed
  byte.
- No data, checkpoint, G2, navigation, runtime, Genesis, hardware, GPU,
  benchmark, or held-out execution occurred.
- Whole-tree archive/worktree export is now explicitly forbidden while legacy
  sealed blobs remain tracked.
- Clean source certification must construct a source-only snapshot from
  exactly the 72 paths bound by the committed closure manifest plus the nine
  checker, guard, and synthetic-test paths explicitly enumerated and
  SHA-256-bound in
  `docs/lewm_go2_g2_runner_source_closure_v1_clean_export_certification_2026-07-24.json`.
  The exporter must validate every path before copying and reject `config/`,
  custody, generated, artifact, checkpoint, and dataset paths.
