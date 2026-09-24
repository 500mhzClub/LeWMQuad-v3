# Storage review outside the current V3 workspace

Read-only review requested by the user, 2026-09-07. Nothing deleted, moved,
archived or modified in these candidate directories. Size walks explicitly
excluded `sealed_test.json`, `sealed/` and `sealed_*` directories. The figures
therefore exclude protected material, and are rounded allocated-space figures.
Directory names alone are not proof that an artifact is safe to delete.

## Volumes and likely benefit

- Workspace drive: approximately 20 GiB free of 3.7 TiB. The visible material
  outside the exact `Workspace/LeWMQuad-v3` directory totals only about 2.8 GiB.
  It includes V3 source-evidence exports, which must not be treated as old
  unrelated repositories. Deleting small old sibling repositories will not
  substantially solve this drive's capacity problem.
- Root/home volume, also backing RecoveryStorage: approximately 104 GiB free.
  Cleanup under home can give the current collection and future studies more
  headroom, but does not directly free space on the separate workspace drive.

## Narrow candidates for explicit approval

1. `/home/andrewknowles/TinyQuadJEPA`: approximately **23 GiB**. Contrary to its
   name, this is a Python 3.12 virtual environment, not a repository or simulation
   dataset: `pyvenv.cfg` records its creation with `python -m venv`, and almost
   all its space is under `lib/python3.12`. Historical June V3 documents use
   this interpreter. Current collector/supervisor mapped-file observations had
   no reference to it; the current 786-source prediction definition's explicit
   input/native bindings also have none. This is evidence of no observed current
   dependency, not proof that every historical script is independent of it.
   Removing it would remove that installed environment and impair historical
   command reproduction until its dependencies are reconstructed.
2. `/home/andrewknowles/.cache/pip`: approximately **1.5 GiB** of package-manager
   cache storage. A possible lower-impact cache cleanup, not simulation-data
   reclamation; package downloads may need to be fetched again. Inspect exact
   target and active package installation processes immediately before removal.

No deletion authorization has been inferred from the user's broad inspection
scope. Ask approval for exact targets, then recheck live users/dependencies.

## Large directories that are NOT blanket cleanup candidates

- `/home/andrewknowles/.cache/lewm_go2_temporal_v03`: **182 GiB**. V3 documents
  directly reference scientific results and raw evidence here, including the
  counterfactual paper package. Outside the checkout is not outside V3 science.
- `/home/andrewknowles/.local/share/lewm_go2_planning_utility_v1_2`: **85 GiB**.
  V3 scripts and preregistrations reference this root. Directory sizes include
  47 GiB counterfactual fidelity, 28 GiB V-JEPA ablation and 10 GiB oracle-scorer
  artifacts. Only directory metadata was inspected, not runtime payloads.
- `/home/andrewknowles/.cache/genesis/gsd`: **82 GiB**. Shared simulator cache;
  leave untouched while the collector is active. It needs a separate explicit
  cache-retirement decision, not deletion based on age or location.
- The two `lewmquad-v12-runtime-*` environments under `.local/share` total about
  **29 GiB**. Current explicit runtime bindings/mapped-file observations did not
  reference them, but they are named V3 historical runtimes: do not classify
  them as outside the V3 iteration without an explicit historical-retention
  decision.
- Games/Steam account for substantial home-volume usage, but are unrelated
  personal applications, not simulation-data cleanup targets.

The current original prediction definition was rechecked unchanged as
`3a1b0d78ba9bb4cc1854bc4d1079a26f21ffa57652a7664aaff6af650318956b`.
No current explicit input/native binding named any of the five examined older
environment/data roots. This narrow check does not authorize deletion of
scientific evidence or replace a complete dependency/retention audit.
