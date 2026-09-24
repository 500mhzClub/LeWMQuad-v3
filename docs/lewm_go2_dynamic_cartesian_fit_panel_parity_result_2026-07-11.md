# Go2 Dynamic-Cartesian Fit-Panel Projection Parity Result

Date: 2026-07-11

Status: pass. This licenses the dynamic Cartesian N32 occupancy-only runner;
it is not a learned-model, G2, memory, exploration, or navigation result.

## Exact Execution

```text
env PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3:/home/andrewknowles/Workspace/LeWMQuad-v3/lewm_worlds:/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/python3 scripts/audit_go2_dynamic_cartesian_fit_panel_parity.py
```

The stdlib float64 reference used six spawned CPU workers. The production
Torch float32 geometry used CPU batches of four with one native numerical
thread. Neither GPU was used.

## Frozen Inputs

- Execution binding SHA-256:
  `42687e80a16fb424be47d49782699bbc3ed549d7826a0ce6e78e92aa37188e1e`
- Fit-panel file/content/fit-row SHA-256:
  `c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c` /
  `f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f` /
  `5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d`
- Attitude-sidecar manifest file/content SHA-256:
  `6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529` /
  `6f1ef7d9ac0c55a42182c3e2c75909f00ab37fffa460aadb549d5cd60d278c1a`
- Authorized train-sidecar file SHA-256:
  `6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6`
- Pure stdlib geometry source SHA-256:
  `ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf`
- Production model source SHA-256:
  `c4006e9804182b077399229d43bc8c9be64b5af12c81fff4076d5a78e6ef359b`
- Audit runner source SHA-256:
  `bf036a58c91335b022783744eb4984f4c4c4b63a6e8f7334fb05541815a69d94`

## Result

- Fit transitions joined by exact global row and row identity: 160/160
- Current plus next frames: 320/320
- Cell decisions compared: 1,310,720/1,310,720
- Stdlib float64 visible decisions: 662,078
- Production Torch float32 visible decisions: 662,078
- Mismatched frames: 0
- Mismatched cells: 0
- Equal ordered support-mask SHA-256:
  `915082a5079ad748361989bfc6645fa89750b0fa9c52249b7fa51843ade82cf1`

Published result:

- Path:
  `.generated/go2_dynamic_cartesian_fit_panel_parity/v1/result.json`
- File SHA-256:
  `72d21aaf5e923126dd3a5022b0ea9775340877a00f40aa22845b244886fde70b`
- Canonical content SHA-256:
  `3729a3fcd61b523d744c476da89fb2f638593145055b52bc96035bb30c3f3cea`

An independent post-run recomputation reproduced the declared result content
hash exactly.

## Access Boundary

The strict sidecar loader opened only the manifest and authorized train role.
Checkpoint-selection, probability-calibration, and G2 sidecar byte opens were
zero. RGB, labels, depth, checkpoints/model outputs, runtime, and sealed
payload byte opens were also zero.

## Consequence

The production float32 dynamic visibility logic now exactly reproduces its
independent stdlib float64 reference on every registered fit-panel attitude.
The geometry bridge is therefore cleared for the bound N32 occupancy-only
qualification. No learned accuracy or navigation claim follows from this
parity result alone.
