# Go2 categorical-radial N32 V3 implementation manifest

Date: 2026-07-11

Status: frozen after implementation, adversarial review, and non-authoritative
smoke; before the authoritative seed-20260710 command.

## Experiment binding

- execution binding:
  `docs/lewm_go2_categorical_radial_n32_v3_token_width_binding_2026-07-11.md`;
- binding SHA-256:
  `a9898d349d82f65ce35443192b555aac4386136032c8fe70c115eda5a788a5ad`;
- sole intervention: token feature width `24 -> 32`; context width remains 64;
- registered parameter count: 2,891,171;
- registered parameter delta from V2: 4,104.

## Frozen implementation

The runner and independent finalizer produce the same 32-entry transitive
source map. Its canonical JSON SHA-256 is
`c2a299fd478e81817bc5e1e692e91537eea6c5f9c5f3c8a0b94338bc66489b08`.

Direct V3 source commitments:

- pure decision module, `lewm/benchmarks/go2_categorical_radial_n32_v3.py`:
  `5d832097927ceca201a8f65b81e53f297ae14a76fd42d173425f99f9501db9cf`;
- token-width model,
  `lewm/models/categorical_radial_perception_full_ray_token32.py`:
  `04bd81c4560644482a76a81d670d1b2767f33e8bdb6a88ab53f42fd47e25b152`;
- authoritative runner, `scripts/run_go2_categorical_radial_n32_v3.py`:
  `6df52ae820d1cf317bd8ed77d70ce8c0ecea42a650c1bca9f02b9b809f46aea3`;
- torch-free finalizer, `scripts/finalize_go2_categorical_radial_n32_v3.py`:
  `807d98b1c154066f8585c5ff09d4d29328cb41aa367607d86f93f053847e6922`.

Bound V3 test commitments:

- model test:
  `346b426cc941bfbd3efbec0d64e83e61052c9656aef5df7c8969041418e038de`;
- pure decision test:
  `0a3784362056437ad60ab330cd6f48babc95fb81c8e70971e674fda7c7c71df6`;
- runner test:
  `96887908f8d4f53dcfd54c14e2806adad5abacd43d149d624dbe17049bc9d450`;
- finalizer test:
  `89c486de98429987d8543a57bb2bd3ea78709b2d1aafb7b3fc8d24950ef332c8`.

## Initialization and schedule

- seed-20260710 V3 initial state:
  `ddb8f6dbfa54a7445c2b4363d9978b0a99a86e6d88a28f480840c5d8d128804b`;
- seed-20260711 V3 initial state:
  `fa9601fb5f658b640c43b50c28587c5129c6f42f8fd4fb09866983130e4954ee`;
- corresponding V2 reference initial states reproduce exactly;
- 130 same-shape state entries are bit-identical;
- exactly three state shapes change, as registered in the binding;
- no trained V2 state is loaded;
- seed-20260710 minibatch schedule:
  `79b6e66d4e90246f9eb045675f2a06eb25ae28d26f0997392b6780518e668156`;
- seed-20260711 minibatch schedule:
  `f621b85716607b7e7b8e1ba931d19cf552eb944feca48d099a2c1a3b8ef801c6`.

## Verification

The complete categorical model plus N32 V1/V2/V3 regression selection passed:
`128 passed in 4.35s`. Compilation and `git diff --check` also passed.

Adversarial review found and closed two pre-run issues:

- smoke-mode holdout authorization is now unconditionally disabled, even if a
  three-step smoke happens to pass its local fit calculation;
- smoke mode now rejects both canonical authoritative result paths, preventing
  an immutable smoke artifact from blocking a registered seed result.

The final focused V3 suite passed 30/30 tests after both fixes. Independent
review found no remaining authorization, schedule, initialization, source-map,
gate, access-ledger, or license finding.

## Dataset-backed smoke

- path:
  `.generated/go2_categorical_radial_n32/v3/smoke_seed_20260710.json`;
- file SHA-256:
  `e8bbd920610c68be9b82d109037745d27a2ebcccc8cb13c6e2de25c7f6b5a2ac`;
- canonical content SHA-256:
  `301b0a9ac486ae4f69d0f34cb46734922ad08a410bdc092b9355acf49aa8ac41`;
- schema: non-authoritative smoke; favorable false; every license false;
- fit access: 320 image decodes, 20 label-shard opens, 3 updates;
- same-scene holdout, cross-scene holdout, selection, calibration, non-train,
  G2, and sealed byte/model-output access: zero.

This manifest authorizes only the frozen authoritative seed-20260710 N32 V3
fit run at its canonical path. Seed 20260711 remains forbidden unless the
strict finalizer finds seed 20260710 fully favorable after its conditionally
authorized train-role holdouts.
