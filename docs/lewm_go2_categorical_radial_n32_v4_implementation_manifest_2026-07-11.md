# Go2 categorical-radial N32 V4 implementation manifest

Date: 2026-07-11

Status: frozen after implementation and adversarial review, before any V4
dataset-backed model output.

## Experiment binding

- execution binding:
  `docs/lewm_go2_categorical_radial_n32_v4_hierarchical_binding_2026-07-11.md`;
- binding SHA-256:
  `bb691c787af0b90f813ced4e5e521f1b15b70b75c836147cd69275c50df6b5d3`;
- sole intervention: replace the width-24 V2 three-logit output with explicit
  KNOWN/UNKNOWN and OCCUPIED/FREE-given-KNOWN factors;
- registered parameter count: 2,887,002;
- registered parameter delta from V2: -65;
- factor-output contract SHA-256:
  `2d56e5958dc99d9bd9a1230081bf626e7cf9add9836e4be75b18dbde32f08c33`.

## Frozen implementation

The runner and independent finalizer produce the same 41-entry transitive
source map. Its canonical JSON SHA-256 is
`fe136d8543a9664417e65ec8e07f052875f9903b5913ac915eb1ec6d68791800`.

Direct V4 source commitments:

- pure decision module, `lewm/benchmarks/go2_categorical_radial_n32_v4.py`:
  `783a2c61dd39ca618b607bbc8fe9455ace60f57db846e10f91d5130280c34f05`;
- explicit-hierarchy model,
  `lewm/models/categorical_radial_perception_full_ray_hierarchical.py`:
  `c88625d70e447bf71922bca606aa03ce02d3636085738a702b5b3c4959dea4b1`;
- authoritative runner, `scripts/run_go2_categorical_radial_n32_v4.py`:
  `bd0d0721281045348e9594b0839330ff34255ad5ab6aecd27b2e1722a28f9b6d`;
- torch-free finalizer, `scripts/finalize_go2_categorical_radial_n32_v4.py`:
  `cd6e7464fc9bb336b6c6af85f17cfd270430fdcd1186619a44a987c219e383d1`.

Bound V4 test commitments:

- model test:
  `e0ea441ab5a0d1e8a045727354b69b89ea4635818b1471d5f2aa92bf98be8976`;
- pure decision test:
  `3a3d0de09580e6a17b5bfdfa5db33146c8d6e76b84fe596ac3ef261cb334a6bf`;
- runner test:
  `71279aa9068b53aa11e5a2e7e6bf5761fbac1962e5d1a48f3732feabc9347209`;
- finalizer test:
  `fc0883b4719be44613e85d118fbc42854a40f42e8d6ccedec96aec14df6b9113`.

## Initialization and schedule

- seed-20260710 V2 reference initial state:
  `8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278`;
- seed-20260710 V4 initial state:
  `0e82e8832eb2c27dc9ef2ea4c6ff35a83dcca181cb1d4172830fb6b2811a9c5e`;
- seed-20260711 V2 reference initial state:
  `989e2db491d199bc544fabe2df40443a39f3ffc6e936f0d28c24625e7bd0ce13`;
- seed-20260711 V4 initial state:
  `55ae2bbeecbe3913c7e886c11a3a14a5c4c435673a6067df45a2cca6d12fbc99`;
- state key count: 133;
- 131 common entries are bit-identical for both seeds;
- only `polar_head.weight` and `polar_head.bias` change shape and remain at
  deterministic PyTorch default initialization;
- seed-20260710 minibatch schedule:
  `79b6e66d4e90246f9eb045675f2a06eb25ae28d26f0997392b6780518e668156`;
- seed-20260711 minibatch schedule:
  `f621b85716607b7e7b8e1ba931d19cf552eb944feca48d099a2c1a3b8ef801c6`.

No trained V2/V3 state, class-prior transform, calibration, or threshold patch
is loaded.

## Verification

- 42/42 focused V4 tests pass;
- 144/144 selected N32 V1/V2/V3/V4 regression tests pass;
- model, decision, runner, and finalizer compilation passes;
- `git diff --check` passes;
- runner and finalizer source maps and bound-evidence maps are exact-equal;
- the role-namespace amendment is bound by both at SHA-256
  `ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370`;
- the finalizer imports under system Python without importing `torch`;
- hierarchy normalization, arbitrary-probability round trip, sentinel/support
  behavior, evaluator parity, one encoder call, parameter/state identity, and
  tamper rejection are covered;
- UNKNOWN/KNOWN-to-occupied-factor and conditional-FREE/OCCUPIED-to-known-factor
  cross-gradients are bounded in float64 at `1e-12` and in the authoritative
  float32 training precision at `1e-7`.

Independent adversarial review found no remaining code-level, access-control,
initialization, schedule, hierarchy-math, or result-acceptance blocker.

## Access state and authorization

At freeze time, V4 image decodes, label-shard opens, holdout model outputs,
checkpoint-selection outputs, probability-calibration outputs, physical
non-train outputs, G2 outputs, and sealed outputs are all zero.

This manifest authorizes only the distinct non-authoritative three-update
seed-20260710 smoke path under
`.generated/go2_categorical_radial_n32/v4/`. The smoke is unconditionally
fit-only and cannot occupy either canonical authoritative result path. A
separate immutable smoke-evidence note is required before the authoritative
seed-20260710 command. Seed 20260711 remains forbidden unless the strict
torch-free finalizer finds the canonical seed-20260710 result fully favorable
after its conditionally authorized train-role holdouts.
