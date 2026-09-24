# RGB Spatial-Contrastive Memory-Role Joint-JEPA V4 Preregistration

Date: 2026-07-30 UTC

## Decision and scope

V3 is terminally closed at initialization with zero completed updates, zero
presentations, zero RGB decodes, and no observation or checkpoint.  Its
committed result is
`docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v3_terminal_infrastructure_failure_result_2026-07-30.json`
at commit `617c44322034b85fe0d89cb88e8f5967fdbe1e42`.

The exact failure was a module-interface omission.  The inherited runtime
reads the public module constant `PROJECTION_INITIALIZATION_SEED_V13` before it
moves the freshly constructed model to the GPU.  The V3 model inherited the
same projection initialization and value, but its new module did not re-export
that constant.  The resulting `AttributeError` message hash exactly matches
the terminal receipt.  No scientific tensor path, training update, or V3
metric was reached.

V4 is one fresh, science-identical infrastructure replacement.  It is not a
V3 retry or resume and may not read or reuse any V3 model state.  It exists
only to let the already reviewed V3 mechanism reach its preregistered
falsification.

## Sole permitted implementation delta

The only scientific-source change from terminal V3 is:

- import and publicly re-export the inherited
  `PROJECTION_INITIALIZATION_SEED_V13`, whose exact value remains `20260729`,
  from `lewm.models.memory_role_spatial_contrastive_joint_jepa_v3`;
- add a source-only regression asserting that the V3 module exposes that exact
  inherited value through the interface consumed by
  `V13ComposedRuntime.initialize_model_v13`.

This adds no parameter, tensor operation, branch in model forward execution,
loss term, gradient, optimizer state, seed change, or data access.  The V4
launcher must use Python isolated no-bytecode mode (`-I -B`) on its first
reservation-capable invocation.

## Frozen science

V4 preserves V3 byte-for-byte in all scientific respects:

- the shared V18 RGB/object-space encoder and `64 x 64 x 64` latent;
- the `Conv2d(64,16,1)`, exact GELU, adaptive `4 x 4` pool, and
  `Linear(256,64)` normalized place key;
- the zero-final-layer, identity-anchored residual place predictor;
- the combined eight-row place graph, eight EMA positives followed by eight
  EMA paired negatives, temperature `0.10`, and exact alignment, contrast,
  variance, and covariance objective;
- the physical route, local route, EMA coefficient and ordering, one AdamW
  optimizer, gradient routing, initialization, data, split, row order,
  microbatch schedule, hardware, runtime, and all non-interface seeds;
- 16 physical, 8 local, and 8 place presentations per update, for 32 total;
- observations only at updates 0, 100, and 400;
- a maximum of 400 updates and 12,800 presentations.

The update-100 continuation gate remains the exact V3 conjunction: structural,
target, and place/local access integrity; place R@5 at least `1.50` times exact
chance; target place-key effective rank at least `2.0`; positive bootstrap
lower 95% place separation; at least six positive place families; and at
least 60 of 189 physical margins.  Failure stops the attempt at update 100.

The update-400 memory-entry gate remains the exact V3 conjunction: structural,
target, and place/local access integrity; place R@5 at least three times exact
chance; at least six scenes above exact chance; target place-key effective
rank at least `4.0`; more than 72 of 189 physical margins; and all twelve
causal controls.  Local learned metrics and the earlier fixed-magnitude gates
remain diagnostic only.

## Lifecycle and stop rules

- Fresh V4 schema, certified source root, output root, reservation, and
  authority are required.
- One V4 attempt only; retry and resume are false.
- Private immutable non-self-authorizing restart states remain required after
  updates 100 and 400.
- Any integrity, source, schedule, accounting, target-gradient, access, or
  adapter-interface violation terminalizes V4.
- A terminal gate pass authorizes only preparation of a separately reviewed
  learned-memory integration experiment.
- Probability calibration, navigation, G2, held-out, sealed, production, and
  promotion work remain unauthorized.  The V4 30-scene sealed benchmark stays
  unopened.
