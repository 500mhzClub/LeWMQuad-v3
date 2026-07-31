# V18 spatial-token delay-line Joint-JEPA V1 — physical-comparison alias-state integrity replacement V4

Date: 2026-07-31

Status: preregistered, source-only. Execution is denied pending implementation
review, a frozen reviewed source closure, narrow clean-export certification,
and a fresh one-shot authority.

## Consumed V3 result

The one-shot overflow-safe route-norm integrity replacement V3, authorized at
commit `5be273ffde19b4e7d5f862c36bd72fee1e05ca98`, completed update 100 and then
terminated at `observe_update_100` with
`V19 comparison observation is not one-shot`. Its update-zero physical
observation had already populated `runtime.causal_comparisons_v19[400]`;
V18's unchanged physical alias selected update 400 again at outer update 100,
and the frozen V19 diagnostic wrapper rejected the repeated key before the
second comparison-scoring pass.

The independently reviewed terminal result is
`docs/lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_overflow_safe_route_norm_integrity_replacement_v3_terminal_observation_alias_infrastructure_failure_result_2026-07-31.json`
at commit `f3d51438b87ac47af3046b5f9a23434dd5f3d1e3`. It is 10,212 bytes,
has file SHA-256
`430004e93e36456a7ec7a7ae4f311e7b45b17ceca7a4bb1fb7f872238b4d5e6b`,
and content SHA-256
`539a068c809a6ad7b77ff0f0993ac8a0541cad876f21eb99e12e0ebb43d54fc0`.
The V3 result itself grants no source or execution authority. This V4
preregistration authorizes only the source correction and CPU tests below; it
grants no V3 retry or resume and no V4 execution.

## Exactly authorized correction

Authorize one V18-local adapter that:

- keeps `PHYSICAL_OBSERVATION_ALIAS` equal to 400 for all six outer
  observations and runs the inherited physical evaluator on the current model
  every time;
- before each inherited physical observation, requires the V19 comparison
  cache to be absent or an exact empty plain dictionary and fails closed on a
  pre-existing key or malformed state;
- only after the inherited observation returns successfully, requires an
  exact plain-dictionary cache containing only
  `causal_comparisons_v19[400]`, validates its frozen four-control structure,
  pops exactly key 400, and verifies that the cache is empty;
- fails closed if the successful inherited observation leaves the cache
  missing, malformed, differently keyed, or otherwise non-empty;
- leaves the shared V19/V25 sources, V19 comparison scoring and sanitization,
  returned gate and controls, physical metrics, and current-model
  re-evaluation unchanged.

The cache is inherited V19 diagnostic retention that V18 never consumes.
Releasing it after validation changes lifecycle state only; it must not cache
or reuse a prior physical result.

## Required CPU-only tests

- A synthetic sequential outer-update 0 then 100 test must show two inherited
  comparison-scoring calls, alias 400 on both calls, current-model evaluation
  on both calls, unchanged returned values, and an empty cache after each.
- One synthetic sequence must exercise all outer updates
  `(0, 100, 250, 500, 750, 1000)` exactly once with the same assertions.
- A synthetic inherited-observer exception after partial cache population must
  propagate unchanged and leave that evidence intact; cleanup is permitted
  only after a fully successful inherited call.
- Focused negative tests must fail closed for a pre-existing key, a non-dict
  cache, a missing post-success cache/key, a malformed four-control payload,
  an unexpected key, or extra retained state.

No dataset, RGB, GPU, training, checkpoint, or runtime-artifact access is
authorized for these tests.

## Frozen scientific identity

V4 must preserve V3's model and initialization, data and roles, seeds,
training and observation schedules, losses and gradient recipients, V18-local
overflow-safe route norm, optimizer and EMA behavior, wrong-action/reset/
reverse/shuffle/persistence/HOLD controls, physical and place aliases, scoring,
metrics, gates, thresholds, snapshots, and presentation accounting. It keeps
16 memory and 8 physical presentations per update; observations at
0, 100, 250, 500, 750, and 1000; snapshots at 250, 500, 750, and 1000; a
500-update stage-A cap; and caps of 16,000 memory, 8,000 physical, and 24,000
combined presentations.

Only lifecycle identity, the V18-local alias-cache adapter, its failure
receipts, and focused CPU tests may change. V4 requires fresh initialization
and a fresh output root. It grants no retry, resume, recovery, checkpoint
reuse, probability calibration, G2, navigation, held-out, sealed, production,
or promotion access.

## Execution boundary

Execution remains denied until the V3 terminal result is committed and exactly
bound here, the V4 implementation and tests are independently reviewed, the
reviewed source is frozen and narrowly certified under repository custody, and
a fresh V4 one-shot authority is committed. Any V4 run must be a new attempt;
the consumed V3 output must not be reopened.
