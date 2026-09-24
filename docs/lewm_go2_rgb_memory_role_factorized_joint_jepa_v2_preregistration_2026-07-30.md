# Go2 RGB Memory-Role Factorized Joint-JEPA V2

Date: 2026-07-30

Status: preregistered distinct evaluation-contract successor only. This
document grants no RGB, checkpoint, GPU, training, memory, navigation, G2,
held-out, sealed, benchmark, promotion, production, or deployment access.

## Closed predecessors and trigger

The original V1 attempt is terminal and bound by:

- `docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_terminal_infrastructure_failure_result_2026-07-30.json`;
- commit `291a7bcfaf95f24d5c84bd3d590afd54556d5b3d`;
- file SHA-256
  `80eaeb508a988b54e655df5b530fa3adab6a89bb13b6f5c45902ac851bc464f4`;
  and
- byte count `6060`.

Its one integrity replacement is also terminal and exactly bound by:

- `docs/lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_integrity_replacement_v1_terminal_infrastructure_failure_result_2026-07-30.json`;
- commit `79c83b21e6447881cb43961eea404b28ec6ad87a`;
- file SHA-256
  `bedfafa247ee0c39697b16327eff96ed420204000f25f0255f5de26128f1c548`;
  and
- byte count `9867`.

The replacement loaded all 320 registered checkpoint-selection place triplets
and successfully verified and decoded all 960 RGB references, then stopped in
the update-0 evaluator before local evaluation, training, any completed
observation, optimizer or EMA step, presentation, checkpoint publication, or
scientific classification. Its exception-message SHA-256
`cc6f93b9982aaf7d04ca997311d0761c75e84d437c2925c84e77979562bf025f`
binds the exact message `retrieval scene candidate count left [40,64]`.

Both predecessor attempts are consumed. V2 is not a retry, resume, recovery,
extension, or second integrity replacement, and it may open or reuse no
predecessor runtime file, tensor, model state, optimizer state, RNG state,
trace, or checkpoint.

## Decision and sole scientific-scope delta

V2 changes only the structural lower bound for the deterministic within-scene
retrieval candidate panel from `40` to `32`; the allowed range is therefore
exactly `[32,64]`. The frozen selection metadata produces the following exact
candidate counts under the unchanged positive-then-negative-then-anchor,
unique-identity, maximum-64 construction:

| family | candidate count |
|---|---:|
| large_enclosed_maze | 63 |
| local_composite_motifs | 64 |
| loop_alias_stress | 59 |
| medium_enclosed_maze | 59 |
| open_obstacle_field | 64 |
| rough_local_dynamics | 64 |
| small_enclosed_maze | 32 |
| visual_sensor_stress | 48 |

The V1 index builder guaranteed unique selected positive identities within
each scene but did not guarantee 40 distinct identities after deduplicating
the three reference roles. The old lower bound was therefore incompatible
with its own frozen selection panel. Candidate count is metadata and not a
learned metric.

All learned retrieval requirements remain unchanged: pessimistic tie handling,
actual per-scene exact chance `5/N`, equal-scene-mean recall@5 at least `0.40`,
equal-scene-mean recall@5 at least three times equal-scene-mean exact chance,
and at least six of eight scenes strictly above their own exact chance. The
absolute recall gate remains stricter than three times the frozen panel's
equal-scene-mean chance. No candidate is added, removed, reordered, or opened
by this correction.

## Frozen scientific identity

Except for that single evaluator bound and fresh lifecycle identifiers, V2
incorporates and preserves exactly:

- the V1 preregistration at commit
  `01d78284a22a52816a41f31a78411491714b4f9c`, file SHA-256
  `a9deae0b3335540b26791302566cdcb6a7d8397e96618b691dba1fa8db0c85c7`,
  and byte count `11170`;
- the V1 split-integrity amendment at commit
  `5a1535567bf00b8e47d67d8966ef42a52726bd5b`, file SHA-256
  `8350289c0288f9f98d18b17f401318247bd4ecf8ae0597f14a6641606aa77c1f`,
  and byte count `3136`; and
- the V1 integrity-replacement preregistration at commit
  `ba6e37d63f099cd51184642dea39808ae1f2f99e`, file SHA-256
  `a7c757f4a58b9a7d068ceb2e6676573843d58e72606b55713868ddfe86b97820`,
  and byte count `7211`, including the corrected exact 224-by-168 place-RGB
  source geometry and passive place-reference access accounting.

Thus V2 preserves the shared V18 RGB/object-space encoder, role factorizer,
place and local predictors, EMA targets, accepted N320 initialization, every
seed, frozen data/index/row ordering and split, one AdamW optimizer, parameter
groups, learning rates, gradient routing, every physical/place/local loss,
coefficient, margin, diagnostic, and terminal learned threshold. It preserves
the exact 4+2+2 microbatch schedule, observations at updates 0, 100, and 400,
one optimizer and EMA step per completed update, exactly 400 maximum updates,
and exactly 12,800 maximum presentations: 6,400 physical, 3,200 local, and
3,200 place.

There is no architecture, model input, loss, threshold, data, index, seed,
initialization, optimizer, schedule, accounting, cap, candidate construction,
checkpoint, navigation, or memory-intervention change.

## Mandatory metadata-only preflight

After this preregistration is frozen and before V2 source review, clean export,
authority, RGB access, checkpoint access, GPU use, or attempt reservation, one
bounded metadata-only preflight must consume only the frozen
checkpoint-selection triplet rows and their manifest/index bindings. It must
not dereference or decode any RGB path.

The preflight must independently reproduce the unchanged retrieval candidate
construction and publish an immutable receipt proving: 320 rows; eight scenes
and eight families; the exact family counts above; every paired positive
present in its scene candidate panel; every count in `[32,64]`; and no train,
calibration, held-out, sealed, RGB, checkpoint, GPU, training, memory, or
navigation access. Any mismatch closes V2 before execution authority.

Source-only tests must also include one dataset-shaped synthetic panel with
the same anchor/positive/negative identity overlap pattern and candidate counts
as the frozen panel. It must prove that the 32-candidate scene is accepted,
actual `5/N` chance is used, and every learned gate and threshold is unchanged.

## One-shot identity and lifecycle

- Schema/evidence prefix:
  `lewm_go2_rgb_memory_role_factorized_joint_jepa_v2`.
- Fresh attempt root:
  `.generated/go2_rgb_memory_role_factorized_joint_jepa_v2/attempt_v1`.
- Fresh clean source root:
  `/home/andrewknowles/Workspace/LeWMQuad-v3-memory-role-factorized-joint-jepa-v2-source`.
- Both roots must initially be absent. Exactly one V2 attempt is allowed;
  retry, resume, recovery, extension, and a second V2 attempt are false.

After the metadata preflight, V2 still requires a recursive source closure,
independent source review, exact enumerated clean-export certification, and a
separately committed hash-bound one-shot authority before any reservation or
runtime access. Failure publishes complete immutable receipts and no
checkpoint. A terminal pass may publish only the registered update-400
development checkpoint and receipts and makes V2 eligible only for a newly
preregistered learned-memory integration test. It grants no memory,
navigation, calibration, G2, held-out, sealed, benchmark, promotion,
production, or deployment authority.
