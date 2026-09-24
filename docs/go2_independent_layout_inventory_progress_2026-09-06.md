# Independent layout inventory frozen; collection adapter implemented

The previous goal turn completed the twelve-episode tracker-independent pilot
and corrected raw audit. This turn advances the next data stage, without fitting
another one-layout model or relabelling the pilot as evaluation. The full RGB+
deployment-valid-sensor JEPA navigation goal remains active and unachieved.

## Completed construction and evidence

The frozen inventory contains twelve connected sixteen-cell layouts. Each has
straight passages, corners, branches, dead ends and loops. Four layouts have each
cycle rank1,2and3. Construction required13candidates; one was rejected for missing
a context cell type. No physical outcome or model score influenced the inventory.

Prospective roles and planned episodes:

| Role | Distinct layout/topology units | Planned episodes |
| --- | ---: | ---: |
| Train |6|720|
| Selection |3|360|
| Development evaluation |3|360|

Each layout has five contexts: open passage, corner turn, junction, dead-end
approach and near wall. Quiet/recent-forward histories and nominal/lower-friction
support are crossed with all six existing pulse-duration cells. The240six-action
groups give1,440explicit episode definitions, in twelve120-episode layout batches.
This is a prospective development inventory, not actual collected data or a final
benchmark. Three evaluation topology units are insufficient for strong final
generalization claims; keep layout-level uncertainty and later confirmation.

Exact graph canonicalization groups abstract-isomorphic layouts independently of
vertex labels, metric embedding, appearance and spawn. A separate adjacency-
bijection matcher confirms all66layout pairs are non-isomorphic. Metric identities
also remove translation, reflection and right-angle rotations. Differently bent
versions of the same abstract path are conservatively one topology group; they
cannot cross roles. Neither a degree sequence nor a refinement hash alone counts
as identity. Search-budget exhaustion raises an unresolved-identity error.

Twenty-three focused tests pass (10575, exit0,3.24s). They include all720vertex
permutations of each of two regular six-node graphs, an independent small-graph
permutation oracle, distinct regular graphs, transformed metric copies, cross-role
isomorphic-path rejection, all1440episode cells, exact wall boundaries and schema/
role corruption. Synthetic nominal articulated poses fit all60context spawns with
the full4cm-padded setup envelope. This does not establish actual settled support,
common-prefix survival, collision outcomes or future safety.

The explicitly enumerated215-file regression completes with2,720passed in206.77s
(7151, exit0). It includes the completed pilot and its preserved audit-correction
tests. No old frozen source or experiment was changed.

## Frozen artifacts

Writer94258 is terminal exit0. Output:
`.generated/go2_independent_layout_inventory_v1_attempt_001`.

- Launch SHA256: `6497bad46510ebf49f95c2907b7ae992b937b667374f43da0ffad977742894db`.
- Inventory SHA256: `714161041c6db96270d91b53749a982542a5b32fbbee155ec0d5ab98d7afd426`.
- Structural audit SHA256: `5107bda784a4c37b98f42976f5d9f8971ad6c68a3f9a363cdb329bacd4c2059d`.
- Result SHA256: `525382c0e14819fbc5c9b1633641e9f9b46210e03831cf450b2e20904b5c34f1`.

Post-completion verifier31112 rechecks the source/predecessor/output bindings,
reconstructs the exact serialized inventory and independently repeats all66
pairwise comparisons. The [frozen protocol](go2_independent_layout_inventory_v1_2026-09-06.md)
defines the construction, exclusions, scope and interpretation.

## Implemented next-stage adapter, not yet launched

`lewm/independent_layout_collection_development.py` compiles all1,440definitions
into exact native scene packs, retaining roles, seeds, friction, robot spawn and
the entire wall roster. Its selector takes only action index, history kind, tick
and a valid causal RGB/body packet. Geometry, friction, native state and future
targets are not selector parameters. Quiet/recent conditions change the eight
common-history ticks, not the candidate pulse/brake schedule.

`scripts/independent_layout_physical_init_development.py` and
`scripts/independent_layout_session_development.py` connect that construction to
the frozen gait, body/depth/fast-gyro recording and native guard chain. The new
initializer overrides the old exact-pilot constructor; its superclass order is
tested. No simulation has been started with it. These adapter sources are not
part of the already frozen inventory-writer closure and must be separately bound
before execution.

Sixteen adapter tests pass (76257, exit0,7.67s), including compiling all1,440packs,
rejecting mutated geometry/roles/commands, quiet/recent schedule semantics, packet
causality, per-spec mutation isolation and recorder/initializer ordering. These
tests are additional to the2,720-test regression, not a claim that the latter
included the subsequently added adapter.

## Immediate next work

1. Implement a bounded per-layout batch driver and raw auditor for the exact
   frozen inventory. The twelve batches are l00–l11, each120episodes; do not choose
   successful contexts after observing outcomes. Bind batch membership, sources,
   gait/native dependencies, storage allowance and output root before launch.
2. Use the new scene/session adapter and full articulated native setup checks.
   Retain tracker-independent simulation excitation, partial-stop sensor records
   and all planned siblings. Preserve40GiB external free storage. Use explicit
   batch roots and capacity checks; no predecessor deletion, overwrite or resume.
3. Adapt the corrected exact-float64 requested-command audit to both histories;
   retain the separate applied-command slew audit. Reuse the existing1,150native/
   nine-frame prefix witness and target-only pulse labels, but group comparisons
   by layout/context/history/support. No future pose or friction enters inference.
4. Audit every attempted context before fitting. Near-wall0.16m spawns are only
   hazard hypotheses: the prior0.08m-spawn pilot's long nominal pulse approached
   within6mm of the wall, but all its contact labels were negative. Require actual
   native setup/warm-up survival and fresh contact outcomes, not shifted old
   trajectories. Keep missing/failed cases and zero-positive coverage explicit.
5. After sufficient independent data, test action/history/RGB utility and matched
   direct, supervised-rollout and JEPA training, then online rollout and memory
   contributions. Reliable physical local execution, observed branches, executed
   backtracking, complete novel-maze outcomes, realistic sensing/deadlines and
   bounded hardware evidence remain required. Neither the inventory nor pack
   compilation changes the previous0/3room-return result.

No new physics, RGB data, model training, sealed access, source export, hardware
control or navigation promotion occurred in this inventory/adapter turn.
