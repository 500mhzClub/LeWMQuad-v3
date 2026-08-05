# Go2 G4 V2 two-resolution frontier/viewpoint independent review

Date: 2026-07-13

Verdict: **PASS as an additive, synthetic navigation dependency; no G4 result or promotion**

## Review boundary

This different-agent review covered only the immutable G4 V2 candidate source,
its focused synthetic tests, its design and implementation handoff, the legacy
G4 source/tests, and the frozen G3 V2 projection/planner source and adjacent
tests. It did not open or execute an audit, result, model, dataset, checkpoint,
RGB, held-out, GPU, G5, or V5 input.

No implementation, candidate test, legacy source, or frozen G3 V2 source was
edited. This document is the only review artifact added.

## Frozen candidate identities

| Artifact | Independently recomputed SHA-256 |
|---|---|
| `lewm/planning/two_resolution_frontier_viewpoint_v2.py` | `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` |
| `lewm/tests/test_two_resolution_frontier_viewpoint_v2.py` | `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e` |
| `docs/lewm_go2_g4_two_resolution_frontier_viewpoint_v2_design_2026-07-13.md` | `de6cb956d97b9187281da948abcf700904969c3f91486e0c5390024fdd4ddc7f` |
| `docs/lewm_go2_g4_two_resolution_frontier_viewpoint_v2_implementation_handoff_2026-07-13.md` | `bac65cb1622a6b839784ab52099a2d7844a184e1ca5d95dd7ae676156e65cc6a` |

The four values exactly match the review handoff.

## Frozen dependency identities

| Artifact | Independently recomputed SHA-256 |
|---|---|
| legacy G4 source | `2ef20e8213a384e0f514705ca14c058eb7fbd81dcc4f6a53407414c1ba79e08e` |
| legacy G4 tests | `02d5a0b0459f6fde43e046b2b9f86d13d21e7392119b57626f0a398ce4c5241e` |
| G3 V2 projection/planner | `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` |
| G3 V2 exact-control core | `a626a726b2837c6dd8cfacd6d7be3b796278b127ea998ff3a3b894bbf7d69823` |
| G3 V2 captured runner | `d759cb7fa395646d435bdd0af220a098d7d1e908970a30c4f17fc9e391c296e8` |
| G3 V2 one-shot launcher | `3f6fedf1614e01770fa080e870730da32864c65e5fc9e2bae12abdc52d79bad3` |

These values exactly match the implementation handoff. The launcher identity
was hashed as a file only; it was not opened or executed.

## First-principles findings

### Lattice ownership and conversion

- Configuration components, frontiers, routes, goals, and `(cell, yaw, step)`
  view history remain on the `0.10 m` configuration lattice.
- Physical occupancy classes, ray supercovers, visible cells, swept cells,
  entropy, and discovery opportunity remain on the `0.05 m` physical lattice.
- Both frames must carry one exact shared world boundary origin, distinct frame
  identities, exact frozen cell sizes, and shapes in a 2:1 ratio on both axes.
- A configuration centre is converted through its configuration frame to world
  metres and then through the physical frame. The high-index translated test
  proves configuration `(30, 35)` becomes continuous physical-grid coordinate
  `(61, 71)`, not `(30, 35)`.
- A route step is exactly `0.10 m`; an adjacent physical-cell step is exactly
  `0.05 m`. No route index is admitted as physical visibility evidence.

### Live G3 authority and invalidation

- Candidate generation obtains the complete current exact-live G3 V2 connected
  component and its deterministic frontier artifact. Every retained route is a
  current exact-live G3 V2 A* path through confirmed configuration FREE.
- Validation delegates component, frontier, and path checks back to their
  issuing G3 planner. Hash equality alone is insufficient.
- State, candidate set, and candidate bindings cover both frame identities,
  both revisions, memory configuration, physical content, projection source,
  profile, FREE support, OCCUPIED support, both shapes, and the physical view
  state. The view state in turn binds the current view-memory identity.
- Exact-object registries reject copy, deep-copy, reconstruction, replay, and
  foreign-issuer objects. Physical evidence change, view-history change, or a
  new configuration projection invalidates the old chain before prediction,
  scoring, selection, or executable-path validation.

### UNKNOWN, occupancy, and target boundary

- Candidate goals are selected only from the current confirmed configuration-
  FREE component. Configuration UNKNOWN and OCCUPIED cells cannot become a
  route goal or path cell.
- Physical ray groups fail closed at missing-domain or OCCUPIED cells before
  admitting them. The first UNKNOWN group is visible/countable and terminates
  the ray; no cell behind it contributes sweep, entropy, or discovery gain.
- G4 V2 accepts no external beacon/target cell. It therefore cannot reinterpret
  an unknown or high-index target as a configuration route goal. Converting a
  claimed beacon target into a route is a downstream target-router concern,
  outside this exploration-only unit and outside this review. No target success
  or promotion claim is made here.

### Determinism, scoring, and promotion boundary

- Spatially diverse goals, 16 yaw candidates per goal, candidate ordering,
  physical ray sampling, gain terms, score ordering, tie-breaking, and final
  selection are deterministic for one bound state.
- Coverage, entropy, and discovery terms use physical cells; route and turn
  costs use configuration paths. Score-cache values are integrity checked and
  a mutated score fails closed.
- Development view recording keeps physical swept cells separate from
  configuration view history and rejects duplicate observation identities.
  It is unavailable when physical memory is marked promoted.
- `production_promotion_authorized` is unconditionally false. The source has no
  learned-head, checkpoint, result, held-out, G5, V5, or GPU dependency.

## Independent verification

Every command used this prefix:

```text
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3:/home/andrewknowles/Workspace/LeWMQuad-v3/lewm_worlds:/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages
```

Commands and outcomes:

```text
python3 -m py_compile lewm/planning/two_resolution_frontier_viewpoint_v2.py lewm/tests/test_two_resolution_frontier_viewpoint_v2.py
# PASS

python3 -m pytest -q lewm/tests/test_two_resolution_frontier_viewpoint_v2.py
# 8 passed in 50.77s

python3 -m pytest -q lewm/tests/test_frontier_viewpoint_information_gain.py
# 11 passed in 18.90s

python3 -m pytest -q lewm/tests/test_go2_g3_exact_physical_equivalence_v2.py
# 5 passed in 5.33s

python3 -m pytest -q lewm/tests/test_two_resolution_configuration_projection_v2.py
# 14 passed in 38.89s
```

The legacy G4 and first-principles G3 V2 adjacent total is **30/30 passed**.
The legacy G4 and exact-equivalence shards were run concurrently; each process
retained the one-thread numerical-library caps and blank GPU visibility.

## Verdict

**PASS.** The reviewed bytes close the G4 two-resolution source gap for
synthetic navigation integration: safe route/history ownership remains at
`0.10 m`, visibility and information gain remain at `0.05 m`, conversion is
world-frame explicit, and all live authority chains fail closed under stale or
replayed inputs. This verdict does not authorize learned G4 evidence, a scene
result, held-out evaluation, target routing, or production promotion.
