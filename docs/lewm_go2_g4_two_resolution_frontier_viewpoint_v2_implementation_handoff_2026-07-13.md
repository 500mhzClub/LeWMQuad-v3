# Go2 G4 V2 two-resolution frontier/viewpoint implementation handoff

Date: 2026-07-13

Status: **candidate source complete; requires different-agent review**

## Scope

This additive candidate implements the preregistered two-resolution G4 unit
without modifying the legacy same-grid G4 module or any frozen G3 V2 source.
It consumes only current exact-live G3 V2 snapshots, projection, planner,
components, frontiers, paths, and their bound physical memory.

The candidate provides:

- an exact-live physical view-state issuer with physical `0.05 m` FREE,
  OCCUPIED, UNKNOWN, sweep, entropy, and discovery maps;
- configuration `0.10 m` `(cell, yaw, step)` view history;
- shared-origin configuration-centre -> world -> physical-grid conversion;
- deterministic complete-component/frontier viewpoint candidates over 16
  headings, retaining exact-live G3 V2 components, frontier artifacts, and A*
  paths;
- configuration route costs at `0.10 m` per step and physical closed-cell ray
  traversal at `0.05 m` per cell;
- conservative missing/OCCUPIED/first-UNKNOWN ray occlusion;
- separate physical coverage, entropy, and discovery gain, deterministic
  scoring/selection, and fail-closed score-cache integrity;
- exact-object and full frame/revision/profile/support revalidation at every
  state, candidate-set, candidate, prediction, score, selection, and execution
  boundary.

Development view recording remains unavailable for promoted physical memory.
No qualified production camera-view receipt, learned head, G4 result, or
promotion is claimed.

## Candidate identities

| Artifact | SHA-256 |
|---|---|
| `lewm/planning/two_resolution_frontier_viewpoint_v2.py` | `5c84e79e558f51b75b00cf2baa26d7860302d6e3912ac14432dfc010efdc4f82` |
| `lewm/tests/test_two_resolution_frontier_viewpoint_v2.py` | `c50e0d26be068228fe33530d3b2fa42b7520d20d93a0f6e7dc35a6c567ef963e` |
| governing design | `de6cb956d97b9187281da948abcf700904969c3f91486e0c5390024fdd4ddc7f` |

Frozen dependency/legacy identities remained:

| Artifact | SHA-256 |
|---|---|
| legacy G4 source | `2ef20e8213a384e0f514705ca14c058eb7fbd81dcc4f6a53407414c1ba79e08e` |
| legacy G4 tests | `02d5a0b0459f6fde43e046b2b9f86d13d21e7392119b57626f0a398ce4c5241e` |
| G3 V2 projection/planner | `3c858a89170f78a73f401c9534e231f24d6d91bb0469ea95eb00002158146107` |
| G3 V2 exact-control core | `a626a726b2837c6dd8cfacd6d7be3b796278b127ea998ff3a3b894bbf7d69823` |
| G3 V2 captured runner | `d759cb7fa395646d435bdd0af220a098d7d1e908970a30c4f17fc9e391c296e8` |
| G3 V2 one-shot launcher | `3f6fedf1614e01770fa080e870730da32864c65e5fc9e2bae12abdc52d79bad3` |

## Verification

All commands used
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
with `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` empty.

- focused G4 V2 synthetic suite: **8/8 passed** in `50.45 s`;
- legacy G4 plus frozen G3 V2 adjacent suite: **30/30 passed** in `64.04 s`;
- the candidate source and test compiled;
- high translated indices proved configuration `(30,35)` maps through world
  coordinates to continuous physical-grid boundary `(61,71)`;
- explicit adjacent route and physical steps measured `0.10 m` and `0.05 m`;
- wrong/foreign frame, changed origin, support, physical revision,
  configuration revision, copy, deep-copy, reconstruction/replay, view-stale,
  and reprojection-stale cases rejected;
- legacy G4 and frozen G3 V2 hashes stayed byte-identical.

This workstream did not invoke or inspect an authoritative audit, scene result,
dataset, checkpoint, model, RGB input, GPU, G5, V5, or held-out input. A
canonical G3 V2 result path created by a separate workstream was present at
closeout and was not read or changed here.

Different-agent source and adversarial review is required before this unit can
be treated as a navigation dependency. Learned G4 training and any result
remain unauthorized.
