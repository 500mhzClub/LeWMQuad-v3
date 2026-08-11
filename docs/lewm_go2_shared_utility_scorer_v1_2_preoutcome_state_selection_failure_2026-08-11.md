# Shared utility scorer V1.2: pre-outcome state-selection failure

Status: `BLOCKED_PRE_OUTCOME_SCIENTIFIC_CONTRACT_INFEASIBILITY`

This is not a failed scorer result. No scorer was trained, qualified, or applied. The frozen counterfactual predictor-qualification result at `ee47b47e7964c16360f265c4cfbe7f8181d16402` remains unchanged.

## Prospective allocation amendment

The earlier candidate-allocation contradiction at commit `6a4d6a66c93d9461bdfb8bf4c2ccb5b882dcdb78` was corrected prospectively, before scientific operations. Its receipt remains preserved with digest `550c52f9a3ff04f8a564f6f28e75e9d36fc8bc0f73da4795b95dedc3ad2e3cab` and raw SHA-256 `3e224158d43a4e75fc7a60436feaeb00cd538a5fabfae5a92983f7ede612df99`.

The narrow amendment digest is `4dde3562cdd9e503d6e264a5d4982a189a9f43d338c3d6b87ee20de352bc3cbc`. Its deterministic pre-identity validation passed:

- 120 six-candidate subsets and 720 assignments;
- every candidate exactly 60 times;
- every candidate 48 times in fitting and 12 in calibration;
- every candidate 20 times in each state stratum;
- candidate 10 in 60 distinct subsets, with the other 60 subsets legitimately lacking reverse behavior;
- every subset contains forward and turning behavior;
- each family count is seven or eight per candidate.

The amended source was committed cleanly at `38e7fc84b83d815311ead732afa138c2179ccb11`. The corrected scorer-contract digest is `06263907d8f8df0fe735f95da26c10fab9dff4af6827562622aa66463b456c0b`; its artifact digest is `116a7e77a7888788048a9fddcb3b7a1eaf62ea655890503ea09e08ebc91b898d`. The clean launch receipt digest is `7ab90a7fc6cdde04a0982701b008bc9d00b47ea8c0baecf47f775dcef6d64520`.

The previous self-verifying but ineligible `0fc7a3db0ca86ae206050ee6da2894208fa11707e840b112a8a6810e18ac3e21` contract was preserved under `superseded_pre_run/`. The corrected contract binds the abandoned-45 scene-list digest `5d5c4fef96e5132ad443c4fbd2778ad7d13fb9190328a498ca56490d53e041fe` and exclusion digest `6d644c34b822fb5fb8e30906875047d1677aa730c2db584470cabdbe8bf6abc3`.

## Blocking selection-contract defect

The frozen scorer-fit contract requires five completion-enriched states in every family. Completion enrichment requires snapshot-time metric geodesic distance at most 0.75 m and absolute goal bearing at most 75 degrees. The same selector requires the bound landmark to have graph hop distance at least one.

That conjunction is unattainable in `rough_local_dynamics` and `open_obstacle_field`:

1. `SceneGraph.locate` assigns a pose to the nearest graph-node centre.
2. In every allowed scene in both affected families, a landmark goal node is 2.4 m from the nearest distinct graph node.
3. A pose assigned to an admissible non-goal node is therefore at least 1.2 m from the goal-node centre.
4. The frozen 0.75 m completion threshold is below that 1.2 m lower bound.
5. A pose assigned to the goal cell has hop distance zero, so that landmark is excluded and cannot provide a completion-enriched state.

The exact post-exclusion rough pool contains 66 scenes and 132 landmark goals; static qualifying goals are 0/132. The runtime selector correspondingly stopped at `completion_enriched: [0, 5]`. The rough-family allow-list digest is `7c3d4cd719a1e847b06a3a922cbcc042790fe82d931b2602f484ab5f1a2e736a`; the complete eight-family allow-list digest is `ed2f30caf302061d9560e7dca6df7bf63331883fd9336e1cfae22ff72fb5b86f`.

The exact post-exclusion open-field pool contains 116 scenes and 232 landmark goals; static qualifying goals are 0/232. The other six families have geometric lower bounds from 0.400 to 0.525 m and are not subject to this particular impossibility.

Changing hop eligibility, the 0.75 m threshold, the geodesic definition, family quotas, strata, or scene families would change the frozen scientific selection contract. Reordering the stratum allocator cannot overcome the geometric lower bound. Therefore there is no authorized contract-preserving implementation repair in this pass.

## Durable work and recovery

Three outcome-free identity shards completed before the defect was established:

| Family | States | Fit/calibration | Per stratum | Shard digest | Raw SHA-256 |
|---|---:|---:|---:|---|---|
| large_enclosed_maze | 15 | 12/3 | 5 | `3b1cabb8cf104f0edc19e27d9f11922655a5424b32a4ad41ec3d5466b4193914` | `289f63e5a607fb49c62a996043019ca8b6ef2def31ae39afc0a87ddbca71e866` |
| local_composite_motifs | 15 | 12/3 | 5 | `e066820d0a85e53a8b7e30a4b4fe1386df3e315674ebd094b0beb64c74242321` | `eca93582615e946b0a3c4692dc9353cb826f6e3d4b5e6ccd2c706ee9f53ee0dd` |
| loop_alias_stress | 15 | 12/3 | 5 | `9aee1729903e555273a792c03a5290f7547645995164042f9f021fec48de3985` | `9b7d7ee1f0029b3cd64eb3127687ba03e67eab4c75e4c4600a375afbf721acf6` |

The complete 120-state identity manifest was never issued, so no branch identities exist. `medium_enclosed_maze` ended in a native Genesis SIGSEGV, and `small_enclosed_maze` and `open_obstacle_field` experienced infrastructure interruptions before atomic shard issuance. `visual_sensor_stress` was not launched after its preceding sequential command was interrupted. None produced a durable shard. These interruptions are not the stopping reason; the independently proven rough/open contract impossibility is.

The invalid interrupted 45-state attempt remains preserved and excluded. None of its identities entered this run.

## Scientific operation counts

- Durable pre-outcome states: 45, in three family shards.
- Complete 120-state manifest: 0.
- Branch identities: 0.
- Attempted branches or inspected outcomes: 0.
- Rendered frames or encoded target latents: 0.
- Scorer training, checkpoints, or qualification runs: 0.
- Predictor checkpoints opened: 0.
- Development-transfer runs: 0.
- Final-evaluation states or branches: 0.

No predictor-qualification computation was rerun, no utility scorer was trained or invoked, and no final 200-state corpus was generated. Nothing was running when this report was issued.

The machine-readable receipt is `docs/lewm_go2_shared_utility_scorer_v1_2_preoutcome_state_selection_failure_2026-08-11.json`.
