# JEPA Local Waypoint Planning Cost Qualification V1

## Bindings and custody

Source-freeze commit: `b06905c2724a1ecb825db25b54ec1b3a43336bbf`; seed `2026080901`. Development-only and non-claim-bearing.

Genesis, rsl_rl, and tensordict are closed by exact dist-info RECORD plus every listed present-file digest. Torch, NumPy, SciPy, Pillow, and PyYAML are bound to the exact interpreter, version, package root, and live import resolution; residual same-version mutation in those foundational packages is explicitly not byte-closed.

## Goal-view and predictor input reconstruction

`{"candidate_evidence_rows": 1728, "candidates": 576, "current_authority_payload_copies": 48, "current_blocks": 432, "current_token_reencodes": 0, "latent_tensors": 5424, "new_encoder_batches": 9, "new_encoder_frames": 144, "oracle_fanout_blocks": 4320, "oracle_fanout_physics_frames": 1080000, "paired_effect_evidence_rows": 32, "physics_frames": 1080000, "predicted_tensors": 3456, "reconstruction_physics_frames": 480000, "reconstruction_prefix_blocks": 1920, "selection_evidence_rows": 720, "snapshot_reproductions": 48, "states": 48, "successor_blocks": 3888, "total_oracle_blocks": 4320, "total_simulator_blocks": 6240, "total_simulator_physics_frames": 1560000}`

All 48 states use replay boundaries 38/39/40, source-frame offsets -480/-240/0, command-tick offsets -10/-5/0, and one candidate-independent path[2] goal view. Current RGB and raw FP16 token authority reproduce exactly.

The `FROZEN_STATE_RECONSTRUCTION_REPLAY` invokes the production collector scheduler/RouteTeacher and frozen PPO for exactly 48×40 prefix blocks solely to reproduce the committed post-block-40 state identities. It performs no experimental candidate selection, executes zero JEPA-cost actions, and is not a navigation qualification.

## Goal-view amendment and virtual-pose limitation

Goal-cell classification: `{"beacon_endpoint": 13, "endpoint_reachable": 48, "low_clearance_transit_blocked": 1, "nav_blocked": 14, "states": 48, "unblocked": 34}`.

Pose semantics: `{"block_classification_ids": ["UNBLOCKED", "BEACON_ENDPOINT", "LOW_CLEARANCE_TRANSIT_BLOCKED"], "block_classification_precedence": "BEACON_ENDPOINT when path[2] is a beacon cell; otherwise LOW_CLEARANCE_TRANSIT_BLOCKED when path[2] is nav-blocked; otherwise UNBLOCKED", "candidate_independent": true, "counterfactual_render_only": true, "endpoint_reachability_rule": "SceneGraph.bfs_distance(path[0], path[2], transit_blocked=nav_blocked_cells) is not None; a blocked goal may be reached as an endpoint but is never asserted free or transit-safe", "goal_render_semantics": "VIRTUAL_COUNTERFACTUAL_GOAL_VIEW_NOT_A_PHYSICALLY_EXECUTABLE_SENSOR_POSE", "historical_floor_plane_only_renderer_limitation_preserved": true, "physical_executability_claim": false, "physical_sensor_pose_claim": false, "pitch_rad": 0.0, "position_world_xy": "exact frozen SceneGraph cell_center(waypoint_path_cells[2])", "position_world_z": "exact frozen snapshot base z", "robot_reachability_or_stopping_claim": false, "roll_rad": 0.0, "yaw": "atan2 from cell_center(path[0]) to cell_center(path[1])"}`.

The exact path[2] cell centre is a candidate-independent virtual counterfactual render pose, not a physically executable robot or sensor pose. Nav-blocked cells remain valid reachable endpoints: 13 are beacon endpoints and one is low-clearance transit-blocked. No path[1] substitution, alternate standoff search, state drop, or prior failed-shard reuse occurred.

## Current-token authority amendment and BF16 cohort limitation

Amendment binding: `{"bytes": 11084, "content_digest": "3b426c945ac827a2e1e49cd312cd6d40d8af92b4cdec968afa867b3f51d907db", "path": "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_current_token_amendment_2026-08-26.json", "sha256": "2ccad35e809cccff52973ccddc6acf0f0a624173d71c0f7c00113d31bf7301cd"}`.

The GPU encoder newly processes exactly 144 frames in nine fixed batches of 16: 96 context slots 0/1 and 48 goal views. Each of the 48 frozen dense current FP16 authorities is byte-validated and copied once; CONTEXT horizon 0 and CURRENT are logical path/SHA/bytes aliases of that single copy. Current-token re-encodes are zero, the current RGB byte gate remains exact, and no scientific cost, gate, or classification changed.

BF16 cohort limitation: BF16 encoder output bytes can depend on device, runtime, kernels and complete batch cohort. The historical 7154-frame cohort/order and exact token bytes are persisted and bound, but bitwise re-execution equivalence under a changed cohort was never established. Re-encoding is unnecessary and is not an authority; the bound historical current payload is reused without numerical reinterpretation.

## GPU receipt serialization amendment and failed-attempt custody

Amendment binding: `{"bytes": 8678, "content_digest": "75be21487c3f051f0495ab53788c8b43e7e8fe91dfc513f226117ef9085a4def", "path": "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_gpu_receipt_serialization_amendment_2026-08-26.json", "sha256": "6970b4f4b238f04d217fbaad776627402a2d7c56e40f83ea2d0bd6d0aca5449d"}`.

The prior fresh run completed its nonreusable tensor materialisation but failed closed before publishing `receipts/gpu_inference.json`: `source_closure_binding.path` was a `pathlib.PosixPath`, which the canonical JSON serializer correctly rejected. The correction converts only that construction-site field with `str(...)`; the serializer remains strict, and no tensor, cost, metric, gate, classification, checkpoint, goal, candidate, or route-outcome rule changed. The full failed archive and its durable child error streams are byte-bound, no prior phase/shard/tensor/receipt is reused, and the corrected run is wholly fresh.

## GPU child receipt order amendment and failed-attempt custody

Amendment binding: `{"bytes": 9673, "content_digest": "4c708696bc9e30f26ebecb43fc2ca0350f8e89bef788f5b3d91aafa0bfddce9c", "path": "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_gpu_child_receipt_order_amendment_2026-08-26.json", "sha256": "481bcc88876267261fcff0e794b95330222ba18b32a0d3ada4311d063f20904b"}`.

The prior fresh run completed all scientific phases and created hidden result, persistence, and report artifacts, but failed closed before canonical publication because canonical JSON reload sorted the `gpu_child_execution_receipts` object keys. The corrected terminal validator treats JSON object member order as nonsemantic, requires the exact phase key set, and validates each frozen phase binding separately. The canonical serializer and all scientific tensors, costs, metrics, gates, classifications, checkpoints, goals, candidates, and route-outcome rules remain unchanged; the complete failed archive is byte-bound and none of it is reused.

## Markdown report order amendment and failed-attempt custody

Amendment binding: `{"bytes": 9970, "content_digest": "d64bc54f75b5305b5c70a0d354c9e66c6801a162afadfc090ef0c529272f2d44", "path": "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_markdown_report_order_amendment_2026-08-26.json", "sha256": "51023a3a3938d197a1ee6ca4e8f441fa7bebfbf6fcdd23fcaf27000d916bf7ae"}`.

The prior fresh run completed all scientific phases and created hidden result, persistence, and report artifacts, but failed closed before canonical publication because one Markdown JSON fragment inherited in-memory object insertion order while canonical result reload sorted object keys. The corrected renderer sorts only the `diagnostic_flags` fragment keys. Canonical result serialization and all scientific tensors, costs, metrics, gates, classifications, checkpoints, goals, candidates, and route-outcome rules remain unchanged; the complete failed archive is byte-bound and none of it is reused.

## Atomic relocation amendment and failed-attempt custody

Amendment binding: `{"bytes": 13241, "content_digest": "f4df32419c5e9d45d5dc631db02dbb98cd18dba345fb5959160dc3eff168b10b", "path": "docs/lewm_go2_jepa_local_waypoint_planning_cost_qualification_v1_atomic_relocation_amendment_2026-08-26.json", "sha256": "b00074759f68dc724b4defea8c5b76b068dae8fbfce718b6034924ab776f48e6"}`.

The prior fresh run passed the complete deep prepublication check and atomically renamed its hidden attempt to the canonical output path, but the postpublication validator incorrectly compared each immutable GPU child command origin with the renamed path. The corrected validator preserves the exact hidden command origin and accepts the canonical path only after an authorized same-parent, same-filesystem atomic relocation with the origin absent and target present. Child receipts are never rewritten; scientific tensors, costs, metrics, gates, classifications, checkpoints, goals, candidates, and route-outcome rules remain unchanged; the complete failed archive is byte-bound and none of it is reused.

## Historical renderer limitation

The frozen renderer receives `genesis_scene.json`, where structural geometry is under `objects`, while its historical builder reads only top-level `walls`, `obstacles`, and `landmarks`. Those keys are absent, so the effective rendered scene geometry is the textured floor plane only. This is preserved to require byte-identical current/true-future token compatibility; the experiment makes no explicit wall or landmark visual-reasoning claim.

Renderer limitation receipt: `{"builder_schema": "scripts.render_replay_v03.build_scene reads walls, obstacles and landmarks", "current_true_future_byte_compatibility_preserved": true, "effective_scene_geometry": "FLOOR_PLANE_ONLY", "explicit_wall_visual_reasoning_claim": false, "input_schema": "genesis_scene.json with structural geometry under objects", "interpretation": "HISTORICAL_RENDERER_LATENT_ROUTE_RANKING_ONLY", "structural_walls_obstacles_landmarks_rendered": false}`.

## Population and materialisation counts

States/candidates/tensors/rows: `{"candidate_evidence_rows": 1728, "candidates": 576, "current_authority_payload_copies": 48, "current_blocks": 432, "current_token_reencodes": 0, "latent_tensors": 5424, "new_encoder_batches": 9, "new_encoder_frames": 144, "oracle_fanout_blocks": 4320, "oracle_fanout_physics_frames": 1080000, "paired_effect_evidence_rows": 32, "physics_frames": 1080000, "predicted_tensors": 3456, "reconstruction_physics_frames": 480000, "reconstruction_prefix_blocks": 1920, "selection_evidence_rows": 720, "snapshot_reproductions": 48, "states": 48, "successor_blocks": 3888, "total_oracle_blocks": 4320, "total_simulator_blocks": 6240, "total_simulator_physics_frames": 1560000}`.

## Controller execution custody

The frozen PPO/controller executed 6,240 blocks (1,920 deterministic prefix-reconstruction blocks plus 4,320 fixed open-loop candidate/fanout blocks), totaling 1,560,000 physics frames. It was not trained or qualified here, made no experimental candidate selection, and no JEPA/MPC/navigation planner executed actions.

Controller custody receipt: `{"candidate_or_state_selection_changes": 0, "classification": "FROZEN_CONTROLLER_REPLAY_AND_FIXED_ORACLE_FANOUT_ONLY", "experimental_candidate_selecting_jepa_mpc_navigation_planner_executions": 0, "fixed_oracle_fanout_ppo_blocks": 4320, "fixed_oracle_fanout_ppo_physics_frames": 1080000, "frozen_ppo_controller_total_blocks": 6240, "frozen_ppo_controller_total_physics_frames": 1560000, "navigation_system_training_or_qualification": false, "reconstruction_route_teacher_ppo_blocks": 1920, "reconstruction_route_teacher_ppo_physics_frames": 480000, "reporting_rule": "disclose both positive frozen-controller uses and state that no navigation system was trained, evaluated or run as the experimental candidate-selecting planner; never claim that no controller or navigation code executed"}`.

## True-future and predicted ranking metrics

| source | population | n | pairwise | Spearman | regret | top-3 | progress ratio | contact | nonviable | stuck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TRUE_FUTURE_LATENT_COST | ALL_CANDIDATES | 8 | 0.575342 | 0.161713 | 0.448624 | 0.125 | 0.385008 | 2 | 1 | 4 |
| TRUE_FUTURE_LATENT_COST | ORACLE_CONTACT_FREE | 8 | 0.586777 | 0.096516 | 0.436775 | 0.25 | 0.396271 | 0 | 1 | 3 |
| TRUE_FUTURE_LATENT_COST | ORACLE_VIABILITY_ADMISSIBLE | 8 | 0.579096 | 0.0786588 | 0.380532 | 0.25 | 0.449231 | 0 | 0 | 4 |
| ONE_STEP_PREDICTED_LATENT_COST | ALL_CANDIDATES | 8 | 0.569472 | 0.174825 | 0.485309 | 0.125 | 0.269592 | 1 | 0 | 3 |
| ONE_STEP_PREDICTED_LATENT_COST | ORACLE_CONTACT_FREE | 8 | 0.578512 | 0.16546 | 0.502225 | 0.25 | 0.251398 | 0 | 1 | 2 |
| ONE_STEP_PREDICTED_LATENT_COST | ORACLE_VIABILITY_ADMISSIBLE | 8 | 0.573446 | 0.183317 | 0.445981 | 0.25 | 0.304358 | 0 | 0 | 3 |
| TWO_STEP_PREDICTED_LATENT_COST | ALL_CANDIDATES | 8 | 0.555773 | 0.127622 | 0.485309 | 0.125 | 0.269592 | 1 | 0 | 3 |
| TWO_STEP_PREDICTED_LATENT_COST | ORACLE_CONTACT_FREE | 8 | 0.553719 | 0.108483 | 0.502225 | 0.375 | 0.251398 | 0 | 1 | 2 |
| TWO_STEP_PREDICTED_LATENT_COST | ORACLE_VIABILITY_ADMISSIBLE | 8 | 0.548023 | 0.147769 | 0.445981 | 0.375 | 0.304358 | 0 | 0 | 3 |
| KINEMATIC_ROUTE_BASELINE | ALL_CANDIDATES | 8 | 0.851272 | 0.782343 | 0.0400781 | 1 | 0.965591 | 1 | 0 | 2 |
| KINEMATIC_ROUTE_BASELINE | ORACLE_CONTACT_FREE | 8 | 0.856749 | 0.732755 | 0.0892054 | 1 | 0.910758 | 0 | 0 | 3 |
| KINEMATIC_ROUTE_BASELINE | ORACLE_VIABILITY_ADMISSIBLE | 8 | 0.858757 | 0.714898 | 0.0892054 | 1 | 0.910758 | 0 | 0 | 3 |
| RANDOM | ALL_CANDIDATES | 8 | 0.412916 | -0.271853 | 0.597826 | 0.125 | 0.0908375 | 2 | 2 | 4 |
| RANDOM | ORACLE_CONTACT_FREE | 8 | 0.421488 | -0.32157 | 0.630142 | 0.125 | 0.129473 | 0 | 0 | 3 |
| RANDOM | ORACLE_VIABILITY_ADMISSIBLE | 8 | 0.420904 | -0.303713 | 0.630142 | 0.25 | 0.129473 | 0 | 0 | 3 |

## Per-family results and collapse audit

| source | family | n | pairwise | regret | top-3 | progress m | collapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRUE_FUTURE_LATENT_COST | large_enclosed_maze | 2 | 0.362069 | 0.755996 | 0 | -0.134086 | PASS |
| TRUE_FUTURE_LATENT_COST | medium_enclosed_maze | 2 | 0.651376 | 0.333428 | 0 | 0.515063 | FAIL |
| TRUE_FUTURE_LATENT_COST | small_enclosed_maze | 2 | 0.661417 | 0.345033 | 0.5 | 0.467932 | FAIL |
| TRUE_FUTURE_LATENT_COST | loop_alias_stress | 2 | 0.483333 | 0.0876683 | 0.5 | -0.00369391 | FAIL |
| ONE_STEP_PREDICTED_LATENT_COST | large_enclosed_maze | 2 | 0.448276 | 0.56212 | 0 | -0.0768105 | PASS |
| ONE_STEP_PREDICTED_LATENT_COST | medium_enclosed_maze | 2 | 0.697248 | 0.469003 | 0 | 0.360919 | FAIL |
| ONE_STEP_PREDICTED_LATENT_COST | small_enclosed_maze | 2 | 0.527559 | 0.411171 | 0.5 | 0.398056 | FAIL |
| ONE_STEP_PREDICTED_LATENT_COST | loop_alias_stress | 2 | 0.566667 | 0.341632 | 0.5 | -0.109523 | FAIL |
| TWO_STEP_PREDICTED_LATENT_COST | large_enclosed_maze | 2 | 0.517241 | 0.56212 | 0.5 | -0.0768105 | FAIL |
| TWO_STEP_PREDICTED_LATENT_COST | medium_enclosed_maze | 2 | 0.605505 | 0.469003 | 0 | 0.360919 | FAIL |
| TWO_STEP_PREDICTED_LATENT_COST | small_enclosed_maze | 2 | 0.519685 | 0.411171 | 0.5 | 0.398056 | FAIL |
| TWO_STEP_PREDICTED_LATENT_COST | loop_alias_stress | 2 | 0.533333 | 0.341632 | 0.5 | -0.109523 | FAIL |
| KINEMATIC_ROUTE_BASELINE | large_enclosed_maze | 2 | 0.603448 | 0.196509 | 1 | 0.0442306 | FAIL |
| KINEMATIC_ROUTE_BASELINE | medium_enclosed_maze | 2 | 0.926606 | 0 | 1 | 0.90918 | FAIL |
| KINEMATIC_ROUTE_BASELINE | small_enclosed_maze | 2 | 0.944882 | 0 | 1 | 0.794121 | FAIL |
| KINEMATIC_ROUTE_BASELINE | loop_alias_stress | 2 | 0.8 | 0.160312 | 1 | -0.0339654 | FAIL |
| RANDOM | large_enclosed_maze | 2 | 0.344828 | 0.806124 | 0 | -0.202349 | PASS |
| RANDOM | medium_enclosed_maze | 2 | 0.394495 | 0.424268 | 0 | 0.407601 | FAIL |
| RANDOM | small_enclosed_maze | 2 | 0.480315 | 0.435247 | 0.5 | 0.374864 | FAIL |
| RANDOM | loop_alias_stress | 2 | 0.416667 | 0.854927 | 0.5 | -0.336516 | FAIL |

## Selected route outcomes

The held-out table reports selected immediate-contact, successor-nonviable and stuck decisions for every source/comparator and population; all 720 selections are persisted row-wise.

## Paired comparisons and descriptive bootstrap

- `TWO_STEP_MINUS_ONE_STEP`: progress 0 m (95% 0..0); regret improvement 0 (95% 0..0); material=FAIL.
- `TWO_STEP_MINUS_KINEMATIC`: progress -0.142616 m (95% -0.246159..-0.0509431); regret improvement -0.356776 (95% -0.549648..-0.169972); material=FAIL.
- `TWO_STEP_MINUS_TRUE_FUTURE`: progress -0.0340718 m (95% -0.0937207..0.0233936); regret improvement -0.0654499 (95% -0.248317..0.11893); material=FAIL.
- `TRUE_FUTURE_MINUS_KINEMATIC`: progress -0.108544 m (95% -0.176382..-0.040744); regret improvement -0.291326 (95% -0.538785..-0.0818919); material=FAIL.

## Latent progression and all-candidate tendencies

- `TRUE_FUTURE`: endpoint current→H3 nonincrease=0.420139; all-adjacent-step monotonic=0.0451389; contact down-ranking=0.489524; successor-nonviable=0.542576; stuck=0.571004; no-progress=0.413793.
- `ONE_STEP_PREDICTED`: endpoint current→H3 nonincrease=0.951389; all-adjacent-step monotonic=0.269097; contact down-ranking=0.624762; successor-nonviable=0.628821; stuck=0.594796; no-progress=0.434077.
- `TWO_STEP_PREDICTED`: endpoint current→H3 nonincrease=0.949653; all-adjacent-step monotonic=0.310764; contact down-ranking=0.590476; successor-nonviable=0.622271; stuck=0.588848; no-progress=0.432049.

## Gates and classifications

True-future gate `FAIL`; complete two-step gate `FAIL`.
Primary `RAW_LATENT_GOAL_COST_NO_GO`; secondary `[]`; diagnostics `{"ONE_STEP_BASE_SCREEN_ONLY": false, "both_predicted_base_screens_failed": true, "two_step_base_screen_passed_but_full_gate_failed": false}`.
Next experiment: `PLAN_AWARE_MONOTONE_JEPA_COST_V1`.

## Requirements boundary and next decision

Deployment hard-contact requirements, consequences and recovery criteria remain unresolved. No further deployment-safety scope reduction, sensor qualification or learned hard-safety model is authorised.

## Runtime, storage and prohibitions

Runtime `{"cpu_materialization": 160.47972559928894, "evaluation_reduction": 35.41194033622742, "gpu_materialization": 89.44607734680176, "preflight": 24.209078788757324, "terminal_validation_excluded_from_embedded_total": true, "total": 309.54682207107544}`; storage `{"bytes": 8488700226, "files": 5736, "final_ceiling_bytes": 12000000000, "gb_decimal": 8.488700226, "peak_rss_bytes": 8187883520, "peak_vram_bytes": 2350332416, "within_final_ceiling": true}`.

The frozen PPO/controller executed 1,920 deterministic prefix-reconstruction blocks and 4,320 fixed open-loop candidate/fanout physics blocks. No experimental candidate-selecting JEPA, MPC, or navigation planner was executed or evaluated. No training, optimizer, fresh panel, G2, memory, novelty, routing, or beacon capture was executed. Future targets and oracle viability are evaluation-only and unavailable before action execution.
