# Mass Data-Gen Readiness TODO

Purpose: track the work required before using Genesis-generated quadruped
rollouts as mass training data for a LeWM-style JEPA and downstream maze
solving.

Current status: **not ready for mass training**. The route-teacher probe is
improved (`247/312` on the 3-scene-per-family gate), but corpus validity and
vision integrity blockers remain.

## P0 Blockers

- [ ] Regenerate/reroll the production corpus until `audit_scene_corpus.py`
      reports `0 failure(s)`.
  - Current failed corpus: `.generated/scene_corpus/production_20260518T231208Z`
  - Observed failures: `7` scenes with `no_component_contains_all_beacons`,
    concentrated in `visual_sensor_stress`.
  - Acceptance:
    ```bash
    PYTHONPATH=lewm_genesis:lewm_worlds python3 scripts/audit_scene_corpus.py \
      .generated/scene_corpus/<new_production>
    ```

- [ ] Do not use inline multi-env RGB for training until per-env camera
      correspondence is fixed.
  - Risk: `RolloutRunner._render_and_emit_rgb` sets a camera pose from env 0,
    so batched rollouts can produce missing or non-corresponding RGB for envs
    1..N.
  - Interim production path: run bulk rollout with `--no-rgb`, convert with
    `raw_training`, then render every env via render-replay.
  - Acceptance: multi-env RGB smoke where env 0 and env 1 have distinct poses
    and distinct, pose-correct frames.

- [x] Remove default privileged label leakage from render-replay RGB.
  - Fixed: target-route text is opt-in via `--overlay-target-label`.
  - Acceptance: training renders have no text overlay by default; any target
    overlay is opt-in for QA only, while `frames_rendered.jsonl` retains the
    label metadata.

- [x] Update the mass data-gen runbook to prefer MCAP-only rollout plus
      render-replay over inline multi-env RGB.

## P1 Training Corpus Requirements

- [ ] Scale beyond diagnostic corpus size.
  - Minimum serious split: `1000 train / 150 val / 150 test_id / 150 test_hard`.
  - Preferred split: `2400 train / 300 val / 300 test_id / 300 test_hard`.
  - Current production probe has `200` train scenes and is diagnostic only.

- [ ] Use the spec collection mix for the main JEPA corpus:
  - `route_teacher=0.30`
  - `frontier=0.20`
  - `primitive_curriculum=0.20`
  - `ou_noise=0.10`
  - `recovery=0.10`
  - `loop_revisit=0.10`
  - Avoid using the route-heavy probe mix as the only training data.

- [ ] Produce optional balanced shards:
  - `success_route`: route-heavy expert success trajectories.
  - `explore_memory`: frontier and loop-revisit heavy.
  - `action_support`: primitive curriculum and OU heavy.
  - `contact_recovery`: recovery/contact heavy.

- [ ] Gate every raw rollout with `raw_training`.
  - Acceptance: contract audit pass, no missing command/executed/reset streams,
    no timestamp regressions, and critical topic rates/gaps within policy.

- [ ] Gate every rendered rollout with render-quality checks.
  - Acceptance: `invalid_frame_count == 0` or below an explicitly approved
    threshold; `camera_safety_unresolved_count == 0`; no QA overlays in
    training frames.

- [ ] Run derived-label generation for every raw rollout.
  - Required labels: `cell_id`, `yaw_bin`, `local_graph_type`, clearance,
    traversability, landmark visibility/bearing/range, integrated body motion,
    command context, and reset-safe episode ids.

## P2 JEPA/H-JEPA Audits

- [ ] Publish effective LeWM sequence counts after reset/window filtering.
- [ ] Publish per-family, per-primitive, and per-command histograms.
- [ ] Publish local-graph-type coverage.
- [ ] Publish same-place positive and same-scene hard-negative pair counts.
- [ ] Publish loop-closure and false-loop pair counts.
- [ ] Publish goal-image observations per landmark and yaw bin.
- [ ] Audit relaxed route-teacher claims separately from rendered landmark
      visibility labels.

## Route-Teacher Patch Follow-Up

- [ ] Keep the `small_enclosed_maze` arrival relaxation as a collection
      heuristic only.
- [ ] Add telemetry for relaxed claims:
  - `relaxed_claim_count`
  - relaxed claim family/scene/env
  - whether actual derived/rendered landmark visibility passed
- [ ] Do not broaden relaxation to `medium_enclosed_maze` or
      `local_composite_motifs`; prior sweeps regressed those families.

## Suggested Execution Order

1. Fix/reroll corpus validation failures.
2. Run one full smoke suite over the new corpus.
3. Generate a small pilot shard with the spec mix.
4. Convert, render, derive labels, and run all quality audits.
5. Only then scale to minimum or full training corpus size.
