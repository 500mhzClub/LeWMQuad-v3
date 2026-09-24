# Go2 DINO true-successor goal-cost V1 preregistration

Date: 2026-08-04

Status: protocol fixed before any claim-bearing result. This is a
development-only target/readout ceiling, not learned-model evaluation,
held-out evaluation, or deployment evidence.

## Why this precedes training

The planner-oracle assay established substantial H1 action-ranking headroom,
but it did not qualify a visual terminal cost. The repository has also already
shown that a train-only relational readout of actual frozen-DINO successors did
not reliably beat task/action-only or relational-persistence controls. A
predictor cannot make its target space more useful than the actual target under
the same downstream cost.

This assay therefore gives the proposed planner the actual next DINOv2 grid for
every candidate and applies exactly the proposed goal cost. If that privileged
ceiling fails, no dense-DINO temporal predictor will be trained for this route.

## Fixed scope

- Corpus: the same 24 sorted scenes in
  `.generated/scene_corpus/go2_generalization_v4/development`, family
  `go2_deployment_medium_maze`, offset 0, one trial per scene. V4 is
  development-only; the legacy internal `candidate` label grants no held-out
  status.
- Task: local `visible-beacon`, one goal view, approach distance 1.2 m,
  standoff 0.85 m, yaw jitter 0.7 rad, goal radius 0.35 m.
- Execution: kinematic CPU, H1, 12-block cap, seed 7, 0.05 m occupancy cells
  and 0.20 m inflation.
- Ordered candidates: `hold`, `forward_slow`, `forward_medium`,
  `forward_fast`, `arc_left`, `arc_right`, `yaw_left`, `yaw_right`, `backward`.
- Frozen encoder: local `dinov2_vits14`, repository commit
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- DINO checkpoint:
  `/home/andrewknowles/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth`,
  88,283,115 bytes, SHA-256
  `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- Encoder is frozen, in evaluation mode, and called under `no_grad` on the
  discrete R9700 GPU. No feature cache is written.
- Preprocessing: exact 224x224 RGB, ImageNet normalization, DINO patch output
  `[256,384]`, and per-token L2 normalization.
- Cost: mean same-position patch cosine distance,
  `mean_j(1 - dot(z_successor[j], z_goal[j]))`. No learned readout, matching,
  semantic label, pose, or physical outcome enters this cost.

## Arms

1. `oracle_mpc`: privileged geometric endpoint-distance positive control.
2. `dino_true_successor`: reset to the observed state for every candidate,
   execute that nominal H1 candidate, render its actual successor RGB, restore
   the observed state, batch-encode all successors, and select the minimum DINO
   goal cost.
3. `dino_true_successor_shuffled`: identical DINO score multiset assigned by
   the already frozen deterministic score permutation to unchanged candidate
   rows.
4. `dino_persistence`: repeat the current DINO grid for every candidate. Equal
   costs resolve by the registered candidate order.
5. `bearing`: privileged task/geometric ceiling.
6. `hold` and `random`: negative controls.

The true-successor arms are privileged simulator counterfactual ceilings. They
are not deployable policies and make no self-supervision claim.

## Analysis

All seven arms must be present for all 24 scenes and no scene may be skipped.
Per-policy decision regret is first averaged within scene; comparisons then use
paired scene means. Closed-loop progress/final distance is already one value per
scene. Use 10,000 whole-scene bootstrap resamples, seed 2026080402, and
2.5/97.5 percentiles. Per-tick pseudo-replication is forbidden.

Primary contrasts are `dino_true_successor` against
`dino_true_successor_shuffled` and `dino_persistence`:

- paired closed-loop progress advantage (higher is better);
- paired mean geometric-oracle first-action-regret difference (lower is
  better).

Success and path efficiency are secondary. Geometric oracle and bearing are
ceilings, not superiority targets. Kinematic fall rate is not safety evidence.

## Fixed gate

The exact DINO target/cost earns predictor training only if every item passes:

1. 24 complete seven-arm scenes, no skips, exact source/checkpoint/cost
   provenance, finite scores, and geometric `oracle_mpc` regret at or below its
   fixed tolerance.
2. Against shuffled DINO scores, true-successor DINO improves mean progress by
   at least 0.10 m with bootstrap lower 95% bound above 0.
3. Against shuffled DINO scores, true-successor DINO reduces scene-mean
   first-action regret by at least 0.02 m with the bootstrap upper 95% bound for
   `(true - shuffled)` below 0.
4. Against DINO persistence, true-successor DINO has positive progress and
   lower scene-mean first-action regret, with both paired bootstrap intervals
   excluding 0 in the favorable direction.
5. True-successor DINO mean progress exceeds random and hold.

Beating only hold/random is insufficient. Beating bearing or the geometric
oracle is neither expected nor required.

## Smoke and stopping rule

A two-scene run may check only plumbing, exact resets, finite DINO scores,
candidate order, and intervention nonidentity. Its effects cannot alter this
gate. After a clean smoke, run the fixed 24-scene panel once.

- Pass: freeze the already implemented dense temporal predictor/runner and run
  its separately preregistered H6 association/retention experiment.
- Fail: stop this frozen-DINO target plus same-patch goal-cost route. Do not
  tune the cost, retry a seed, train the predictor, or reinterpret token
  prediction accuracy as planning evidence. A nonlinear task-coupled readout,
  embodiment-supervised target, or dense V-JEPA successor is materially new
  work.

## Freeze record

Completed after source review and before the 24-scene command:

- benchmark source SHA-256:
  `c926d54c81bfdf149c1d79baf41b78c1bd05f206414536827e9ddc2d5603d52f`
- analyzer source SHA-256:
  `7fa397cce5d0c76e4d2cf8a6203a34fb528683de1e78d97adc20b587238ccfae`
- focused benchmark-test SHA-256:
  `3f5c700a577afe8ac110b9157d1734b2b3bc3788b7ce62ed36f4a48c4accc2e3`
- focused analyzer-test SHA-256:
  `f1c51ce7b0fb864cc6392a2abacc4e1d292e6fb2d7847a19348c8dc9ceb1f418`
- focused combined validation: `47 passed in 1.40s`; compilation and
  `git diff --check` clean
- reviewed smoke:
  `.generated/dino_true_successor_goal_cost_v1/smoke_development_2scene_h1_seed7.json`,
  SHA-256
  `5ded64be73442584bed0864378b565516f43fff397c29a0a142e1facb2b508d7`
- smoke audit: two scenes, fourteen result rows, no skips, exact DINO/ROCm
  provenance, finite nine-candidate score vectors, identity true-successor
  mappings, valid non-identity shuffled mappings, exact persistence ties with
  registered-row-zero selection, zero oracle regret, and identical initial
  state across arms
